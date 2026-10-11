"""
Etape 4 du curriculum : Global Reward Prediction (GRP) avec regles
Mahjong Soul completes.

[TONPUUSEN ONLY] Le modele est entraine sur des parties East-only
(4 manches max, sans enchousen). Pour rester coherent :
    - Le self-play worker utilise n_rounds=4
    - La progression du GRP est normalisee sur 8 (East 1-4 + repeats max)

Regles implementees :
    - Renchan, Honba (illimite)
    - Tobi : fin si un joueur passe sous 0 absolu
    - Agari Yame : all last + dealer 1er + gagne -> fin
    - Tenpai Yame : all last + dealer 1er + tenpai sur ryuukyoku -> fin
    - Oorasu : fin East 4 si quelqu'un >= 30k
    - Enchousen (Sud 1-4) : implemente mais DESACTIVE en pratique
      (le worker passe n_rounds=4, donc round_num ne depasse jamais 4)
"""
import numpy as np

from mahjong_ai.envs.multi_agent import MultiAgentMahjongEnv, REWARD_SCALE
from mahjong_ai.envs.tiles import shanten_of
from mahjong_ai.agents.reward_predictor import FEATURE_DIM


# ─── Constantes Mahjong Soul ────────────────────────────────────────────
N_ROUNDS_NORMAL = 4     # East 1-4 (fin normale)
N_ROUNDS_MAX = 8        # East 1-4 + Sud 1-4 (max enchousen)
MAX_KYOKU = 30          # securite anti-boucle

TARGET_SCORE = 5.0      # 30000 absolus - 25000 = +5.0 en milliers
TOBI_THRESHOLD = -25.0  # 0 absolu - 25000 = -25.0 en milliers

RANK_REWARDS = {1: 15.0, 2: 5.0, 3: -5.0, 4: -15.0}

# [FIX] Progression normalisee sur 8 (tonpuusen : East 1-4 + repeats)
N_ROUNDS_REF = 8


class PredictorHolder:
    def __init__(self):
        self.model = None


class GameLogBuffer:
    def __init__(self, max_size=20000):
        self.max_size = max_size
        self.sequences = []
        self.final_rewards = []

    def add(self, seq, reward):
        self.sequences.append(seq)
        self.final_rewards.append(reward)
        if len(self.sequences) > self.max_size:
            self.sequences.pop(0)
            self.final_rewards.pop(0)

    def __len__(self):
        return len(self.sequences)


def _rank_rewards_from_scores(scores):
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    rewards = {}
    for rank, player in enumerate(order, start=1):
        rewards[player] = RANK_REWARDS[rank]
    return rewards


def _decide_dealer_repeat(outcome, dealer_round_reward):
    """Retourne (dealer_repeats, honba_increments)."""
    if outcome.startswith("abortive"):
        return True, False
    if outcome == "exhaustive":
        repeats = dealer_round_reward >= 0
        return repeats, repeats
    repeats = dealer_round_reward > 0
    return repeats, repeats


class GameEnv:
    def __init__(self, seed=None, predictor_holder=None,
                 game_log_buffer=None, n_rounds=N_ROUNDS_MAX,
                 game_format=None):
        self.round_env = MultiAgentMahjongEnv(seed=seed)
        self.predictor_holder = predictor_holder or PredictorHolder()
        self.game_log_buffer = game_log_buffer
        self.n_rounds = n_rounds
        self.game_format = game_format or (
            "tonpuu" if n_rounds <= N_ROUNDS_NORMAL else "hanchan")
        self._reset_game_state()

    def _reset_game_state(self):
        self.round_num = 1
        self.honba = 0
        self.dealer_seat = 0
        self.kyoku_count = 0
        self.game_scores = [0.0] * 4
        self.feature_history = [[] for _ in range(4)]
        self.hidden_states = [None] * 4
        self.prev_phi = [0.0] * 4
        self.is_enchousen = False
        self.end_reason = None

    def _seat_mapping(self):
        return [(self.dealer_seat + i) % 4 for i in range(4)]

    def reset_round(self):
        # n_rounds indique le nombre de manches normales avant l'enchousen,
        # pas une limite de reset. En tonpuu, round 5+ est l'extension de la
        # meme partie et ne doit pas demarrer une nouvelle partie.
        if (self.kyoku_count >= MAX_KYOKU
                or self.end_reason is not None):
            self._reset_game_state()
        mapping = self._seat_mapping()
        round_wind_idx = (self.round_num - 1) // 4
        self.round_env.set_game_context(
            round_wind_idx=round_wind_idx,
            round_idx=self.round_num - 1,
            scores=[self.game_scores[mapping[i]] for i in range(4)],
            honba=self.honba,
        )
        req = self.round_env.reset_round()
        return self._translate_request(req)

    def _translate_request(self, req):
        mapping = self._seat_mapping()
        translated = dict(req)
        translated["seat"] = mapping[req["seat"]]
        return translated

    def _is_all_last(self):
        if not self.is_enchousen:
            return self.round_num == N_ROUNDS_NORMAL
        return self.round_num == N_ROUNDS_MAX

    def _check_game_end(self, outcome, dealer_repeats, dealer_real,
                         dealer_round_reward):
        """Regles Mahjong Soul (tobi, agari yame, tenpai yame, oorasu,
        enchousen, max_kyoku)."""
        ranks = self._current_ranks()
        dealer_rank = ranks[dealer_real]

        # 1. Tobi
        if any(s <= TOBI_THRESHOLD for s in self.game_scores):
            return True, "tobi"

        # 2-3. Agari yame / Tenpai yame : all last + dealer 1er
        if self._is_all_last() and dealer_rank == 1:
            dealer_won = (outcome in ("tsumo", "ron", "multi_ron")
                          and dealer_round_reward > 0)
            if dealer_won:
                return True, "agari_yame"

            if outcome == "exhaustive":
                try:
                    dealer_tenpai = shanten_of(
                        self.round_env._shanten_counts(0).tolist()) == 0
                    if dealer_tenpai:
                        return True, "tenpai_yame"
                except Exception:
                    pass

        # 4. Enchousen : agari qui porte un joueur a 30k+
        if self.is_enchousen and outcome in ("tsumo", "ron", "multi_ron"):
            if any(s >= TARGET_SCORE for s in self.game_scores):
                return True, "enchousen_30k"

        # 5. Fin d'East 4
        current_round = self.round_num
        if current_round == N_ROUNDS_NORMAL and not dealer_repeats:
            if any(s >= TARGET_SCORE for s in self.game_scores):
                return True, "oorasu"
            return False, None

        # 6. Fin forcee a Sud 4 sans repeat
        if (self.is_enchousen and current_round == N_ROUNDS_MAX
                and not dealer_repeats):
            return True, "enchousen_max"

        # 7. Limite securite
        if self.kyoku_count + 1 >= MAX_KYOKU:
            return True, "max_kyoku"

        return False, None

    def step(self, action):
        result = self.round_env.step(action)
        if not result["round_done"]:
            translated = dict(result)
            translated["next_request"] = self._translate_request(
                result["next_request"])
            if result["acting_seat"] is not None:
                translated["acting_seat"] = \
                    self._seat_mapping()[result["acting_seat"]]
            translated["game_done"] = False
            return translated

        # ---- Fin de manche ----
        mapping = self._seat_mapping()
        real_round_rewards = {
            mapping[k]: v for k, v in result["final_rewards"].items()
        }
        is_dealer_flag = [
            1.0 if p == self.dealer_seat else 0.0 for p in range(4)
        ]
        riichi_flags = self._riichi_flags(mapping)

        for p in range(4):
            self.game_scores[p] += real_round_rewards[p]

        # ---- GRP reward ----
        ranks = self._current_ranks()
        grp_rewards = {}
        format_rounds = (4 if self.game_format == "tonpuu" else 8)
        progression = min(self.kyoku_count + 1, format_rounds) / float(format_rounds)

        for p in range(4):
            gap_leader = max(self.game_scores) - self.game_scores[p]
            gap_last = self.game_scores[p] - min(self.game_scores)
            feat = np.array([
                real_round_rewards[p],
                self.game_scores[p],
                is_dealer_flag[p],
                progression,
                riichi_flags[p],
                ranks[p] / 4.0,
                gap_leader,
                gap_last,
                1.0 if self.game_format == "tonpuu" else 0.0,
                1.0 if self.game_format == "hanchan" else 0.0,
                1.0 if self.round_num == format_rounds else 0.0,
                1.0 if self.round_num > format_rounds else 0.0,
                min(max(self.honba, 0), 4) / 4.0,
                min(max(self.round_env.riichi_deposits, 0), 4) / 4.0,
                max(format_rounds - self.round_num, 0)
                / float(format_rounds),
            ], dtype=np.float32)
            self.feature_history[p].append(feat)

            model = self.predictor_holder.model
            if model is None:
                grp_rewards[p] = real_round_rewards[p]
            else:
                phi_k, new_hidden = model.predict_step(
                    feat, self.hidden_states[p])
                self.hidden_states[p] = new_hidden
                grp_rewards[p] = phi_k - self.prev_phi[p]
                self.prev_phi[p] = phi_k

        # ---- Decide repeat dealer ----
        outcome = result.get('outcome', 'unknown')
        dealer_real = mapping[0]
        dealer_round_reward = real_round_rewards[dealer_real]
        dealer_repeats, honba_increments = _decide_dealer_repeat(
            outcome, dealer_round_reward)

        # ---- [MAHJONG SOUL] Regles de fin ----
        game_done, end_reason = self._check_game_end(
            outcome, dealer_repeats, dealer_real, dealer_round_reward)

        # ---- Mise a jour du round si pas fin ----
        if not game_done:
            if dealer_repeats:
                if honba_increments:
                    self.honba += 1
            else:
                self.honba = 0
                self.dealer_seat = (self.dealer_seat + 1) % 4
                self.round_num += 1
                if self.round_num == N_ROUNDS_NORMAL + 1:
                    self.is_enchousen = True

        self.kyoku_count += 1
        if game_done:
            self.end_reason = end_reason

        # ---- Log final ----
        if game_done and self.game_log_buffer is not None:
            final_rewards_by_rank = _rank_rewards_from_scores(self.game_scores)
            for p in range(4):
                seq = np.stack(self.feature_history[p], axis=0)
                self.game_log_buffer.add(seq, final_rewards_by_rank[p])

        translated = dict(result)
        translated["final_rewards"] = grp_rewards
        translated["acting_seat"] = None
        translated["game_done"] = game_done
        translated["end_reason"] = end_reason
        return translated

    def _current_ranks(self):
        order = sorted(range(4), key=lambda p: (-self.game_scores[p], p))
        ranks = [0] * 4
        for rank, player in enumerate(order, start=1):
            ranks[player] = rank
        return ranks

    def _riichi_flags(self, mapping):
        flags = [0.0] * 4
        for internal_seat, real_player in enumerate(mapping):
            flags[real_player] = (
                1.0 if self.round_env.riichi[internal_seat] else 0.0)
        return flags

    def _shanten_counts(self, real_player):
        mapping = self._seat_mapping()
        internal_seat = mapping.index(real_player)
        return self.round_env._shanten_counts(internal_seat)

    @property
    def obs_dim(self):
        return self.round_env.obs_dim

    @property
    def oracle_dim(self):
        return self.round_env.oracle_dim

    @property
    def riichi(self):
        mapping = self._seat_mapping()
        flags = [False] * 4
        for internal_seat, real_player in enumerate(mapping):
            flags[real_player] = self.round_env.riichi[internal_seat]
        return flags

    @property
    def melds(self):
        mapping = self._seat_mapping()
        out = [None] * 4
        for internal_seat, real_player in enumerate(mapping):
            out[real_player] = self.round_env.melds[internal_seat]
        return out

    def get_stats(self):
        return self.round_env.get_stats()

    def reset_stats(self):
        self.round_env.reset_stats()