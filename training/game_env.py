"""
Etape 4 du curriculum : Global Reward Prediction (GRP).

[FIDELITE PAPIER] Ajout du repeat dealer et honba (Tenhou rules).
Layout : East 1 -> repeat -> East 1 honba 1 -> ... -> East 4 -> fin.

Conditions repeat dealer :
    - Dealer win (tsumo/ron)         -> repeat, honba+1
    - Dealer tenpai at exhaustive    -> repeat, honba+1
    - Abortive draw                  -> repeat, honba inchange
    - Sinon                          -> dealer passe, round_num+1

[FIX 2025-XX] Progression GRP : utilise kyoku_count (nombre de manches
jouees, repeats inclus) normalise sur 4, au lieu de (round_num - 1) / 4.
Coherent avec train_grp_sl.py qui utilise min(round_idx + 1, N_ROUNDS_REF)
/ N_ROUNDS_REF. Le GRP East-4 attend {0.25, 0.5, 0.75, 1.0}.
"""
import numpy as np

from mahjong_ai.envs.multi_agent import MultiAgentMahjongEnv, REWARD_SCALE
from mahjong_ai.agents.reward_predictor import FEATURE_DIM

N_ROUNDS = 4        # East 1-4 (tonpuusen)
MAX_KYOKU = 30      # safety anti-boucle-infinie

RANK_REWARDS = {1: 15.0, 2: 5.0, 3: -5.0, 4: -15.0}

MAX_REPEATS_PER_ROUND = 2

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


def _decide_dealer_repeat(outcome, dealer_round_reward, current_honba=0):
    if current_honba >= MAX_REPEATS_PER_ROUND:
        return False, False
    if outcome.startswith("abortive"):
        return True, False
    if outcome == "exhaustive":
        repeats = dealer_round_reward >= 0
        return repeats, repeats
    repeats = dealer_round_reward > 0
    return repeats, repeats

class GameEnv:
    def __init__(self, seed=None, predictor_holder=None,
                 game_log_buffer=None, n_rounds=N_ROUNDS):
        self.round_env = MultiAgentMahjongEnv(seed=seed)
        self.predictor_holder = predictor_holder or PredictorHolder()
        self.game_log_buffer = game_log_buffer
        self.n_rounds = n_rounds
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

    def _seat_mapping(self):
        return [(self.dealer_seat + i) % 4 for i in range(4)]

    def reset_round(self):
        if (self.round_num > self.n_rounds
                or self.kyoku_count >= MAX_KYOKU):
            self._reset_game_state()
        mapping = self._seat_mapping()
        round_wind_idx = (self.round_num - 1) // 4  # East=0, South=1
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

        ranks = self._current_ranks()
        grp_rewards = {}

        # [FIX] Progression normalisee sur n_rounds (=4), comptant les
        # manches jouees (repeats inclus) pour matcher train_grp_sl.py.
        # kyoku_count est le nombre de manches DEJA terminees, donc
        # l'indice de la manche courante est kyoku_count + 1 (1-indexe).
        progression = min(self.kyoku_count + 1, self.n_rounds) \
            / float(self.n_rounds)

        for p in range(4):
            gap_leader = max(self.game_scores) - self.game_scores[p]
            gap_last = self.game_scores[p] - min(self.game_scores)
            feat = np.array([
                real_round_rewards[p],
                self.game_scores[p],
                is_dealer_flag[p],
                progression,           # <-- CORRIGE : etait (round_num-1)/4
                riichi_flags[p],
                ranks[p] / 4.0,
                gap_leader,
                gap_last,
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

        # Decide repeat
        outcome = result.get('outcome', 'unknown')
        dealer_real = mapping[0]
        dealer_round_reward = real_round_rewards[dealer_real]
        dealer_repeats, honba_increments = _decide_dealer_repeat(
            outcome, dealer_round_reward, self.honba)

        if dealer_repeats:
            if honba_increments:
                self.honba += 1
        else:
            self.honba = 0
            self.dealer_seat = (self.dealer_seat + 1) % 4
            self.round_num += 1

        self.kyoku_count += 1
        game_done = (self.round_num > self.n_rounds
                     or self.kyoku_count >= MAX_KYOKU)

        if game_done and self.game_log_buffer is not None:
            final_rewards_by_rank = _rank_rewards_from_scores(self.game_scores)
            for p in range(4):
                seq = np.stack(self.feature_history[p], axis=0)
                self.game_log_buffer.add(seq, final_rewards_by_rank[p])

        translated = dict(result)
        translated["final_rewards"] = grp_rewards
        translated["acting_seat"] = None
        translated["game_done"] = game_done
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