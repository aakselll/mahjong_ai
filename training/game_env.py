"""
Etape 4 du curriculum : Global Reward Prediction (GRP), comme dans Suphx.

Probleme que ca resout : depuis l'etape 1, chaque manche est un episode
INDEPENDANT -- l'agent optimise l'esperance de points de CETTE manche, sans
aucune notion de score cumule ou de classement sur plusieurs manches. En vrai
Mahjong, un joueur avec une grosse avance en fin de partie doit parfois jouer
DEFENSIVEMENT meme avec une main vivante, pour proteger son classement final
plutot que maximiser l'esperance de la manche en cours. Cette logique
n'existe nulle part dans le reward actuel.

Solution (section 3.2 du papier Suphx) : un predicteur Phi (GRU) apprend a
predire le reward final de partie (base sur le CLASSEMENT, une fonction non
lineaire du score cumule) a partir de l'historique des manches jouees. On
utilise ensuite Phi(x^k) - Phi(x^{k-1}) comme reward de la k-eme manche
pendant l'entrainement RL, au lieu du score brut de la manche.

Design : GameEnv expose EXACTEMENT la meme API que MultiAgentMahjongEnv
(reset_round()/step()), donc reutilisable tel quel dans vectorized_runner.py
via env_factory=GameEnv, SANS modifier ce fichier. En interne, GameEnv :
    - enchaine N_ROUNDS manches par partie, avec rotation du dealer
      (siege interne 0 = dealer de la manche en cours -> joueur reel
      (i + round_idx) % 4, cf. round_idx qui avance a chaque manche) ;
    - garde un score cumule par JOUEUR REEL sur la duree de la partie ;
    - a la fin de chaque manche, calcule le vecteur de features x^k pour
      chaque joueur et l'attribution de reward Phi(x^k)-Phi(x^{k-1}),
      remplace `final_rewards` en consequence ;
    - a la fin d'une PARTIE COMPLETE, calcule le reward de classement final
      et enregistre la sequence complete (x^1..x^N_ROUNDS, reward final)
      dans un buffer partage pour le reentrainement periodique de Phi.

Le predicteur Phi est fourni via un `PredictorHolder` partage entre TOUS les
GameEnv d'un run vectorise (memes poids, mis a jour en place par la boucle
d'entrainement) -- cf. train.py et agents/reward_predictor.py.

[Reorganisation] Ce module importait auparavant MultiAgentMahjongEnvV3 depuis
multi_env3old -- un instantane de l'environnement pris AVANT l'ajout du Ron
decisionnel et du canal furiten (cf. envs/multi_agent.py, qui a ete edite sur
place apres coup sans que cet import soit mis a jour). Consequence : la tete
RON du reseau n'etait jamais entrainee pendant l'entrainement GRP, puisque
l'environnement utilise ici ne demandait jamais de decision de Ron. Corrige :
on importe maintenant l'environnement actif (avec Ron), donc GRP et Ron
s'entrainent ensemble desormais. obs_dim passe de 484 a 485 en consequence --
pour reprendre un ancien checkpoint entraine avant ce correctif, utiliser
--warm-start (cf. agents/policy.py:load_partial_weights_from_step4), pas
--resume qui exige des formes identiques.
"""
import numpy as np

from mahjong_ai.envs.multi_agent import MultiAgentMahjongEnv, REWARD_SCALE
from mahjong_ai.agents.reward_predictor import FEATURE_DIM

N_ROUNDS = 8  # East 1-4 + South 1-4 (pas de repeat-dealer-sur-victoire, simplification)

# reward de classement final (unites REWARD_SCALE, coherent avec les rewards
# de manche existants -- ordre de grandeur d'une grosse main)
RANK_REWARDS = {1: 15.0, 2: 5.0, 3: -5.0, 4: -15.0}


class PredictorHolder:
    """Conteneur mutable partage : toutes les GameEnv lisent `.model` a
    chaque manche, la boucle d'entrainement le remplace en place apres
    chaque reentrainement de Phi (cf. train_step4.py). Avant le premier
    reentrainement, `.model` est None -> GameEnv retombe sur le reward brut
    de la manche (equivalent a l'etape 3) pour ne pas bloquer le demarrage."""

    def __init__(self):
        self.model = None


class GameLogBuffer:
    """Buffer partage (liste bornee) des parties completes, pour le
    reentrainement supervise periodique de Phi. Chaque entree = (sequence de
    features (N_ROUNDS, FEATURE_DIM), reward final de classement) pour UN
    joueur (chaque partie fournit 4 entrees, une par joueur)."""

    def __init__(self, max_size=20000):
        self.max_size = max_size
        self.sequences = []
        self.final_rewards = []

    def add(self, feature_sequence, final_reward):
        self.sequences.append(feature_sequence)
        self.final_rewards.append(final_reward)
        if len(self.sequences) > self.max_size:
            self.sequences.pop(0)
            self.final_rewards.pop(0)

    def __len__(self):
        return len(self.sequences)


def _rank_rewards_from_scores(scores):
    """scores: liste de 4 scores cumules (unites REWARD_SCALE). Renvoie un
    dict {joueur: reward de classement}. Egalite tranchee par index de
    siege (simplification -- le vrai jeu utilise l'ordre des places initiales)."""
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    rewards = {}
    for rank, player in enumerate(order, start=1):
        rewards[player] = RANK_REWARDS[rank]
    return rewards


class GameEnv:
    def __init__(self, seed=None, predictor_holder=None, game_log_buffer=None, n_rounds=N_ROUNDS):
        self.round_env = MultiAgentMahjongEnv(seed=seed)
        self.predictor_holder = predictor_holder if predictor_holder is not None else PredictorHolder()
        self.game_log_buffer = game_log_buffer  # peut etre None (pas de logging, ex. evaluation)
        self.n_rounds = n_rounds

        self.round_idx = 0
        self.game_scores = [0.0] * 4
        self.feature_history = [[] for _ in range(4)]   # liste de vecteurs par joueur, cette partie
        self.hidden_states = [None] * 4                  # etat cache du GRU par joueur, cette partie
        self.prev_phi = [0.0] * 4

    # ------------------------------------------------------------------ #
    def _seat_mapping(self):
        """internal_seat i -> joueur reel, pour la manche en cours."""
        return [(i + self.round_idx) % 4 for i in range(4)]

    def _start_new_game(self):
        self.round_idx = 0
        self.game_scores = [0.0] * 4
        self.feature_history = [[] for _ in range(4)]
        self.hidden_states = [None] * 4
        self.prev_phi = [0.0] * 4

    # ------------------------------------------------------------------ #
    def reset_round(self):
        if self.round_idx >= self.n_rounds:
            self._start_new_game()
        # [FIX] contexte de partie visible par la politique : vent du tour (Est pour la
        # 1ere moitie de la partie, Sud ensuite), numero de manche et scores cumules
        # (indexes par siege INTERNE ; le siege interne 0 est le dealer).
        mapping = self._seat_mapping()
        round_wind_idx = 0 if self.round_idx < max(self.n_rounds // 2, 1) else 1
        self.round_env.set_game_context(
            round_wind_idx=round_wind_idx,
            round_idx=self.round_idx,
            scores=[self.game_scores[mapping[i]] for i in range(4)],
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
            translated["next_request"] = self._translate_request(result["next_request"])
            if result["acting_seat"] is not None:
                translated["acting_seat"] = self._seat_mapping()[result["acting_seat"]]
            return translated

        # ---- une manche vient de se terminer : traduire + attribuer le reward GRP ----
        mapping = self._seat_mapping()
        real_round_rewards = {mapping[k]: v for k, v in result["final_rewards"].items()}
        is_dealer_flag = [1.0 if mapping[0] == p else 0.0 for p in range(4)]
        riichi_flags = self._riichi_flags(mapping)

        grp_rewards = {}
        for p in range(4):
            self.game_scores[p] += real_round_rewards[p]

        ranks = self._current_ranks()
        for p in range(4):
            gap_leader = max(self.game_scores) - self.game_scores[p]
            gap_last = self.game_scores[p] - min(self.game_scores)
            feat = np.array([
                real_round_rewards[p],
                self.game_scores[p],
                is_dealer_flag[p],
                self.round_idx / self.n_rounds,
                riichi_flags[p],
                ranks[p] / 4.0,
                gap_leader,
                gap_last,
            ], dtype=np.float32)
            self.feature_history[p].append(feat)

            model = self.predictor_holder.model
            if model is None:
                # avant le premier entrainement de Phi : reward brut de la manche
                # (equivalent a l'etape 3, pour ne pas bloquer le tout premier cycle)
                grp_rewards[p] = real_round_rewards[p]
            else:
                phi_k, new_hidden = model.predict_step(feat, self.hidden_states[p])
                self.hidden_states[p] = new_hidden
                grp_rewards[p] = phi_k - self.prev_phi[p]
                self.prev_phi[p] = phi_k

        self.round_idx += 1
        game_done = self.round_idx >= self.n_rounds

        if game_done and self.game_log_buffer is not None:
            final_rewards_by_rank = _rank_rewards_from_scores(self.game_scores)
            for p in range(4):
                seq = np.stack(self.feature_history[p], axis=0)  # (n_rounds, FEATURE_DIM)
                self.game_log_buffer.add(seq, final_rewards_by_rank[p])

        translated = dict(result)
        translated["final_rewards"] = grp_rewards
        translated["acting_seat"] = None
        # round_done reste True et next_request reste None (deja le cas dans `result`,
        # exactement comme MultiAgentMahjongEnv._final()) : c'est a l'appelant de
        # rappeler reset_round() quand il est pret pour la manche suivante -- reset_round()
        # decide alors lui-meme si on continue la partie en cours ou si on en demarre une
        # nouvelle (cf. son propre code, base sur self.round_idx deja incremente ici).
        return translated

    # ------------------------------------------------------------------ #
    def _current_ranks(self):
        order = sorted(range(4), key=lambda p: (-self.game_scores[p], p))
        ranks = [0] * 4
        for rank, player in enumerate(order, start=1):
            ranks[player] = rank
        return ranks

    def _riichi_flags(self, mapping):
        # self.round_env.riichi est indexe par siege INTERNE ; mapping[i] = joueur reel
        flags = [0.0] * 4
        for internal_seat, real_player in enumerate(mapping):
            flags[real_player] = 1.0 if self.round_env.riichi[internal_seat] else 0.0
        return flags

    def _shanten_counts(self, real_player):
        """Passthrough pour vectorized_runner.py (suivi mean_shanten_exhaustive),
        qui appelle _shanten_counts(s) directement sur l'environnement, quel
        qu'il soit. `s` est ici un INDICE DE JOUEUR REEL -> on le retraduit
        vers le siege interne correspondant avant de deleguer."""
        mapping = self._seat_mapping()
        internal_seat = mapping.index(real_player)
        return self.round_env._shanten_counts(internal_seat)

    @property
    def obs_dim(self):
        return self.round_env.obs_dim

    @property
    def oracle_dim(self):
        return self.round_env.oracle_dim

    # exposes pour compat avec les scripts d'entrainement qui inspectent
    # directement l'environnement (ex: comptage riichi/kong pour le logging)
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
        melds_by_real = [None] * 4
        for internal_seat, real_player in enumerate(mapping):
            melds_by_real[real_player] = self.round_env.melds[internal_seat]
        return melds_by_real

    # [Bug corrige] ces deux methodes manquaient completement -- vectorized_runner.py
    # les appelle sur `self.envs[i]` (donc sur GameEnv en pratique, cf.
    # train.py:env_factory) pour agreger ankan/riichi/tsumo/ron/yaku dans le
    # log, mais comme GameEnv ne les relayait pas vers round_env, l'appel
    # levait un AttributeError silencieusement avale par un `except
    # AttributeError: pass` dans vectorized_runner.py -- resultat : TOUTES
    # les stats ankan/shouminkan/daiminkan/riichi_tot/tsumo_tot/ron_tot du
    # log etaient a zero en permanence, quoi qu'il se passe reellement en jeu.
    def get_stats(self):
        return self.round_env.get_stats()

    def reset_stats(self):
        self.round_env.reset_stats()
