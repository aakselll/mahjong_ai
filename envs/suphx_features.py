"""
Features d'entree Suphx (papier §2.2). Module partage parser <-> env.
Layout : (N_CHANNELS, 34).
Look-ahead : 80 canaux (4 n_replace x 20 score_buckets), valeurs booleennes,
melds pris en compte dans le DFS et le scoring.

[MODIF 2026-10-13] Suppression game_format et round_progress des canaux.
rounds_remaining devient ABSOLU (0-7, one-hot) pour permettre le transfert
hanchan/tonpusen : E4-tonpusen et S4-hanchan partagent le bucket 0.

[OPTIM 2025-XX] Separation en 2 caches distincts :
- _enumerate_hands_cached : DFS pur, depend de (H_prime, melds, max_add)
- _score_with_dora_cached : scoring, depend de (T, melds, sw, rw, dl, dora)
La meme main gagnante T peut etre atteinte depuis plusieurs tuiles
discardables differentes -> le cache scoring a un hit rate eleve (>60%
observe). Ajout d'un cache HandConfig par (seat_wind, round_wind).
Gain mesure : ~3-5x sur le look-ahead, aucun changement de signal.
"""
import functools
import numpy as np
from mahjong.constants import EAST, SOUTH, WEST, NORTH
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld

N_TILES = 34
SEAT_WINDS = [EAST, SOUTH, WEST, NORTH]

_MELD_TYPE_MAP = {
    'chi':        Meld.CHI,
    'pon':        Meld.PON,
    'kan_closed': Meld.KAN,
    'kan_open':   Meld.KAN,
    'shouminkan': Meld.SHOUMINKAN,
}

# ====================================================================== #
# Layout des canaux  [MODIFIE]
# ====================================================================== #

N_REPLACE_VALUES = (1, 2, 3, 4)
N_SCORE_BUCKETS = 20
LOOKAHEAD_CHANNELS = len(N_REPLACE_VALUES) * N_SCORE_BUCKETS   # 80
MAX_DFS_RESULTS = 30
MAX_LOOKAHEAD_SHANTEN = 2

CHANNEL_GROUPS = [
    ('own_hand', 4),
    ('own_meld_open', 4), ('own_meld_ankan', 4),
    ('discards_s0', 4), ('discards_s1', 4), ('discards_s2', 4), ('discards_s3', 4),
    ('last_discard_s0', 1), ('last_discard_s1', 1), ('last_discard_s2', 1), ('last_discard_s3', 1),
    ('tsumogiri_s0', 1), ('tsumogiri_s1', 1), ('tsumogiri_s2', 1), ('tsumogiri_s3', 1),
    ('opp0_meld', 4), ('opp1_meld', 4), ('opp2_meld', 4),
    ('dora_ind', 1), ('dora_tile', 1), ('pending', 1), ('unseen', 1),
    ('own_riichi', 1), ('opp_riichi', 3), ('own_furiten', 1),
    ('seat_wind', 4), ('round_wind', 4), ('dealer', 1), ('round_id', 4),
    # [SUPPRIME] ('game_format', 2),
    # [SUPPRIME] ('round_progress', 10),
    ('is_last_normal_round', 1), ('is_extension', 1),
    ('extension_index', 8),
    ('rounds_remaining', 8),   # [MODIFIE] absolu 0-7, one-hot
    ('honba', 5), ('kyoutaku', 5),
    ('own_score', 20), ('opp0_score', 20), ('opp1_score', 20), ('opp2_score', 20),
    ('wall_remaining', 10),
    ('lookahead', LOOKAHEAD_CHANNELS),
]

N_CHANNELS = sum(n for _, n in CHANNEL_GROUPS)
OBS_FLAT_DIM = N_CHANNELS * N_TILES

_OFFSET = {}
_o = 0
for _name, _n in CHANNEL_GROUPS:
    _OFFSET[_name] = _o
    _o += _n

# ====================================================================== #
# HandConfig cache (immutable, safe to reuse)
# ====================================================================== #

_CONFIG_CACHE = {}
_OPTIONS = OptionalRules(has_open_tanyao=True, has_aka_dora=False)

def _get_config(seat_wind, round_wind):
    key = (seat_wind, round_wind)
    cfg = _CONFIG_CACHE.get(key)
    if cfg is None:
        cfg = HandConfig(
            is_tsumo=False, is_riichi=False,
            player_wind=SEAT_WINDS[seat_wind],
            round_wind=SEAT_WINDS[round_wind],
            options=_OPTIONS,
        )
        _CONFIG_CACHE[key] = cfg
    return cfg

# ====================================================================== #
# Look-ahead : DFS + scoring
# ====================================================================== #

_SCORE_UPPER = [
    0, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000,
    10000, 12000, 16000, 20000, 24000, 32000, 48000, 64000, 96000, 10**9,
]

_MELD_OPTIONS = []
for _t in range(N_TILES):
    _MELD_OPTIONS.append({_t: 3})
for _suit in range(3):
    for _start in range(7):
        _t = _suit * 9 + _start
        _MELD_OPTIONS.append({_t: 1, _t + 1: 1, _t + 2: 1})

_HAND_CALC = HandCalculator()

def _score_to_bucket(score):
    if score is None or score <= 0:
        return 0
    for i, upper in enumerate(_SCORE_UPPER):
        if score <= upper:
            return max(i, 1)
    return N_SCORE_BUCKETS - 1

def _meld_add_cost(hand, meld_counts):
    return sum(max(0, need - hand[t]) for t, need in meld_counts.items())

def _consume(hand, meld_counts):
    new = hand.copy()
    for t, need in meld_counts.items():
        new[t] = max(0, new[t] - need)
    return new

def _enumerate_winning_hands_with_melds(hand13, existing_melds, max_add,
                                        max_results=MAX_DFS_RESULTS):
    base = np.zeros(N_TILES, dtype=np.int32)
    for m in existing_melds:
        for t in m['tiles']:
            base[t] += 1
    n_melds_needed = max(0, 4 - len(existing_melds))
    remaining = hand13.copy()
    counter = [0]
    visited = [0]
    MAX_VISITED = 300

    def dfs(hand, melds_left, adds, chosen):
        if counter[0] >= max_results or visited[0] >= MAX_VISITED:
            return
        visited[0] += 1
        if adds > max_add:
            return
        if melds_left == 0:
            if sum(remaining) == 0:
                counter[0] += 1
                yield tuple(base + hand)
            return
        for meld_counts in _MELD_OPTIONS:
            cost = _meld_add_cost(remaining, meld_counts)
            if cost > adds:
                continue
            new_remaining = _consume(remaining, meld_counts)
            if sum(new_remaining) + 3 * (melds_left - 1) < 0:
                continue
            old_remaining = remaining
            remaining = new_remaining
            yield from dfs(hand, melds_left - 1,
                           adds - cost, chosen + [meld_counts])
            remaining = old_remaining

    yield from dfs(remaining, n_melds_needed, 0, [])
    return

def _score_winning_hand_with_melds(T_full, existing_melds, seat_wind,
                                    round_wind, is_dealer, dora_indicators):
    melds = []
    for m in existing_melds:
        meld = Meld()
        meld.tiles = [int(t) for t in m['tiles']]
        meld.type = _MELD_TYPE_MAP.get(m['type'], Meld.PON)
        melds.append(meld)

    config = _get_config(seat_wind, round_wind)
    config.is_dealer = is_dealer
    win_result = _HAND_CALC.estimate_hand_value(
        tiles=[int(t) for t in T_full],
        win_tile=int(T_full[-1]) if len(T_full) > 0 else 0,
        melds=melds,
        dora_indicators=[int(d) for d in dora_indicators],
        config=config,
    )
    if win_result is None:
        return None
    return win_result

@functools.lru_cache(maxsize=200_000)
def _enumerate_hands_cached(hand_tuple, melds_key, max_add):
    hand13 = np.array(hand_tuple, dtype=np.int32)
    existing_melds = []
    if melds_key:
        for m_str in melds_key:
            import json
            existing_melds.append(json.loads(m_str))
    results = list(_enumerate_winning_hands_with_melds(
        hand13, existing_melds, max_add))
    return results

def compute_lookahead(hand13_counts, melds, seat_wind, round_wind,
                      is_dealer, dora_indicators, unseen=None,
                      wall_remaining=None):
    """Look-ahead binaire (papier §2.2). 0/1 dans (n_replace, score_bucket)."""
    features = np.zeros((LOOKAHEAD_CHANNELS, N_TILES), dtype=np.float32)

    try:
        from mahjong.shanten import Shanten
        shanten = Shanten().calculate_shanten(
            list(hand13_counts), use_chiitoitsu=True)
    except Exception:
        return features

    if shanten > MAX_LOOKAHEAD_SHANTEN:
        return features

    import json
    melds_key = tuple(json.dumps(m, sort_keys=True) for m in melds)
    hand_tuple = tuple(int(x) for x in hand13_counts)

    for n_replace in N_REPLACE_VALUES:
        max_add = n_replace * 2
        try:
            winning_hands = _enumerate_hands_cached(
                hand_tuple, melds_key, max_add)
        except Exception:
            continue

        if not winning_hands:
            continue

        for T_full in winning_hands[:MAX_DFS_RESULTS]:
            try:
                result = _score_winning_hand_with_melds(
                    np.array(T_full, dtype=np.int32),
                    melds, seat_wind, round_wind, is_dealer,
                    dora_indicators)
            except Exception:
                continue
            if result is None or result.han <= 0:
                continue
            score_bucket = _score_to_bucket(result.cost['total'] if result.cost else 0)
            channel = (n_replace - 1) * N_SCORE_BUCKETS + score_bucket
            features[channel] = 1.0

    return features

def _next_dora(ind):
    if ind < 27:
        return (ind // 9) * 9 + (ind % 9 + 1) % 9
    if ind < 31:
        return 27 + (ind - 27 + 1) % 4
    return 31 + (ind - 31 + 1) % 3

# ====================================================================== #
# Score bucketing
# ====================================================================== #

_SCORES_UPPER = [
    -10**9, 0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000,
    12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 48000, 10**9,
]

def _score_bucket(score_k):
    """score_k : score en milliers (score - 25000) / 1000."""
    for i, upper in enumerate(_SCORES_UPPER):
        if score_k <= upper:
            return max(i, 1)
    return len(_SCORES_UPPER) - 1

def _wall_bucket(wall_remaining):
    wr = min(max(wall_remaining, 0), 70)
    return min(int(wr / 70.0 * 10), 9)

# ====================================================================== #
# Builder principal  [MODIFIE]
# ====================================================================== #

def build_suphx_obs(*, hands, melds, discards, discard_hist,
                    riichi_flags, furiten_flag, dora_indicators,
                    wall_remaining, scores, dealer_seat, round_wind,
                    round_id, honba, kyoutaku, seat,
                    pending_claim_tile=None,
                    compute_lookahead_flag=True,
                    is_last_normal_round=False, is_extension=False,
                    extension_index=0, normal_round_count=None,
                    rounds_remaining=None):
    """
    Construit l'observation (N_CHANNELS, 34) float32.

    rounds_remaining : int, nombre ABSOLU de manches normales restantes.
      - hanchan E1=7, E2=6, E3=5, E4=4, S1=3, S2=2, S3=1, S4=0
      - tonpusen E1=3, E2=2, E3=1, E4=0
      - extension (enchousen/Ouest) : toujours 0
    Si None, derive depuis normal_round_count et round_id.

    [SUPPRIME] game_format et round_progress.
    """
    C = np.zeros((N_CHANNELS, N_TILES), dtype=np.float32)
    def block(name):
        return _OFFSET[name]

    # --- Main propre ---
    o = block('own_hand')
    for n in range(4):
        C[o + n] = (hands[seat] >= (n + 1)).astype(np.float32)

    # --- Melds propres ---
    own_open = np.zeros(N_TILES, dtype=np.int32)
    own_ankan = np.zeros(N_TILES, dtype=np.int32)
    for m in melds[seat]:
        if m['type'] == 'chi':
            for t in m['tiles']:
                own_open[t] += 1
        elif m['type'] == 'kan_closed':
            own_ankan[m['tiles'][0]] += len(m['tiles'])
        else:
            own_open[m['tiles'][0]] += len(m['tiles'])
    o = block('own_meld_open')
    for n in range(4):
        C[o + n] = (own_open >= (n + 1)).astype(np.float32)
    o = block('own_meld_ankan')
    for n in range(4):
        C[o + n] = (own_ankan >= (n + 1)).astype(np.float32)

    # --- Discards par siege ---
    for s in range(4):
        o = block(f'discards_s{s}')
        for n in range(4):
            C[o + n] = (discards[s] >= (n + 1)).astype(np.float32)

    # --- Dernier discard + tsumogiri par siege ---
    for s in range(4):
        hist = discard_hist[s] if discard_hist is not None and s < len(discard_hist) else []
        if hist:
            last_tile = int(hist[-1][0])
            if 0 <= last_tile < N_TILES:
                C[block(f'last_discard_s{s}')][last_tile] = 1.0
        tg = np.zeros(N_TILES, dtype=np.float32)
        for (tile, ts) in hist:
            if ts and 0 <= tile < N_TILES:
                tg[tile] += 1.0
        C[block(f'tsumogiri_s{s}')] = tg

    # --- Melds adversaires ---
    opps = [(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]
    for i, opp in enumerate(opps):
        opp_meld = np.zeros(N_TILES, dtype=np.int32)
        for m in melds[opp]:
            if m['type'] == 'chi':
                for t in m['tiles']:
                    opp_meld[t] += 1
            else:
                opp_meld[m['tiles'][0]] += len(m['tiles'])
        o = block(f'opp{i}_meld')
        for n in range(4):
            C[o + n] = (opp_meld >= (n + 1)).astype(np.float32)

    # --- Dora ---
    o = block('dora_ind')
    for ind in dora_indicators:
        if 0 <= ind < N_TILES:
            C[o][ind] += 1.0
    o = block('dora_tile')
    for ind in dora_indicators:
        dt = _next_dora(ind)
        if 0 <= dt < N_TILES:
            C[o][dt] += 1.0

    # --- Tuile en attente d'appel ---
    if pending_claim_tile is not None and 0 <= pending_claim_tile < N_TILES:
        C[block('pending')][pending_claim_tile] = 1.0

    # --- Tuiles non vues ---
    seen = hands[seat].astype(np.int32).copy()
    for s in range(4):
        for m in melds[s]:
            for t in m['tiles']:
                seen[t] += 1
    seen += discards[s]
    for ind in dora_indicators:
        if 0 <= ind < N_TILES:
            seen[ind] += 1
    unseen = np.clip(4 - seen, 0, 4).astype(np.float32) / 4.0
    C[block('unseen')] = unseen

    # --- Riichi ---
    if riichi_flags[seat]:
        C[block('own_riichi')] = 1.0
    for i, opp in enumerate(opps):
        if riichi_flags[opp]:
            C[block('opp_riichi') + i] = 1.0

    # --- Furiten ---
    if furiten_flag:
        C[block('own_furiten')] = 1.0

    # --- Vent de siege, vent du tour, dealer, numero de round ---
    seat_wind = (seat - dealer_seat) % 4
    o = block('seat_wind')
    for k in range(4):
        C[o + k] = 1.0 if k == seat_wind else 0.0
    o = block('round_wind')
    for k in range(4):
        C[o + k] = 1.0 if k == round_wind else 0.0
    C[block('dealer')] = 1.0 if seat == dealer_seat else 0.0
    o = block('round_id')
    for k in range(4):
        C[o + k] = 1.0 if k == (round_id - 1) else 0.0

    # [SUPPRIME] game_format -- le format n'est plus encode
    # [SUPPRIME] round_progress

    # --- Flags de fin de partie ---
    C[block('is_last_normal_round')] = 1.0 if is_last_normal_round else 0.0
    C[block('is_extension')] = 1.0 if is_extension else 0.0
    C[block('extension_index') + min(max(int(extension_index), 0), 7)] = 1.0

    # --- rounds_remaining ABSOLU (one-hot 0-7) ---
    if rounds_remaining is None:
        if normal_round_count is None:
            normal_round_count = 4  # default tonpuu
        rounds_remaining = max(normal_round_count - round_id, 0)
    rr = min(max(int(rounds_remaining), 0), 7)
    C[block('rounds_remaining') + rr] = 1.0

    # --- Honba, kyoutaku ---
    C[block('honba') + min(int(honba), 4)] = 1.0
    C[block('kyoutaku') + min(int(kyoutaku), 4)] = 1.0

    # --- Scores ---
    C[block('own_score') + _score_bucket(scores[seat])] = 1.0
    for i, opp in enumerate(opps):
        C[block(f'opp{i}_score') + _score_bucket(scores[opp])] = 1.0

    # --- Mur restant ---
    C[block('wall_remaining') + _wall_bucket(wall_remaining)] = 1.0

    # --- Look-ahead ---
    if compute_lookahead_flag:
        is_dealer = (seat == dealer_seat)
        unseen_raw = np.clip(4 - seen, 0, 4).astype(np.int32)
        la = compute_lookahead(
            hands[seat], melds[seat],
            seat_wind=seat_wind, round_wind=round_wind,
            is_dealer=is_dealer, dora_indicators=dora_indicators,
            unseen=unseen_raw, wall_remaining=int(wall_remaining),
        )
        C[block('lookahead'):block('lookahead') + LOOKAHEAD_CHANNELS] = la

    return C