"""
Features d'entree Suphx (papier §2.2). Module partage parser <-> env.

Layout : (N_CHANNELS, 34).
Look-ahead : 80 canaux (4 n_replace x 20 score_buckets), valeurs booleennes,
melds pris en compte dans le DFS et le scoring.

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
# Layout des canaux
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
    ('game_format', 2), ('is_last_normal_round', 1), ('is_extension', 1),
    ('extension_index', 8), ('round_progress', 10), ('rounds_remaining', 10),
    ('honba', 5), ('kyoutaku', 5),
    ('own_score', 20), ('opp0_score', 20), ('opp1_score', 20), ('opp2_score', 20),
    ('wall_remaining', 10),
    ('lookahead', LOOKAHEAD_CHANNELS),
]
N_CHANNELS = sum(n for _, n in CHANNEL_GROUPS)
OBS_FLAT_DIM = N_CHANNELS * N_TILES                # 8500

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
            for t in range(N_TILES):
                pair_add = max(0, 2 - hand[t])
                if adds + pair_add <= max_add:
                    T = base.copy()
                    for md in chosen:
                        for tt, need in md.items():
                            T[tt] += need
                    T[t] += 2
                    counter[0] += 1
                    yield T, adds + pair_add
                    if counter[0] >= max_results or visited[0] >= MAX_VISITED:
                        return
            return
        for md in _MELD_OPTIONS:
            if counter[0] >= max_results or visited[0] >= MAX_VISITED:
                return
            md_add = _meld_add_cost(hand, md)
            if adds + md_add > max_add:
                continue
            if adds > 2 and not any(hand[t] > 0 for t in md):
                continue
            yield from dfs(_consume(hand, md), melds_left - 1,
                            adds + md_add, chosen + [md])

    yield from dfs(remaining, n_melds_needed, 0, [])


def _score_winning_hand_with_melds(T_full, existing_melds, seat_wind,
                                     round_wind, is_dealer, dora_indicators):
    if int(np.sum(T_full)) != 14:
        return None

    tiles_136 = []
    nid = [0] * N_TILES
    T = T_full.copy()
    meld_objs = []

    for m in existing_melds:
        tile_ids = []
        for t in m['tiles']:
            if t < 0 or t >= N_TILES or T[t] <= 0:
                return None
            tid = 4 * t + nid[t]
            nid[t] += 1
            T[t] -= 1
            tiles_136.append(tid)
            tile_ids.append(tid)
        mt = _MELD_TYPE_MAP.get(m['type'])
        if mt is None:
            return None
        opened = (m['type'] != 'kan_closed')
        meld_objs.append(Meld(meld_type=mt, tiles=tile_ids, opened=opened))

    for t in range(N_TILES):
        for _ in range(int(T[t])):
            tid = 4 * t + nid[t]
            nid[t] += 1
            tiles_136.append(tid)

    if len(tiles_136) != 14:
        return None
    win_tile_136 = tiles_136[-1]

    dora_136 = [t * 4 for t in dora_indicators if 0 <= t < N_TILES]
    try:
        res = _HAND_CALC.estimate_hand_value(
            tiles_136, win_tile_136, melds=meld_objs,
            dora_indicators=dora_136,
            config=_get_config(seat_wind, round_wind),
        )
        if res.error is not None:
            return None
        base = int(res.cost.get('main', 0)) if isinstance(res.cost, dict) else int(res.cost)
        return int(base * 1.5) if is_dealer else base
    except Exception:
        return None


def _melds_cache_key(melds):
    return tuple(sorted(
        (str(m['type']), tuple(sorted(int(t) for t in m['tiles'])))
        for m in melds
    ))


# ---------------------------------------------------------------------- #
# Cache 1 : DFS pur, independant du dora et du wind
# ---------------------------------------------------------------------- #
@functools.lru_cache(maxsize=512)
def _enumerate_hands_cached(hand_tuple, melds_key, max_add):
    """DFS seulement. Retourne tuple de (T_tuple, n_add).
    Independant de dora/winds -> cache plus efficace."""
    hand = np.array(hand_tuple, dtype=np.int32)
    melds = [{'type': k[0], 'tiles': list(k[1])} for k in melds_key]
    out = []
    for T, n_add in _enumerate_winning_hands_with_melds(hand, melds, max_add):
        if n_add < 1 or n_add > len(N_REPLACE_VALUES):
            continue
        out.append((tuple(int(x) for x in T.tolist()), int(n_add)))
    return tuple(out)


# ---------------------------------------------------------------------- #
# Cache 2 : scoring, independant de H_prime (donc partage entre discards)
# ---------------------------------------------------------------------- #
@functools.lru_cache(maxsize=8192)
def _score_with_dora_cached(T_tuple, melds_key, seat_wind, round_wind,
                              is_dealer, dora_tuple):
    """Score d'une main gagnante T. Le T est identique pour plusieurs
    discards -> hit rate eleve."""
    T = np.array(T_tuple, dtype=np.int32)
    melds = [{'type': k[0], 'tiles': list(k[1])} for k in melds_key]
    return _score_winning_hand_with_melds(
        T, melds, seat_wind, round_wind, is_dealer, list(dora_tuple))


def compute_lookahead(hand13_counts, melds, seat_wind, round_wind,
                       is_dealer, dora_indicators, unseen=None, wall_remaining=None):
    """Look-ahead binaire (papier §2.2). 0/1 dans (n_replace, score_bucket)."""
    features = np.zeros((LOOKAHEAD_CHANNELS, N_TILES), dtype=np.float32)

    if wall_remaining is not None and wall_remaining <= 0:
        return features

    hand = np.asarray(hand13_counts, dtype=np.int32)

    # ---- Skip si trop loin du tenpai ----
    try:
        from mahjong_ai.envs.tiles import shanten_of
        melds_counts = np.zeros(N_TILES, dtype=np.int32)
        for m in melds:
            if m['type'] in ('kan_closed', 'shouminkan', 'kan_open'):
                melds_counts[m['tiles'][0]] += 3
            else:
                for t in m['tiles']:
                    melds_counts[t] += 1
        full = hand + melds_counts
        sh = shanten_of(full.tolist())
        if sh < 0 or sh > MAX_LOOKAHEAD_SHANTEN:
            return features
    except Exception:
        pass

    dora_t = tuple(sorted(int(x) for x in dora_indicators if 0 <= x < N_TILES))
    melds_key = _melds_cache_key(melds)
    sw, rw, dl = int(seat_wind), int(round_wind), bool(is_dealer)
    max_add = max(N_REPLACE_VALUES)

    for d in range(N_TILES):
        if hand[d] == 0:
            continue
        H_prime = hand.copy()
        H_prime[d] -= 1
        H_prime_tuple = tuple(int(x) for x in H_prime.tolist())

        # DFS (cache par H_prime + melds)
        dfs_results = _enumerate_hands_cached(H_prime_tuple, melds_key, max_add)

        # Scoring (cache par T + melds + winds + dora -> hits elevees)
        for T_t, n_add in dfs_results:
            score = _score_with_dora_cached(
                T_t, melds_key, sw, rw, dl, dora_t)
            if score is None:
                continue
            bucket = _score_to_bucket(score)
            features[(n_add - 1) * N_SCORE_BUCKETS + bucket][d] = 1.0

    return features


# ====================================================================== #
# Bucketing
# ====================================================================== #
_SCORE_LOW, _SCORE_HIGH = -30.0, 30.0
_SCORE_BUCKETS = 20


def _score_bucket(score_k):
    if score_k <= _SCORE_LOW:
        return 0
    if score_k >= _SCORE_HIGH:
        return _SCORE_BUCKETS - 1
    frac = (score_k - _SCORE_LOW) / (_SCORE_HIGH - _SCORE_LOW)
    return min(int(frac * _SCORE_BUCKETS), _SCORE_BUCKETS - 1)


def _wall_bucket(wall_remaining):
    wr = min(max(wall_remaining, 0), 70)
    return min(int(wr / 70.0 * 10), 9)


def _next_dora(ind):
    if ind < 0 or ind >= N_TILES:
        return -1
    if ind < 27:
        s = (ind // 9) * 9
        return s + (ind % 9 + 1) % 9
    if ind < 31:
        return 27 + (ind - 27 + 1) % 4
    return 31 + (ind - 31 + 1) % 3


# ====================================================================== #
# Builder principal
# ====================================================================== #
def build_suphx_obs(*, hands, melds, discards, discard_hist,
                     riichi_flags, furiten_flag, dora_indicators,
                     wall_remaining, scores, dealer_seat, round_wind,
                     round_id, honba, kyoutaku, seat, pending_claim_tile=None,
                     compute_lookahead_flag=True, game_format='hanchan',
                     is_last_normal_round=False, is_extension=False,
                     extension_index=0, normal_round_count=None,
                     round_progress=None, rounds_remaining=None):
    C = np.zeros((N_CHANNELS, N_TILES), dtype=np.float32)

    def block(name):
        return _OFFSET[name]

    o = block('own_hand')
    for n in range(4):
        C[o + n] = (hands[seat] >= (n + 1)).astype(np.float32)

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

    for s in range(4):
        o = block(f'discards_s{s}')
        for n in range(4):
            C[o + n] = (discards[s] >= (n + 1)).astype(np.float32)

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

    o = block('dora_ind')
    for ind in dora_indicators:
        if 0 <= ind < N_TILES:
            C[o][ind] += 1.0
    o = block('dora_tile')
    for ind in dora_indicators:
        dt = _next_dora(ind)
        if 0 <= dt < N_TILES:
            C[o][dt] += 1.0

    if pending_claim_tile is not None and 0 <= pending_claim_tile < N_TILES:
        C[block('pending')][pending_claim_tile] = 1.0

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

    if riichi_flags[seat]:
        C[block('own_riichi')] = 1.0
    for i, opp in enumerate(opps):
        if riichi_flags[opp]:
            C[block('opp_riichi') + i] = 1.0
    if furiten_flag:
        C[block('own_furiten')] = 1.0

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

    if game_format not in ('tonpuu', 'hanchan'):
        raise ValueError(f'unknown game_format: {game_format}')
    o = block('game_format')
    C[o + (0 if game_format == 'tonpuu' else 1)] = 1.0
    C[block('is_last_normal_round')] = 1.0 if is_last_normal_round else 0.0
    C[block('is_extension')] = 1.0 if is_extension else 0.0
    C[block('extension_index') + min(max(int(extension_index), 0), 7)] = 1.0
    if normal_round_count is None:
        normal_round_count = 4 if game_format == 'tonpuu' else 8
    if round_progress is None:
        round_progress = round_id
    progress = min(max(int(round_progress), 1), int(normal_round_count))
    progress_bucket = min(
        int((progress - 1) * 10 / max(int(normal_round_count), 1)), 9
    )
    C[block('round_progress') + progress_bucket] = 1.0
    if rounds_remaining is None:
        rounds_remaining = max(int(normal_round_count) - progress, 0)
    remaining_bucket = min(
        int(max(int(rounds_remaining), 0) * 10
            / max(int(normal_round_count), 1)),
        9,
    )
    C[block('rounds_remaining') + remaining_bucket] = 1.0

    C[block('honba') + min(int(honba), 4)] = 1.0
    C[block('kyoutaku') + min(int(kyoutaku), 4)] = 1.0

    C[block('own_score') + _score_bucket(scores[seat])] = 1.0
    for i, opp in enumerate(opps):
        C[block(f'opp{i}_score') + _score_bucket(scores[opp])] = 1.0

    C[block('wall_remaining') + _wall_bucket(wall_remaining)] = 1.0

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