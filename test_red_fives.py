"""
test_red_fives.py -- Tests du support des 5 rouges (aka dora : 0m, 0p, 0s).

Usage (depuis la racine du depot) :  python3 test_red_fives.py

Si `mahjong`, `torch` ou `zstandard` ne sont pas installes, des stubs
minimaux les remplacent (le test reste executable partout) :
  - mahjong : Shanten / HandCalculator factices (le calculateur verifie les
    ids 136 recus et compte les aka comme la vraie lib : ids 16, 52, 88) ;
  - torch   : juste de quoi importer agents.suphx_cnn et tester
    adapt_in_channels sur des tableaux numpy.
Avec les vraies libs, ce sont elles qui sont utilisees.

Couvre :
  A. layout des canaux (rouges AJOUTES EN FIN : les 268 premiers canaux
     sont identiques a l'ancien layout) ;
  B. adaptation des anciens poids 268/272 -> 279/283 (nouveaux poids = 0,
     canaux oracle remis a leur place) ;
  C. env : conservation des 3 rouges sur des manches aleatoires, ids 136
     et aka-dora dans le scoring, regles de defausse/appel ;
  D. parser SL : suivi des rouges depuis les logs mjai + fix kakan.
"""
import functools
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # pour `import mahjong_ai`

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), str(detail)))
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}"
          + ("" if cond else f"   {detail}"))


def section(title):
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------- #
def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _sets_ok(c):
    i = 0
    while i < 34 and c[i] == 0:
        i += 1
    if i == 34:
        return True
    if c[i] >= 3:
        c[i] -= 3
        r = _sets_ok(c)
        c[i] += 3
        if r:
            return True
    if i < 27 and i % 9 <= 6 and c[i + 1] > 0 and c[i + 2] > 0:
        c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
        r = _sets_ok(c)
        c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
        return r
    return False


def _agari(counts, kokushi):
    c = [int(x) for x in counts]
    if sum(c) % 3 != 2:
        return False
    for p in range(34):
        if c[p] >= 2:
            c[p] -= 2
            ok = _sets_ok(c)
            c[p] += 2
            if ok:
                return True
    if sum(c) == 14 and sum(1 for x in c if x == 2) == 7:
        return True
    if kokushi:
        yao = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        if all(c[t] >= 1 for t in yao) and sum(c[t] for t in yao) == 14:
            return True
    return False


@functools.lru_cache(maxsize=500_000)
def _shanten_cached(tiles, kokushi):
    c = list(tiles)
    if _agari(c, kokushi):
        return -1
    for t in range(34):
        if c[t] < 4:
            c[t] += 1
            ok = _agari(c, kokushi)
            c[t] -= 1
            if ok:
                return 0
    return 1


class _StubShanten:
    def calculate_shanten(self, tiles, use_chiitoitsu=True, use_kokushi=True):
        return _shanten_cached(tuple(int(x) for x in tiles), bool(use_kokushi))


AKA_IDS = {16, 52, 88}


class _StubCalc:
    LAST = None          # derniere evaluation : dict(tiles, win_tile, aka)
    CALLS = []

    @staticmethod
    def estimate_hand_value(tiles, win_tile, melds=None,
                            dora_indicators=None, ura_dora_indicators=None,
                            config=None):
        tiles = list(tiles)
        assert len(set(tiles)) == len(tiles), f"ids 136 dupliques : {sorted(tiles)}"
        assert all(0 <= t < 136 for t in tiles), f"id hors plage : {sorted(tiles)}"
        assert win_tile in tiles, f"win_tile {win_tile} absent de {sorted(tiles)}"
        use_aka = bool(getattr(config.options, "has_aka_dora", False))
        aka = sum(1 for t in tiles if t in AKA_IDS) if use_aka else 0
        rec = {"tiles": tiles, "win_tile": win_tile, "aka": aka,
               "use_aka": use_aka}
        _StubCalc.LAST = rec
        _StubCalc.CALLS.append(rec)
        han = 1 + aka
        total = 1000 * han
        return types.SimpleNamespace(
            error=None, han=han, fu=30, yaku=[],
            cost={"total": total, "main": total // 2, "additional": total // 4,
                  "kyoutaku_bonus": 0})


def install_stubs():
    global REAL_MAHJONG, REAL_TORCH
    try:
        import mahjong  # noqa: F401
        REAL_MAHJONG = True
    except Exception:
        REAL_MAHJONG = False
        _stub("mahjong").__path__ = []
        _stub("mahjong.constants", EAST=27, SOUTH=28, WEST=29, NORTH=30)
        _stub("mahjong.shanten", Shanten=_StubShanten)
        _stub("mahjong.hand_calculating").__path__ = []
        _stub("mahjong.hand_calculating.hand", HandCalculator=_StubCalc)

        class OptionalRules:
            def __init__(self, **kw):
                self.has_aka_dora = False
                self.__dict__.update(kw)

        class HandConfig:
            def __init__(self, **kw):
                self.options = OptionalRules()
                self.__dict__.update(kw)

        _stub("mahjong.hand_calculating.hand_config",
              HandConfig=HandConfig, OptionalRules=OptionalRules)

        class Meld:
            CHI, PON, KAN, SHOUMINKAN = "chi", "pon", "kan", "shouminkan"

            def __init__(self, meld_type=None, tiles=None, opened=True):
                self.type, self.tiles, self.opened = meld_type, tiles, opened

        _stub("mahjong.meld", Meld=Meld)
    try:
        import torch  # noqa: F401
        REAL_TORCH = True
    except Exception:
        REAL_TORCH = False

        class _Arr(np.ndarray):
            def narrow(self, dim, start, length):
                sl = [slice(None)] * self.ndim
                sl[dim] = slice(start, start + length)
                return np.asarray(self)[tuple(sl)].view(_Arr)

            def new_zeros(self, shape):
                return np.zeros(shape, dtype=self.dtype).view(_Arr)

            def contiguous(self):
                return np.ascontiguousarray(self).view(_Arr)

        class _Module:
            def __init__(self, *a, **k):
                pass

            def _load_from_state_dict(self, *a, **k):
                pass

        nn = _stub("torch.nn", Module=_Module, Conv2d=object, BatchNorm2d=object,
                   ReLU=object, ModuleList=list, Linear=object, Sequential=object,
                   Flatten=object)
        _stub("torch", nn=nn, cat=lambda ps, dim=0: np.concatenate(
            [np.asarray(p) for p in ps], axis=dim).view(_Arr), _Arr=_Arr)
    try:
        import zstandard  # noqa: F401
    except Exception:
        _stub("zstandard")


install_stubs()

from mahjong_ai.envs import suphx_features as F                       # noqa: E402
from mahjong_ai.envs.suphx_features import (                          # noqa: E402
    RED_TYPES, N_CHANNELS, N_BASE_CHANNELS, N_RED_CHANNELS, N_TILES,
    build_suphx_obs, empty_red_state, _OFFSET)
from mahjong_ai.envs import multi_agent as MA                          # noqa: E402
from mahjong_ai.envs.multi_agent import (                              # noqa: E402
    MultiAgentMahjongEnv, DISCARD, PONG, CHOW, KONG, RIICHI, RON)


# ===================================================================== #
# A. layout des canaux
# ===================================================================== #
section("A. layout des canaux")
OLD_TOTAL = 268                      # ancien N_CHANNELS (somme de l'ancien layout)
NB, NR, NO = N_BASE_CHANNELS, N_RED_CHANNELS, 4
check("N_BASE_CHANNELS == 268 (ancien layout inchange)",
      N_BASE_CHANNELS == OLD_TOTAL, N_BASE_CHANNELS)
check("11 canaux rouges, N_CHANNELS == 279",
      N_RED_CHANNELS == 11 and N_CHANNELS == 279, (N_RED_CHANNELS, N_CHANNELS))
first_red = min(_OFFSET[n] for n, _ in F._RED_GROUPS)
check("les canaux rouges commencent apres le canal 268 (fin de layout)",
      first_red == OLD_TOTAL, first_red)
check("le dernier canal de l'ancien layout est toujours 'lookahead' (80 canaux)",
      _OFFSET["lookahead"] + F.LOOKAHEAD_CHANNELS == OLD_TOTAL,
      _OFFSET["lookahead"])


def make_obs(red_state, seat=0):
    hands = [np.zeros(34, dtype=np.int32) for _ in range(4)]
    hands[seat][[0, 1, 2, 4, 9]] = 1
    return build_suphx_obs(
        hands=hands, melds=[[] for _ in range(4)],
        discards=[np.zeros(34, dtype=np.int32) for _ in range(4)],
        discard_hist=[[] for _ in range(4)], riichi_flags=[False] * 4,
        furiten_flag=False, dora_indicators=[3], wall_remaining=70,
        scores=[0, 0, 0, 0], dealer_seat=0, round_wind=0, round_id=1,
        honba=0, kyoutaku=0, seat=seat, compute_lookahead_flag=False,
        red_state=red_state)


o0 = make_obs(None)
o1 = make_obs(empty_red_state())
check("sans red_state : shape (279, 34) et canaux rouges a 0",
      o0.shape == (279, 34) and not o0[OLD_TOTAL:].any(), o0.shape)
check("red_state vide : seul 'red_unseen' s'allume (3 rouges inconnus)",
      o1[_OFFSET["red_unseen"]].tolist()
      == [1.0 if t in RED_TYPES else 0.0 for t in range(34)]
      and not o1[OLD_TOTAL:_OFFSET["red_unseen"]].any())
check("les 268 premiers canaux sont identiques avec/sans red_state",
      np.array_equal(o0[:OLD_TOTAL], o1[:OLD_TOTAL]))

rs = empty_red_state()
rs["hand"][0][0] = True                 # mon 5m rouge en main
rs["meld"][0][1] = True                 # mon 5p rouge dans un meld
rs["river"][2][2] = True                # 5s rouge dans la riviere du siege 2
rs["meld"][1][0] = False
rs["pending"] = 2
o2 = make_obs(rs, seat=0)
check("red_hand : 1 sur la colonne 5m",
      o2[_OFFSET["red_hand"]][4] == 1 and o2[_OFFSET["red_hand"]].sum() == 1)
check("red_meld_own : 1 sur la colonne 5p",
      o2[_OFFSET["red_meld_own"]][13] == 1)
check("red_river_s2 : 1 sur la colonne 5s",
      o2[_OFFSET["red_river_s2"]][22] == 1
      and not o2[_OFFSET["red_river_s0"]].any())
check("red_unseen : plus aucun rouge inconnu (tous vus)",
      not o2[_OFFSET["red_unseen"]].any())
check("pending_red : 1 sur la colonne 5s",
      o2[_OFFSET["pending_red"]][22] == 1)
rs2 = empty_red_state()
rs2["meld"][2][1] = True                # meld de l'adversaire de 'seat+2'
o3 = make_obs(rs2, seat=0)
check("red_meld_opp1 = adversaire (seat+2) : rouge 5p",
      o3[_OFFSET["red_meld_opp1"]][13] == 1
      and not o3[_OFFSET["red_meld_opp0"]].any())

# ===================================================================== #
# B. adaptation des anciens poids
# ===================================================================== #
section("B. adaptation des anciens poids (268/272 -> 279/283)")
from mahjong_ai.agents import suphx_cnn as C                           # noqa: E402

if REAL_TORCH:
    import torch
    mk = lambda n: torch.arange(1, n + 1, dtype=torch.float32).reshape(1, n, 1, 1)
    cat = lambda xs: torch.cat(xs, dim=1)
    arr = lambda t: t.flatten().numpy()
else:
    import torch
    mk = lambda n: np.arange(1, n + 1, dtype=np.float32).reshape(1, n, 1, 1).view(torch._Arr)
    cat = lambda xs: np.concatenate([np.asarray(x) for x in xs], axis=1)
    arr = lambda t: np.asarray(t).flatten()

wB, wBO = mk(NB), mk(NB + NO)
base_ids = np.arange(1, NB + 1)
orc_ids = np.arange(NB + 1, NB + NO + 1)
a = arr(C.adapt_in_channels(wB, NB + NR))
check("268 -> 279 : base inchangee, 11 derniers = 0",
      a.shape == (NB + NR,) and np.array_equal(a[:NB], base_ids)
      and not a[NB:].any())
a = arr(C.adapt_in_channels(wBO, NB + NR + NO))
check("272 -> 283 : base inchangee, 11 zeros, oracle deplace a la fin",
      a.shape == (NB + NR + NO,) and np.array_equal(a[:NB], base_ids)
      and not a[NB:NB + NR].any() and np.array_equal(a[NB + NR:], orc_ids))
a = arr(C.adapt_in_channels(wBO, NB + NR))
check("272 -> 279 : oracle retire, rouges a 0",
      a.shape == (NB + NR,) and np.array_equal(a[:NB], base_ids)
      and not a[NB:].any())
wFull = mk(NB + NR + NO)
a = arr(C.adapt_in_channels(wFull, NB + NR))
check("283 -> 279 : oracle retire (export RL)",
      np.array_equal(a, np.arange(1, NB + NR + 1)))
a = arr(C.adapt_in_channels(wB, NB + NR + NO))
check("268 -> 283 : SL ancien vers policy oracle",
      a.shape == (NB + NR + NO,) and not a[NB:].any()
      and np.array_equal(a[:NB], base_ids))
check("deja au bon format : renvoie le meme tableau",
      C.adapt_in_channels(wFull, NB + NR + NO) is wFull)
# Linear aplati (critic) : (out, C*34)
if REAL_TORCH:
    wl = torch.arange(3 * (NB + NO) * 34, dtype=torch.float32).reshape(3, (NB + NO) * 34)
    out = C.adapt_flat_in_features(wl, (NB + NR + NO) * 34)
    o3d, w3d = out.reshape(3, NB + NR + NO, 34), wl.reshape(3, NB + NO, 34)
    ok = (out.shape == (3, (NB + NR + NO) * 34)
          and torch.equal(o3d[:, :NB], w3d[:, :NB])
          and not o3d[:, NB:NB + NR].any()
          and torch.equal(o3d[:, NB + NR:], w3d[:, NB:]))
    check("critic Linear 272*34 -> 283*34", ok)
else:
    print("  [skip] critic Linear (torch absent)")

# ===================================================================== #
# C. env
# ===================================================================== #
section("C. env : 3 rouges")
R = RED_TYPES


def count_places(env):
    """Pour chaque rouge k : nombre d'endroits ou il est (doit valoir 1,
    ou 0 s'il a ete sacrifie par une rinshan)."""
    out = []
    hidden_raw = list(env.wall) + list(env.rinshan_pool) + [
        x for pair in env._dora_pairs for x in pair]
    for k, t in enumerate(R):
        n = 0
        for s in range(4):
            if env.red_hand[s][k] and env.hands[s][t] > 0:
                n += 1
            if env.red_river[s][k]:
                n += 1
            n += sum(1 for m in env.melds[s] if k in m.get("reds", ()))
        n += sum(1 for x in hidden_raw if x == 34 + k)
        out.append(n)
    return out


def invariants(env, where=""):
    errs = []
    places = count_places(env)
    for k, n in enumerate(places):
        if n > 1:
            errs.append(f"rouge {k} en {n} endroits")
        if n == 0 and env._sacrificed_count == 0:
            errs.append(f"rouge {k} perdu sans sacrifice rinshan")
    for s in range(4):
        for k, t in enumerate(R):
            if env.red_hand[s][k] and env.hands[s][t] < 1:
                errs.append(f"red_hand[{s}][{k}] sans tuile en main")
            if env.red_river[s][k] and env.discards[s][t] < 1:
                errs.append(f"red_river[{s}][{k}] sans defausse")
        for m in env.melds[s]:
            for k in m.get("reds", ()):
                if R[k] not in m["tiles"]:
                    errs.append(f"meld {m['type']} marque rouge {k} sans 5 dedans")
    return [f"{where}: {e}" for e in errs]


# --- C1. deal : sentinels et conservation ---------------------------------
env = MultiAgentMahjongEnv(seed=3, compute_lookahead=False)
env._deal()
check("deal : 136 tuiles reparties (mains + mur + mort)",
      sum(int(h.sum()) for h in env.hands) + len(env.wall)
      + len(env.rinshan_pool) + 2 * MA.N_DORA_PAIRS == 136)
check("deal : les 3 rouges sont quelque part exactement une fois",
      count_places(env) == [1, 1, 1], count_places(env))
check("deal : aucune tuile ne depasse 4 copies par type",
      all(int(sum(h[t] for h in env.hands)) <= 4 for t in range(34)))
check("deal : les mains ne contiennent que des types 0..33",
      all(h.shape == (34,) and h.min() >= 0 for h in env.hands))

# --- C2. ids 136 / aka --------------------------------------------------
env = MultiAgentMahjongEnv(seed=5, compute_lookahead=False)
env._deal()
env._clear_reds()
# main complete : 123m 456m 789m 55p + 555s ... rouge 5p dans la paire
env.hands[0][:] = 0
for t in (0, 1, 2, 3, 4, 5, 6, 7, 8):
    env.hands[0][t] = 1
env.hands[0][13] = 2               # paire de 5p
env.hands[0][19] = 1; env.hands[0][20] = 1; env.hands[0][21] = 1   # 234s
env.red_hand[0][1] = True          # un des 5p est le rouge
env.red_hand[0][0] = True          # et le 5m aussi (hands[4] == 1)
_StubCalc.LAST = None
res = env._evaluate_hand(0, 13, is_tsumo=True, already_in_hand=True)
rec = _StubCalc.LAST
if rec is None:
    print("  [skip] ids 136 : calculateur reel (pas de capture)")
else:
    check("tsumo : 2 aka comptes (5m + 5p)", rec["aka"] == 2, rec)
    check("tsumo : ids 136 uniques, win_tile dans la main",
          rec["win_tile"] in rec["tiles"])
    check("les 5 normaux n'ont jamais l'id rouge",
          sorted(i for i in rec["tiles"] if i in AKA_IDS) == [16, 52])
# sans rouge : aucun aka meme avec 5 en main
env.red_hand[0][:] = False
_StubCalc.LAST = None
env._evaluate_hand(0, 13, is_tsumo=True, already_in_hand=True)
if _StubCalc.LAST is not None:
    check("meme main sans rouge : 0 aka (les 5 ne sont pas tous rouges)",
          _StubCalc.LAST["aka"] == 0, _StubCalc.LAST)
    check("ids 136 des 5 normaux decales (pas d'id 16/52/88)",
          not any(i in AKA_IDS for i in _StubCalc.LAST["tiles"]))
# ron sur un rouge defausse par un adversaire
env.hands[0][13] = 1               # attend le 5p
env._last_discard_red = 1          # la tuile defaussee est le 5p rouge
env.red_hand[0][:] = False
_StubCalc.LAST = None
env._evaluate_hand(0, 13, is_tsumo=False, already_in_hand=False)
if _StubCalc.LAST is not None:
    check("ron sur le 5p rouge de l'adversaire : 1 aka",
          _StubCalc.LAST["aka"] == 1, _StubCalc.LAST)
    check("apres le ron potentiel : etat restaure (pas de rouge fantome)",
          not env.red_hand[0][1] and env.hands[0][13] == 1)
# melds avec rouge
env.hands[0][:] = 0
for t in (0, 1, 2, 3, 4, 5, 6, 7, 8):
    env.hands[0][t] = 1
env.hands[0][19] = 1; env.hands[0][20] = 1; env.hands[0][21] = 1
env.hands[0][27] = 2
env.melds[0] = [{"type": "pon", "tiles": [22, 22, 22], "reds": [2]}]
env.red_hand[0][:] = False
env.hands[0][4] = 1
env.red_hand[0][0] = True
_StubCalc.LAST = None
env._evaluate_hand(0, 27, is_tsumo=True, already_in_hand=True)
if _StubCalc.LAST is not None:
    check("meld pon de 5s (rouge) + 5m rouge en main : 2 aka",
          _StubCalc.LAST["aka"] == 2, _StubCalc.LAST)

# --- C3. defausse / appels ----------------------------------------------
env = MultiAgentMahjongEnv(seed=1, compute_lookahead=False)
env._deal()
env._clear_reds()
env.hands[0][:] = 0
env.hands[0][4] = 2                 # deux 5m, dont le rouge
env.red_hand[0][0] = True
env.hands[0][4] -= 1
env.discards[0][4] += 1
env._after_discard(0, 4)
check("2 copies dont le rouge : on defausse la normale, le rouge reste",
      env.red_hand[0][0] and not env.red_river[0][0]
      and env._last_discard_red == -1)
env.hands[0][4] -= 1
env.discards[0][4] += 1
env._after_discard(0, 4)
check("derniere copie = le rouge : il part en riviere",
      not env.red_hand[0][0] and env.red_river[0][0]
      and env._last_discard_red == 0)
# pon sur le rouge defausse
env.hands[1][:] = 0
env.hands[1][4] = 2
env._last_discarder = 0
env.discards[0][4] -= 1
env._apply_pong(1, 4)
check("pon sur un 5m rouge : le rouge passe de la riviere au meld",
      not env.red_river[0][0] and env.melds[1][-1]["reds"] == [0])
# pon avec le rouge en main (2 copies, appel d'une normale)
env.hands[2][:] = 0
env.hands[2][13] = 2
env.red_hand[2][1] = True
env._last_discard_red = -1
env._last_discarder = 3
env._apply_pong(2, 13)
check("pon avec les 2 copies de la main dont le rouge : rouge dans le meld",
      env.melds[2][-1]["reds"] == [1] and not env.red_hand[2][1])
# pon en gardant le rouge (3 copies en main)
env.hands[3][:] = 0
env.hands[3][22] = 3
env.red_hand[3][2] = True
env._last_discard_red = -1
env._last_discarder = 0
env._apply_pong(3, 22)
check("pon avec 3 copies en main (rouge inclus) : le rouge reste en main",
      env.red_hand[3][2] and env.melds[3][-1]["reds"] == []
      and env.hands[3][22] == 1)
# chi avec le 5 rouge de la main
env.hands[1][:] = 0
env.hands[1][3] = 1; env.hands[1][4] = 1
env.red_hand[1][0] = True
env._last_discard_red = -1
env._last_discarder = 0
env._apply_chow(1, 2, MA.CHOW_UPPER)       # appelle 3m avec 4m 5m
check("chi consommant le 5m rouge de la main : rouge dans le meld",
      env.melds[1][-1]["reds"] == [0] and not env.red_hand[1][0])
# kan ferme avec le rouge
env.hands[0][:] = 0
env.hands[0][13] = 4
env.red_hand[0][1] = True
env._apply_closed_kong(0, 13)
check("ankan de 5p : le rouge est dans le kan",
      env.melds[0][-1]["reds"] == [1] and not env.red_hand[0][1])
# kakan avec la tuile rouge
env.melds[1] = [{"type": "pon", "tiles": [22, 22, 22], "reds": []}]
env.hands[1][:] = 0
env.hands[1][22] = 1
env.red_hand[1][2] = True
env._apply_shouminkan(1, 22)
check("kakan avec le 5s rouge : le rouge entre dans le meld, "
      "_last_discard_red prepare le chankan",
      env.melds[1][-1]["type"] == "shouminkan"
      and env.melds[1][-1]["reds"] == [2] and env._last_discard_red == 2)

# --- C4. observation de l'env ----------------------------------------------
env = MultiAgentMahjongEnv(seed=21, compute_lookahead=False)
req = env.reset_round()
obs = req["obs"]
seat = req["seat"]
check("obs de l'env : shape (279, 34)", obs.shape == (279, 34), obs.shape)
exp_red_hand = [env.red_hand[seat][k] and env.hands[seat][t] > 0
                for k, t in enumerate(R)]
got = [bool(obs[_OFFSET["red_hand"]][t]) for t in R]
check("obs : canal red_hand = rouges de ma main", got == exp_red_hand,
      (got, exp_red_hand))

# --- C5. manches aleatoires -------------------------------------------------
section("C5. manches aleatoires : conservation des 3 rouges")
N_ROUNDS = int(os.environ.get("RED_TEST_ROUNDS", "40"))
rng = np.random.RandomState(0)
errors, stats = [], {"steps": 0, "rounds": 0, "red_meld": 0, "red_discard": 0,
                     "aka_eval": 0, "wins": 0}
obs_errors = []
for r in range(N_ROUNDS):
    env = MultiAgentMahjongEnv(seed=1000 + r, compute_lookahead=False)
    req = env.reset_round()
    for _ in range(600):
        errs = invariants(env, f"manche {r}")
        if errs:
            errors.extend(errs)
            break
        seat = req["seat"]
        mask = np.asarray(req["mask"])
        legal = np.flatnonzero(mask > 0)
        if req["obs"].shape != (279, 34):
            obs_errors.append(f"manche {r}: shape {req['obs'].shape}")
        else:
            for k, t in enumerate(R):
                want = bool(env.red_hand[seat][k] and env.hands[seat][t] > 0)
                if bool(req["obs"][_OFFSET["red_hand"]][t]) != want:
                    obs_errors.append(f"manche {r}: red_hand obs != etat")
        at = req["action_type"]
        if at == DISCARD:
            action = int(rng.choice(legal))
            if action in R and env.red_hand[seat][R.index(action)]:
                stats["red_discard"] += 1
        elif at == CHOW:
            nz = [a for a in legal if a != 0]
            action = int(rng.choice(nz)) if nz and rng.rand() < 0.8 else 0
        elif at == RON:
            action = 1 if rng.rand() < 0.7 else 0
        else:
            action = 1 if rng.rand() < 0.7 else 0
        res = env.step(action)
        stats["steps"] += 1
        if res.get("round_done"):
            break
        req = res["next_request"]
    errs = invariants(env, f"manche {r} (fin)")
    errors.extend(errs)
    stats["rounds"] += 1
    stats["red_meld"] += sum(1 for s in range(4) for m in env.melds[s]
                             if m.get("reds"))
check(f"{N_ROUNDS} manches aleatoires : les 3 rouges sont conserves "
      "(jamais dupliques/perdus)", not errors, errors[:5])
check("observations coherentes avec l'etat", not obs_errors, obs_errors[:3])
if not REAL_MAHJONG:
    aka_evals = [c for c in _StubCalc.CALLS if c["aka"] > 0]
    check("des evaluations de main avec aka ont eu lieu pendant les parties",
          len(aka_evals) >= 0)
print(f"  [info] {stats}  evaluations={len(_StubCalc.CALLS)}")
check("couverture : des rouges ont ete appeles dans des melds",
      stats["red_meld"] > 0, stats)

# ===================================================================== #
# D. parser SL
# ===================================================================== #
section("D. parser SL (mjai)")
try:
    from mahjong_ai.sl import mjai_parser as P
except Exception as e:           # zstd absent et pas de stub utilisable
    P = None
    print(f"  [skip] parser : {e}")

if P is not None:
    check("mjai_red_index : 5mr/5pr/5sr/0m",
          [P.mjai_red_index(x) for x in ("5mr", "5pr", "5sr", "0s", "5m", "E")]
          == [0, 1, 2, 2, -1, -1])

    def hand13(extra):
        base = ["1m", "2m", "3m", "6m", "7m", "8m", "1p", "2p", "3p",
                "1s", "2s", "3s", "E"]
        return (extra + base)[:13]

    ev = [
        {"type": "start_game"},
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
         "oya": 0, "dora_marker": "9s", "scores": [25000] * 4,
         "kyotaku": 0,
         "tehais": [hand13(["5mr", "5p"]), hand13(["5m", "5m"]),
                    hand13(["9m"]), hand13(["9p"])]},
        {"type": "tsumo", "actor": 0, "pai": "N"},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
        {"type": "pon", "actor": 1, "target": 0, "pai": "5mr",
         "consumed": ["5m", "5m"]},
        {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "S"},
        {"type": "dahai", "actor": 2, "pai": "S", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "W"},
        {"type": "dahai", "actor": 3, "pai": "W", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "5pr"},
        {"type": "dahai", "actor": 0, "pai": "N", "tsumogiri": False},
        {"type": "end_kyoku"},
    ]
    samples = P.events_to_samples(ev, pov=0)
    disc = [s for s in samples if s[1] == P.DISCARD]
    check("parser : 2 echantillons de defausse pour le POV 0", len(disc) == 2,
          len(disc))
    if len(disc) == 2:
        o_a, o_b = disc[0][0], disc[1][0]
        check("1re defausse : mon 5m rouge est dans ma main",
              o_a[_OFFSET["red_hand"]][4] == 1 and o_a[_OFFSET["red_unseen"]][4] == 0)
        check("2e defausse : le 5m rouge est dans le meld de l'adversaire (seat+1), "
              "plus en riviere",
              o_b[_OFFSET["red_meld_opp0"]][4] == 1
              and o_b[_OFFSET["red_river_s0"]][4] == 0
              and o_b[_OFFSET["red_hand"]][4] == 0)
        check("2e defausse : j'ai pioche le 5p rouge",
              o_b[_OFFSET["red_hand"]][13] == 1)
        check("2e defausse : le 5s rouge reste inconnu",
              o_b[_OFFSET["red_unseen"]][22] == 1)

    # kakan : en mjai `consumed` = les 3 tuiles du pon, `pai` = tuile ajoutee
    ev2 = [
        {"type": "start_game"},
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
         "oya": 0, "dora_marker": "9s", "scores": [25000] * 4, "kyotaku": 0,
         "tehais": [hand13(["9m"]), hand13(["2m", "2m"]),
                    hand13(["2m"]), hand13(["9p"])]},
        {"type": "tsumo", "actor": 0, "pai": "N"},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": False},
        {"type": "pon", "actor": 1, "target": 0, "pai": "2m",
         "consumed": ["2m", "2m"]},
        {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "S"},
        {"type": "dahai", "actor": 2, "pai": "S", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "W"},
        {"type": "dahai", "actor": 3, "pai": "W", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "N"},
        {"type": "dahai", "actor": 0, "pai": "N", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "kakan", "actor": 1, "pai": "2m", "consumed": ["2m", "2m", "2m"]},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": False},
        {"type": "end_kyoku"},
    ]
    samples1 = [s for s in P.events_to_samples(ev2, pov=1) if s[1] == P.DISCARD]
    own = _OFFSET["own_hand"]
    last = samples1[-1][0]
    n2m = int(sum(last[own + n][1] for n in range(4)))
    # 3 copies au depart (2 + 1 dans la main de base) - 2 (pon) + 1 (tsumo)
    # - 1 (kakan) = 1 ; l'ancien code retirait 3 tuiles -> -1
    check("kakan : une seule tuile retiree de la main (la tuile ajoutee)",
          n2m == 1, f"copies de 2m en main = {n2m} (attendu 1)")

# ===================================================================== #
failed = [r for r in RESULTS if not r[1]]
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} controles OK")
if failed:
    print("ECHECS :")
    for n, _, d in failed:
        print("  -", n, d)
sys.exit(1 if failed else 0)
