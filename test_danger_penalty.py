"""Penalite de danger dans le worker (stubs, sans torch reel)."""
import sys, types
import numpy as np
import test_stubs  # noqa: F401
def _mod(name, **kw):
    m = types.ModuleType(name); m.__dict__.update(kw); sys.modules[name] = m
_mod('mahjong_ai.training.game_env', GameEnv=object, PredictorHolder=object)
_mod('mahjong_ai.agents.oracle_agent', OracleDiscardAgent=object)
_mod('mahjong_ai.training.parameter_server', fetch_from_state=None)
_mod('mahjong_ai.agents.reward_predictor', RewardPredictor=object, FEATURE_DIM=12)
from mahjong_ai.training import self_play_worker as W

ok = True
def check(c, m):
    global ok
    print(("  [OK  ] " if c else "  [FAIL] ") + m); ok &= bool(c)

class FakeEnv:
    def __init__(self, with_danger):
        self.game_scores=[0.,0.,0.,0.]; self.honba=0; self.wd=with_danger; self.rd=0
    def get_stats(self): return {}
    def reset_round(self):
        self.n=0; self.rd+=1; return self._req()
    def _req(self):
        r = dict(seat=self.n%4, action_type=W.DISCARD, obs=np.zeros(3),
                 oracle=np.zeros(2), mask=np.ones(34))
        if self.wd:
            d = np.zeros(34, dtype=np.float32); d[1] = 1.0   # tuile 1 = dangereuse
            r['danger'] = d
        return r
    def step(self, a):
        self.n += 1
        if self.n >= 8:
            fr = {0:0., 1:0., 2:0., 3:0.}
            return dict(round_done=True, outcome='exhaustive', final_rewards=fr, game_done=self.rd>=1)
        return dict(round_done=False, next_request=self._req())

class FakeAgent:
    def __init__(self, act): self.act=act
    def select_action(self,o,oo,m,g): return self.act,0.,0.,oo

for act, exp_pen in [(1, True), (2, False)]:
    tr, st = W._play_one_game(FakeEnv(True), FakeAgent(act), 1.0, danger_penalty=0.7)
    rew = [t['reward'] for t in tr]
    if exp_pen:
        check(all(abs(r + 0.7) < 1e-6 for r in rew), f"action dangereuse : reward = -0.7 sur chaque decision ({rew[:2]}...)")
        check(st['n_danger_discards'] == st['n_rl_discards'] == len(tr), "compteurs dng = 100 %")
    else:
        check(all(r == 0.0 for r in rew), "action sure : pas de penalite")
        check(st['n_danger_discards'] == 0 and st['n_rl_discards'] == len(tr), "compteurs dng = 0 %")
# flag inactif : pas de cle 'danger' dans les requetes -> comportement inchange
tr, st = W._play_one_game(FakeEnv(False), FakeAgent(1), 1.0, danger_penalty=0.0)
check(all(t['reward'] == 0.0 for t in tr) and st['n_rl_discards'] == 0, "penalite desactivee : comportement inchange")
print("OK" if ok else "ECHEC"); sys.exit(0 if ok else 1)
