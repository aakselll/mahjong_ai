================================================================================
MAHJONG AI - SUPHX REPRODUCTION (REDUCED SCALE)
================================================================================

A PyTorch implementation of Suphx (Microsoft Research Asia, 2020), an AI for
4-player Japanese Riichi Mahjong based on deep reinforcement learning, at a
reduced parameter scale (~7M params per SL model instead of ~21M in the paper).

Reference: Suphx: Mastering Mahjong with Deep Reinforcement Learning
           arXiv:2003.13590 (https://arxiv.org/abs/2003.13590)


================================================================================
TABLE OF CONTENTS
================================================================================

  1. Overview
  2. Pipeline
  3. Architecture
  4. Input Features
  5. Directory Structure
  6. Installation
  7. Data Preparation
  8. Training
  9. Inference / Play
 10. Deviations from the Paper
 11. Performance Notes
 12. References


================================================================================
1. OVERVIEW
================================================================================

Suphx is a Mahjong AI that reached 10-dan on Tenhou.net (the highest rank) and
is rated above 99.99% of ranked human players. Its contributions are:

  1. Global Reward Prediction (GRP)
     A GRU-based predictor that maps the final game reward (rank-based) to
     per-round signals, enabling tactical play (e.g. defend when leading in the
     last round) without hand-coded rules.

  2. Oracle Guiding
     An oracle agent with access to perfect information (opponents' hands,
     wall) is gradually transitioned to a normal agent via feature dropout,
     accelerating RL training.

  3. Parametric Monte-Carlo Policy Adaptation (pMCPA)
     At run time, the offline policy is fine-tuned to the current round's
     initial hand using simulated rollouts, without building a regular game
     tree.

This repository reproduces the supervised learning (SL) stage and the GRP
component at reduced scale, using the public dataset:
    nikketryhard/tenhou-houou-mjai  (HuggingFace)


================================================================================
2. PIPELINE
================================================================================

  +----------------------------------------------------------------------+
  |  STAGE 1 - Supervised Learning (SL)                                  |
  |                                                                      |
  |   .mjai.json.zst  -->  mjai_parser  -->  PolicyDataset               |
  |                                              |                       |
  |                                              v                       |
  |                         5 independent CNNs (Suphx architecture)      |
  |                          * discard  (34 logits)                      |
  |                          * riichi   (2 logits)                       |
  |                          * chow     (4 logits: pass/lower/mid/upper) |
  |                          * pong     (2 logits)                       |
  |                          * kong     (2 logits)                       |
  |                                                                      |
  |   Loss: cross-entropy, split train/val by FILE (no leakage)          |
  +----------------------------------------------------------------------+
                                    |
                                    v
  +----------------------------------------------------------------------+
  |  STAGE 2 - Global Reward Predictor (GRP)                             |
  |                                                                      |
  |   GameLogBuffer (per-round feature sequence + final rank reward)     |
  |                                              |                       |
  |                                              v                       |
  |                2-layer GRU + 2 FC -> predicts final game reward      |
  |                                                                      |
  |   Per-round RL reward:  r_k = Phi(x^k) - Phi(x^{k-1})                |
  |   (replaces raw round score, captures "rank protection" behaviour)   |
  +----------------------------------------------------------------------+
                                    |
                                    v
  +----------------------------------------------------------------------+
  |  STAGE 3 - Reinforcement Learning (RL)                               |
  |                                                                      |
  |   Distributed PPO self-play (vectorized_runner.py)                   |
  |   + entropy regularization                                           |
  |   + optional oracle guiding (perfect-info dropout schedule)          |
  +----------------------------------------------------------------------+


================================================================================
3. ARCHITECTURE
================================================================================

CNN trunk (shared by all 5 SL models)
-------------------------------------

Faithful to Suphx section 2.2 / Figures 4 and 5:

  * Input:  (N_CHANNELS, 34, 1)  - no pooling
  * 3x1 conv, C=256, BatchNorm, ReLU
  * N_blocks residual blocks of two 3x1 conv C -> C with skip connections
  * Discard head: 1x1 conv 1 -> flatten -> 34 logits
  * Binary heads (riichi/chow/pong/kong):
        3x1 conv 32 -> flatten -> 2 FC layers -> 2 or 4 logits

Default (reduced scale):  C=256, N_blocks=16  ->  ~6.7M params per model
Reference paper scale:    C=256, N_blocks=50  ->  ~21M params

Reward Predictor (GRP)
----------------------

  * Input: per-round feature vector (FEATURE_DIM=8) sequence
  * 2-layer GRU (hidden=64 or 128) + 2 FC layers -> scalar output
  * Trained by MSE on final rank rewards
  * Frozen during RL (as in the paper)


================================================================================
4. INPUT FEATURES
================================================================================

The observation is a matrix (N_CHANNELS=250, 34), one row per tile type.
This follows Suphx section 2.2: each feature occupies a 34-wide channel.

  Group                          Channels    Encoding
  ---------------------------------------------------------------------------
  Own hand                          4         "n copies" (Fig. 3):
                                              channel n = 1 if count >= n+1
  Own open melds                    4         same
  Own closed kan                    4         same
  Discards per seat (x4)            4 each    "n copies" per seat
  Last discard per seat (x4)        1 each    one-hot
  Tsumogiri count per seat (x4)     1 each    count per tile
  Opponent melds (x3)               4 each    "n copies"
  Dora indicators                   1         count
  Dora tiles (derived)              1         count
  Pending claim tile                1         one-hot
  Unseen tile counts                1         normalized [0, 1]
  Own riichi                        1         all-ones if active
  Opponents' riichi (x3)            1 each    all-ones if active
  Own furiten                       1         all-ones if permanent furiten
  Seat wind                         4         one-hot
  Round wind                        4         one-hot
  Dealer                            1         binary
  Round id (1-4)                    4         one-hot
  Honba (repeat dealer)             5         bucket (0,1,2,3,4+)
  Kyoutaku (riichi sticks)          5         bucket
  Own score                        20         bucketed
  Each opponent's score (x3)       20 each    bucketed
  Wall remaining                   10         bucketed
  Look-ahead                       80         see below

Look-ahead features (Suphx section 2.2)
---------------------------------------

The paper says:
  "a feature represents whether discarding a specific tile can lead to a
   winning hand of 12,000 round score with replacing 3 hand tiles by tiles
   drawn from the wall or discarded by other players."

Implemented as 4 x 20 = 80 channels:

  * 4 replacement depths: n_replace in {1, 2, 3, 4}
  * 20 score buckets (0, 1k, 1.5k, 2k, ..., 96k, inf)

For each discardable tile d, we run a bounded DFS to enumerate reachable
winning hands, score each with mahjong.hand_calculator.HandCalculator
(including existing melds), and mark the corresponding bucket.

Optimizations applied (speed vs. fidelity trade-offs):
  * Skip DFS when shanten > 2 (mains too far from tenpai)
  * MAX_DFS_RESULTS = 30 per DFS call
  * MAX_VISITED = 300 nodes per DFS call
  * Two-layer lru_cache: DFS results cached independently from scoring results
  * HandConfig cached per (seat_wind, round_wind)

Known deviations:
  * Paper says "100+ look-ahead features"; we use 80
  * Paper example uses "3 replacements"; we use up to 4
  * MAX_DFS_RESULTS = 30 may truncate long tails of rare winning hands


================================================================================
5. DIRECTORY STRUCTURE
================================================================================

mahjong_ai/
|
+-- envs/
|   +-- tiles.py                 # shanten, tile constants
|   +-- multi_agent.py           # active game environment (RL)
|   +-- outcomes.py              # round outcome classification
|   +-- suphx_features.py        # (N_CHANNELS, 34) builder - shared SL/RL
|
+-- sl/                          # SUPERVISED LEARNING
|   +-- mjai_parser.py           # .mjai.json.zst -> (obs, action, mask)
|   +-- download_mjai.py         # downloads annual shards from HuggingFace
|   +-- dataset.py               # PolicyDataset, split-by-file
|   +-- train_policy_sl.py       # trains ONE head
|   +-- train_all_sl.py          # orchestrates the 5 heads
|
+-- agents/
|   +-- suphx_cnn.py             # CNN trunk + heads (Fig. 4, 5)
|   +-- reward_predictor.py      # GRU Phi for GRP
|   +-- train_grp_sl.py          # GRP supervised training on human logs
|
+-- training/                    # RL
|   +-- game_env.py              # multi-round wrapper + GRP attribution
|   +-- ppo_core.py              # PPO single-model
|   +-- vectorized_runner.py     # distributed self-play
|   +-- train_rl.py              # orchestrates RL for the 5 heads
|
+-- play/
    +-- game.py                  # local inference
    +-- sniffer_mahjong.py       # adapter for online play


================================================================================
6. INSTALLATION
================================================================================

Requires Python 3.10+ (tested on 3.14). Tested on Windows and Linux.

  git clone <this-repo>
  cd MahjongE
  pip install -e .

Dependencies: torch, numpy, zstandard, mahjong, huggingface_hub.


================================================================================
7. DATA PREPARATION
================================================================================

Download one or more annual shards from HuggingFace:

  python -m mahjong_ai.sl.download_mjai --year 2015
  python -m mahjong_ai.sl.download_mjai --year 2016

This produces data/tenhou/2015/*.mjai.json.zst (~150k files per year, each
containing 1 complete hanchan with all 4 players' perspectives).


================================================================================
8. TRAINING
================================================================================

8.1 Supervised Learning (5 heads)
---------------------------------

  python -m mahjong_ai.sl.train_all_sl \
      --data-dir data/tenhou/2015 \
      --output-dir checkpoints/sl_2015 \
      --cache-dir cache/sl_2015 \
      --max-files 1000 --epochs 2 --batch-size 128 --num-workers 0

Options:
  --heads discard riichi       train a subset
  --skip-existing              skip heads whose .pt already exists
  --channels 256 --n-blocks 16 architecture knobs

Caching: each head has its own samples_<head>.pkl in --cache-dir.
Re-running with the SAME arguments loads the cache (seconds instead of
minutes). Changing --max-files, --data-dir, or N_CHANNELS does NOT invalidate
the cache - delete cache/sl_2015/ manually to force re-parse.

8.2 Global Reward Predictor
---------------------------

  python -m mahjong_ai.agents.train_grp_sl \
      --data-dir data/tenhou/2015 \
      --output checkpoints/grp_sl.pt \
      --max-files 50000 --epochs 10

Then use --warm-start-predictor checkpoints/grp_sl.pt in RL training.

8.3 Reinforcement Learning
--------------------------

  python -m mahjong_ai.training.train_rl \
      --heads discard \
      --warm-start checkpoints/sl_2015 \
      --warm-start-predictor checkpoints/grp_sl.pt \
      --steps 1000000


================================================================================
9. INFERENCE / PLAY
================================================================================

  python -m mahjong_ai.play.game \
      --checkpoint checkpoints/sl_2015 \
      --opponents random


================================================================================
10. DEVIATIONS FROM THE PAPER
================================================================================

This is a reduced-scale, partial reproduction. Honest list of gaps:

  #   Paper                          This repo                    Impact
  --------------------------------------------------------------------------
  1   50 residual blocks, 21M params 16 blocks, 6.7M params      Intentional
  2   "100+ look-ahead features"     80 channels                  Reduced
  3   Look-ahead depth: up to 3      up to 4                      Comparable
  4   100M+ rounds dataset           150k files/year              Smaller
  5   Discards as full sequence      4-canaux count + last +      Slight loss
                                     tsumogiri                    of ordering
  6   Temporary furiten tracked      only permanent furiten       Negligible
  7   Oracle guiding, pMCPA          not yet implemented          Missing
  8   CNN without pooling            same                         -
  9   5 models + rule-based winner   same                         -
 10   GRP: 2-layer GRU               same                         -

Known bugs fixed:
  * multi_agent.py used to import a stale environment snapshot without Ron
    decision; RL now trains Ron properly.
  * game_env.py get_stats() / reset_stats() were missing, silently swallowing
    an AttributeError and logging zeros for all RL stats.


================================================================================
11. PERFORMANCE NOTES
================================================================================

Parsing speed
-------------

Parsing dominates when look-ahead is enabled:

  Config                            Files/s    1000 files
  -----------------------------------------------------------------
  Look-ahead OFF                    3-10       3-5 min
  Look-ahead ON, aggressive opt.    0.03-0.05  6-8 h
  Look-ahead ON, paper-faithful     0.01-0.02  15-20 h

Recommendation: for the first runs, set compute_lookahead_flag=False in both
mjai_parser.build_obs and multi_agent._build_obs. Validate the training loop
converges, then enable look-ahead if you have GPU hours to spare.

Training speed
--------------

Per head, on a single mid-range GPU:

  Head                          1 epoch on 1000 files
  ---------------------------------------------------
  discard                       ~3-5 min
  riichi / chow / pong / kong   ~1-2 min

Memory
------

  * Model:                ~27 MB (float32)
  * Batch (256, 250, 34): ~8.5 MB
  * Dataset cache pickle: ~200 MB per head per 1000 files
  * LRU caches:           ~30 MB total


================================================================================
12. REFERENCES
================================================================================

  * Li, J., et al. Suphx: Mastering Mahjong with Deep Reinforcement Learning.
    arXiv:2003.13590, 2020.

  * Silver, D., et al. Mastering the game of Go without human knowledge.
    Nature 550, 2017.

  * nikketryhard/tenhou-houou-mjai dataset on HuggingFace.

  * Tenhou.net (https://tenhou.net/) - the platform used for evaluation.


================================================================================
LICENSE & ACKNOWLEDGEMENTS
================================================================================

This project is a research reproduction. It relies on the public dataset
nikketryhard/tenhou-houou-mjai. All credit for the original Suphx architecture
and training pipeline goes to the Microsoft Research Asia team.

================================================================================