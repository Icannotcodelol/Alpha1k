# Alpha1kMASTER PROMPT — Alpha‑1000 (Tysiąc 1v1) — Engine + Minimal UI + PPO‑LSTM (with optional Bombing)

You are a senior Python engineer. Build a runnable project called alpha‑1000 that implements the 2‑player Polish card game Tysiąc (Thousand) exactly as specified below, with a minimal Streamlit UI, unit tests, baseline bots, and a PPO‑LSTM agent for self‑play training. Include an optional Bombing (Bomba) rule as a toggle in the rules YAML (default: off). Keep the code clean and pass all tests.

⸻

1) Authoritative Rules (implement exactly)

Deck: 24 cards (ranks per suit: 9, J, Q, K, 10, A).
Card points: 9=0, J=2, Q=3, K=4, 10=10, A=11.
Ranks high→low in a suit: A > 10 > K > Q > J > 9.

Deal (2 players): 10 cards each. The remaining 4 cards form two face‑down musiki (2+2). Dealer alternates each hand.

Bidding (auction):
	•	Starts at 100, increments of 10, or Pass. Highest bid wins; winner is the playing player.
	•	Proof requirement: any bid > 120 may be challenged; the bidder must show a meld (K+Q of the same suit) whose value ≥ (bid − 100).
Meld values: ♠=40, ♣=60, ♦=80, ♥=100.
Showing proof during bidding does not set trump and does not consume the meld.
	•	After winning the auction, the playing player chooses one musik (turns face‑up), adds both cards (hand size 12), then returns exactly 2 cards face‑down to the unused musik (default) so both players have 10 cards again. (Make this target configurable; default = “return to unused musik”.)

Trump & Melds (dynamic):
	•	Initially no trump.
	•	Declaring a meld (holding K+Q of a suit) is only allowed by the trick leader and only when playing one of the pair (the other is shown).
	•	Declaring a meld immediately sets trump to that suit; any later meld by either player immediately overrides trump.
	•	Each player may declare each suit at most once per hand.
	•	Meld points are scored by the player who declares them.

Play (tricks):
	•	Playing player leads the first trick.
	•	Must follow suit if possible.
	•	Must overtake the current highest card in the led suit if possible (i.e., if you follow suit and can beat the currently winning card, you must).
	•	Must overtrump if a trump is currently winning and you cannot follow suit but can play a higher trump.
	•	If you cannot follow suit, you may play any card; the trick is won by the highest trump, otherwise by the highest card of the led suit.
	•	A meld declaration changes trump before the current trick is resolved.

Scoring per hand:
	•	Each player tallies card points + meld points from tricks they won.
	•	Defender: adds their total rounded up to the nearest 10 to their game score.
	•	Playing player: does not add raw points; instead they declare a contract (must be ≥ winning bid and a multiple of 10).
	•	If achieved or exceeded, add the contract to game score.
	•	If failed, subtract the contract from game score.
	•	800‑lock: once a player reaches ≥800, they may only advance further by being the playing player in future hands (defender rounding does not move them beyond 800).
	•	Game end: first to ≥1000 wins. (No overshoot penalty by default.)

Bombing (Bomba) — optional rule (default: disabled):
	•	Toggle via YAML: bomba.enabled: false|true.
	•	If enabled, immediately after the playing player returns 2 cards and before the first lead, the defender may call “Bomb” to double all scoring for this hand (both the playing player’s contract result and defender’s rounded points).
	•	Optional redouble: bomba.allow_redouble: false|true (default: false). If true, the playing player may respond “Re‑Bomb” to redouble (×4). No further levels.
	•	Doubling applies after defender rounding and after contract success/failure is determined.
	•	Bombing does not change 800‑lock behavior.

All of the above must be configurable in rules_tysiac.yaml with safe defaults matching this spec.

⸻

2) Deliverables / Repository

alpha-1000/
  engine/
    cards.py                # Suit/Rank/Card, order, points, helpers
    rules_schema.py         # Pydantic model + validation for rules YAML
    rules_tysiac.yaml       # Encodes the rules (including bomba options)
    state.py                # GameState (hands, trick, trump, melds, bids, musiki, scores, history, rng)
    bidding.py              # Auction loop, proof check, musik take/return, set contract
    marriages.py            # declare_meld() → sets/overrides trump immediately; track per-player per-suit
    mechanics.py            # legal_moves(): follow, must-overtake, must-overtrump; action masking
    trick.py                # Trick structure + resolve_trick(); winner logic
    scoring.py              # tally, defender rounding, contract apply, 800-lock, bomba multipliers
    encode.py               # observation encoding + action masks (bidding vs play phases)
  bots/
    baseline_greedy.py
    baseline_trump_manager.py
    bot_arena.py            # CLI to pit bots; deterministic seeds for tests
  ui/
    app_streamlit.py        # Minimal UI: human vs bot, bidding console (proof UI), trump badge, meld log, trick viewer
  rl/ppo_lstm/
    net.py                  # Shared MLP + 1×LSTM; heads: bid policy, play policy, value
    selfplay.py             # Parallel workers; batched inference (MPS/CPU); save trajectories
    train.py                # PPO-clip + entropy + value; GAE; checkpoints; logs
    eval_arena.py           # Round-robin vs baselines and prior checkpoints; CSV report
  tests/
    test_bidding_justify.py
    test_bidding_increments.py
    test_musik_take_return.py
    test_meld_sets_trump_now.py
    test_trump_override_midhand.py
    test_must_overtake_and_overtrump.py
    test_scoring_contract_and_rounding.py
    test_800_lock.py
    test_bomba_optional.py   # Only active when bomba.enabled=true
    golden_deals/            # Fixed hands/seeds for regression
  scripts/
    quickstart_mac.sh
    run_tests.sh
  pyproject.toml
  README.md
  PROJECT_LOG.md            # Append brief dev logs automatically

  Quality constraints
	•	Python 3.11; clean, typed, documented functions.
	•	All tests must pass locally (macOS; CPU OK; use MPS if available).
	•	Illegal moves must be masked in both UI and RL action space.
	•	Keep the engine deterministic for given seeds.
	•	Keep modules small and readable; avoid premature optimization.

⸻

3) Tests (must implement & make green)
	1.	Bidding Proof
	•	bid=200 without proof ⇒ MissingMeldProof
	•	bid=200 with ♥K+♥Q ⇒ OK
	•	bid=180 with ♠K+♠Q (40) ⇒ InvalidMeldProof; with ♦K+♦Q (80) ⇒ OK
	•	bid=130 with ♠K+♠Q (40) ⇒ OK (threshold policy ≥ bid−100).
	•	bid≤120 requires no proof.
	•	Proof must be K+Q same suit and both cards in hand; showing proof does not set trump or mark meld as used.
	2.	Bidding increments
	•	Bids must rise by exactly 10; jumping/flat bids invalid.
	•	Once a player passes, they cannot re-enter.
	3.	Musik handling
	•	Winner reveals chosen musik (2), hand→12, returns exactly 2 face‑down to unused musik ⇒ hands back to 10/10.
	4.	Meld → trump timing (same trick)
	•	If leader declares ♦, trump=♦ immediately before resolving the current trick; a ♦J played by follower can now beat non‑♦ A.
	5.	Trump override midhand
	•	Later ♥ meld overrides ♦; next comparisons use ♥ as trump.
	6.	Must overtake / overtrump
	•	If following suit and able to beat the current winner, lower cards are illegal.
	•	If trump is winning and you can overtrump, playing a lower trump is illegal.
	7.	Scoring & rounding
	•	Defender rounds up to nearest 10; playing player adds/subtracts declared contract (≥ winning bid, multiple of 10).
	8.	800‑lock
	•	Defender at 800+ does not advance from rounding; must be playing player to progress.
	9.	Bombing (optional)
	•	With bomba.enabled=true: after musik return and before first lead, defender may “Bomb”:
	•	If contract 150 succeeds ⇒ playing player +300; defender rounded points ×2.
	•	If contract 150 fails ⇒ playing player −300; defender rounded points ×2.
	•	With redouble enabled (allow_redouble=true): “Re‑Bomb” ⇒ ×4.
	•	Bombing window closes once the first trick begins.

Provide fixtures (“golden deals”) to lock tricky timing cases.

⸻

4) Minimal UI (Streamlit)
	•	Hand display with illegal cards greyed/disabled.
	•	Bidding console with challenge button (enforces proof if bid >120); showing proof does not affect trump.
	•	Live trump badge that updates immediately on meld; Meld log (who, suit, points, trick index).
	•	Trick viewer (current trick + previous).
	•	Menu to toggle rules profile (load from YAML; optional enable Bomba).
	•	Human vs Greedy bot; bot vs bot fast‑forward for quick sanity runs.

⸻

5) PPO‑LSTM Agent (rl/ppo_lstm)

Action spaces
	•	Bidding phase: {Pass, 100, 110, 120, …, MAX} (mask invalid/too‑low; mask proof actions, which are UI‑only).
	•	Play phase: card index actions; when leader and holding K+Q not yet declared, include composite action “Declare‑Meld‑and‑Play [first-of-pair]” (environment applies meld points & immediate trump switch).

Observations (mask illegal moves)
	•	Hand (24‑hot); seen/played cards (24‑hot).
	•	Current trick encoding (led suit/card, who leads, current winner).
	•	Trump (None/♠/♣/♦/♥).
	•	Declared melds (2×4 binary).
	•	Bidding context (high bid, you/opp passed, are you bidder).
	•	Role flag (playing player vs defender).
	•	Phase flag (bidding/play).
	•	Optional features: suit counts remaining (public inference), trick index, musiki taken.

Network
	•	Shared MLP trunk → 1×LSTM (hidden 128–256) → heads:
	•	π_bid (bidding)
	•	π_play (masked over legal plays)
	•	V (value)

Training
	•	PPO‑clip + entropy + value loss; GAE; parallel CPU workers; batched inference (MPS if available, CPU fallback).
	•	Reward = game outcome (win=+1, loss=−1) or normalized score margin; tiny shaping allowed (+ε contract made, −ε overshoot if ever enabled).
	•	Evaluate every N updates vs Greedy and TrumpManager; write CSV.

Expose CLI:
python rl/ppo_lstm/train.py --rules engine/rules_tysiac.yaml --workers 6 --steps-per-update 4096 --updates 50
python rl/ppo_lstm/eval_arena.py --checkpoint data/checkpoints/latest.pt

6) README & scripts

README.md must include:
	•	Rule summary (exactly this spec), including Bombing toggle.
	•	How to run tests, UI, bots, and training.
	•	MPS note for Apple Silicon; CPU fallback.
	•	“Assumptions” section for any unspecified minor choices.
	•	“Troubleshooting” (common errors and fixes).

scripts/quickstart_mac.sh
	•	Create venv, install deps, run tests, launch UI.

PROJECT_LOG.md
	•	Append a 3–6 line entry after each milestone (engine green, UI ok, PPO first win >55%, etc.).

⸻

7) Run locally (print these at the end)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
streamlit run ui/app_streamlit.py

# Bots
python bots/bot_arena.py --rules engine/rules_tysiac.yaml --n 1000

# Train PPO-LSTM
python rl/ppo_lstm/train.py --rules engine/rules_tysiac.yaml --workers 6 --steps-per-update 4096 --updates 50
python rl/ppo_lstm/eval_arena.py --checkpoint data/checkpoints/latest.pt


⸻

8) Development policy (very important)
	•	Do not change rules unless explicitly stated.
	•	If a detail is missing, choose the simplest conventional option and document it in README → “Assumptions”.
	•	When tests fail, fix only the failing parts; do not add features.
	•	Keep functions small, typed, and covered by tests.
