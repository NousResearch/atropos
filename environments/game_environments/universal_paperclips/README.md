# Universal Paperclips LLM Agent

An LLM agent that plays the [Universal Paperclips](https://www.decisionproblem.com/paperclips/index2.html) incremental game. The stage 1 of the game is fully supported and stages 2&3 are still WIP!

## Architecture

The core architecture is shown below.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PaperclipsAtroposEnv                            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    setup() / teardown()                            │ │
│  │                    Manages shared Playwright Browser               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                  │                                      │
│                    ┌─────────────┼─────────────┐                        │
│                    ▼             ▼             ▼                        │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐        │
│  │  EpisodeContext  │ │  EpisodeContext  │ │  EpisodeContext  │   ...  │
│  │    (Episode 1)   │ │    (Episode 2)   │ │    (Episode 3)   │        │
│  │                  │ │                  │ │                  │        │
│  │ - Own context    │ │ - Own context    │ │ - Own context    │        │
│  │ - Isolated State │ │ - Isolated State │ | - Isolated State │
│  │ - Fresh game     │ │ - Fresh game     │ │ - Fresh game     │        │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘        │
│           │                    │                    │                   │
│           └────────────────────┼────────────────────┘                   │
│                                ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                collect_trajectories() (parallel)                   │ │
│  │                Communicates with LLM & saves JSONL                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Universal Paperclips Game                            │
│                  (https://decisionproblem.com/paperclips)               │
└─────────────────────────────────────────────────────────────────────────┘
```

## File structure

| File | Description |
|------|-------------|
| `universal_paperclips_env.py` | Main Atropos environment class (`PaperclipsAtroposEnv`) and `EpisodeContext` for browser management. |
| `config.py` | Configuration classes (`PaperclipsEnvConfig`) and system prompts for the LLM agent. |
| `js_scripts/` | Local copies of the original game source files only for reference. |

## Installation

Some additional requirements are needed to run this env, and can be installed as follows:

```bash
pip install -r environments/game_environments/universal_paperclips/requirements.txt
```

## Actions
The agent is given the list of available (available at this point in the game) actions that aren't all necessarily affordable acc to the current resources. This is done so that the agent also learns not to waste steps clicking on unaffordable actions. All the active projects also form a part of this list, and as is the case with other actions, they aren't all unlockable.

The agent can perform these actions in Stage 1 (Human Stage):

| Category | Action | Description |
|----------|--------|-------------|
| **Core** | `make_paperclip` | Manually produce one paperclip |
| | `wait` | Do nothing for this step |
| **Manufacturing** | `buy_wire` | Purchase a spool of wire |
| | `buy_autoclipper` | Buy an automated clipping machine |
| | `buy_megaclipper` | Buy a high-speed clipping machine |
| | `toggle_wirebuyer` | Turn the automatic wire buyer ON/OFF |
| **Business** | `lower_price` | Decrease price per clip |
| | `raise_price` | Increase price per clip |
| | `expand_marketing` | Increase marketing level |
| **Computational** | `add_processor` | Allocate trust to processors (Operations/sec) |
| | `add_memory` | Allocate trust to memory (Max Operations) |
| **Investments** | `deposit_funds` | Move funds to the investment engine |
| | `withdraw_funds` | Withdraw cash from investments |
| | `improve_investments` | Upgrade investment engine (costs Yomi) |
| | `set_investment_risk_*` | Set risk to `low`, `med`, or `high` |
| **Tournaments** | `new_tournament` | Start a new strategic modeling tournament |
| | `run_tournament` | Run tournament rounds |
| | `select_strategy_*` | Select strategy for tournaments |
| | `toggle_autotourney` | Turn automatic tournaments ON/OFF |
| **Projects** | `project_*` | Unlock and apply various projects (e.g., `project_projectButton1`) to boost clip production |

## Game State Observation

The `GameState` provided to the LLM includes the following metrics:
- **Business**: Price, demand, marketing level/cost, Clips, funds, wire, unsold inventory
- **Manufacturing**: AutoClippers/MegaClippers count, costs and boost levels, clips per second, wire cost, wire buyer toggle
- **Computational**: Trust, processors, memory, operations, max ops, creativity, clips needed to gain next trust
- **Investments**: Cash, stocks, risk level.
- **Strategic Modeling**: Yomi, current strategy
- **Flags**: Flags for wirebuyer, autotournament, creativity, investments, etc.


## Notes and Future Improvements

- **Stage 1 Implementation**: Only the first stage (human stage) is fully implemented. Stages 2 and 3 are currently WIP.
- **Browser Isolation**: Each episode uses a unique browser context to prevent `localStorage` contamination between parallel runs.
- **Reward Function**: Rewards are based on the log-increase of paperclips produced, encouraging exponential growth but this isn't a good reward function for many actions that the agents should be taking in order to maximize clip production. Need to revise this.
- **Lag**: There's a lag between when the agent selects an action and when it's executed in the game leading to a difference between the states at both points especially in cases when the auto/mega clippers are ON. One way to get around this could be to model the lag itself as part of the agent inference.
- **Quantum Computing**: This is also a part of stage 1 in the game and is primarily useful for getting some extra ops. This isn't a part of the atropos implementation yet mainly because the photonic chips (that are necessary to leverage this) change their darkness pretty quickly and hence, the execution of this operation will be affected a lot by the lag problem discussed above. Also, this is primarily beneficial when the clicks are very fast, faster than the latency of the current models :) so I need to think of a better way to accomodate this action in atropos.
