"""
Environment for Universal Paperclips game: https://www.decisionproblem.com/paperclips/index2.html

This environment wraps the browser-based Universal Paperclips game and provides
a standard Atropos interface to generate trajectories.

PLEASE NOTE: ONLY the first stage (human stage) of the game is implemented for now, and further stages (2&3) are WIP.

Key features:
- Parallel episode collection using isolated browser contexts
- Fresh game for each episode (localStorage is cleared)
- LLM-based action generation with state observations
- Reward based on paperclip production progress
"""

import asyncio
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from .config import (
    PAPERCLIPS_SYSTEM_PROMPT,
    PaperclipsEnvConfig,
    get_action_prompt,
)


@dataclass
class GameState:
    """Structured representation of the current game state."""

    clips: float = 0.0
    is_human_stage: bool = True
    ticks: float = 0.0

    # Business
    funds: float = 0.0
    price_per_clip: float = 0.25
    demand_percent: float = 0.0
    unsold_inventory: float = 0.0
    marketing_level: int = 1
    marketing_cost: float = 100.0

    # Manufacturing
    clips_per_second: float = 0.0
    wire: float = 0.0
    wire_cost: float = 20.0
    has_wirebuyer: bool = False
    use_wirebuyer: bool = False
    autoclippers: int = 0
    has_autoclippers: bool = False
    autoclipper_cost: float = 5.0
    autoclipper_boost: float = 1.0
    megaclippers: int = 0
    has_megaclippers: bool = False
    megaclipper_cost: float = 500.0
    megaclipper_boost: float = 1.0

    # computational resources
    comp_flag: bool = False
    processors: int = 1
    memory: int = 1
    operations: int = 0
    max_operations: int = 1000
    trust: int = 2
    creativity: int = 0
    creativity_on: bool = False
    next_trust: str = "3000"

    # Investments
    has_investments: bool = False
    cash: str = ""
    stocks: str = ""
    investment_risk: str = "low"

    # Strategic modelling (tournaments)
    has_strategy_engine: bool = False
    yomi: float = 0
    tournament_in_progress: bool = False
    has_auto_tournament: bool = False
    use_auto_tournament: bool = False
    current_strategy: int = 10

    def to_prompt_string(self) -> str:
        """Get state representation for prompts"""
        lines = [
            f"Paperclips Made: {self.clips:,.0f}",
            f"Ticks: {self.ticks:.0f}",
            "",
            "-- Manufacturing --",
            f"Clips per Second: {self.clips_per_second:.1f}",
            f"Wire: {self.wire:.0f} inches",
            f"Wire Cost: ${self.wire_cost:.2f}",
        ]

        if self.has_wirebuyer:
            lines.append(f"Wire Buyer: {'ON' if self.use_wirebuyer else 'OFF'}")
        else:
            lines.append("Wire Buyer: Not available yet!")
        if self.has_autoclippers:
            lines.extend(
                [
                    f"AutoClippers: {self.autoclippers} (Cost: ${self.autoclipper_cost:.2f})",
                    f"AutoClipper Boost: {self.autoclipper_boost:.2f}x",
                ]
            )
        else:
            lines.append("AutoClippers: Not available yet!")
        if self.has_megaclippers:
            lines.extend(
                [
                    f"MegaClippers: {self.megaclippers} (Cost: ${self.megaclipper_cost:.2f})",
                    f"MegaClipper Boost: {self.megaclipper_boost:.2f}x",
                ]
            )
        else:
            lines.append("MegaClippers: Not available yet!")

        lines.extend(
            [
                "",
                "-- Business --",
                f"Funds: ${self.funds:.2f}",
                f"Price per Clip: ${self.price_per_clip:.2f}",
                f"Public Demand: {self.demand_percent:.0f}%",
                f"Unsold Inventory: {self.unsold_inventory:,.0f}",
                f"Marketing Level: {self.marketing_level} (Next: ${self.marketing_cost:.2f})",
            ]
        )

        if self.comp_flag:
            lines.extend(
                [
                    "",
                    "-- Computational Resources --",
                    f"Trust: {self.trust} |  Trust increases by 1 at: {self.next_trust} clips",
                    f"Processors: {self.processors} | Memory: {self.memory}",
                    f"Operations: {self.operations:,} / {self.max_operations:,}",
                ]
            )
            if self.creativity_on:
                lines.append(f"Creativity: {self.creativity}")
            else:
                lines.append("Creativity: Not available yet!")
        else:
            lines.append(
                """Computational resources project (Trust-Constrained Self-Modification):
                Not available yet! Become available when 2000 clips are made."""
            )

        if self.has_investments:
            lines.extend(
                [
                    "",
                    "-- Investments --",
                    f"Cash: ${self.cash}",
                    f"Stocks: ${self.stocks}",
                    f"Risk Level: {self.investment_risk}",
                ]
            )
        else:
            lines.append(
                "Algorithmic trading project (Investment engine): Not available yet!"
            )

        if self.has_strategy_engine:
            lines.extend(
                [
                    "",
                    "-- Strategic Modeling (tournaments) --",
                    f"Yomi: {self.yomi:.0f}",
                    f"Current strategy: {self.current_strategy}"
                    f"Tournament: {'In Progress' if self.tournament_in_progress else 'Ready'}",
                ]
            )
            if self.has_auto_tournament:
                lines.append(
                    f"Auto-tournament: {'ON' if self.use_auto_tournament else 'OFF'}"
                )
            else:
                lines.append("Auto-tournament: Not available yet!")
        else:
            lines.append("Strategic modeling project (Tournaments): Not available yet!")


class EpisodeContext:
    """
    Manages a single episode's browser context.

    Each parallel episode gets its own isolated browser context to ensure
    fresh game state and no localStorage interference between any two episodes.
    """

    def __init__(
        self, browser: Browser, game_url: str, episode_id: int, headless: bool = True
    ):
        self.browser = browser
        self.game_url = game_url
        self.headless = headless
        self.episode_id = episode_id
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Create a fresh browser context and load a new game."""
        if self._initialized:
            return

        self._context = await self.browser.new_context()
        self._page = await self._context.new_page()

        print(
            f" Episode {self.episode_id}: Navigating to {self.game_url}...Clearing localStorage and reloading"
        )
        await self._page.goto(self.game_url, wait_until="networkidle")
        await self._page.evaluate("localStorage.clear()")
        await self._page.reload(wait_until="networkidle")
        await self._page.wait_for_selector("#clips", timeout=10000)
        print(f" Episode {self.episode_id}: Game ready!")

        self._initialized = True

    async def close(self) -> None:
        """Close this episode's context"""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        self._initialized = False

    async def get_state(self) -> dict:
        """Extract current game state from the browser.
        State consists of several game parameters useful
        for the agent to make a next decision.
        """
        if not self._initialized:
            raise RuntimeError(
                f"Episode {self.episode_id}: Episode context not initialized"
            )

        state = await self._page.evaluate("""
            () => {
                const state = {
                    clips: clips || 0,

                    // Business
                    funds: funds || 0,
                    avgRev: avgRev || 0,
                    unusedClips: unusedClips || 0, // not for stage 1
                    unsoldClips: unsoldClips || 0,
                    margin: margin || 0.25,
                    demand: demand || 0,
                    marketing: marketing || 1,
                    marketingLvl: marketingLvl || 1,
                    adCost: adCost || 100,

                    // Manufacturing
                    clipRate: clipRate || 0, // clips per Second on the ui

                    wire: wire || 0,
                    wireCost: wireCost || 20,
                    wireSupply: wireSupply || 1000,
                    wireBuyerStatus: wireBuyerStatus || 0,
                    wireBuyerFlag: wireBuyerFlag || 0,

                    autoClipperFlag: autoClipperFlag || 0,
                    clipmakerLevel: clipmakerLevel || 0,
                    clipperCost: clipperCost || 5,
                    clipperBoost: clipperBoost || 1,

                    megaClipperFlag: megaClipperFlag || 0,
                    megaClipperLevel: megaClipperLevel || 0,
                    megaClipperCost: megaClipperCost || 500,
                    megaClipperBoost: megaClipperBoost || 1,


                    // computational resources (useful for projects, strategic modeling etc.)
                    compFlag: compFlag || 0,
                    processors: processors || 1,
                    memory: memory || 1,
                    operations: operations || 0,
                    trust: trust || 2,
                    nextTrust: nextTrust || 3000,
                    creativity: creativity || 0,
                    creativityOn: creativityOn || false,

                    // tournament (strategic modeling)
                    strategyEngineFlag: strategyEngineFlag || 0,
                    tourneyCost: tourneyCost || 1000,
                    tourneyInProg: tourneyInProg || 0,
                    autoTourneyFlag: autoTourneyFlag || 0,
                    autoTourneyStatus: autoTourneyStatus || 0,
                    // Strategy selection (pick = index into strats array, 10 means no selection)
                    currentStrategyPick: typeof pick !== 'undefined' ? pick : 10,
                    yomi: typeof yomi !== 'undefined' ? yomi : 0,

                    // investments
                    investmentEngineFlag: investmentEngineFlag || 0,
                    bankroll: bankroll || 0,
                    secTotal: secTotal || 0,
                    portTotal: portTotal || 0,
                    riskiness: riskiness || 7,

                    // stage flags
                    humanFlag: humanFlag || 1,

                    // Stage 2 variables (post-human)
                    // factoryLevel: factoryLevel || 0,
                    // harvesterLevel: harvesterLevel || 0,
                    // wireDroneLevel: wireDroneLevel || 0,
                    // availableMatter: availableMatter || 0,
                    // acquiredMatter: acquiredMatter || 0,

                    // Stage 3 variables (space stage)
                    // probeCount: probeCount || 0,
                    // spaceFlag: spaceFlag || 0,

                    // Game ticks (for time tracking)
                    ticks: ticks || 0
                };

                // unlocked strategies for tournaments
                state.availableStrategies = [];
                if (typeof strats !== 'undefined' && strats && typeof allStrats !== 'undefined') {
                    state.availableStrategies = strats.map(s => ({
                        index: allStrats.indexOf(s),
                        name: s.name
                    }));
                }

                return state;
            }
        """)
        return state

    async def get_available_actions(self) -> List[dict]:
        """
        Get actions based on game stage.
        Actions for a particular stage are always returned regardless of their
        affordability (attribute `available`) to be able to teach
        the agent not to click them if they aren't affordable yet.

        Notes:
        - some actions don't cost any resource
        - some actions are available given enough resources
        (e.g., make paperclip if wire available > 0)
        but these are always added to the list, helps
        teach the agent what's imposible to click!
        - Some actions become available only after a certain stage in game is
          reached (investment, strategic modeling etc.)
        - The available projects also consitute the actions available
          as the user can click and unlock them to
          eg. gather resources that enable more clip production
        - we also have an additional wait action (not present in the actual game),
          this is for the agent to just observe/let the game state evolve without
          taking any new action!


        Returns:
            List of action dictionaries with name, description, and availability.
        """
        if not self._initialized:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        actions = await self._page.evaluate("""
            () => {
                const actions = [];

                actions.push({
                    name: 'make_paperclip',
                    description: 'Make a paperclip manually',
                    available: wire >= 1,
                    cost: '1 wire'
                });

                // Manufacturing
                actions.push({
                    name: 'buy_wire',
                    description: 'Buy a spool of wire if needed. Remember if you run out ' +
                        'of wire, you cannot produce more paperclips!',
                    available: funds >= wireCost,
                    cost: '$' + Math.ceil(wireCost).toFixed()
                });
                if (autoClipperFlag === 1) {
                    actions.push({
                        name: 'buy_autoclipper',
                        description: 'AutoClippers are automated clipping systems to ' +
                            'generate paperclips for sale automatically rather than ' +
                            'doing it manually. Produces 1 clip/second/clipper',
                        available: funds >= clipperCost,
                        cost: '$' + clipperCost.toFixed(4)
                    });
                }
                if (megaClipperFlag === 1) {
                    actions.push({
                        name: 'buy_megaclipper',
                        description: 'Buy a MegaClipper. A megaclipper can do 500 clips/sec/clipper ' +
                            '(500x a standard autoclipper!)',
                        available: funds >= megaClipperCost,
                        cost: '$' + megaClipperCost.toFixed(4)
                    });
                }
                if (wireBuyerFlag === 1) {
                    actions.push({
                        name: 'toggle_wirebuyer',
                        description: 'Toggle automatic wire buying to turn it on or off.' +
                        'If the wire buyer is on, this toggle turns it off and vice versa.',
                        available: true,
                        cost: 'none'
                    });
                }

                // business
                if (humanFlag === 1) {
                    actions.push({
                        name: 'lower_price',
                        description: 'Lower the price per clip',
                        available: margin > 0.01,
                        cost: 'none'
                    });
                    actions.push({
                        name: 'raise_price',
                        description: 'Raise the price per clip',
                        available: true,
                        cost: 'none'
                    });
                    actions.push({
                        name: 'expand_marketing',
                        description: 'Expand marketing reach to sell more paperclips',
                        available: funds >= adCost,
                        cost: '$' + adCost.toFixed(2)
                    });
                }

                // comp resources
                if (compFlag === 1) {
                    const swarmBonus = typeof swarmGifts !== 'undefined' ? swarmGifts : 0;
                    const canAllocate = trust >= 1 && (processors + memory) < (trust + swarmBonus);
                    actions.push({
                        name: 'add_processor',
                        description: 'Add a processor (uses 1 trust) to increase operations or capacity per second',
                        available: canAllocate,
                        cost: '1 trust allocation'
                    });
                    actions.push({
                        name: 'add_memory',
                        description: 'Add memory (uses 1 trust) to increase max operations capacity',
                        available: canAllocate,
                        cost: '1 trust allocation'
                    });
                }

                // Investments
                if (investmentEngineFlag === 1) {
                    actions.push({
                        name: 'deposit_funds',
                        description: 'Deposit all current funds into investments',
                        available: funds >= 1000,
                        cost: 'variable'
                    });
                    actions.push({
                        name: 'withdraw_funds',
                        description: 'Withdraw funds from investments',
                        available: bankroll > 0,
                        cost: 'none'
                    });
                    actions.push({
                        name: 'improve_investments',
                        description: 'Upgrade investment engine for a better profit to loss ratio',
                        available: yomi >= investUpgradeCost,
                        cost: investUpgradeCost.toFixed() + ' yomi'
                    });
                    actions.push({
                        name: 'set_investment_risk_low',
                        description: 'Set investment risk to LOW (riskiness=7, most conservative)',
                        available: true,
                        cost: 'none',
                        currentlySelected: riskiness === 7
                    });
                    actions.push({
                        name: 'set_investment_risk_med',
                        description: 'Set investment risk to MEDIUM (riskiness=5, balanced)',
                        available: true,
                        cost: 'none',
                        currentlySelected: riskiness === 5
                    });
                    actions.push({
                        name: 'set_investment_risk_high',
                        description: 'Set investment risk to HIGH (riskiness=1, most aggressive)',
                        available: true,
                        cost: 'none',
                        currentlySelected: riskiness === 1
                    });
                }

                // tournaments (strategic modelling)
                if (strategyEngineFlag === 1) {
                    actions.push({
                        name: 'new_tournament',
                        description: 'Start a new tournament to pick your strategy.',
                        available: (operations >= tourneyCost) && (tourneyInProg == 0),
                        cost: tourneyCost.toFixed() + ' ops'
                    });
                    actions.push({
                    name: 'run_tournament',
                    description: 'Run the tournament rounds and earn yomi',
                    available: tourneyInProg === 1,
                    cost: 'none'
                    });
                }
                // Add individual strategy selection actions for each unlocked strategy
                if (strategyEngineFlag === 1 && typeof strats !== 'undefined' && typeof allStrats !== 'undefined') {
                    strats.forEach(s => {
                        const stratIndex = allStrats.indexOf(s);
                        actions.push({
                            name: 'select_strategy_' + stratIndex,
                            description: 'Select strategy: ' + s.name + ' (bet on this strategy in tournaments)',
                            available: true,
                            cost: 'none',
                            strategyIndex: stratIndex,
                            strategyName: s.name
                        });
                    });
                }
                if (autoTourneyFlag === 1) {
                    actions.push({
                        name: 'toggle_autotourney',
                        description: 'Toggle automatic strategy tournaments to turn it on or off' +
                        ' If the auto-tournament is on, this toggle turns it off and vice versa.',
                        available: true,
                        cost: 'none'
                    });
                }

                // add each active project as an action regarless of availability
                if (typeof activeProjects !== 'undefined' && activeProjects) {
                    activeProjects.forEach(project => {
                        actions.push({
                            name: 'project_' + project.id,
                            description: project.title.trim() + ': ' + project.description,
                            available: project.cost && project.cost(),
                            cost: project.priceTag
                        });
                    });
                }

                actions.push({
                    name: 'wait',
                    description: 'Do nothing for this step!',
                    available: true,
                    cost: 'none'
                });

                return actions;
            }
        """)
        return actions

    async def execute_action(self, action_name: str, state: GameState) -> bool:
        """
        Execute an action by calling the corresponding js function.
        Args:
            action_name: action to execute
        Returns:
            True if action was executed, else False
        """

        if not self._initialized:
            raise RuntimeError(
                "Error: cannot execute action as browser isn't initialized yet! "
                "Call method PaperclipsBrowserWrapper.initialize() first."
            )

        action_map = {
            "make_paperclip": "clipClick(1)",
            # business
            "lower_price": "lowerPrice()",
            "raise_price": "raisePrice()",
            "expand_marketing": "buyAds()",
            # manufacturing
            "buy_wire": "buyWire()",
            "buy_autoclipper": "makeClipper()",
            "buy_megaclipper": "makeMegaClipper()",
            "toggle_wirebuyer": "toggleWireBuyer()",
            # computational resources
            "add_processor": "addProc()",
            "add_memory": "addMem()",
            # strategic modeling
            "new_tournament": "newTourney()",
            "run_tournament": "runTourney()",
            "toggle_autotourney": "toggleAutoTourney()",
            # investments
            "improve_investments": "investUpgrade()",
            "deposit_funds": "investDeposit()",
            "withdraw_funds": "investWithdraw()",
            "wait": "null",
        }

        # dropdowns (investment risk, strategy selection)
        if action_name.startswith("set_investment_risk_"):
            risk_level = action_name.replace("set_investment_risk_", "").lower()
            if risk_level not in ["low", "med", "high"]:
                return False
            if (
                risk_level == state.investment_risk
            ):  # agent selected the same risk level again
                return False
            js_code = f"""
                (() => {{
                    if (typeof investStratElement !== 'undefined' && investStratElement) {{
                        investStratElement.value = '{risk_level}';
                        return true;
                    }}
                    return false;
                }})()
            """
        elif action_name.startswith("select_strategy_"):
            strategy_index = int(action_name.replace("select_strategy_", ""))
            if strategy_index < 0 or strategy_index > 7:
                return False
            if (
                strategy_index == state.current_strategy
            ):  # agent selected the same strategy again
                return False
            js_code = f"""
                (() => {{
                    if (typeof stratPickerElement !== 'undefined' && stratPickerElement) {{
                        stratPickerElement.value = {strategy_index};
                        return true;
                    }}
                    return false;
                }})()
            """

        # projects
        elif action_name.startswith("project_"):
            project_id = action_name.replace("project_", "")
            project_number = int(project_id.replace("projectButton", ""))
            if project_number < 1 or project_number > 219:
                return False
            # todo: fetch both activeProjects and project.cost from the action
            # list sent to the llm rather than executing new code for it
            # why? bec by the time this action gets executed, values might change.
            js_code = f"""
                (() => {{
                    const project = activeProjects.find(p => p.id === '{project_id}');
                    if (project && project.cost && project.cost() && project.flag === 0) {{
                        project.effect();
                        return true;
                    }}
                    return false;
                }})()
            """

        elif action_name in action_map:
            js_code = action_map[action_name]
            if js_code == "null":
                return True
            js_code = f"(() => {{ try {{ {js_code}; return true; }} catch(e) {{ return false; }} }})()"
        else:
            return False

        try:
            result = await self._page.evaluate(js_code)
            return bool(result) if result is not None else True
        except Exception as e:
            print(f"Error executing action {action_name}: {e}")
            return False

    async def wait_ticks(self, ticks: int = 5) -> None:
        """Wait for game ticks to elapse"""
        await asyncio.sleep(ticks * 0.1)


class PaperclipsAtroposEnv(BaseEnv):
    """
    Atropos-compatible environment for Universal Paperclips.

    This environment supports parallel episode collection by creating isolated
    browser contexts for each episode. Each episode starts with a fresh game state.

    The LLM agent observes the game state and selects actions to maximize
    paperclip production.
    """

    name = "universal_paperclips"
    env_config_cls = PaperclipsEnvConfig

    def __init__(
        self,
        config: PaperclipsEnvConfig,
        server_configs: Union[List[APIServerConfig], Any],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: PaperclipsEnvConfig = config
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._iter = 0

    async def setup(self) -> None:
        """Initialize Playwright browser instance."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless
        )

    async def teardown(self) -> None:
        """Clean up Playwright resources."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        print(
            "Environment teardown complete. Browser closed, playwright engine stopped!"
        )

    async def get_next_item(self) -> Item:
        """
        Get the next item to be processed.

        For this environment, one item is one episode config
        We use a simple counter to track episodes
        """
        self._iter += 1
        initial_prompt = frozenset(
            {"role": "system", "content": PAPERCLIPS_SYSTEM_PROMPT}.items()
        )
        return (initial_prompt, self._iter)

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Collect a single episode trajectory.

        This method runs one complete episode in an isolated browser context
        and returns the tokenized trajectory with its reward.

        Args:
            item: Tuple of (initial_prompt, episode_id)

        Returns:
            Tuple of (ScoredDataItem, backlog_items)
        """
        initial_prompt_frozenset, episode_id = item
        print(
            f" collect_trajectory() - episode_id={episode_id}, max_steps={self.config.max_steps_per_episode}"
        )

        episode = EpisodeContext(
            browser=self._browser,
            game_url=self.config.game_url,
            headless=self.config.headless,
            episode_id=episode_id,
        )

        try:
            print(f" Episode {episode_id}: Initializing browser context...")
            await episode.initialize()
            print(f" Episode {episode_id}: Browser context ready, game loaded")

            total_reward = 0.0
            prev_clips = 0.0
            steps = 0
            done = False
            start_time = datetime.now()
            transitions: List[dict] = []

            while steps < self.config.max_steps_per_episode and not done:
                raw_state = await episode.get_state()
                state = self._parse_state(raw_state)
                available_actions = await episode.get_available_actions()
                state_text = state.to_prompt_string()
                actions_text = "\n".join(map(str, available_actions))
                user_msg = {
                    "role": "user",
                    "content": get_action_prompt(state_text, actions_text),
                }
                messages: List[dict] = [dict(initial_prompt_frozenset)]
                messages.append(user_msg)
                print(
                    f"   Episode {episode_id}, Step {steps+1}/"
                    f"{self.config.max_steps_per_episode}.\nAgent taking action..."
                )
                try:
                    response = await self.server.chat_completion(
                        messages=messages,
                        n=1,
                        max_tokens=64,
                        timeout=30,
                    )
                    action_text = response.choices[0].message.content.strip()
                    print(
                        f"   Episode {episode_id}, Step {steps+1}, Agent response: '{action_text}'"
                    )
                except Exception as e:
                    print(
                        f"   Episode {episode_id}, Step {steps+1}, Agent ERROR: {e}, defaulting to 'wait'"
                    )
                    action_text = "wait"

                action_name = self._parse_action(action_text, available_actions)
                action_info = next(
                    (a for a in available_actions if a["name"] == action_name), None
                )
                is_affordable = action_info and action_info.get("available", False)
                messages.append({"role": "assistant", "content": action_name})
                print(
                    f"   Episode {episode_id}, Step {steps+1}, "
                    f"Parsed action: '{action_name}', affordable={is_affordable}"
                )

                success = False
                if is_affordable:
                    success = await episode.execute_action(action_name, state)

                # wait for ticks_per_step to let the action show effect then fetch a new state
                await episode.wait_ticks(self.config.ticks_per_step)
                raw_state = await episode.get_state()
                current_clips = raw_state.get("clips", 0)
                if success:
                    step_reward = math.log(
                        current_clips + self.config.reward_eps
                    ) - math.log(prev_clips + self.config.reward_eps)
                else:
                    step_reward = 0
                prev_clips = current_clips
                total_reward += step_reward

                transitions.append(
                    {
                        "step": steps + 1,
                        "observation": asdict(state),
                        "action": action_name,
                        "action_success": success if is_affordable else False,
                        "step_reward": step_reward,
                        "done": False,  # updated below if episode ends w this step
                        "truncated": False,
                    }
                )

                steps += 1
                print(
                    f"   Episode {episode_id}, After step {steps+1}: "
                    f"clips={current_clips:.0f}, step_reward={step_reward:.2f}"
                )

                # check termination at this step
                if (
                    self.config.target_clips
                    and current_clips >= self.config.target_clips
                ):
                    done = True
                    total_reward += 10.0  # Bonus for reaching target
                    print(
                        f"   Episode {episode_id}, After step {steps+1}, Target reached!"
                    )
                if not raw_state.get("humanFlag", 1):
                    done = True
                    print(
                        f"   Episode {episode_id}, After step {steps+1}, Stage 1 complete!"
                    )

            print(f"""==============Episode {episode_id}:
                COMPLETED --- total clips={current_clips:.0f},
                steps={steps},
                total_reward={total_reward:.2f}==============""")

            # Mark last transition as done if episode terminated
            if transitions and done:
                transitions[-1]["done"] = True
            elif transitions and steps >= self.config.max_steps_per_episode:
                transitions[-1]["truncated"] = True

            end_time = datetime.now()
            self._save_trajectory_to_jsonl(
                episode_id=episode_id,
                transitions=transitions,
                total_reward=total_reward,
                final_clips=current_clips,
                start_time=start_time,
                end_time=end_time,
                terminated=done,
            )

            out_dict = tokenize_for_trainer(self.tokenizer, tuple(messages))
            return (
                ScoredDataItem(
                    tokens=out_dict["tokens"],
                    masks=out_dict["masks"],
                    scores=total_reward,
                    advantages=None,
                    ref_logprobs=None,
                    group_overrides=None,
                    overrides=None,
                    images=None,
                ),
                [],
            )

        except Exception as e:
            print(f" Episode {episode_id} FAILED: {e}")
            import traceback

            traceback.print_exc()
            return None, []

        finally:
            print(f" Episode {episode_id}: Closing browser context...")
            await episode.close()

    def _parse_state(self, raw_state: dict) -> GameState:
        """Convert raw browser state to GameState object."""
        riskiness = int(raw_state.get("riskiness", 7))
        investment_risk = (
            "low" if riskiness == 7 else ("med" if riskiness == 5 else "high")
        )

        return GameState(
            clips=float(raw_state.get("clips", 0)),
            is_human_stage=bool(raw_state.get("humanFlag", 1)),
            ticks=float(raw_state.get("ticks", 0)),
            funds=float(raw_state.get("funds", 0)),
            price_per_clip=float(raw_state.get("margin", 0.25)),
            demand_percent=float(raw_state.get("demand", 0)) * 10,
            unsold_inventory=float(raw_state.get("unsoldClips", 0)),
            marketing_level=int(raw_state.get("marketingLvl", 1)),
            marketing_cost=float(raw_state.get("adCost", 100)),
            clips_per_second=float(raw_state.get("clipRate", 0)),
            wire=float(raw_state.get("wire", 0)),
            wire_cost=float(raw_state.get("wireCost", 20)),
            autoclipper_cost=float(raw_state.get("clipperCost", 5)),
            megaclipper_cost=float(raw_state.get("megaClipperCost", 500)),
            has_autoclippers=bool(raw_state.get("autoClipperFlag", 0)),
            has_megaclippers=bool(raw_state.get("megaClipperFlag", 0)),
            has_wirebuyer=bool(raw_state.get("wireBuyerFlag", 0)),
            use_wirebuyer=bool(raw_state.get("wireBuyerStatus", 0)),
            autoclippers=int(raw_state.get("clipmakerLevel", 0)),
            megaclippers=int(raw_state.get("megaClipperLevel", 0)),
            autoclipper_boost=float(raw_state.get("clipperBoost", 1.0)),
            megaclipper_boost=float(raw_state.get("megaClipperBoost", 1.0)),
            comp_flag=bool(raw_state.get("compFlag", 0)),
            processors=int(raw_state.get("processors", 1)),
            memory=int(raw_state.get("memory", 1)),
            operations=int(raw_state.get("operations", 0)),
            max_operations=int(raw_state.get("memory", 1)) * 1000,
            trust=int(raw_state.get("trust", 2)),
            creativity=int(raw_state.get("creativity", 0)),
            creativity_on=bool(raw_state.get("creativityOn", False)),
            next_trust=str(raw_state.get("nextTrust", 3000)),
            has_investments=bool(raw_state.get("investmentEngineFlag", 0)),
            cash=str(int(raw_state.get("bankroll", 0))),
            stocks=str(int(raw_state.get("portTotal", 0))),
            investment_risk=investment_risk,
            has_strategy_engine=bool(raw_state.get("strategyEngineFlag", 0)),
            current_strategy=int(raw_state.get("currentStrategyPick", 10)),
            yomi=float(raw_state.get("yomi", 0)),
            tournament_in_progress=bool(raw_state.get("tourneyInProg", 0)),
            has_auto_tournament=bool(raw_state.get("autoTourneyFlag", 0)),
            use_auto_tournament=bool(raw_state.get("autoTourneyStatus", 0)),
        )

    def _parse_action(self, response_text: str, available_actions: List[dict]) -> str:
        """Parse LLM response to extract action name."""
        all_action_names = [a["name"] for a in available_actions]
        response_clean = response_text.lower().strip()
        response_clean = re.sub(
            r"^(action:|selected:|i choose|i select)\s*", "", response_clean
        )
        response_clean = re.sub(r'[\'"`]', "", response_clean)

        for action in all_action_names:
            if action.lower() == response_clean:
                return action

        # partial match
        for action in all_action_names:
            if action.lower() in response_clean or response_clean in action.lower():
                return action

        # match 1st word
        first_word = response_clean.split()[0] if response_clean.split() else ""
        for action in all_action_names:
            if first_word in action.lower():
                return action

        return "wait"

    def _save_trajectory_to_jsonl(
        self,
        episode_id: int,
        transitions: List[dict],
        total_reward: float,
        final_clips: float,
        start_time: datetime,
        end_time: datetime,
        terminated: bool,
    ) -> None:
        """
        Save a single episode to JSONL file.

        Args:
            episode_id: The episode identifier
            transitions: List of step transitions with observations, actions, rewards
            total_reward: Total accumulated reward for the episode
            final_clips: Final paperclip count
            start_time: Episode start timestamp
            end_time: Episode end timestamp
            terminated: Whether episode terminated (vs truncated)
        """
        if not self.config.trajectory_output_dir:
            return

        os.makedirs(self.config.trajectory_output_dir, exist_ok=True)

        trajectory_record = {
            "episode_id": episode_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_steps": len(transitions),
            "total_reward": total_reward,
            "final_clips": final_clips,
            "terminated": terminated,
            "transitions": transitions,
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            self.config.trajectory_output_dir, f"trajectories_{timestamp}.jsonl"
        )
        with open(filepath, "a") as f:
            f.write(json.dumps(trajectory_record) + "\n")

        print(f"Saved trajectory for episode {episode_id} to {filepath}")

    async def _run_single_eval_episode(self) -> Optional[float]:
        """
        Run a single evaluation episode. Each episode gets a fresh game state
        via its own EpisodeContext (isolated browser context with cleared localStorage).

        Returns:
            The episode score (total reward), None if episode failed
        """
        item = await self.get_next_item()
        result, _ = await self.collect_trajectory(item)
        if result:
            return result["scores"]
        return None

    async def evaluate(self, *args, **kwargs) -> None:
        """
        Runs episodes in parallel (limited by max_eval_workers).
        Each episode starts with a fresh game state via isolated browser contexts.
        """
        num_episodes = self.config.num_eval_episodes
        max_parallel = min(num_episodes, self.config.max_eval_workers)
        print(
            f"Running {num_episodes} eval episodes with up to {max_parallel} in parallel"
        )

        tasks = [self._run_single_eval_episode(i) for i in range(num_episodes)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        eval_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Episode {i} failed with exception: {result}")
            elif result is not None:
                eval_results.append(result)

        if eval_results:
            avg_reward = sum(eval_results) / len(eval_results)
            print(
                f"""Evaluation complete: {len(eval_results)}/{num_episodes} episodes succeeded,
                avg_reward={avg_reward:.2f}"""
            )

    @classmethod
    def config_init(cls) -> Tuple[PaperclipsEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = PaperclipsEnvConfig(
            wandb_name="universal_paperclips",
            tokenizer_name="Qwen/Qwen2.5-3B",
            group_size=2,
            use_wandb=False,
            max_num_workers=2,
            rollout_server_url="https://api.openai.com/v1",
            total_steps=1,
            batch_size=1,
            steps_per_eval=100,
            max_token_length=4096,
            headless=True,
            max_steps_per_episode=10,
            ticks_per_step=5,
            num_eval_episodes=2,
            max_eval_workers=2,
        )

        server_configs = [
            APIServerConfig(
                model_name="gpt-4o-mini",
                base_url="https://api.openai.com/v1",
                api_key="x",
                num_requests_for_eval=1,
            )
        ]

        return env_config, server_configs


if __name__ == "__main__":
    PaperclipsAtroposEnv.cli()
