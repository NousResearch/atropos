"""
Crypto DeFi Reasoning Environment for Atropos
Author: Investorquab
Purpose: Train LLMs to analyze DeFi protocols and token metrics,
         reason through tokenomics math, and identify healthy vs
         unhealthy market conditions with justified explanations.
"""

import random
import re
from typing import List, Optional, Tuple

from atroposlib.utils.tokenize_and_truncate import tokenize_and_truncate

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.type_definitions import GameHistory, Item

SYSTEM_PROMPT = """You are a DeFi and crypto market analyst with deep expertise in tokenomics,
on-chain data analysis, and decentralized finance protocols. When given a scenario,
reason carefully through the data using mathematical analysis and logical deduction.

Always structure your response exactly like this:
<reasoning>
Step through the key metrics mathematically. Calculate ratios, compare to benchmarks,
and identify red flags or positive signals. Show your work clearly.
</reasoning>
<verdict>
HEALTHY, OVERVALUED, UNDERVALUED, or RISKY — then a 1-2 sentence justification.
</verdict>"""

SCENARIO_TEMPLATES = [
    {
        "type": "tokenomics_health",
        "template": (
            "Token: {name}\n"
            "Circulating Supply: {circ_supply}M tokens\n"
            "Total Supply: {total_supply}M tokens\n"
            "Current Price: ${price}\n"
            "Market Cap: ${mcap}M\n"
            "Fully Diluted Valuation (FDV): ${fdv}M\n"
            "30-Day Trading Volume: ${volume}M\n"
            "Is this token HEALTHY, OVERVALUED, UNDERVALUED, or RISKY? Reason through the data."
        ),
        "answer_fn": "tokenomics_verdict",
    },
    {
        "type": "liquidity_analysis",
        "template": (
            "DeFi Protocol: {name}\n"
            "Total Value Locked (TVL): ${tvl}M\n"
            "Token Market Cap: ${mcap}M\n"
            "Daily Volume: ${volume}M\n"
            "Annual Protocol Revenue: ${revenue}M\n"
            "Token Inflation Rate: {inflation}% per year\n"
            "Is this protocol HEALTHY, OVERVALUED, UNDERVALUED, or RISKY? Analyze the metrics."
        ),
        "answer_fn": "liquidity_verdict",
    },
    {
        "type": "defi_yield",
        "template": (
            "Yield Farming Opportunity:\n"
            "Protocol: {name}\n"
            "Advertised APY: {apy}%\n"
            "TVL: ${tvl}M\n"
            "Token emission rate: {emission}M tokens/month\n"
            "Token price: ${price}\n"
            "Monthly token emissions value: ${emissions_value}M\n"
            "Is this yield opportunity HEALTHY, OVERVALUED, UNDERVALUED, or RISKY?"
        ),
        "answer_fn": "yield_verdict",
    },
    {
        "type": "onchain_activity",
        "template": (
            "On-Chain Analysis:\n"
            "Token: {name}\n"
            "Market Cap: ${mcap}M\n"
            "30-Day On-Chain Transaction Volume: ${onchain_vol}M\n"
            "Number of Active Wallets (30d): {wallets}K\n"
            "Average Transaction Value: ${avg_tx}\n"
            "Is this token HEALTHY, OVERVALUED, UNDERVALUED, or RISKY?"
        ),
        "answer_fn": "onchain_verdict",
    },
]


def tokenomics_verdict(params: dict) -> str:
    fdv_mcap_ratio = params["fdv"] / params["mcap"]
    volume_mcap_ratio = params["volume"] / params["mcap"]
    circ_ratio = params["circ_supply"] / params["total_supply"]
    if fdv_mcap_ratio > 10:
        return "RISKY"
    if circ_ratio < 0.2 and fdv_mcap_ratio > 5:
        return "OVERVALUED"
    if 0.1 <= volume_mcap_ratio <= 2.0 and fdv_mcap_ratio < 3 and circ_ratio > 0.5:
        return "HEALTHY"
    if volume_mcap_ratio < 0.02:
        return "UNDERVALUED"
    return "RISKY"


def liquidity_verdict(params: dict) -> str:
    tvl_mcap_ratio = params["tvl"] / params["mcap"]
    pe_ratio = params["mcap"] / max(params["revenue"], 0.1)
    real_yield = (params["revenue"] / params["tvl"]) * 100
    if params["inflation"] > 50:
        return "RISKY"
    if tvl_mcap_ratio > 1.5 and pe_ratio < 20 and real_yield > 5:
        return "HEALTHY"
    if tvl_mcap_ratio < 0.1 and pe_ratio > 100:
        return "OVERVALUED"
    if tvl_mcap_ratio > 2.0 and pe_ratio < 10:
        return "UNDERVALUED"
    return "RISKY"


def yield_verdict(params: dict) -> str:
    emissions_to_tvl = params["emissions_value"] / max(params["tvl"], 1)
    if params["apy"] > 500:
        return "RISKY"
    if emissions_to_tvl > 0.1:
        return "RISKY"
    if params["apy"] > 20 and emissions_to_tvl < 0.02:
        return "HEALTHY"
    if params["apy"] < 5:
        return "UNDERVALUED"
    return "OVERVALUED"


def onchain_verdict(params: dict) -> str:
    vol_mcap_ratio = params["onchain_vol"] / params["mcap"]
    mcap_per_wallet = (params["mcap"] * 1_000_000) / (params["wallets"] * 1000)
    if vol_mcap_ratio > 0.5 and params["wallets"] > 100:
        return "HEALTHY"
    if mcap_per_wallet > 100_000 and vol_mcap_ratio < 0.05:
        return "OVERVALUED"
    if vol_mcap_ratio < 0.01 and params["wallets"] < 10:
        return "RISKY"
    if vol_mcap_ratio > 0.1 and mcap_per_wallet < 1000:
        return "UNDERVALUED"
    return "RISKY"


VERDICT_FNS = {
    "tokenomics_verdict": tokenomics_verdict,
    "liquidity_verdict": liquidity_verdict,
    "yield_verdict": yield_verdict,
    "onchain_verdict": onchain_verdict,
}

PROTOCOL_NAMES = [
    "NexusFi",
    "SolarSwap",
    "ArcadeDAO",
    "VaultX",
    "ChainPulse",
    "DeepLiquidity",
    "OmegaStake",
    "NileProtocol",
    "ZeroFee DEX",
    "PrimeLend",
    "AfriSwap",
    "SahelDAO",
    "LagosFinance",
    "AbujaDeFi",
]


def generate_scenario(scenario_type: Optional[str] = None) -> dict:
    template_info = (
        random.choice(SCENARIO_TEMPLATES)
        if not scenario_type
        else next(
            (t for t in SCENARIO_TEMPLATES if t["type"] == scenario_type),
            random.choice(SCENARIO_TEMPLATES),
        )
    )
    name = random.choice(PROTOCOL_NAMES)

    if template_info["type"] == "tokenomics_health":
        circ = random.choice([50, 100, 200, 500, 800])
        total = random.choice([500, 1000, 2000, 5000, 10000])
        total = max(total, circ * 2)
        price = round(random.uniform(0.01, 50), 4)
        mcap = round(circ * price, 2)
        fdv = round(total * price, 2)
        volume = round(random.choice([0.5, 2, 10, 50, 200, 500]), 2)
        params = dict(
            name=name,
            circ_supply=circ,
            total_supply=total,
            price=price,
            mcap=mcap,
            fdv=fdv,
            volume=volume,
        )
    elif template_info["type"] == "liquidity_analysis":
        tvl = random.choice([1, 5, 10, 50, 100, 500])
        mcap = random.choice([1, 5, 20, 100, 500])
        volume = round(tvl * random.uniform(0.01, 0.5), 2)
        revenue = round(tvl * random.uniform(0.01, 0.3), 2)
        inflation = random.choice([5, 15, 30, 60, 100, 200])
        params = dict(
            name=name,
            tvl=tvl,
            mcap=mcap,
            volume=volume,
            revenue=revenue,
            inflation=inflation,
        )
    elif template_info["type"] == "defi_yield":
        tvl = random.choice([1, 5, 20, 100, 500])
        apy = random.choice([3, 8, 25, 80, 200, 1000])
        emission = random.choice([0.1, 0.5, 2, 10, 50])
        price = round(random.uniform(0.01, 10), 3)
        emissions_value = round(emission * price, 3)
        params = dict(
            name=name,
            tvl=tvl,
            apy=apy,
            emission=emission,
            price=price,
            emissions_value=emissions_value,
        )
    else:
        mcap = random.choice([5, 20, 100, 500, 2000])
        onchain_vol = round(mcap * random.uniform(0.005, 0.8), 2)
        wallets = random.choice([2, 10, 50, 200, 500, 2000])
        avg_tx = round((onchain_vol * 1_000_000) / (wallets * 1000 * 30), 2)
        params = dict(
            name=name,
            mcap=mcap,
            onchain_vol=onchain_vol,
            wallets=wallets,
            avg_tx=avg_tx,
        )

    prompt = template_info["template"].format(**params)
    correct_answer = VERDICT_FNS[template_info["answer_fn"]](params)
    return {"prompt": prompt, "correct_answer": correct_answer, "params": params}


def extract_verdict(text: str) -> Optional[str]:
    verdict_match = re.search(r"<verdict>(.*?)</verdict>", text, re.DOTALL)
    content = verdict_match.group(1).upper() if verdict_match else text.upper()
    for label in ["HEALTHY", "OVERVALUED", "UNDERVALUED", "RISKY"]:
        if label in content:
            return label
    return None


def extract_reasoning(text: str) -> Optional[str]:
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def score_response(response: str, correct_answer: str) -> float:
    """
    Score model response 0.0-1.0:
      0.5 - correct verdict label
      0.2 - has <reasoning> block
      0.1 - reasoning contains numerical calculations
      0.1 - reasoning length > 100 chars
      0.1 - has <verdict> block
    """
    score = 0.0
    verdict = extract_verdict(response)
    reasoning = extract_reasoning(response)
    if verdict == correct_answer:
        score += 0.5
    if reasoning is not None:
        score += 0.2
        if re.search(r"\d+\.?\d*\s*[\/x*]\s*\d+\.?\d*|\d+%|\$\d+", reasoning):
            score += 0.1
        if len(reasoning) > 100:
            score += 0.1
    if "<verdict>" in response.lower():
        score += 0.1
    return round(score, 2)


class CryptoDeFiReasoningConfig(BaseEnvConfig):
    tokenizer_name: str = "NousResearch/Hermes-3-Llama-3.1-8B"
    group_size: int = 8
    max_token_length: int = 1024
    use_wandb: bool = False
    rollout_server_url: str = "http://localhost:8000"
    total_steps: int = 500
    batch_size: int = 8
    steps_per_eval: int = 50
    wandb_name: str = "crypto_defi_reasoning"


class CryptoDeFiReasoningEnv(BaseEnv):
    """
    Crypto DeFi Reasoning RL Environment for Atropos.

    Trains LLMs to reason about DeFi protocol health, tokenomics,
    liquidity metrics, and on-chain data. The model must produce
    structured reasoning and a correct categorical verdict:
    HEALTHY / OVERVALUED / UNDERVALUED / RISKY.

    Scenario types:
      - Tokenomics health (FDV/MCap ratio, circulating supply pressure)
      - Liquidity analysis (TVL/MCap, protocol revenue, inflation)
      - DeFi yield sustainability (APY vs emission pressure)
      - On-chain activity (active wallets, volume vs market cap)
    """

    name = "crypto_defi_reasoning"
    env_config_cls = CryptoDeFiReasoningConfig

    def __init__(self, config, server_configs, slurm=False, testing=False):
        super().__init__(config, server_configs, slurm=slurm, testing=testing)
        self.config = config
        self._scenarios: List[dict] = []
        self._eval_scenarios: List[dict] = []

    async def setup(self):
        random.seed(42)
        self._scenarios = [generate_scenario() for _ in range(500)]
        random.seed(99)
        self._eval_scenarios = [generate_scenario() for _ in range(50)]

    def _build_messages(self, scenario: dict) -> GameHistory:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": scenario["prompt"]},
        ]

    async def get_next_item(self) -> Item:
        scenario = random.choice(self._scenarios)
        messages = self._build_messages(scenario)
        prompt = tokenize_and_truncate(
            messages, self.tokenizer, self.config.max_token_length
        )
        return (prompt, scenario["correct_answer"])

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, list]:
        prompt, correct_answer = item
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=512,
        )
        scored = ScoredDataGroup()
        scored["tokens"] = []
        scored["masks"] = []
        scored["scores"] = []
        for choice in completions.choices:
            text = choice.message.content or ""
            s = score_response(text, correct_answer)
            tokens = self.tokenizer.encode(text)
            scored["tokens"].append(prompt + tokens)
            scored["masks"].append([0] * len(prompt) + [1] * len(tokens))
            scored["scores"].append(s)
        return scored, []

    async def evaluate(self, *args, **kwargs):
        correct = 0
        total = len(self._eval_scenarios)
        for scenario in self._eval_scenarios:
            messages = self._build_messages(scenario)
            prompt = tokenize_and_truncate(
                messages, self.tokenizer, self.config.max_token_length
            )
            result = await self.server.completion(prompt=prompt, n=1, max_tokens=512)
            text = result.choices[0].message.content or ""
            if extract_verdict(text) == scenario["correct_answer"]:
                correct += 1
        accuracy = correct / total if total > 0 else 0
        print(f"[Eval] DeFi Reasoning Accuracy: {accuracy:.2%} ({correct}/{total})")
        return {"defi_reasoning_accuracy": accuracy}


if __name__ == "__main__":
    CryptoDeFiReasoningEnv.cli()
