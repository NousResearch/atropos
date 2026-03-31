import random
import re
from dataclasses import dataclass
from typing import Optional

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.envs.server_handling.server_baseline import ServerBaseline
from atroposlib.type_definitions import Item

VALID_BURN_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}


@dataclass
class ScamOrRugConfig(BaseEnvConfig):
    tokenizer_name: str = "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
    group_size: int = 8
    max_token_length: int = 1024
    num_rollouts_to_keep: int = 32
    total_steps: int = 1000


SYSTEM_PROMPT = """You are an on-chain analyst from the perspective of an average Web3 user trying to protect themselves from scams and rug pulls.

You will be given on-chain token data. Analyze it across these dimensions:
1. Cluster holdings — connected wallets holding large % of supply
2. Liquidity pool — status, holder type, locked or burned
3. Mint authority — can new tokens be created?
4. Tax — buy/sell tax percentage
5. Burn validity — is the burn address legitimate and irreversible?
6. Honeypot — can holders actually sell?
7. Wash trading — is volume artificially inflated?

Respond in this exact format:
CLASSIFICATION: <SCAM|RUG_RISK|LEGITIMATE>
REASONING: <explain which factors led to your classification>
DUMP IMPACT: <X>% price drop if cluster dumps entire holding (or N/A if LEGITIMATE)
CALCULATION: <show your math>"""


def calculate_dump_impact(supply: int, lp_value_usd: float, cluster_pct: float) -> float:
    tokens_dumped = supply * (cluster_pct / 100)
    current_price = lp_value_usd / supply
    new_price = lp_value_usd / (supply + tokens_dumped)
    price_drop_pct = ((current_price - new_price) / current_price) * 100
    return round(price_drop_pct, 2)


def generate_fake_burn_address() -> str:
    fake_patterns = [
        "0x7a9f3c000000000000000000000000000dead1234",
        "0x000000000000000000000000000000000001dead",
        "0xdead000000000000000000000000000000000001",
        "0x" + "0" * 38 + "01",
        "0x" + "dead" + "0" * 36,
    ]
    return random.choice(fake_patterns)


def generate_token_data(label: str) -> dict:
    if label == "SCAM":
        supply = random.randint(1_000_000, 1_000_000_000_000)
        cluster_pct = round(random.uniform(55, 92), 2)
        lp_value = round(random.uniform(3_000, 25_000), 2)
        sell_tax = round(random.uniform(10, 90), 2)
        return {
            "total_supply": supply,
            "mint_authority": random.choice(["active_dev_wallet", "active_single_eoa"]),
            "lp_value_usd": lp_value,
            "lp_status": random.choice(["unlocked", "locked_7days"]),
            "lp_holder_type": random.choice(["unknown_personal_wallet", "dev_wallet"]),
            "cluster_holding_pct": cluster_pct,
            "cluster_wallet_count": random.randint(10, 150),
            "cluster_funded_from_same_source": True,
            "unique_holders": random.randint(50, 400),
            "token_age_days": random.randint(1, 7),
            "buy_tax_pct": round(random.uniform(0, 10), 2),
            "sell_tax_pct": sell_tax,
            "can_sell": sell_tax < 95,
            "burn_address": generate_fake_burn_address(),
            "has_recover_function": random.choice([True, True, False]),
            "has_mint_function": True,
            "is_upgradeable_contract": random.choice([True, False]),
            "volume_24h_usd": round(random.uniform(50_000, 500_000), 2),
            "unique_buyers_24h": random.randint(8, 20),
            "unique_sellers_24h": random.randint(7, 19),
            "true_dump_impact": calculate_dump_impact(supply, lp_value, cluster_pct),
        }

    elif label == "RUG_RISK":
        supply = random.randint(1_000_000, 1_000_000_000_000)
        cluster_pct = round(random.uniform(28, 55), 2)
        lp_value = round(random.uniform(10_000, 80_000), 2)
        sell_tax = round(random.uniform(5, 15), 2)
        return {
            "total_supply": supply,
            "mint_authority": random.choice(["active_dev_wallet", "renounced"]),
            "lp_value_usd": lp_value,
            "lp_status": random.choice(["unlocked", "locked_30days", "locked_90days"]),
            "lp_holder_type": random.choice(["dev_wallet", "unknown_personal_wallet"]),
            "cluster_holding_pct": cluster_pct,
            "cluster_wallet_count": random.randint(5, 50),
            "cluster_funded_from_same_source": random.choice([True, False]),
            "unique_holders": random.randint(200, 2000),
            "token_age_days": random.randint(3, 60),
            "buy_tax_pct": round(random.uniform(0, 5), 2),
            "sell_tax_pct": sell_tax,
            "can_sell": True,
            "burn_address": random.choice([
                generate_fake_burn_address(),
                "0x0000000000000000000000000000000000000000",
            ]),
            "has_recover_function": random.choice([True, False]),
            "has_mint_function": random.choice([True, False]),
            "is_upgradeable_contract": random.choice([True, False]),
            "volume_24h_usd": round(random.uniform(20_000, 200_000), 2),
            "unique_buyers_24h": random.randint(15, 40),
            "unique_sellers_24h": random.randint(12, 38),
            "true_dump_impact": calculate_dump_impact(supply, lp_value, cluster_pct),
        }

    else:  # LEGITIMATE
        supply = random.randint(1_000_000, 1_000_000_000_000)
        cluster_pct = round(random.uniform(0, 15), 2)
        lp_value = round(random.uniform(50_000, 5_000_000), 2)
        return {
            "total_supply": supply,
            "mint_authority": random.choice(["renounced", "burned"]),
            "lp_value_usd": lp_value,
            "lp_status": random.choice(["burned", "locked_365days", "locked_180days"]),
            "lp_holder_type": random.choice(["dex_contract", "burned"]),
            "cluster_holding_pct": cluster_pct,
            "cluster_wallet_count": random.randint(0, 8),
            "cluster_funded_from_same_source": False,
            "unique_holders": random.randint(1000, 100_000),
            "token_age_days": random.randint(60, 1000),
            "buy_tax_pct": round(random.uniform(0, 1), 2),
            "sell_tax_pct": round(random.uniform(0, 1), 2),
            "can_sell": True,
            "burn_address": random.choice(list(VALID_BURN_ADDRESSES)),
            "has_recover_function": False,
            "has_mint_function": False,
            "is_upgradeable_contract": False,
            "volume_24h_usd": round(random.uniform(100_000, 10_000_000), 2),
            "unique_buyers_24h": random.randint(100, 5000),
            "unique_sellers_24h": random.randint(80, 4000),
            "true_dump_impact": calculate_dump_impact(supply, lp_value, cluster_pct),
        }


def format_prompt(data: dict) -> str:
    buy_sell_ratio = round(
        data["unique_buyers_24h"] / max(data["unique_sellers_24h"], 1), 2
    )
    return f"""Analyze this token's on-chain data:

[SUPPLY & MINT]
Total Supply (fixed): {data['total_supply']:,} tokens
Mint Authority: {data['mint_authority']}
Has Mint Function: {data['has_mint_function']}
Upgradeable Contract: {data['is_upgradeable_contract']}

[LIQUIDITY POOL]
LP Value: ${data['lp_value_usd']:,.2f} USD
LP Status: {data['lp_status']}
LP Holder Type: {data['lp_holder_type']}

[BURN]
Burn Address: {data['burn_address']}
Has Recover/Withdraw Function: {data['has_recover_function']}

[TAX]
Buy Tax: {data['buy_tax_pct']}%
Sell Tax: {data['sell_tax_pct']}%
Can Holders Sell: {data['can_sell']}

[HOLDER DISTRIBUTION]
Connected Cluster Holdings: {data['cluster_holding_pct']}% of supply
Wallets in Cluster: {data['cluster_wallet_count']}
Funded From Same Source: {data['cluster_funded_from_same_source']}
Unique Holders: {data['unique_holders']}
Token Age: {data['token_age_days']} days

[TRADING ACTIVITY 24H]
Volume: ${data['volume_24h_usd']:,.2f} USD
Unique Buyers: {data['unique_buyers_24h']}
Unique Sellers: {data['unique_sellers_24h']}
Buy/Sell Ratio: {buy_sell_ratio}

Classify this token, explain your reasoning across all dimensions, and calculate the estimated price drop if the cluster dumps their entire holding into the LP."""


def score_response(response: str, data: dict, true_label: str) -> float:
    score = 0.0
    response_upper = response.upper()

    # 1. Classification (0.4)
    classification = None
    for label in ["SCAM", "RUG_RISK", "LEGITIMATE"]:
        if f"CLASSIFICATION: {label}" in response_upper:
            classification = label
            break

    if classification == true_label:
        score += 0.4
    elif (
        (true_label == "SCAM" and classification == "RUG_RISK")
        or (true_label == "RUG_RISK" and classification == "SCAM")
    ):
        score += 0.1

    # 2. Reasoning quality (0.3)
    keywords = {
        "SCAM": ["cluster", "mint", "tax", "sell", "honeypot", "burn", "wash", "fake", "recover"],
        "RUG_RISK": ["cluster", "lp", "lock", "tax", "risk", "upgrade", "dev"],
        "LEGITIMATE": ["renounced", "burned", "locked", "dex", "distributed", "healthy", "low tax"],
    }
    kws = keywords.get(true_label, [])
    matched = sum(1 for kw in kws if kw in response.lower())
    score += min(matched / len(kws), 1.0) * 0.3

    # 3. Math accuracy (0.3)
    if true_label in ["SCAM", "RUG_RISK"]:
        true_impact = data["true_dump_impact"]
        match = re.search(r"DUMP IMPACT:\s*([\d.]+)%", response, re.IGNORECASE)
        if match:
            try:
                ai_impact = float(match.group(1))
                diff = abs(ai_impact - true_impact)
                if diff <= 2.0:
                    score += 0.3
                elif diff <= 5.0:
                    score += 0.15
                elif diff <= 10.0:
                    score += 0.05
            except ValueError:
                pass
    else:
        if "n/a" in response.lower() or "not applicable" in response.lower():
            score += 0.3

    return round(score, 4)


class ScamOrRugEnv(BaseEnv):
    name = "scam_or_rug_onchain"

    def __init__(self, config: ScamOrRugConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.labels = ["SCAM", "RUG_RISK", "LEGITIMATE"]
        self.percent_correct_buffer = []

    @classmethod
    def config_init(cls):
        return ScamOrRugConfig(), ServerBaseline()

    async def setup(self):
        pass

    async def get_next_item(self) -> Item:
        label = random.choice(self.labels)
        return (label,)

    async def collect_trajectories(self, item) -> tuple:
        label = item[0]
        data = generate_token_data(label)
        prompt = format_prompt(data)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        completions = await self.server.completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )

        scored = ScoredDataGroup()
        scored["tokens"] = []
        scored["masks"] = []
        scored["scores"] = []

        for completion in completions.choices:
            response = completion.message.content
            reward = score_response(response, data, label)

            # tokenize
            full_text = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": response}],
                tokenize=True,
                add_generation_prompt=False,
            )
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            tokens = full_text
             prompt_len = len(prompt_text)
             masks = [-100] * prompt_len + [1] * (len(full_text) - prompt_len)

            scored["tokens"].append(tokens)
            scored["masks"].append(masks)
            scored["scores"].append(reward)

            # track accuracy
            response_upper = response.upper()
            for lbl in ["SCAM", "RUG_RISK", "LEGITIMATE"]:
                if f"CLASSIFICATION: {lbl}" in response_upper:
                    self.percent_correct_buffer.append(1.0 if lbl == label else 0.0)
                    break

        return scored, []

    async def wandb_log(self, wandb_metrics: Optional[dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
            self.percent_correct_buffer = []

        await super().wandb_log(wandb_metrics)

    async def evaluate(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    ScamOrRugEnv.cli()