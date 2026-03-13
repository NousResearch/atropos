"""
Capital City Quiz Environment for Atropos.
Trains LLMs to correctly identify capital cities of countries.
"""

import random
from typing import Optional

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from atroposlib.type_definitions import Item

CAPITAL_CITIES = {
    "France": "Paris",
    "Germany": "Berlin",
    "Japan": "Tokyo",
    "Brazil": "Brasília",
    "Australia": "Canberra",
    "Canada": "Ottawa",
    "India": "New Delhi",
    "China": "Beijing",
    "Egypt": "Cairo",
    "South Africa": "Pretoria",
    "Argentina": "Buenos Aires",
    "Mexico": "Mexico City",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Russia": "Moscow",
    "Nigeria": "Abuja",
    "Kenya": "Nairobi",
    "Indonesia": "Jakarta",
    "Pakistan": "Islamabad",
    "Bangladesh": "Dhaka",
    "Turkey": "Ankara",
    "Saudi Arabia": "Riyadh",
    "Sweden": "Stockholm",
    "Norway": "Oslo",
    "Netherlands": "Amsterdam",
    "Belgium": "Brussels",
    "Switzerland": "Bern",
    "Portugal": "Lisbon",
    "Greece": "Athens",
    "Poland": "Warsaw",
}

SYSTEM_PROMPT = (
    "You are a geography expert. When given a country name, "
    "respond with ONLY the capital city name. Nothing else."
)


class CapitalCityEnvConfig(BaseEnvConfig):
    """Configuration for the Capital City Quiz Environment."""

    env_name: str = "CapitalCityEnv"


class CapitalCityEnv(BaseEnv):
    """
    An environment that trains LLMs to identify capital cities.
    The model receives a country name and must respond with the correct capital.
    Rewards are binary: 1.0 for correct, 0.0 for incorrect.
    """

    name = "CapitalCityEnv"

    def __init__(self, config: CapitalCityEnvConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.countries = list(CAPITAL_CITIES.keys())

    @classmethod
    def config_init(cls) -> CapitalCityEnvConfig:
        return CapitalCityEnvConfig()

    async def collect_trajectories(
        self, item: Item
    ) -> tuple[Optional[ScoredDataGroup], list[Item]]:
        country = item[0]
        correct_capital = CAPITAL_CITIES[country]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What is the capital city of {country}?"},
        ]

        completions = await self.server.completion(
            messages=messages,
            n=self.config.group_size,
        )

        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []

        for completion in completions.choices:
            answer = completion.message.content.strip()
            score = 1.0 if correct_capital.lower() in answer.lower() else 0.0

            tokens, masks = self.tokenize_for_trainer(messages, answer)
            scored_data["tokens"].append(tokens)
            scored_data["masks"].append(masks)
            scored_data["scores"].append(score)

        return scored_data, []

    async def get_next_item(self) -> Item:
        country = random.choice(self.countries)
        return (country,)

    async def evaluate(self):
        correct = 0
        total = len(self.countries)
        for country in self.countries:
            correct_capital = CAPITAL_CITIES[country]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"What is the capital city of {country}?"},
            ]
            completion = await self.server.completion(messages=messages, n=1)
            answer = completion.choices[0].message.content.strip()
            if correct_capital.lower() in answer.lower():
                correct += 1

        accuracy = correct / total
        return {"eval/capital_city_accuracy": accuracy}


if __name__ == "__main__":
    CapitalCityEnv.cli()
