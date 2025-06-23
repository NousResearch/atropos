"""
Economic Analyzer for LLM Society Simulation Phase β
Provides comprehensive economic analysis and metrics
"""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .banking_system import BankingSystem
from .market_system import MarketSystem, ResourceType

logger = logging.getLogger(__name__)


@dataclass
class EconomicIndicators:
    """Economic health indicators for the society"""

    gini_coefficient: float = 0.0  # Wealth inequality (0 = equal, 1 = max inequality)
    gdp_growth_rate: float = 0.0  # Economic growth rate
    unemployment_rate: float = 0.0  # Unemployment percentage
    inflation_rate: float = 0.0  # Price inflation rate
    trade_balance: float = 0.0  # Export - import balance
    market_efficiency: float = 0.0  # Overall market efficiency
    financial_stability: float = 0.0  # Banking system stability

    # Derived metrics
    economic_health_score: float = 0.0  # Overall economic health (0-1)
    wealth_mobility_index: float = 0.0  # Social mobility based on wealth changes
    market_liquidity_index: float = 0.0  # Market liquidity across all resources


@dataclass
class WealthDistribution:
    """Analysis of wealth distribution across agents"""

    total_wealth: float = 0.0
    median_wealth: float = 0.0
    mean_wealth: float = 0.0
    wealth_percentiles: Dict[int, float] = field(
        default_factory=dict
    )  # 10th, 25th, 50th, 75th, 90th

    # Inequality metrics
    gini_coefficient: float = 0.0
    wealth_concentration_top_10: float = 0.0  # % of wealth held by top 10%
    wealth_concentration_top_1: float = 0.0  # % of wealth held by top 1%

    # Distribution categories
    wealthy_agents: int = 0  # Top 10%
    middle_class_agents: int = 0  # 25th-75th percentile
    poor_agents: int = 0  # Bottom 25%


class EconomicAnalyzer:
    """
    Analyzes economic conditions and provides insights
    """

    def __init__(self, market_system: MarketSystem, banking_system: BankingSystem):
        self.market_system = market_system
        self.banking_system = banking_system

        # Historical tracking
        self.historical_indicators: List[EconomicIndicators] = []
        self.historical_wealth_distributions: List[WealthDistribution] = []
        self.price_history: Dict[ResourceType, List[Tuple[float, float]]] = defaultdict(
            list
        )  # (timestamp, price)

        # Analysis parameters
        self.analysis_window = 30  # Days for trend analysis
        self.inflation_basket = [
            ResourceType.FOOD,
            ResourceType.MATERIALS,
            ResourceType.ENERGY,
        ]  # CPI basket

        logger.info("Economic Analyzer initialized")

    def analyze(self, agent_states: Dict[str, Any]) -> EconomicIndicators:
        """Perform comprehensive economic analysis"""

        # Analyze wealth distribution
        wealth_distribution = self._analyze_wealth_distribution(agent_states)

        # Calculate economic indicators
        indicators = EconomicIndicators()

        # Wealth inequality (Gini coefficient)
        indicators.gini_coefficient = wealth_distribution.gini_coefficient

        # GDP growth rate
        indicators.gdp_growth_rate = self._calculate_gdp_growth_rate(agent_states)

        # Unemployment rate
        indicators.unemployment_rate = self._calculate_unemployment_rate(agent_states)

        # Inflation rate
        indicators.inflation_rate = self._calculate_inflation_rate()

        # Trade balance
        indicators.trade_balance = self._calculate_trade_balance()

        # Market efficiency
        indicators.market_efficiency = self.market_system._calculate_market_efficiency()

        # Financial stability
        indicators.financial_stability = self._calculate_financial_stability()

        # Wealth mobility
        indicators.wealth_mobility_index = self._calculate_wealth_mobility(agent_states)

        # Market liquidity
        indicators.market_liquidity_index = self._calculate_market_liquidity()

        # Overall economic health score
        indicators.economic_health_score = self._calculate_economic_health_score(
            indicators
        )

        # Store for historical analysis
        self.historical_indicators.append(indicators)
        self.historical_wealth_distributions.append(wealth_distribution)

        # Update price history
        self._update_price_history()

        # Trim historical data to maintain performance
        max_history = 365  # Keep 1 year of data
        if len(self.historical_indicators) > max_history:
            self.historical_indicators = self.historical_indicators[-max_history:]
            self.historical_wealth_distributions = self.historical_wealth_distributions[
                -max_history:
            ]

        logger.debug(
            f"Economic analysis complete - Health Score: {indicators.economic_health_score:.3f}"
        )
        return indicators

    def _analyze_wealth_distribution(
        self, agent_states: Dict[str, Any]
    ) -> WealthDistribution:
        """Analyze wealth distribution across all agents"""

        # Collect wealth data
        agent_wealth = []

        for agent_id, agent in agent_states.items():
            # Calculate total wealth (cash + assets - debt)
            wealth = 0.0

            # Bank account balances
            if agent_id in self.banking_system.agent_accounts:
                for account_id in self.banking_system.agent_accounts[agent_id]:
                    account = self.banking_system.accounts[account_id]
                    wealth += account.balance

            # Agent resources (simplified valuation)
            if hasattr(agent, "resources"):
                for resource_type, quantity in agent.resources.items():
                    if resource_type != "currency" and quantity > 0:
                        # Get current market price
                        try:
                            resource_enum = ResourceType(resource_type)
                            price = self.market_system.get_market_price(resource_enum)
                            wealth += quantity * price
                        except ValueError:
                            # Unknown resource type, use default valuation
                            wealth += quantity * 1.0

            # Subtract debt
            agent_loans = [
                loan
                for loan in self.banking_system.loans.values()
                if loan.borrower_id == agent_id and loan.status.value == "active"
            ]
            debt = sum(loan.remaining_balance for loan in agent_loans)
            wealth -= debt

            agent_wealth.append(
                max(0.0, wealth)
            )  # Wealth can't be negative for distribution analysis

        if not agent_wealth:
            return WealthDistribution()

        # Sort wealth data
        sorted_wealth = sorted(agent_wealth)
        n = len(sorted_wealth)

        # Calculate distribution metrics
        distribution = WealthDistribution()
        distribution.total_wealth = sum(sorted_wealth)
        distribution.median_wealth = sorted_wealth[n // 2] if n > 0 else 0.0
        distribution.mean_wealth = distribution.total_wealth / n if n > 0 else 0.0

        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            idx = int((p / 100.0) * (n - 1))
            distribution.wealth_percentiles[p] = sorted_wealth[idx]

        # Calculate Gini coefficient
        distribution.gini_coefficient = self._calculate_gini_coefficient(sorted_wealth)

        # Calculate wealth concentration
        top_10_percent_count = max(1, n // 10)
        top_1_percent_count = max(1, n // 100)

        top_10_wealth = sum(sorted_wealth[-top_10_percent_count:])
        top_1_wealth = sum(sorted_wealth[-top_1_percent_count:])

        distribution.wealth_concentration_top_10 = (
            top_10_wealth / distribution.total_wealth
            if distribution.total_wealth > 0
            else 0
        )
        distribution.wealth_concentration_top_1 = (
            top_1_wealth / distribution.total_wealth
            if distribution.total_wealth > 0
            else 0
        )

        # Categorize agents
        bottom_25_threshold = distribution.wealth_percentiles.get(25, 0)
        top_25_threshold = distribution.wealth_percentiles.get(75, float("in"))

        distribution.poor_agents = sum(
            1 for w in agent_wealth if w <= bottom_25_threshold
        )
        distribution.wealthy_agents = sum(
            1 for w in agent_wealth if w >= top_25_threshold
        )
        distribution.middle_class_agents = (
            n - distribution.poor_agents - distribution.wealthy_agents
        )

        return distribution

    def _calculate_gini_coefficient(self, sorted_wealth: List[float]) -> float:
        """Calculate Gini coefficient for wealth inequality"""

        if len(sorted_wealth) <= 1:
            return 0.0

        n = len(sorted_wealth)
        total_wealth = sum(sorted_wealth)

        if total_wealth == 0:
            return 0.0

        # Calculate Gini coefficient using the formula
        cumulative_wealth = 0.0
        gini_sum = 0.0

        for i, wealth in enumerate(sorted_wealth):
            cumulative_wealth += wealth
            # Area under Lorenz curve
            gini_sum += cumulative_wealth / total_wealth

        # Gini = 1 - 2 * (Area under Lorenz curve) / n
        gini = 1.0 - (2.0 * gini_sum) / n

        return max(0.0, min(1.0, gini))

    def _calculate_gdp_growth_rate(self, agent_states: Dict[str, Any]) -> float:
        """Calculate GDP growth rate based on economic activity"""

        if len(self.historical_indicators) < 2:
            return 0.0

        # Simple GDP proxy: total trade volume + total wealth creation
        current_trade_volume = sum(
            sum(txn.total_cost for txn in market.recent_transactions)
            for market in self.market_system.markets.values()
        )

        # Get previous period's trade volume
        if len(self.historical_indicators) >= 2:
            # This is simplified - in reality we'd track this more carefully
            previous_volume = current_trade_volume * 0.95  # Assume 5% growth baseline

            if previous_volume > 0:
                growth_rate = (current_trade_volume - previous_volume) / previous_volume
                return max(-0.5, min(0.5, growth_rate))  # Cap at ±50%

        return 0.02  # Default 2% growth

    def _calculate_unemployment_rate(self, agent_states: Dict[str, Any]) -> float:
        """Calculate unemployment rate based on agent employment status"""

        if not agent_states:
            return 0.0

        unemployed_count = 0
        working_age_population = 0

        for agent_id, agent in agent_states.items():
            # Check if agent is working age (simplified)
            age = getattr(agent, "age", 25)
            if 18 <= age <= 65:
                working_age_population += 1

                # Check employment status (simplified)
                employed = getattr(agent, "employed", True)  # Default to employed
                if not employed:
                    unemployed_count += 1

        if working_age_population == 0:
            return 0.0

        return unemployed_count / working_age_population

    def _calculate_inflation_rate(self) -> float:
        """Calculate inflation rate based on price changes"""

        if len(self.historical_indicators) < 2:
            return 0.0

        # Calculate price changes for inflation basket
        current_prices = {}
        for resource_type in self.inflation_basket:
            current_prices[resource_type] = self.market_system.get_market_price(
                resource_type
            )

        # Compare with prices from previous period
        if (
            resource_type in self.price_history
            and len(self.price_history[resource_type]) >= 2
        ):
            total_price_change = 0.0
            valid_resources = 0

            for resource_type in self.inflation_basket:
                history = self.price_history[resource_type]
                if len(history) >= 2:
                    previous_price = history[-2][1]  # (timestamp, price)
                    current_price = history[-1][1]

                    if previous_price > 0:
                        price_change = (current_price - previous_price) / previous_price
                        total_price_change += price_change
                        valid_resources += 1

            if valid_resources > 0:
                inflation_rate = total_price_change / valid_resources
                return max(-0.2, min(0.2, inflation_rate))  # Cap at ±20%

        return 0.01  # Default 1% inflation

    def _calculate_trade_balance(self) -> float:
        """Calculate trade balance (simplified)"""

        # This is simplified - in a real economy we'd track imports/exports
        # For now, we'll use net trade activity as a proxy

        total_trade_value = 0.0
        for market in self.market_system.markets.values():
            # Recent transactions indicate trade activity
            recent_volume = sum(
                txn.total_cost for txn in market.recent_transactions[-10:]
            )
            total_trade_value += recent_volume

        # Simplified trade balance calculation
        # Positive indicates net export economy, negative indicates net import
        return total_trade_value * 0.1  # 10% of trade volume as proxy

    def _calculate_financial_stability(self) -> float:
        """Calculate financial system stability score"""

        banking_stats = self.banking_system.get_banking_statistics()

        # Factors affecting stability
        factors = []

        # Default rate (lower is better)
        default_rate = banking_stats.get("default_rate", 0.0)
        default_factor = max(
            0.0, 1.0 - default_rate * 5
        )  # 20% default rate = 0 stability
        factors.append(default_factor)

        # Loan-to-deposit ratio (around 0.8 is ideal)
        loan_deposit_ratio = banking_stats.get("loan_to_deposit_ratio", 0.0)
        optimal_ratio = 0.8
        ratio_factor = 1.0 - abs(loan_deposit_ratio - optimal_ratio) / optimal_ratio
        factors.append(max(0.0, ratio_factor))

        # Credit score health (higher average is better)
        avg_credit_score = banking_stats.get("average_credit_score", 700)
        credit_factor = (avg_credit_score - 300) / (850 - 300)  # Normalize to 0-1
        factors.append(max(0.0, min(1.0, credit_factor)))

        # Average the factors
        return sum(factors) / len(factors) if factors else 0.5

    def _calculate_wealth_mobility(self, agent_states: Dict[str, Any]) -> float:
        """Calculate wealth mobility index"""

        if len(self.historical_wealth_distributions) < 2:
            return 0.5  # Neutral mobility

        # This is simplified - would need to track individual agent wealth changes
        # For now, use changes in wealth distribution as proxy

        current_gini = self.historical_wealth_distributions[-1].gini_coefficient
        previous_gini = self.historical_wealth_distributions[-2].gini_coefficient

        # Decreasing inequality suggests mobility
        gini_change = previous_gini - current_gini
        mobility_factor = 0.5 + gini_change  # Baseline 0.5 + change

        return max(0.0, min(1.0, mobility_factor))

    def _calculate_market_liquidity(self) -> float:
        """Calculate overall market liquidity index"""

        total_liquidity = 0.0
        valid_markets = 0

        for market in self.market_system.markets.values():
            # Liquidity based on order book depth and recent activity
            order_count = len(market.buy_orders) + len(market.sell_orders)
            recent_activity = len(market.recent_transactions)

            # Normalize factors
            order_factor = min(1.0, order_count / 20.0)  # 20 orders = full liquidity
            activity_factor = min(
                1.0, recent_activity / 10.0
            )  # 10 transactions = full activity

            market_liquidity = (order_factor + activity_factor) / 2.0
            total_liquidity += market_liquidity
            valid_markets += 1

        return total_liquidity / valid_markets if valid_markets > 0 else 0.0

    def _calculate_economic_health_score(self, indicators: EconomicIndicators) -> float:
        """Calculate overall economic health score"""

        # Weight different factors
        weights = {
            "gini": 0.15,  # Lower inequality is better
            "gdp_growth": 0.20,  # Positive growth is good
            "unemployment": 0.15,  # Lower unemployment is better
            "inflation": 0.10,  # Moderate inflation is good
            "market_efficiency": 0.15,
            "financial_stability": 0.15,
            "wealth_mobility": 0.10,
        }

        # Normalize each indicator to 0-1 scale (higher is better)
        scores = {}

        # Gini coefficient (0 = perfect equality, 1 = maximum inequality)
        scores["gini"] = 1.0 - indicators.gini_coefficient

        # GDP growth rate (-50% to +50%)
        scores["gdp_growth"] = max(
            0.0, min(1.0, (indicators.gdp_growth_rate + 0.5) / 1.0)
        )

        # Unemployment rate (0% to 50%)
        scores["unemployment"] = max(0.0, 1.0 - indicators.unemployment_rate / 0.5)

        # Inflation rate (target 2%, range -20% to +20%)
        ideal_inflation = 0.02
        inflation_deviation = abs(indicators.inflation_rate - ideal_inflation)
        scores["inflation"] = max(0.0, 1.0 - inflation_deviation / 0.2)

        # Market efficiency (0 to 1)
        scores["market_efficiency"] = indicators.market_efficiency

        # Financial stability (0 to 1)
        scores["financial_stability"] = indicators.financial_stability

        # Wealth mobility (0 to 1)
        scores["wealth_mobility"] = indicators.wealth_mobility_index

        # Calculate weighted average
        health_score = sum(scores[key] * weights[key] for key in scores.keys())

        return max(0.0, min(1.0, health_score))

    def _update_price_history(self):
        """Update price history for all resources"""

        current_time = time.time()

        for resource_type, market in self.market_system.markets.items():
            price = market.current_price
            self.price_history[resource_type].append((current_time, price))

            # Keep only recent history
            max_history = 365  # Days
            cutoff_time = current_time - (max_history * 24 * 3600)
            self.price_history[resource_type] = [
                (t, p) for t, p in self.price_history[resource_type] if t > cutoff_time
            ]

    def get_economic_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get economic trends over specified period"""

        if len(self.historical_indicators) < 2:
            return {"error": "Insufficient historical data"}

        # Get indicators from the specified period
        recent_indicators = (
            self.historical_indicators[-days:]
            if len(self.historical_indicators) >= days
            else self.historical_indicators
        )

        if len(recent_indicators) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Calculate trends
        trends = {}

        # GDP growth trend
        gdp_values = [ind.gdp_growth_rate for ind in recent_indicators]
        trends["gdp_trend"] = (
            (gdp_values[-1] - gdp_values[0]) / len(gdp_values)
            if len(gdp_values) > 1
            else 0
        )

        # Unemployment trend
        unemployment_values = [ind.unemployment_rate for ind in recent_indicators]
        trends["unemployment_trend"] = (
            (unemployment_values[-1] - unemployment_values[0])
            / len(unemployment_values)
            if len(unemployment_values) > 1
            else 0
        )

        # Inflation trend
        inflation_values = [ind.inflation_rate for ind in recent_indicators]
        trends["inflation_trend"] = (
            (inflation_values[-1] - inflation_values[0]) / len(inflation_values)
            if len(inflation_values) > 1
            else 0
        )

        # Inequality trend
        gini_values = [ind.gini_coefficient for ind in recent_indicators]
        trends["inequality_trend"] = (
            (gini_values[-1] - gini_values[0]) / len(gini_values)
            if len(gini_values) > 1
            else 0
        )

        # Economic health trend
        health_values = [ind.economic_health_score for ind in recent_indicators]
        trends["health_trend"] = (
            (health_values[-1] - health_values[0]) / len(health_values)
            if len(health_values) > 1
            else 0
        )

        # Interpretation
        trends["interpretation"] = self._interpret_trends(trends)

        return trends

    def _interpret_trends(self, trends: Dict[str, float]) -> Dict[str, str]:
        """Interpret economic trends"""

        interpretations = {}

        # GDP interpretation
        gdp_trend = trends["gdp_trend"]
        if gdp_trend > 0.01:
            interpretations["gdp"] = "Strong economic growth"
        elif gdp_trend > 0.0:
            interpretations["gdp"] = "Moderate economic growth"
        elif gdp_trend > -0.01:
            interpretations["gdp"] = "Economic stagnation"
        else:
            interpretations["gdp"] = "Economic recession"

        # Unemployment interpretation
        unemployment_trend = trends["unemployment_trend"]
        if unemployment_trend < -0.01:
            interpretations["unemployment"] = "Improving employment"
        elif unemployment_trend < 0.01:
            interpretations["unemployment"] = "Stable employment"
        else:
            interpretations["unemployment"] = "Rising unemployment"

        # Inflation interpretation
        inflation_trend = trends["inflation_trend"]
        if abs(inflation_trend) < 0.005:
            interpretations["inflation"] = "Stable prices"
        elif inflation_trend > 0.01:
            interpretations["inflation"] = "Rising inflation"
        elif inflation_trend < -0.01:
            interpretations["inflation"] = "Deflationary pressure"
        else:
            interpretations["inflation"] = "Moderate price changes"

        # Inequality interpretation
        inequality_trend = trends["inequality_trend"]
        if inequality_trend < -0.01:
            interpretations["inequality"] = "Decreasing inequality"
        elif inequality_trend < 0.01:
            interpretations["inequality"] = "Stable inequality"
        else:
            interpretations["inequality"] = "Increasing inequality"

        # Overall health interpretation
        health_trend = trends["health_trend"]
        if health_trend > 0.01:
            interpretations["overall"] = "Improving economic conditions"
        elif health_trend > -0.01:
            interpretations["overall"] = "Stable economic conditions"
        else:
            interpretations["overall"] = "Deteriorating economic conditions"

        return interpretations

    def get_policy_recommendations(self, indicators: EconomicIndicators) -> List[str]:
        """Generate policy recommendations based on economic conditions"""

        recommendations = []

        # Inequality recommendations
        if indicators.gini_coefficient > 0.6:
            recommendations.append("Implement wealth redistribution policies")
            recommendations.append("Increase progressive taxation")

        # Growth recommendations
        if indicators.gdp_growth_rate < 0.0:
            recommendations.append(
                "Stimulate economic activity through public investment"
            )
            recommendations.append("Reduce interest rates to encourage borrowing")

        # Unemployment recommendations
        if indicators.unemployment_rate > 0.1:  # 10%
            recommendations.append("Implement job creation programs")
            recommendations.append("Provide job training and reskilling programs")

        # Inflation recommendations
        if indicators.inflation_rate > 0.05:  # 5%
            recommendations.append("Implement monetary tightening policies")
            recommendations.append("Monitor supply chain disruptions")
        elif indicators.inflation_rate < -0.02:  # -2%
            recommendations.append("Stimulate demand through fiscal policy")
            recommendations.append("Consider monetary easing")

        # Financial stability recommendations
        if indicators.financial_stability < 0.5:
            recommendations.append("Strengthen banking regulations")
            recommendations.append("Increase bank capital requirements")

        # Market efficiency recommendations
        if indicators.market_efficiency < 0.5:
            recommendations.append("Improve market transparency")
            recommendations.append("Reduce trade barriers and regulations")

        return recommendations[:5]  # Return top 5 recommendations
