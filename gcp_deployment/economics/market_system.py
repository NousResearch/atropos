"""
Market System for LLM Society Simulation Phase β
Implements dynamic markets, supply/demand pricing, and trade networks
"""

import logging
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of tradeable resources"""

    FOOD = "food"
    MATERIALS = "materials"
    ENERGY = "energy"
    LUXURY = "luxury"
    TOOLS = "tools"
    KNOWLEDGE = "knowledge"
    SERVICES = "services"
    CURRENCY = "currency"


class TradeOrderType(Enum):
    """Types of trade orders"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Status of trade orders"""

    PENDING = "pending"
    PARTIAL = "partial"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TradeOrder:
    """Individual trade order in the market"""

    order_id: str
    agent_id: str
    resource_type: ResourceType
    order_type: TradeOrderType
    quantity: float
    price_per_unit: float
    max_price: Optional[float] = None  # For buy orders
    min_price: Optional[float] = None  # For sell orders

    # Order state
    quantity_filled: float = 0.0
    status: OrderStatus = OrderStatus.PENDING

    # Metadata
    created_time: float = field(default_factory=time.time)
    expiry_time: Optional[float] = None
    priority: float = 1.0  # Higher priority = processed first

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["resource_type"] = self.resource_type.value
        data["order_type"] = self.order_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeOrder":
        data["resource_type"] = ResourceType(data["resource_type"])
        data["order_type"] = TradeOrderType(data["order_type"])
        data["status"] = OrderStatus(data["status"])
        return cls(**data)


@dataclass
class Transaction:
    """Completed trade transaction"""

    transaction_id: str
    buyer_id: str
    seller_id: str
    resource_type: ResourceType
    quantity: float
    price_per_unit: float
    total_cost: float

    # Market context
    market_id: str
    buy_order_id: str
    sell_order_id: str

    # Metadata
    timestamp: float = field(default_factory=time.time)
    location: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["resource_type"] = self.resource_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transaction":
        data["resource_type"] = ResourceType(data["resource_type"])
        return cls(**data)


@dataclass
class PriceHistory:
    """Historical price data for a resource"""

    resource_type: ResourceType
    prices: deque = field(default_factory=lambda: deque(maxlen=1000))
    volumes: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add_price_point(self, price: float, volume: float):
        """Add new price data point"""
        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(time.time())

    def get_moving_average(self, window: int = 10) -> float:
        """Get moving average price"""
        if len(self.prices) < window:
            window = len(self.prices)
        if window == 0:
            return 0.0

        recent_prices = list(self.prices)[-window:]
        return sum(recent_prices) / len(recent_prices)

    def get_price_volatility(self, window: int = 20) -> float:
        """Calculate price volatility (standard deviation)"""
        if len(self.prices) < window:
            window = len(self.prices)
        if window < 2:
            return 0.0

        recent_prices = list(self.prices)[-window:]
        mean_price = sum(recent_prices) / len(recent_prices)
        variance = sum((p - mean_price) ** 2 for p in recent_prices) / len(
            recent_prices
        )
        return math.sqrt(variance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type.value,
            "prices": list(self.prices),
            "volumes": list(self.volumes),
            "timestamps": list(self.timestamps),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], maxlen: int = 1000) -> "PriceHistory":
        return cls(
            resource_type=ResourceType(data["resource_type"]),
            prices=deque(data.get("prices", []), maxlen=maxlen),
            volumes=deque(data.get("volumes", []), maxlen=maxlen),
            timestamps=deque(data.get("timestamps", []), maxlen=maxlen),
        )


class Market:
    """Individual market for a specific resource type"""

    def __init__(self, resource_type: ResourceType, base_price: float = 1.0):
        self.resource_type = resource_type
        self.base_price = base_price
        self.current_price = base_price

        # Order books
        self.buy_orders: List[TradeOrder] = []
        self.sell_orders: List[TradeOrder] = []

        # Market state
        self.total_supply = 0.0
        self.total_demand = 0.0
        self.daily_volume = 0.0
        self.price_history = PriceHistory(resource_type)

        # Market parameters
        self.price_elasticity = 0.1  # How much supply/demand affects price
        self.volatility_factor = 0.05  # Random price movement
        self.transaction_fee = 0.01  # 1% transaction fee

        # Recent transactions for market analysis
        self.recent_transactions: List[Transaction] = []

        logger.debug(
            f"Created market for {resource_type.value} with base price {base_price}"
        )

    def add_order(self, order: TradeOrder) -> bool:
        """Add trade order to the market"""
        try:
            if order.resource_type != self.resource_type:
                logger.warning(
                    f"Order resource type {order.resource_type} doesn't match market {self.resource_type}"
                )
                return False

            # Add to appropriate order book
            if order.order_type == TradeOrderType.BUY:
                self.buy_orders.append(order)
                self.buy_orders.sort(
                    key=lambda x: (-x.price_per_unit, x.created_time)
                )  # Highest price first
                self.total_demand += order.quantity
            else:
                self.sell_orders.append(order)
                self.sell_orders.sort(
                    key=lambda x: (x.price_per_unit, x.created_time)
                )  # Lowest price first
                self.total_supply += order.quantity

            logger.debug(
                f"Added {order.order_type.value} order for {order.quantity} {self.resource_type.value} at {order.price_per_unit}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error adding order to market {self.resource_type.value}: {e}",
                exc_info=True,
            )
            return False

    def process_trades(self) -> List[Transaction]:
        """Process matching buy and sell orders"""
        transactions = []
        try:
            while self.buy_orders and self.sell_orders:
                buy_order = self.buy_orders[0]
                sell_order = self.sell_orders[0]

                # Check if orders can match
                if buy_order.price_per_unit >= sell_order.price_per_unit:
                    # Orders match - execute trade
                    trade_quantity = min(
                        buy_order.quantity - buy_order.quantity_filled,
                        sell_order.quantity - sell_order.quantity_filled,
                    )

                    # Price discovery - use average of bid and ask
                    trade_price = (
                        buy_order.price_per_unit + sell_order.price_per_unit
                    ) / 2.0

                    # Create transaction
                    transaction = Transaction(
                        transaction_id=f"txn_{self.resource_type.value}_{len(self.recent_transactions):06d}_{int(time.time())}",
                        buyer_id=buy_order.agent_id,
                        seller_id=sell_order.agent_id,
                        resource_type=self.resource_type,
                        quantity=trade_quantity,
                        price_per_unit=trade_price,
                        total_cost=trade_quantity * trade_price,
                        market_id=f"market_{self.resource_type.value}",
                        buy_order_id=buy_order.order_id,
                        sell_order_id=sell_order.order_id,
                    )

                    transactions.append(transaction)
                    self.recent_transactions.append(transaction)

                    # Update order quantities
                    buy_order.quantity_filled += trade_quantity
                    sell_order.quantity_filled += trade_quantity

                    # Update market state
                    self.daily_volume += trade_quantity
                    self.current_price = trade_price
                    self.price_history.add_price_point(trade_price, trade_quantity)

                    # Remove or update completed orders
                    if buy_order.quantity_filled >= buy_order.quantity:
                        buy_order.status = OrderStatus.COMPLETED
                        self.buy_orders.pop(0)
                        self.total_demand -= buy_order.quantity
                    else:
                        buy_order.status = OrderStatus.PARTIAL

                    if sell_order.quantity_filled >= sell_order.quantity:
                        sell_order.status = OrderStatus.COMPLETED
                        self.sell_orders.pop(0)
                        self.total_supply -= sell_order.quantity
                    else:
                        sell_order.status = OrderStatus.PARTIAL

                    logger.debug(
                        f"Executed trade in {self.resource_type.value} market: {trade_quantity} units at {trade_price}"
                    )
                else:
                    # No more matching orders
                    break
        except Exception as e:
            logger.error(
                f"Error processing trades in market {self.resource_type.value}: {e}",
                exc_info=True,
            )
        return transactions

    def calculate_market_price(self) -> float:
        """Calculate current market price based on supply and demand"""
        try:
            if self.total_supply == 0 and self.total_demand == 0:
                return self.current_price

            # Supply/demand ratio affects price
            if self.total_supply > 0:
                demand_supply_ratio = self.total_demand / self.total_supply
            else:
                demand_supply_ratio = 2.0  # High demand, no supply

            # Price adjustment based on ratio
            price_multiplier = 1.0 + (demand_supply_ratio - 1.0) * self.price_elasticity

            # Apply volatility
            volatility = random.gauss(0, self.volatility_factor)
            price_multiplier *= 1.0 + volatility

            new_price = self.base_price * price_multiplier

            # Keep price within reasonable bounds
            min_price = self.base_price * 0.1
            max_price = self.base_price * 10.0
            new_price = max(min_price, min(max_price, new_price))

            return new_price
        except Exception as e:
            logger.error(
                f"Error calculating market price for {self.resource_type.value}: {e}",
                exc_info=True,
            )
            return self.current_price

    def update_market_price(self):
        """Update market price based on current conditions"""
        try:
            self.current_price = self.calculate_market_price()
        except Exception as e:
            logger.error(
                f"Error updating market price for {self.resource_type.value}: {e}",
                exc_info=True,
            )

    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        try:
            best_bid = self.buy_orders[0].price_per_unit if self.buy_orders else 0.0
            best_ask = (
                self.sell_orders[0].price_per_unit if self.sell_orders else float("in")
            )
            spread = best_ask - best_bid if best_ask != float("inf") else 0.0

            return {
                "resource_type": self.resource_type.value,
                "current_price": self.current_price,
                "base_price": self.base_price,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "total_supply": self.total_supply,
                "total_demand": self.total_demand,
                "daily_volume": self.daily_volume,
                "active_buy_orders": len(self.buy_orders),
                "active_sell_orders": len(self.sell_orders),
                "recent_transactions": len(self.recent_transactions),
                "price_moving_average": self.price_history.get_moving_average(),
                "price_volatility": self.price_history.get_price_volatility(),
            }
        except Exception as e:
            logger.error(
                f"Error getting market summary for {self.resource_type.value}: {e}",
                exc_info=True,
            )
            return {
                "resource_type": self.resource_type.value,
                "current_price": self.current_price,
                "base_price": self.base_price,
                "best_bid": 0.0,
                "best_ask": float("in"),
                "spread": 0.0,
                "total_supply": self.total_supply,
                "total_demand": self.total_demand,
                "daily_volume": self.daily_volume,
                "active_buy_orders": 0,
                "active_sell_orders": 0,
                "recent_transactions": 0,
                "price_moving_average": 0.0,
                "price_volatility": 0.0,
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type.value,
            "base_price": self.base_price,
            "current_price": self.current_price,
            "buy_orders": [order.to_dict() for order in self.buy_orders],
            "sell_orders": [order.to_dict() for order in self.sell_orders],
            "total_supply": self.total_supply,
            "total_demand": self.total_demand,
            "daily_volume": self.daily_volume,
            "price_history": self.price_history.to_dict(),
            "price_elasticity": self.price_elasticity,
            "volatility_factor": self.volatility_factor,
            "transaction_fee": self.transaction_fee,
            "recent_transactions": [txn.to_dict() for txn in self.recent_transactions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Market":
        resource_type = ResourceType(data["resource_type"])
        market = cls(resource_type, data.get("base_price", 1.0))
        market.current_price = data.get("current_price", market.base_price)
        market.buy_orders = [
            TradeOrder.from_dict(order_data)
            for order_data in data.get("buy_orders", [])
        ]
        market.sell_orders = [
            TradeOrder.from_dict(order_data)
            for order_data in data.get("sell_orders", [])
        ]
        market.total_supply = data.get("total_supply", 0.0)
        market.total_demand = data.get("total_demand", 0.0)
        market.daily_volume = data.get("daily_volume", 0.0)
        market.price_history = PriceHistory.from_dict(data["price_history"])
        market.price_elasticity = data.get("price_elasticity", 0.1)
        market.volatility_factor = data.get("volatility_factor", 0.05)
        market.transaction_fee = data.get("transaction_fee", 0.01)
        market.recent_transactions = [
            Transaction.from_dict(txn_data)
            for txn_data in data.get("recent_transactions", [])
        ]
        return market


class MarketSystem:
    """
    Manages all markets and coordinates trading activities
    """

    def __init__(self):
        # Create markets for each resource type
        self.markets: Dict[ResourceType, Market] = {}

        # Default base prices for different resources
        base_prices = {
            ResourceType.FOOD: 1.0,
            ResourceType.MATERIALS: 2.0,
            ResourceType.ENERGY: 1.5,
            ResourceType.LUXURY: 10.0,
            ResourceType.TOOLS: 5.0,
            ResourceType.KNOWLEDGE: 3.0,
            ResourceType.SERVICES: 2.5,
            ResourceType.CURRENCY: 1.0,
        }

        for resource_type in ResourceType:
            base_price = base_prices.get(resource_type, 1.0)
            self.markets[resource_type] = Market(resource_type, base_price)

        # Global market state
        self.all_transactions: List[Transaction] = []
        self.trade_networks: Dict[str, Set[str]] = defaultdict(
            set
        )  # Agent trading relationships

        # Market regulation parameters
        self.market_hours = {"open": 6, "close": 18}  # Market operates 6 AM to 6 PM
        self.price_change_limits = {"min": 0.5, "max": 2.0}  # Max price change per day

        logger.info(f"Market System initialized with {len(self.markets)} markets")

    def submit_order(
        self,
        agent_id: str,
        resource_type: ResourceType,
        order_type: TradeOrderType,
        quantity: float,
        price_per_unit: float,
        **kwargs,
    ) -> Optional[str]:
        """Submit a trade order to the market"""
        try:
            if resource_type not in self.markets:
                logger.error(f"Market for {resource_type} not found")
                return None

            market = self.markets[resource_type]

            # Create order
            order_id = f"order_{agent_id}_{resource_type.value}_{int(time.time())}_{random.randint(1000,9999)}"
            order = TradeOrder(
                order_id=order_id,
                agent_id=agent_id,
                resource_type=resource_type,
                order_type=order_type,
                quantity=quantity,
                price_per_unit=price_per_unit,
                **kwargs,
            )

            # Add order to market
            if market.add_order(order):
                logger.info(
                    f"Agent {agent_id} submitted {order_type.value} order {order_id}: {quantity} {resource_type.value} at {price_per_unit}"
                )
                return order_id

            return None
        except Exception as e:
            logger.error(
                f"Error submitting order for agent {agent_id} ({resource_type.value}): {e}",
                exc_info=True,
            )
            return None

    def process_all_markets(self) -> Dict[ResourceType, List[Transaction]]:
        """Process trades in all markets"""
        all_market_transactions = {}
        try:
            for resource_type, market in self.markets.items():
                if resource_type == ResourceType.CURRENCY:
                    continue  # No trading currency against itself
                # Process trades
                transactions = market.process_trades()
                all_market_transactions[resource_type] = transactions

                # Add to global transaction history
                self.all_transactions.extend(transactions)

                # Update trade networks
                for transaction in transactions:
                    self.trade_networks[transaction.buyer_id].add(transaction.seller_id)
                    self.trade_networks[transaction.seller_id].add(transaction.buyer_id)

                # Update market prices
                market.update_market_price()
        except Exception as e:
            logger.error(f"Error processing all markets: {e}", exc_info=True)
            # Return what has been processed so far, or an empty dict if critical error
        return all_market_transactions

    def get_market_price(self, resource_type: ResourceType) -> float:
        """Get current market price for a resource"""
        try:
            if resource_type in self.markets:
                return self.markets[resource_type].current_price
            return 0.0
        except Exception as e:
            logger.error(
                f"Error getting market price for {resource_type.value}: {e}",
                exc_info=True,
            )
            return 0.0

    def get_price_quote(
        self, resource_type: ResourceType, quantity: float, order_type: TradeOrderType
    ) -> Dict[str, float]:
        """Get price quote for a potential trade"""
        try:
            if resource_type not in self.markets:
                return {"error": "Market not found"}

            market = self.markets[resource_type]

            if order_type == TradeOrderType.BUY:
                # Check sell orders for best prices
                available_quantity = 0.0
                total_cost = 0.0

                for sell_order in market.sell_orders:
                    available_qty = sell_order.quantity - sell_order.quantity_filled
                    take_qty = min(quantity - available_quantity, available_qty)

                    total_cost += take_qty * sell_order.price_per_unit
                    available_quantity += take_qty

                    if available_quantity >= quantity:
                        break

                if available_quantity > 0:
                    avg_price = total_cost / available_quantity
                    return {
                        "avg_price": avg_price,
                        "total_cost": total_cost,
                        "available_quantity": available_quantity,
                        "market_impact": abs(avg_price - market.current_price)
                        / market.current_price,
                    }

            else:  # SELL order
                # Check buy orders for best prices
                available_quantity = 0.0
                total_revenue = 0.0

                for buy_order in market.buy_orders:
                    available_qty = buy_order.quantity - buy_order.quantity_filled
                    take_qty = min(quantity - available_quantity, available_qty)

                    total_revenue += take_qty * buy_order.price_per_unit
                    available_quantity += take_qty

                    if available_quantity >= quantity:
                        break

                if available_quantity > 0:
                    avg_price = total_revenue / available_quantity
                    return {
                        "avg_price": avg_price,
                        "total_revenue": total_revenue,
                        "available_quantity": available_quantity,
                        "market_impact": abs(avg_price - market.current_price)
                        / market.current_price,
                    }

            return {"error": "Insufficient market liquidity"}
        except Exception as e:
            logger.error(
                f"Error getting price quote for {resource_type.value}: {e}",
                exc_info=True,
            )
            return {"error": f"Error generating quote for {resource_type.value}"}

    def get_agent_trading_history(
        self, agent_id: str, days: int = 7
    ) -> List[Transaction]:
        """Get trading history for a specific agent"""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)

            agent_transactions = [
                txn
                for txn in self.all_transactions
                if (txn.buyer_id == agent_id or txn.seller_id == agent_id)
                and txn.timestamp > cutoff_time
            ]

            return sorted(agent_transactions, key=lambda x: x.timestamp, reverse=True)
        except Exception as e:
            logger.error(
                f"Error getting agent trading history for {agent_id}: {e}",
                exc_info=True,
            )
            return []

    def calculate_agent_trade_reputation(self, agent_id: str) -> Dict[str, float]:
        """Calculate agent's trading reputation metrics"""
        try:
            agent_transactions = self.get_agent_trading_history(agent_id, days=30)

            if not agent_transactions:
                return {"reputation": 0.5, "trade_volume": 0.0, "reliability": 0.5}

            # Calculate metrics
            total_volume = sum(txn.total_cost for txn in agent_transactions)
            trade_count = len(agent_transactions)
            unique_partners = len(self.trade_networks.get(agent_id, set()))

            # Simple reputation calculation
            base_reputation = 0.5
            volume_bonus = min(
                0.3, total_volume / 1000.0
            )  # Max 0.3 bonus for high volume
            diversity_bonus = min(
                0.2, unique_partners / 10.0
            )  # Max 0.2 bonus for trading diversity

            reputation = base_reputation + volume_bonus + diversity_bonus

            return {
                "reputation": min(1.0, reputation),
                "trade_volume": total_volume,
                "trade_count": trade_count,
                "unique_partners": unique_partners,
                "reliability": min(
                    1.0, trade_count / 30.0
                ),  # Based on trading frequency
            }
        except Exception as e:
            logger.error(
                f"Error calculating agent trade reputation for {agent_id}: {e}",
                exc_info=True,
            )
            return {"reputation": 0.5, "trade_volume": 0.0, "reliability": 0.5}

    def get_market_statistics(self) -> Dict[str, Any]:
        """Get comprehensive market system statistics"""
        try:
            total_transactions = len(self.all_transactions)
            total_volume = sum(txn.total_cost for txn in self.all_transactions)

            # Calculate market summaries
            market_summaries = {}
            for resource_type, market in self.markets.items():
                market_summaries[resource_type.value] = market.get_market_summary()

            # Calculate trade network statistics
            active_traders = len(self.trade_networks)
            total_connections = sum(
                len(connections) for connections in self.trade_networks.values()
            )
            avg_connections = total_connections / max(1, active_traders)

            # Recent activity (last 24 hours)
            recent_cutoff = time.time() - 86400
            recent_transactions = [
                txn for txn in self.all_transactions if txn.timestamp > recent_cutoff
            ]
            recent_volume = sum(txn.total_cost for txn in recent_transactions)

            return {
                "total_transactions": total_transactions,
                "total_volume": total_volume,
                "recent_transactions": len(recent_transactions),
                "recent_volume": recent_volume,
                "active_traders": active_traders,
                "average_connections_per_trader": avg_connections,
                "markets": market_summaries,
                "most_traded_resource": self._get_most_traded_resource(),
                "market_efficiency": self._calculate_market_efficiency(),
            }
        except Exception as e:
            logger.error(f"Error getting market statistics: {e}", exc_info=True)
            return {
                "total_transactions": 0,
                "total_volume": 0,
                "recent_transactions": 0,
                "recent_volume": 0,
                "active_traders": 0,
                "average_connections_per_trader": 0,
                "markets": {},
                "most_traded_resource": "none",
                "market_efficiency": 0.0,
            }

    def _get_most_traded_resource(self) -> str:
        """Get the most actively traded resource"""
        try:
            resource_volumes = defaultdict(float)

            for txn in self.all_transactions:
                resource_volumes[txn.resource_type.value] += txn.total_cost

            if resource_volumes:
                return max(resource_volumes.items(), key=lambda x: x[1])[0]
            return "none"
        except Exception as e:
            logger.error(f"Error getting most traded resource: {e}", exc_info=True)
            return "none"

    def _calculate_market_efficiency(self) -> float:
        """Calculate overall market efficiency (0-1 scale)"""
        try:
            if not self.markets:
                return 0.0

            total_efficiency = 0.0

            for market in self.markets.values():
                # Market efficiency based on spread and liquidity
                best_bid = (
                    market.buy_orders[0].price_per_unit if market.buy_orders else 0.0
                )
                best_ask = (
                    market.sell_orders[0].price_per_unit
                    if market.sell_orders
                    else float("in")
                )

                if best_ask == float("in") or best_bid == 0.0:
                    efficiency = 0.5  # Neutral efficiency for illiquid markets
                else:
                    spread = best_ask - best_bid
                    spread_efficiency = max(0.0, 1.0 - (spread / market.current_price))

                    # Liquidity efficiency
                    total_orders = len(market.buy_orders) + len(market.sell_orders)
                    liquidity_efficiency = min(
                        1.0, total_orders / 10.0
                    )  # Good liquidity = 10+ orders

                    efficiency = (spread_efficiency + liquidity_efficiency) / 2.0

                total_efficiency += efficiency

            return total_efficiency / len(self.markets)
        except Exception as e:
            logger.error(f"Error calculating market efficiency: {e}", exc_info=True)
            return 0.0

    def get_resource_market_summary(self, resource_type: ResourceType) -> Optional[str]:
        """
        Provides a concise string summary of a specific resource market for LLM prompts.
        Example: "Food (Price: 2.5, Bid: 2.4, Ask: 2.6, Demand(Qty): 50, Supply(Qty): 30)"
        """
        if resource_type not in self.markets:
            logger.warning(
                f"Market for {resource_type.value} not found when requesting summary."
            )
            return None

        market = self.markets[resource_type]
        summary_data = market.get_market_summary()  # This returns a dictionary

        if not summary_data:
            return f"{resource_type.value.capitalize()}: Market data unavailable."

        # Extract key figures for the concise summary
        price = summary_data.get("current_price", "N/A")
        bid = summary_data.get("best_bid", "N/A")
        ask = summary_data.get("best_ask", "N/A")
        # total_demand is the sum of quantities in active buy orders
        demand_qty = summary_data.get("total_demand", "N/A")
        # total_supply is the sum of quantities in active sell orders
        supply_qty = summary_data.get("total_supply", "N/A")

        # Format numbers nicely
        try:
            price_str = f"{float(price):.2f}" if price != "N/A" else "N/A"
            bid_str = f"{float(bid):.2f}" if bid != "N/A" else "N/A"
            # Handle ask potentially being infinity if no sell orders
            ask_str = (
                f"{float(ask):.2f}" if ask != "N/A" and ask != float("in") else "N/A"
            )
            demand_qty_str = (
                f"{float(demand_qty):.0f}" if demand_qty != "N/A" else "N/A"
            )
            supply_qty_str = (
                f"{float(supply_qty):.0f}" if supply_qty != "N/A" else "N/A"
            )
        except ValueError:
            # Fallback if float conversion fails for some reason
            price_str, bid_str, ask_str, demand_qty_str, supply_qty_str = (
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            )

        return f"{resource_type.value.capitalize()} (Price:{price_str}, Bid:{bid_str}, Ask:{ask_str}, Demand(Qty):{demand_qty_str}, Supply(Qty):{supply_qty_str})"

    def get_detailed_resource_market_info(
        self, resource_type: ResourceType
    ) -> Optional[str]:
        """
        Provides a detailed string summary of a specific resource market for LLM prompts.
        Example: "Food Market: Price: 2.50, Bid: 2.40 (X orders), Ask: 2.60 (Y orders), Demand(Qty): 50, Supply(Qty): 30, Volume: 100"
        """
        if resource_type not in self.markets:
            logger.warning(
                f"Market for {resource_type.value} not found when requesting detailed info."
            )
            return f"{resource_type.value.capitalize()}: No market data found."

        market = self.markets[resource_type]
        summary_data = market.get_market_summary()  # This returns a dictionary

        if not summary_data:
            return f"{resource_type.value.capitalize()}: Market data unavailable."

        try:
            price_str = f"{float(summary_data.get('current_price', 0)):.2f}"
            bid_str = f"{float(summary_data.get('best_bid', 0)):.2f}"
            ask_str = (
                f"{float(summary_data.get('best_ask', float('in"))):.2f}"
                if summary_data.get("best_ask", float("in")) != float("in")
                else "N/A"
            )
            demand_qty_str = f"{float(summary_data.get('total_demand', 0)):.0f}"
            supply_qty_str = f"{float(summary_data.get('total_supply', 0)):.0f}"
            buy_orders_count = summary_data.get("active_buy_orders", 0)
            sell_orders_count = summary_data.get("active_sell_orders", 0)
            volume_str = f"{float(summary_data.get('daily_volume', 0)):.0f}"

            return (
                f"{resource_type.value.capitalize()} Market: Price: {price_str}, "
                f"Best Bid: {bid_str} ({buy_orders_count} orders), Best Ask: {ask_str} ({sell_orders_count} orders), "
                f"Demand(Qty): {demand_qty_str}, Supply(Qty): {supply_qty_str}, Volume: {volume_str}"
            )
        except Exception as e:
            logger.error(
                f"Error formatting detailed market info for {resource_type.value}: {e}"
            )
            return (
                f"{resource_type.value.capitalize()}: Error retrieving market details."
            )

    def get_general_market_overview(self) -> Optional[str]:
        """
        Provides a general overview of all markets, listing resource and current price.
        Example: "Market Overview: Food (Price: 2.50); Tools (Price: 15.75); Materials (Price: 3.20)"
        """
        if not self.markets:
            return "Market Overview: No markets available."

        overview_parts = []
        for resource_type, market in self.markets.items():
            if (
                resource_type == ResourceType.CURRENCY
            ):  # Usually no need to list currency market price against itself
                continue
            price = market.current_price
            overview_parts.append(
                f"{resource_type.value.capitalize()} (Price: {price:.2f})"
            )

        if not overview_parts:
            return "Market Overview: No active resource markets to display."

        return "Market Overview: " + "; ".join(overview_parts) + "."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "markets": {
                res_type.value: market.to_dict()
                for res_type, market in self.markets.items()
            },
            "all_transactions": [txn.to_dict() for txn in self.all_transactions],
            "trade_networks": {
                agent_id: list(partners)
                for agent_id, partners in self.trade_networks.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketSystem":
        system = cls()
        system.markets = {
            ResourceType(res_val): Market.from_dict(market_data)
            for res_val, market_data in data.get("markets", {}).items()
        }
        system.all_transactions = [
            Transaction.from_dict(txn_data)
            for txn_data in data.get("all_transactions", [])
        ]
        system.trade_networks = defaultdict(
            set,
            {
                agent_id: set(partners)
                for agent_id, partners in data.get("trade_networks", {}).items()
            },
        )
        # Re-initialize any markets that might not have been in the saved data (e.g. new ResourceType added after save)
        # This ensures all ResourceTypes have a market instance, using defaults if not in save data.
        base_prices = {
            ResourceType.FOOD: 1.0,
            ResourceType.MATERIALS: 2.0,
            ResourceType.ENERGY: 1.5,
            ResourceType.LUXURY: 10.0,
            ResourceType.TOOLS: 5.0,
            ResourceType.KNOWLEDGE: 3.0,
            ResourceType.SERVICES: 2.5,
            ResourceType.CURRENCY: 1.0,
        }
        for rt_enum in ResourceType:
            if rt_enum not in system.markets:
                logger.info(
                    f"MarketSystem.from_dict: Initializing missing market for {rt_enum.value} from defaults."
                )
                system.markets[rt_enum] = Market(rt_enum, base_prices.get(rt_enum, 1.0))
        return system
