# src/database/database_handler.py
import asyncio  # Added asyncio
import datetime
import logging
import time  # For default real_timestamp in save methods
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.economics.banking_system import AccountType, LoanStatus, LoanType
from src.economics.banking_system import (
    TransactionType as BankingTransactionType,  # For BankingTransaction
)

# Import Enums from other modules to ensure type consistency if storing enum values
# from src.agents.llm_agent import AgentState # Example, if we store Agent state history
from src.economics.market_system import (  # For MarketTransaction if we expand
    OrderStatus,
    ResourceType,
    TradeOrderType,
)

logger = logging.getLogger(__name__)
Base = declarative_base()

# --- Conceptual SQLAlchemy Models (based on DATABASE_SCHEMA.MD) ---


class SimulationRunDB(Base):
    __tablename__ = "simulation_runs"
    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_snapshot = Column(JSONB)
    start_time = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    end_time = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50))
    notes = Column(Text, nullable=True)


class AgentMemoryDB(Base):
    __tablename__ = "agent_memories"
    memory_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id_str = Column(String(255), nullable=False, index=True)
    simulation_run_id = Column(
        UUID(as_uuid=True), ForeignKey("simulation_runs.run_id"), nullable=True
    )
    step_timestamp = Column(Integer, nullable=False, index=True)
    real_timestamp = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    content = Column(Text, nullable=False)
    importance = Column(Float)
    tags = Column(ARRAY(String), nullable=True)
    related_agent_id = Column(String(255), nullable=True)


class MarketTransactionDB(Base):
    __tablename__ = "market_transactions"
    transaction_id_pk = Column(
        Integer, primary_key=True, autoincrement=True
    )  # Changed to Integer for autoincrement by default in many DBs
    transaction_sim_id = Column(String(255), nullable=False, unique=True)
    simulation_run_id = Column(
        UUID(as_uuid=True), ForeignKey("simulation_runs.run_id"), nullable=True
    )
    step_timestamp = Column(Integer, nullable=False, index=True)
    real_timestamp = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    market_id = Column(String(255), nullable=False)
    resource_type = Column(String(50), nullable=False, index=True)
    buyer_agent_id = Column(String(255), nullable=False, index=True)
    seller_agent_id = Column(String(255), nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    price_per_unit = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)
    buy_order_id = Column(String(255), nullable=True)
    sell_order_id = Column(String(255), nullable=True)


class BankingTransactionDB(Base):
    __tablename__ = "banking_transactions"
    transaction_id_pk = Column(Integer, primary_key=True, autoincrement=True)
    transaction_sim_id = Column(String(255), nullable=False, unique=True)
    simulation_run_id = Column(
        UUID(as_uuid=True), ForeignKey("simulation_runs.run_id"), nullable=True
    )
    step_timestamp = Column(Integer, nullable=False, index=True)
    real_timestamp = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    account_id = Column(String(255), nullable=False, index=True)
    agent_id_str = Column(String(255), nullable=False, index=True)
    transaction_type = Column(
        String(50), nullable=False, index=True
    )  # Store enum value as string
    amount = Column(Float, nullable=False)
    balance_after = Column(Float, nullable=False)
    description = Column(Text, nullable=True)
    counterparty_id = Column(String(255), nullable=True)
    reference_id = Column(String(255), nullable=True)


class SimulationEventDB(Base):
    __tablename__ = "simulation_events"
    event_id = Column(
        Integer, primary_key=True, autoincrement=True
    )  # Using Integer for simplicity, UUID also fine
    simulation_run_id = Column(
        UUID(as_uuid=True), ForeignKey("simulation_runs.run_id"), nullable=True
    )
    step_timestamp = Column(Integer, nullable=False, index=True)
    real_timestamp = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    event_type = Column(String(100), nullable=False, index=True)
    agent_id_primary = Column(String(255), nullable=True, index=True)
    agent_id_secondary = Column(String(255), nullable=True, index=True)
    details = Column(JSONB, nullable=True)  # Flexible JSON for event-specific data
    description = Column(Text, nullable=True)


# ... (Potentially more models for Agents, Families, BankAccounts, Loans, etc.)


class DatabaseHandler:
    def __init__(
        self,
        db_url: Optional[str] = None,
        config: Optional[Any] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ):
        self.db_url = db_url or getattr(
            config.output, "database_url", "sqlite:///./llm_society_dynamic_data.db"
        )
        self.engine = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._current_run_id: Optional[uuid.UUID] = None
        self.loop = asyncio.get_event_loop()

        if self.db_url:
            try:
                self.engine = create_engine(self.db_url)
                Base.metadata.create_all(
                    self.engine
                )  # Create tables if they don't exist
                self.SessionLocal = sessionmaker(
                    autocommit=False, autoflush=False, bind=self.engine
                )
                logger.info(f"DatabaseHandler initialized with URL: {self.db_url}")
                # Defer starting simulation run record to an async method to be called by simulator
            except Exception as e:
                logger.error(
                    f"Failed to initialize database at {self.db_url}: {e}",
                    exc_info=True,
                )
                self.engine = None  # Ensure engine is None if init fails
                self.SessionLocal = None
        else:
            logger.warning(
                "DatabaseHandler initialized without a db_url. Database operations will be skipped."
            )

    def _execute_db_operation(self, func, *args, **kwargs):
        # Helper to run synchronous DB code in executor
        if not self.SessionLocal:
            raise ConnectionError("Database not initialized or session not available.")
        db: Session = self.SessionLocal()
        try:
            result = func(db, *args, **kwargs)
            db.commit()
            return result
        except Exception as e:
            db.rollback()
            logger.error(f"Database operation failed: {e}", exc_info=True)
            raise  # Re-raise after rollback to signal failure to caller
        finally:
            db.close()

    async def start_simulation_run(
        self, config_snapshot: Optional[Dict[str, Any]] = None
    ) -> Optional[uuid.UUID]:
        if not self.engine:
            return None

        def _start_run(db: Session, cfg_snap: Optional[Dict[str, Any]]):
            new_run = SimulationRunDB(config_snapshot=cfg_snap, status="running")
            db.add(new_run)
            db.flush()
            db.refresh(new_run)  # Flush to get ID before commit
            self._current_run_id = new_run.run_id
            return new_run.run_id

        try:
            run_id = await self.loop.run_in_executor(
                None, self._execute_db_operation, _start_run, config_snapshot
            )
            logger.info(f"Started new simulation run with ID: {run_id}")
            return run_id
        except Exception as e:
            logger.error(f"Async start_simulation_run failed: {e}")
            return None

    async def end_simulation_run(self, status: str = "completed"):
        if not self.engine or not self._current_run_id:
            return

        def _end_run(db: Session, run_id: uuid.UUID, end_status: str):
            run = (
                db.query(SimulationRunDB)
                .filter(SimulationRunDB.run_id == run_id)
                .first()
            )
            if run:
                run.end_time = datetime.datetime.utcnow()
                run.status = end_status
            else:
                logger.warning(f"Could not find run ID: {run_id} to mark as ended.")

        try:
            await self.loop.run_in_executor(
                None, self._execute_db_operation, _end_run, self._current_run_id, status
            )
            logger.info(
                f"Ended simulation run ID: {self._current_run_id} with status: {status}"
            )
        except Exception as e:
            logger.error(f"Async end_simulation_run failed: {e}")

    # --- AgentMemory Methods ---
    async def save_agent_memory(
        self, agent_id: str, memory_data: Dict[str, Any], step: int
    ):
        if not self.engine:
            return

        def _save_memory(
            db: Session,
            ag_id: str,
            mem_data: Dict[str, Any],
            s: int,
            run_id: Optional[uuid.UUID],
        ):
            db_mem = AgentMemoryDB(
                agent_id_str=ag_id,
                simulation_run_id=run_id,
                step_timestamp=s,
                content=mem_data.get("content"),
                importance=mem_data.get("importance"),
                tags=mem_data.get("tags"),
                related_agent_id=mem_data.get("related_agent_id"),
            )
            db.add(db_mem)

        try:
            await self.loop.run_in_executor(
                None,
                self._execute_db_operation,
                _save_memory,
                agent_id,
                memory_data,
                step,
                self._current_run_id,
            )
        except Exception as e:
            logger.error(f"Async save_agent_memory for {agent_id} failed: {e}")

    async def get_recent_agent_memories(
        self, agent_id: str, count: int
    ) -> List[Dict[str, Any]]:
        if not self.engine:
            return []

        def _get_memories(db: Session, ag_id: str, c: int):
            mems_db = (
                db.query(AgentMemoryDB)
                .filter(AgentMemoryDB.agent_id_str == ag_id)
                .order_by(AgentMemoryDB.step_timestamp.desc())
                .limit(c)
                .all()
            )
            return [
                {
                    "content": m.content,
                    "timestamp": m.step_timestamp,
                    "importance": m.importance,
                    "tags": m.tags,
                    "related_agent_id": m.related_agent_id,
                }
                for m in mems_db
            ]

        try:
            return await self.loop.run_in_executor(
                None, self._execute_db_operation, _get_memories, agent_id, count
            )
        except Exception as e:
            logger.error(f"Async get_recent_agent_memories for {agent_id} failed: {e}")
            return []

    # --- MarketTransaction Methods ---
    async def save_market_transaction(
        self, transaction_data: Dict[str, Any], step: int
    ):
        if not self.engine:
            return

        def _save_txn(
            db: Session, txn_data: Dict[str, Any], s: int, run_id: Optional[uuid.UUID]
        ):
            db_txn = MarketTransactionDB(
                transaction_sim_id=txn_data["transaction_id"],
                simulation_run_id=run_id,
                step_timestamp=s,
                real_timestamp=datetime.datetime.fromtimestamp(
                    txn_data.get("timestamp", time.time())
                ),
                market_id=txn_data["market_id"],
                resource_type=txn_data["resource_type"],
                buyer_agent_id=txn_data["buyer_id"],
                seller_agent_id=txn_data["seller_id"],
                quantity=txn_data["quantity"],
                price_per_unit=txn_data["price_per_unit"],
                total_cost=txn_data["total_cost"],
                buy_order_id=txn_data["buy_order_id"],
                sell_order_id=txn_data["sell_order_id"],
            )
            db.add(db_txn)

        try:
            await self.loop.run_in_executor(
                None,
                self._execute_db_operation,
                _save_txn,
                transaction_data,
                step,
                self._current_run_id,
            )
        except Exception as e:
            logger.error(
                f"Async save_market_transaction {transaction_data.get('transaction_id')} failed: {e}"
            )

    # --- BankingTransaction Methods ---
    async def save_banking_transaction(
        self, transaction_data: Dict[str, Any], step: int, agent_id: str
    ):
        if not self.engine:
            return

        def _save_bank_txn(
            db: Session,
            txn_data: Dict[str, Any],
            s: int,
            ag_id: str,
            run_id: Optional[uuid.UUID],
        ):
            db_txn = BankingTransactionDB(
                transaction_sim_id=txn_data["transaction_id"],
                simulation_run_id=run_id,
                step_timestamp=s,
                real_timestamp=datetime.datetime.fromtimestamp(
                    txn_data.get("timestamp", time.time())
                ),
                account_id=txn_data["account_id"],
                agent_id_str=ag_id,
                transaction_type=txn_data["transaction_type"],
                amount=txn_data["amount"],
                balance_after=txn_data["balance_after"],
                description=txn_data.get("description"),
                counterparty_id=txn_data.get("counterparty_id"),
                reference_id=txn_data.get("reference_id"),
            )
            db.add(db_txn)

        try:
            await self.loop.run_in_executor(
                None,
                self._execute_db_operation,
                _save_bank_txn,
                transaction_data,
                step,
                agent_id,
                self._current_run_id,
            )
        except Exception as e:
            logger.error(
                f"Async save_banking_transaction {transaction_data.get('transaction_id')} failed: {e}"
            )

    async def save_simulation_event(
        self,
        event_type: str,
        step: int,
        agent_id_primary: Optional[str] = None,
        agent_id_secondary: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        if not self.engine or not self.SessionLocal:
            # logger.debug("DB not available, skipping save_simulation_event")
            return

        def _save_event_sync(
            db: Session,
            ev_type: str,
            s: int,
            run_id: Optional[uuid.UUID],
            ag_id1: Optional[str],
            ag_id2: Optional[str],
            dtls: Optional[Dict[str, Any]],
            desc: Optional[str],
        ):

            db_event = SimulationEventDB(
                simulation_run_id=run_id,
                step_timestamp=s,
                event_type=ev_type,
                agent_id_primary=ag_id1,
                agent_id_secondary=ag_id2,
                details=dtls,  # Stored as JSONB
                description=desc,
            )
            db.add(db_event)
            # Commit is handled by _execute_db_operation

        try:
            await self.loop.run_in_executor(
                None,
                self._execute_db_operation,
                _save_event_sync,
                event_type,
                step,
                self._current_run_id,
                agent_id_primary,
                agent_id_secondary,
                details,
                description,
            )
            # logger.debug(f"Saved simulation event: {event_type} for agent {agent_id_primary or 'N/A'} at step {step}")
        except Exception as e:
            logger.error(
                f"Async save_simulation_event failed for type {event_type}: {e}",
                exc_info=True,
            )


# Example usage (for testing this file directly):
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     # Use a temporary in-memory SQLite for testing
#     test_db_url = "sqlite:///./test_simulation_data.db"
#     if os.path.exists("./test_simulation_data.db"): os.remove("./test_simulation_data.db")

#     db_handler = DatabaseHandler(db_url=test_db_url)

#     if db_handler.engine:
#         print(f"DB Handler initialized with engine: {db_handler.engine}")
#         current_run_id = db_handler._current_run_id
#         print(f"Current Run ID: {current_run_id}")

#         # Test save memory
#         mock_memory = {"content": "Test memory content", "importance": 0.8, "tags": ["test", "important"], "related_agent_id": "agent_test_related"}
#         db_handler.save_agent_memory(agent_id="agent_test_1", memory_data=mock_memory, step=10)
#         db_handler.save_agent_memory(agent_id="agent_test_1", memory_data={"content":"Another memory", "importance":0.5}, step=11)

#         # Test retrieve memory
#         retrieved_memories = db_handler.get_recent_agent_memories(agent_id="agent_test_1", count=5)
#         print(f"Retrieved memories for agent_test_1: {retrieved_memories}")

#         # Test save market transaction
#         mock_market_txn = {
#             "transaction_id": "sim_market_txn_001", "market_id": "market_food", "resource_type": "food",
#             "buyer_id": "agent_test_1", "seller_id": "agent_test_2", "quantity": 10.0, "price_per_unit": 2.5,
#             "total_cost": 25.0, "buy_order_id": "buy_abc", "sell_order_id": "sell_xyz", "timestamp": time.time()
#         }
#         db_handler.save_market_transaction(transaction_data=mock_market_txn, step=12)

#         # Test save banking transaction
#         mock_banking_txn = {
#             "transaction_id": "sim_bank_txn_001", "account_id": "acc_agent_test_1_uuid",
#             "transaction_type": BankingTransactionType.DEPOSIT.value, # Use .value for enums if storing as string
#             "amount": 100.0, "balance_after": 1100.0, "description": "Agent deposit"
#         }
#         db_handler.save_banking_transaction(transaction_data=mock_banking_txn, step=13, agent_id="agent_test_1")

#         db_handler.end_simulation_run(status="test_completed")
#         if os.path.exists("./test_simulation_data.db"): os.remove("./test_simulation_data.db") # Clean up test db
#     else:
#         print("DB Handler engine not created. Check DB URL or dependencies.")
