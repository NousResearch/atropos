"""
Banking System for LLM Society Simulation Phase β
Implements banking accounts, loans, credit scoring, and financial services
"""

import asyncio  # Added for async operations
import logging
import math
import random
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Assuming DatabaseHandler is importable if type hinting is desired for db_handler
# from src.database.database_handler import DatabaseHandler

logger = logging.getLogger(__name__)


class AccountType(Enum):
    """Types of bank accounts"""

    CHECKING = "checking"
    SAVINGS = "savings"
    BUSINESS = "business"
    INVESTMENT = "investment"

    def __str__(self):
        return self.value


class LoanType(Enum):
    """Types of loans"""

    PERSONAL = "personal"
    BUSINESS = "business"
    MORTGAGE = "mortgage"
    EDUCATION = "education"
    EMERGENCY = "emergency"

    def __str__(self):
        return self.value


class LoanStatus(Enum):
    """Loan application and repayment status"""

    PENDING = "pending"
    APPROVED = "approved"
    ACTIVE = "active"
    PAID_OFF = "paid_of"
    DEFAULTED = "defaulted"
    REJECTED = "rejected"

    def __str__(self):
        return self.value


class TransactionType(Enum):
    """Types of banking transactions"""

    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    LOAN_PAYMENT = "loan_payment"
    INTEREST_PAYMENT = "interest_payment"
    FEE = "fee"
    SALARY = "salary"
    TRADE_SETTLEMENT = "trade_settlement"

    def __str__(self):
        return self.value


@dataclass
class BankTransaction:
    """Individual banking transaction"""

    transaction_id: str
    account_id: str
    transaction_type: TransactionType
    amount: float
    balance_after: float

    # Transaction details
    description: str
    counterparty_id: Optional[str] = None
    reference_id: Optional[str] = None  # Reference to external transaction

    # Metadata
    timestamp: float = field(default_factory=time.time)
    processed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["transaction_type"] = self.transaction_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BankTransaction":
        data["transaction_type"] = TransactionType(data["transaction_type"])
        # Ensure all fields required by __init__ are present or provide defaults
        return cls(**data)


@dataclass
class BankAccount:
    """Individual bank account"""

    account_id: str
    owner_id: str
    account_type: AccountType

    # Account balances
    balance: float = 0.0
    available_balance: float = 0.0  # Balance minus holds/pending transactions

    # Account settings
    interest_rate: float = 0.01  # Annual interest rate
    minimum_balance: float = 0.0
    overdraft_limit: float = 0.0

    # Account status
    is_active: bool = True
    credit_score: float = 700.0  # Credit score (300-850)
    risk_level: str = "medium"  # "low", "medium", "high"

    # Transaction history
    transactions: List[BankTransaction] = field(default_factory=list)

    # Loan information
    total_debt: float = 0.0
    monthly_debt_payment: float = 0.0

    # Account metadata
    created_date: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["account_type"] = self.account_type.value
        data["transactions"] = [txn.to_dict() for txn in self.transactions]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BankAccount":
        data["account_type"] = AccountType(data["account_type"])
        data["transactions"] = [
            BankTransaction.from_dict(txn_data)
            for txn_data in data.get("transactions", [])
        ]
        # Handle missing fields that have defaults in dataclass if not in data
        # cls constructor should handle this if data only contains subset of fields
        return cls(**data)


@dataclass
class Loan:
    """Individual loan"""

    loan_id: str
    borrower_id: str
    loan_type: LoanType

    # Loan terms
    principal: float
    interest_rate: float
    term_months: int
    monthly_payment: float

    # Loan status
    status: LoanStatus = LoanStatus.PENDING
    remaining_balance: float = 0.0
    payments_made: int = 0
    missed_payments: int = 0

    # Loan purpose and collateral
    purpose: str = ""
    collateral_value: float = 0.0
    collateral_description: str = ""

    # Risk assessment
    approval_probability: float = 0.0
    risk_rating: str = "medium"

    # Payment history
    payment_history: List[Dict[str, Any]] = field(default_factory=list)

    # Loan metadata
    application_date: float = field(default_factory=time.time)
    approval_date: Optional[float] = None
    first_payment_date: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["loan_type"] = self.loan_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Loan":
        data["loan_type"] = LoanType(data["loan_type"])
        data["status"] = LoanStatus(data["status"])
        # payment_history is already a list of dicts
        return cls(**data)


class CreditScoreCalculator:
    """Calculates and updates credit scores for agents"""

    def __init__(self):
        self.score_factors = {
            "payment_history": 0.35,  # 35% - Most important factor
            "debt_utilization": 0.30,  # 30% - Amount owed vs credit available
            "credit_history_length": 0.15,  # 15% - How long accounts have been open
            "credit_mix": 0.10,  # 10% - Variety of credit types
            "new_credit": 0.10,  # 10% - Recent credit inquiries and accounts
        }

        # Base scores for different risk levels
        self.base_scores = {
            "excellent": (750, 850),
            "good": (700, 749),
            "fair": (650, 699),
            "poor": (600, 649),
            "bad": (300, 599),
        }

    def calculate_credit_score(self, account: BankAccount, loans: List[Loan]) -> float:
        """Calculate credit score based on financial history"""

        # Start with base score
        base_score = account.credit_score

        # Payment history (35%)
        payment_score = self._calculate_payment_history_score(account, loans)

        # Debt utilization (30%)
        utilization_score = self._calculate_debt_utilization_score(account, loans)

        # Credit history length (15%)
        history_score = self._calculate_credit_history_score(account)

        # Credit mix (10%)
        mix_score = self._calculate_credit_mix_score(loans)

        # New credit (10%)
        new_credit_score = self._calculate_new_credit_score(account, loans)

        # Weighted score calculation
        new_score = (
            payment_score * self.score_factors["payment_history"]
            + utilization_score * self.score_factors["debt_utilization"]
            + history_score * self.score_factors["credit_history_length"]
            + mix_score * self.score_factors["credit_mix"]
            + new_credit_score * self.score_factors["new_credit"]
        )

        # Blend with previous score for stability
        final_score = base_score * 0.7 + new_score * 0.3

        # Keep within valid range
        return max(300, min(850, final_score))

    def _calculate_payment_history_score(
        self, account: BankAccount, loans: List[Loan]
    ) -> float:
        """Calculate payment history component of credit score"""

        if not loans:
            return 700.0  # Neutral score for no credit history

        total_payments = 0
        missed_payments = 0

        for loan in loans:
            total_payments += loan.payments_made
            missed_payments += loan.missed_payments

        if total_payments == 0:
            return 700.0

        # Score based on payment reliability
        payment_ratio = 1.0 - (missed_payments / total_payments)

        if payment_ratio >= 0.98:
            return 800.0  # Excellent payment history
        elif payment_ratio >= 0.95:
            return 750.0  # Good payment history
        elif payment_ratio >= 0.90:
            return 700.0  # Fair payment history
        elif payment_ratio >= 0.80:
            return 650.0  # Poor payment history
        else:
            return 500.0  # Bad payment history

    def _calculate_debt_utilization_score(
        self, account: BankAccount, loans: List[Loan]
    ) -> float:
        """Calculate debt utilization component"""

        # Simple utilization based on debt vs available credit
        estimated_credit_limit = account.balance * 2 + 1000  # Rough estimate

        if estimated_credit_limit <= 0:
            return 700.0

        utilization_ratio = account.total_debt / estimated_credit_limit

        if utilization_ratio <= 0.10:
            return 850.0  # Excellent utilization
        elif utilization_ratio <= 0.30:
            return 750.0  # Good utilization
        elif utilization_ratio <= 0.50:
            return 650.0  # Fair utilization
        elif utilization_ratio <= 0.75:
            return 550.0  # Poor utilization
        else:
            return 400.0  # Very high utilization

    def _calculate_credit_history_score(self, account: BankAccount) -> float:
        """Calculate credit history length component"""

        account_age_years = (time.time() - account.created_date) / (365 * 24 * 3600)

        if account_age_years >= 10:
            return 800.0  # Long credit history
        elif account_age_years >= 5:
            return 750.0  # Good credit history
        elif account_age_years >= 2:
            return 700.0  # Fair credit history
        elif account_age_years >= 1:
            return 650.0  # Short credit history
        else:
            return 600.0  # Very short credit history

    def _calculate_credit_mix_score(self, loans: List[Loan]) -> float:
        """Calculate credit mix component"""

        loan_types = set(loan.loan_type for loan in loans)

        if len(loan_types) >= 3:
            return 750.0  # Good credit mix
        elif len(loan_types) == 2:
            return 720.0  # Fair credit mix
        elif len(loan_types) == 1:
            return 700.0  # Limited credit mix
        else:
            return 700.0  # No credit history

    def _calculate_new_credit_score(
        self, account: BankAccount, loans: List[Loan]
    ) -> float:
        """Calculate new credit component"""

        # Count recent loan applications (last 6 months)
        recent_cutoff = time.time() - (6 * 30 * 24 * 3600)
        recent_loans = [loan for loan in loans if loan.application_date > recent_cutoff]

        if len(recent_loans) == 0:
            return 750.0  # No recent credit applications
        elif len(recent_loans) <= 2:
            return 700.0  # Few recent applications
        elif len(recent_loans) <= 4:
            return 650.0  # Several recent applications
        else:
            return 600.0  # Many recent applications (credit seeking behavior)


class BankingSystem:
    """
    Manages all banking operations, accounts, loans, and financial services
    """

    def __init__(
        self,
        database_handler: Optional[Any] = None,
        current_step_getter: Optional[callable] = None,
    ):
        self.accounts: Dict[str, BankAccount] = {}
        self.loans: Dict[str, Loan] = {}
        self.agent_accounts: Dict[str, List[str]] = defaultdict(
            list
        )  # agent_id -> account_ids

        # Credit scoring system
        self.credit_calculator = CreditScoreCalculator()

        # Banking parameters
        self.base_interest_rate = 0.05  # 5% annual interest
        self.reserve_ratio = 0.10  # 10% reserve requirement
        self.loan_loss_provision = 0.02  # 2% provision for loan losses

        # Risk assessment parameters
        self.max_debt_to_income_ratio = 0.36  # 36% maximum debt-to-income
        self.minimum_credit_score = 550  # Minimum score for loan approval

        # Fee structure
        self.fees = {
            "account_maintenance": 5.0,  # Monthly fee
            "overdraft": 35.0,  # Overdraft fee
            "loan_origination": 0.01,  # 1% of loan amount
            "early_payment": 0.02,  # 2% prepayment penalty
        }

        self.db_handler = database_handler
        self.get_current_step = (
            current_step_getter  # Function to get current simulation step
        )

        logger.info("Banking System initialized")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accounts": {
                acc_id: acc.to_dict() for acc_id, acc in self.accounts.items()
            },
            "loans": {loan_id: loan.to_dict() for loan_id, loan in self.loans.items()},
            "agent_accounts": {
                agent_id: list(acc_ids)
                for agent_id, acc_ids in self.agent_accounts.items()
            },
            # Config-like attributes (fees, rates, etc.) are not serialized as they are part of initial setup
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        database_handler: Optional[Any] = None,
        current_step_getter: Optional[callable] = None,
    ) -> "BankingSystem":
        system = cls(
            database_handler=database_handler, current_step_getter=current_step_getter
        )
        # Basic attributes like rates, fees are set by __init__, not from dict for simplicity
        # If these were dynamically configurable during a run & need saving, they'd be in data.

        system.accounts = {
            acc_id: BankAccount.from_dict(acc_data)
            for acc_id, acc_data in data.get("accounts", {}).items()
        }
        system.loans = {
            loan_id: Loan.from_dict(loan_data)
            for loan_id, loan_data in data.get("loans", {}).items()
        }

        # Reconstruct agent_accounts defaultdict
        agent_accounts_data = data.get("agent_accounts", {})
        system.agent_accounts = defaultdict(list)
        for agent_id, acc_ids_list in agent_accounts_data.items():
            system.agent_accounts[agent_id] = list(
                acc_ids_list
            )  # Ensure it's a list of strings

        logger.info(
            f"BankingSystem state restored with {len(system.accounts)} accounts, {len(system.loans)} loans."
        )
        return system

    async def create_account(
        self,
        agent_id: str,
        account_type: AccountType = AccountType.CHECKING,
        initial_deposit: float = 0.0,
        current_step: Optional[int] = None,
    ) -> Optional[BankAccount]:
        """Create a new bank account for an agent"""

        try:
            account_id = f"acc_{agent_id}_{uuid.uuid4().hex[:8]}"

            # Determine initial credit score
            initial_credit_score = max(300, min(850, random.gauss(700, 100)))

            # Set account parameters based on type
            if account_type == AccountType.SAVINGS:
                interest_rate = 0.02  # Higher interest for savings
                minimum_balance = 100.0
            elif account_type == AccountType.BUSINESS:
                interest_rate = 0.005  # Lower interest for business
                minimum_balance = 500.0
            elif account_type == AccountType.INVESTMENT:
                interest_rate = 0.01
                minimum_balance = 1000.0
            else:  # CHECKING
                interest_rate = 0.005
                minimum_balance = 25.0

            account = BankAccount(
                account_id=account_id,
                owner_id=agent_id,
                account_type=account_type,
                balance=initial_deposit,
                available_balance=initial_deposit,
                interest_rate=interest_rate,
                minimum_balance=minimum_balance,
                credit_score=initial_credit_score,
            )

            # Add initial deposit transaction
            if initial_deposit > 0:
                deposit_txn = BankTransaction(
                    transaction_id=f"txn_{account_id}_{uuid.uuid4().hex[:8]}",
                    account_id=account_id,
                    transaction_type=TransactionType.DEPOSIT,
                    amount=initial_deposit,
                    balance_after=initial_deposit,
                    description="Initial deposit",
                )
                account.transactions.append(deposit_txn)

            self.accounts[account_id] = account
            self.agent_accounts[agent_id].append(account_id)

            logger.info(
                f"Created {account_type.value} account {account_id} for agent {agent_id} with ${initial_deposit:.2f}"
            )
            return account
        except Exception as e:
            logger.error(
                f"Error creating account for agent {agent_id}: {e}", exc_info=True
            )
            return None

    async def process_transaction(
        self,
        account_id: str,
        transaction_type: TransactionType,
        amount: float,
        description: str = "",
        counterparty_id: Optional[str] = None,
        current_step: Optional[int] = None,
    ) -> bool:
        """Process a banking transaction"""

        try:
            if account_id not in self.accounts:
                logger.error(f"Account {account_id} not found")
                return False

            account = self.accounts[account_id]

            # Check if account is active
            if not account.is_active:
                logger.warning(f"Account {account_id} is inactive")
                return False

            # Calculate new balance
            if transaction_type in [
                TransactionType.DEPOSIT,
                TransactionType.SALARY,
                TransactionType.INTEREST_PAYMENT,
            ]:
                new_balance = account.balance + amount
            elif transaction_type in [
                TransactionType.WITHDRAWAL,
                TransactionType.FEE,
                TransactionType.LOAN_PAYMENT,
            ]:
                new_balance = account.balance - amount
            else:
                # For transfers, amount can be positive or negative
                new_balance = account.balance + amount

            # Check for overdraft
            if new_balance < 0 and account.overdraft_limit == 0:
                logger.warning(f"Transaction would overdraw account {account_id}")
                return False

            if new_balance < -account.overdraft_limit:
                logger.warning(
                    f"Transaction exceeds overdraft limit for account {account_id}"
                )
                return False

            # Apply overdraft fee if necessary
            overdraft_fee = 0.0
            if new_balance < 0 and account.balance >= 0:
                overdraft_fee = self.fees["overdraft"]
                new_balance -= overdraft_fee

            # Create transaction record
            transaction = BankTransaction(
                transaction_id=f"txn_{account_id}_{uuid.uuid4().hex[:8]}",
                account_id=account_id,
                transaction_type=transaction_type,
                amount=amount,
                balance_after=new_balance,
                description=description,
                counterparty_id=counterparty_id,
            )

            # Update account
            account.balance = new_balance
            account.available_balance = (
                new_balance  # Simplified - in reality would account for holds
            )
            account.last_activity = time.time()
            account.transactions.append(transaction)

            # Add overdraft fee transaction if applied
            if overdraft_fee > 0:
                fee_txn = BankTransaction(
                    transaction_id=f"txn_{account_id}_{uuid.uuid4().hex[:8]}_fee",
                    account_id=account_id,
                    transaction_type=TransactionType.FEE,
                    amount=overdraft_fee,
                    balance_after=account.balance,
                    description="Overdraft fee",
                )
                account.transactions.append(fee_txn)

            # Save primary transaction to DB
            step_to_log = (
                current_step
                if current_step is not None
                else (self.get_current_step() if self.get_current_step else -1)
            )
            if self.db_handler:
                try:
                    await self.db_handler.save_banking_transaction(
                        transaction.to_dict(), step_to_log, account.owner_id
                    )
                except Exception as e_db:
                    logger.error(
                        f"DB Error saving transaction {transaction.transaction_id} for {account_id}: {e_db}",
                        exc_info=True,
                    )

            if fee_txn:
                # Save fee transaction to DB
                if self.db_handler:
                    try:
                        await self.db_handler.save_banking_transaction(
                            fee_txn.to_dict(), step_to_log, account.owner_id
                        )
                    except Exception as e_db_fee:
                        logger.error(
                            f"DB Error saving fee transaction for {account_id}: {e_db_fee}",
                            exc_info=True,
                        )

            logger.debug(
                f"Processed {transaction_type.value}: ${amount:.2f} for account {account_id}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error processing transaction for account {account_id} ({transaction_type.value}, {amount}): {e}",
                exc_info=True,
            )
            return False

    async def transfer_funds(
        self,
        from_account_id: str,
        to_account_id: str,
        amount: float,
        description: str = "Transfer",
        current_step: Optional[int] = None,
    ) -> bool:
        """Transfer funds between accounts"""

        try:
            if (
                from_account_id not in self.accounts
                or to_account_id not in self.accounts
            ):
                logger.error("One or both accounts not found for transfer")
                return False

            from_account = self.accounts[from_account_id]
            to_account = self.accounts[to_account_id]

            # Process withdrawal from source account
            if not await self.process_transaction(
                from_account_id,
                TransactionType.TRANSFER,
                -amount,
                f"Transfer to {to_account_id}: {description}",
                to_account.owner_id,
                current_step=current_step,
            ):
                return False

            # Process deposit to destination account
            if not await self.process_transaction(
                to_account_id,
                TransactionType.TRANSFER,
                amount,
                f"Transfer from {from_account_id}: {description}",
                from_account.owner_id,
                current_step=current_step,
            ):
                # Reverse the withdrawal if deposit fails
                await self.process_transaction(
                    from_account_id,
                    TransactionType.TRANSFER,
                    amount,
                    f"Reversal - Transfer to {to_account_id} failed",
                    current_step=current_step,
                )
                return False

            logger.info(
                f"Transferred ${amount:.2f} from {from_account_id} to {to_account_id}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error transferring funds from {from_account_id} to {to_account_id}: {e}",
                exc_info=True,
            )
            return False

    async def apply_for_loan(
        self,
        agent_id: str,
        loan_type: LoanType,
        amount: float,
        purpose: str,
        term_months: int = 60,
        current_step: Optional[int] = None,
    ) -> Optional[Loan]:
        """Submit loan application"""

        try:
            if agent_id not in self.agent_accounts or not self.agent_accounts[agent_id]:
                logger.warning(f"Agent {agent_id} has no bank accounts")
                return None

            primary_account_id = self.agent_accounts[agent_id][0]
            account = self.accounts[primary_account_id]

            # Create loan application
            loan_id = f"loan_{agent_id}_{uuid.uuid4().hex[:8]}"

            # Calculate interest rate based on loan type and credit score
            base_rate = self.base_interest_rate

            rate_adjustments = {
                LoanType.PERSONAL: 0.03,  # Higher rate for unsecured personal loans
                LoanType.BUSINESS: 0.01,  # Moderate rate for business loans
                LoanType.MORTGAGE: -0.01,  # Lower rate for secured mortgages
                LoanType.EDUCATION: 0.0,  # Standard rate for education
                LoanType.EMERGENCY: 0.05,  # Higher rate for emergency loans
            }

            loan_interest_rate = base_rate + rate_adjustments.get(loan_type, 0.0)

            # Adjust rate based on credit score
            credit_adjustment = (
                750 - account.credit_score
            ) / 1000.0  # Lower score = higher rate
            loan_interest_rate += credit_adjustment

            # Calculate monthly payment
            monthly_rate = loan_interest_rate / 12.0
            monthly_payment = (
                amount
                * (monthly_rate * (1 + monthly_rate) ** term_months)
                / ((1 + monthly_rate) ** term_months - 1)
            )

            loan = Loan(
                loan_id=loan_id,
                borrower_id=agent_id,
                loan_type=loan_type,
                principal=amount,
                interest_rate=loan_interest_rate,
                term_months=term_months,
                monthly_payment=monthly_payment,
                remaining_balance=amount,
                purpose=purpose,
            )

            # Assess loan application
            approval_result = self._assess_loan_application(account, loan)
            loan.approval_probability = approval_result["probability"]
            loan.risk_rating = approval_result["risk_rating"]

            # Auto-approve or reject based on criteria
            if approval_result["probability"] > 0.7:
                loan.status = LoanStatus.APPROVED
                loan.approval_date = time.time()
                logger.info(f"Loan {loan_id} approved for ${amount:.2f}")
            elif approval_result["probability"] < 0.3:
                loan.status = LoanStatus.REJECTED
                logger.info(f"Loan {loan_id} rejected for ${amount:.2f}")
            else:
                loan.status = LoanStatus.PENDING
                logger.info(f"Loan {loan_id} pending review for ${amount:.2f}")

            self.loans[loan_id] = loan
            return loan
        except Exception as e:
            logger.error(
                f"Error applying for loan for agent {agent_id} ({loan_type.value}, {amount}): {e}",
                exc_info=True,
            )
            return None

    async def approve_loan(
        self, loan_id: str, current_step: Optional[int] = None
    ) -> bool:
        """Approve a pending loan and disburse funds"""

        try:
            if loan_id not in self.loans:
                logger.error(f"Loan {loan_id} not found")
                return False

            loan = self.loans[loan_id]

            if loan.status != LoanStatus.APPROVED:
                logger.warning(f"Loan {loan_id} is not in approved status")
                return False

            # Get borrower's account
            if loan.borrower_id not in self.agent_accounts:
                logger.error(f"Borrower {loan.borrower_id} has no bank accounts")
                return False

            primary_account_id = self.agent_accounts[loan.borrower_id][0]
            account = self.accounts[primary_account_id]

            # Calculate loan origination fee
            origination_fee = loan.principal * self.fees["loan_origination"]
            net_disbursement = loan.principal - origination_fee

            # Disburse loan funds
            step = (
                current_step
                if current_step is not None
                else (self.get_current_step() if self.get_current_step else -1)
            )
            if await self.process_transaction(
                primary_account_id,
                TransactionType.DEPOSIT,
                net_disbursement,
                f"Loan disbursement - {loan.loan_id}",
                current_step=step,
            ):

                # Update loan status
                loan.status = LoanStatus.ACTIVE
                loan.first_payment_date = time.time() + (
                    30 * 24 * 3600
                )  # First payment in 30 days

                # Update account debt information
                account.total_debt += loan.principal
                account.monthly_debt_payment += loan.monthly_payment

                # Process origination fee
                if origination_fee > 0:
                    await self.process_transaction(
                        primary_account_id,
                        TransactionType.FEE,
                        origination_fee,
                        f"Loan origination fee - {loan.loan_id}",
                        current_step=step,
                    )

                logger.info(
                    f"Disbursed loan {loan_id}: ${net_disbursement:.2f} (after ${origination_fee:.2f} fee)"
                )
                return True

            return False
        except Exception as e:
            logger.error(
                f"Error approving/disbursing loan {loan_id}: {e}", exc_info=True
            )
            return False

    async def process_loan_payment(
        self, loan_id: str, payment_amount: float, current_step: Optional[int] = None
    ) -> bool:
        """Process a loan payment"""

        try:
            if loan_id not in self.loans:
                logger.error(f"Loan {loan_id} not found")
                return False

            loan = self.loans[loan_id]

            if loan.status != LoanStatus.ACTIVE:
                logger.warning(f"Loan {loan_id} is not active")
                return False

            # Get borrower's account
            if loan.borrower_id not in self.agent_accounts:
                logger.error(f"Borrower {loan.borrower_id} has no bank accounts")
                return False

            primary_account_id = self.agent_accounts[loan.borrower_id][0]
            account = self.accounts[primary_account_id]

            # Check if account has sufficient funds
            if account.available_balance < payment_amount:
                # Record missed payment
                loan.missed_payments += 1
                payment_record = {
                    "date": time.time(),
                    "amount_due": loan.monthly_payment,
                    "amount_paid": 0.0,
                    "status": "missed",
                }
                loan.payment_history.append(payment_record)

                logger.warning(f"Insufficient funds for loan payment {loan_id}")
                return False

            # Process payment
            step = (
                current_step
                if current_step is not None
                else (self.get_current_step() if self.get_current_step else -1)
            )
            if await self.process_transaction(
                primary_account_id,
                TransactionType.LOAN_PAYMENT,
                payment_amount,
                f"Loan payment - {loan_id}",
                current_step=step,
            ):

                # Calculate interest and principal portions
                interest_portion = loan.remaining_balance * (loan.interest_rate / 12.0)
                principal_portion = min(
                    payment_amount - interest_portion, loan.remaining_balance
                )

                # Update loan balance
                loan.remaining_balance -= principal_portion
                loan.payments_made += 1

                # Record payment
                payment_record = {
                    "date": time.time(),
                    "amount_due": loan.monthly_payment,
                    "amount_paid": payment_amount,
                    "principal_portion": principal_portion,
                    "interest_portion": interest_portion,
                    "remaining_balance": loan.remaining_balance,
                    "status": "paid",
                }
                loan.payment_history.append(payment_record)

                # Check if loan is paid off
                if (
                    loan.remaining_balance <= 0.01
                ):  # Account for floating point precision
                    loan.status = LoanStatus.PAID_OFF
                    account.total_debt -= loan.principal
                    account.monthly_debt_payment -= loan.monthly_payment
                    logger.info(f"Loan {loan_id} paid off!")

                logger.debug(f"Processed loan payment {loan_id}: ${payment_amount:.2f}")
                return True

            return False
        except Exception as e:
            logger.error(
                f"Error processing loan payment for {loan_id}: {e}", exc_info=True
            )
            return False

    def _assess_loan_application(
        self, account: BankAccount, loan: Loan
    ) -> Dict[str, Any]:
        """Assess loan application and calculate approval probability"""

        # Get agent's existing loans
        agent_loans = [
            l
            for l in self.loans.values()
            if l.borrower_id == account.owner_id and l.loan_id != loan.loan_id
        ]

        # Update credit score
        account.credit_score = self.credit_calculator.calculate_credit_score(
            account, agent_loans
        )

        # Assessment factors
        credit_factor = self._calculate_credit_factor(account.credit_score)
        income_factor = self._estimate_income_factor(account)
        debt_factor = self._calculate_debt_factor(account, loan)
        collateral_factor = self._calculate_collateral_factor(loan)

        # Risk assessment
        base_probability = 0.5
        probability = (
            base_probability
            * credit_factor
            * income_factor
            * debt_factor
            * collateral_factor
        )

        # Determine risk rating
        if probability >= 0.8:
            risk_rating = "low"
        elif probability >= 0.6:
            risk_rating = "medium"
        elif probability >= 0.4:
            risk_rating = "high"
        else:
            risk_rating = "very_high"

        return {
            "probability": min(0.95, max(0.05, probability)),
            "risk_rating": risk_rating,
            "credit_factor": credit_factor,
            "income_factor": income_factor,
            "debt_factor": debt_factor,
            "collateral_factor": collateral_factor,
        }

    def _calculate_credit_factor(self, credit_score: float) -> float:
        """Calculate credit score factor for loan approval"""

        if credit_score >= 750:
            return 1.2  # Excellent credit
        elif credit_score >= 700:
            return 1.0  # Good credit
        elif credit_score >= 650:
            return 0.8  # Fair credit
        elif credit_score >= 600:
            return 0.6  # Poor credit
        else:
            return 0.3  # Bad credit

    def _estimate_income_factor(self, account: BankAccount) -> float:
        """Estimate income factor based on account activity"""

        # Look at recent deposits to estimate income
        recent_deposits = [
            txn
            for txn in account.transactions[-30:]
            if txn.transaction_type in [TransactionType.DEPOSIT, TransactionType.SALARY]
        ]

        if not recent_deposits:
            return 0.5  # No recent income data

        total_deposits = sum(txn.amount for txn in recent_deposits)
        estimated_monthly_income = total_deposits / max(1, len(recent_deposits))

        # Income factor based on estimated income
        if estimated_monthly_income >= 5000:
            return 1.2  # High income
        elif estimated_monthly_income >= 3000:
            return 1.0  # Good income
        elif estimated_monthly_income >= 2000:
            return 0.8  # Moderate income
        elif estimated_monthly_income >= 1000:
            return 0.6  # Low income
        else:
            return 0.4  # Very low income

    def _calculate_debt_factor(self, account: BankAccount, loan: Loan) -> float:
        """Calculate debt-to-income factor"""

        estimated_monthly_income = max(1000, account.balance / 2)  # Rough estimate

        # Calculate debt-to-income ratio with new loan
        total_monthly_debt = account.monthly_debt_payment + loan.monthly_payment
        debt_to_income_ratio = total_monthly_debt / estimated_monthly_income

        if debt_to_income_ratio <= 0.20:
            return 1.2  # Low debt ratio
        elif debt_to_income_ratio <= 0.30:
            return 1.0  # Acceptable debt ratio
        elif debt_to_income_ratio <= 0.40:
            return 0.7  # High debt ratio
        else:
            return 0.3  # Very high debt ratio

    def _calculate_collateral_factor(self, loan: Loan) -> float:
        """Calculate collateral factor for secured loans"""

        if loan.loan_type == LoanType.MORTGAGE:
            return 1.1  # Secured by real estate
        elif loan.collateral_value > 0:
            collateral_ratio = loan.collateral_value / loan.principal
            return min(1.3, 0.8 + collateral_ratio * 0.5)
        else:
            return 1.0  # Unsecured loan

    async def process_monthly_interest(self, current_step: Optional[int] = None):
        """Process monthly interest payments for all accounts"""

        try:
            logger.info("Processing monthly interest for all eligible accounts.")
            step = (
                current_step
                if current_step is not None
                else (self.get_current_step() if self.get_current_step else -1)
            )
            for account_id, account in list(
                self.accounts.items()
            ):  # Use list copy for safe iteration if modifying dict
                if (
                    account.balance > 0
                    and account.interest_rate > 0
                    and account.is_active
                ):
                    monthly_interest = account.balance * (account.interest_rate / 12.0)
                    if (
                        monthly_interest > 0.001
                    ):  # Process only if interest is meaningful
                        await self.process_transaction(
                            account_id,
                            TransactionType.INTEREST_PAYMENT,
                            monthly_interest,
                            "Monthly interest earned",
                            current_step=step,
                        )
        except Exception as e:
            logger.error(f"Error processing monthly interest: {e}", exc_info=True)

    def get_agent_financial_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive financial summary for an agent"""

        try:
            if agent_id not in self.agent_accounts or not self.agent_accounts[agent_id]:
                return {"error": "Agent has no bank accounts"}

            # Get all accounts
            agent_account_ids = self.agent_accounts[agent_id]
            accounts = [
                self.accounts[acc_id]
                for acc_id in agent_account_ids
                if acc_id in self.accounts
            ]

            # Get all loans
            agent_loans = [
                loan for loan in self.loans.values() if loan.borrower_id == agent_id
            ]

            # Calculate totals
            total_balance = sum(acc.balance for acc in accounts)
            total_debt = sum(
                loan.remaining_balance
                for loan in agent_loans
                if loan.status == LoanStatus.ACTIVE
            )
            net_worth = total_balance - total_debt

            # Get credit score
            primary_account = accounts[0] if accounts else None
            credit_score = primary_account.credit_score if primary_account else 0

            # Recent transaction summary
            all_transactions = []
            for account in accounts:
                all_transactions.extend(
                    account.transactions[-10:]
                )  # Last 10 transactions per account

            all_transactions.sort(key=lambda x: x.timestamp, reverse=True)

            return {
                "agent_id": agent_id,
                "total_balance": total_balance,
                "total_debt": total_debt,
                "net_worth": net_worth,
                "credit_score": credit_score,
                "num_accounts": len(accounts),
                "num_loans": len(agent_loans),
                "recent_transactions": len(all_transactions),
                "account_details": [
                    {
                        "account_id": acc.account_id,
                        "type": acc.account_type.value,
                        "balance": acc.balance,
                        "interest_rate": acc.interest_rate,
                    }
                    for acc in accounts
                ],
                "loan_details": [
                    {
                        "loan_id": loan.loan_id,
                        "type": loan.loan_type.value,
                        "remaining_balance": loan.remaining_balance,
                        "monthly_payment": loan.monthly_payment,
                        "status": loan.status.value,
                    }
                    for loan in agent_loans
                ],
            }
        except Exception as e:
            logger.error(
                f"Error getting financial summary for agent {agent_id}: {e}",
                exc_info=True,
            )
            return {
                "error": f"Could not retrieve financial summary for agent {agent_id}"
            }

    def get_banking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive banking system statistics"""

        try:
            total_deposits = sum(
                acc.balance for acc in self.accounts.values() if acc.balance > 0
            )
            total_loans = sum(
                loan.remaining_balance
                for loan in self.loans.values()
                if loan.status == LoanStatus.ACTIVE
            )

            # Account type distribution
            account_types = defaultdict(int)
            for account in self.accounts.values():
                acc_type = account.account_type.value
                account_types[acc_type] = account_types.get(acc_type, 0) + 1

            # Loan type distribution
            loan_types = defaultdict(int)
            loan_statuses = defaultdict(int)
            for loan in self.loans.values():
                loan_type = loan.loan_type.value
                loan_status = loan.status.value
                loan_types[loan_type] += 1
                loan_statuses[loan_status] += 1

            # Calculate average credit score
            credit_scores = [
                acc.credit_score
                for acc in self.accounts.values()
                if acc.credit_score > 0
            ]
            avg_credit_score = (
                sum(credit_scores) / len(credit_scores) if credit_scores else 0
            )

            # Default rate
            defaulted_loans = [
                loan
                for loan in self.loans.values()
                if loan.status == LoanStatus.DEFAULTED
            ]
            default_rate = len(defaulted_loans) / max(1, len(self.loans))

            return {
                "total_accounts": len(self.accounts),
                "total_deposits": total_deposits,
                "total_loans_outstanding": total_loans,
                "total_loans_issued": len(self.loans),
                "loan_to_deposit_ratio": total_loans / max(1, total_deposits),
                "average_credit_score": avg_credit_score,
                "default_rate": default_rate,
                "account_types": dict(account_types),
                "loan_types": dict(loan_types),
                "loan_statuses": dict(loan_statuses),
                "reserve_ratio": self.reserve_ratio,
                "base_interest_rate": self.base_interest_rate,
            }
        except Exception as e:
            logger.error(f"Error getting banking statistics: {e}", exc_info=True)
            return {"error": "Could not retrieve banking statistics"}

    def get_concise_account_summary_for_llm(self, agent_id: str) -> Optional[str]:
        """
        Provides a concise string summary of an agent's banking status for LLM prompts.
        Example: "Bank: Checking Balance: 123.45. Active Loans: 1, Total Debt: 500.00"
        """
        try:
            financial_summary = self.get_agent_financial_summary(agent_id)

            if not financial_summary or "error" in financial_summary:
                return "Bank: No account information available."

            parts = []

            # Account type and balance
            if financial_summary.get("account_details"):
                primary_account = financial_summary["account_details"][
                    0
                ]  # Use first account as primary for summary
                acc_type = primary_account.get("type", "N/A").capitalize()
                balance = primary_account.get("balance", 0.0)
                parts.append(f"{acc_type} Balance: {balance:.2f}")
            else:
                parts.append(
                    f"Balance: {financial_summary.get('total_balance', 0.0):.2f}"
                )  # Fallback to total balance

            # Loan information
            active_loans = []
            if financial_summary.get("loan_details"):
                for loan_detail in financial_summary["loan_details"]:
                    if (
                        loan_detail.get("status") == LoanStatus.ACTIVE.value
                    ):  # Ensure comparison with enum value if needed
                        active_loans.append(loan_detail)

            if active_loans:
                num_active_loans = len(active_loans)
                total_active_debt = sum(
                    loan.get("remaining_balance", 0.0) for loan in active_loans
                )
                parts.append(
                    f"Active Loans: {num_active_loans}, Total Debt: {total_active_debt:.2f}"
                )
            else:
                parts.append("No active loans")

            if not parts:
                return "Bank: No specific details available."

            return "Bank: " + ". ".join(parts) + "."
        except Exception as e:
            logger.error(
                f"Error getting concise account summary for {agent_id}: {e}",
                exc_info=True,
            )
            return "Bank: Error retrieving account summary."
