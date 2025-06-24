"""
Pay-to-Play Environment with Mixture of Judges

A reinforcement learning environment where an AI agent must strategically select and pay 
judges before each evaluation, implementing economic constraints and strategic decision-making 
in AI training.

This environment creates genuine economic pressure by requiring real USDC payments on the 
Base blockchain before each evaluation, encouraging efficient learning and high-quality responses.

Author: OpenBlock Labs
License: MIT
"""

import json
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

import wandb
from eth_account import Account
from pydantic import Field
from web3 import Web3

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Import agent card configurations
from agent_cards_config import (
    AgentCardSpecialty,
    get_all_agent_card_configs,
)

# Blockchain configuration
BASE_RPC_URL = "https://mainnet.base.org"
BASE_CHAIN_ID = 8453
USDC_CONTRACT_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_DECIMALS = 6

# Minimal ERC-20 ABI for USDC operations
USDC_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    }
]



@dataclass
class AgentCardMetadata:
    """
    Metadata for each agent card including pricing, specialties, and wallet info.
    
    This combines the agent card configuration from agent_cards_config.py with wallet
    credentials from secrets.json to create a complete agent card instance.
    """
    name: str
    price_usd: Decimal
    specialties: List[AgentCardSpecialty]
    description: str
    system_prompt: str
    model_name: str
    address: str
    private_key: str
    
    # Performance tracking
    total_evaluations: int = 0
    average_score_given: float = 0.0
    consistency_score: float = 1.0  # How consistent the agent card is
    agent_satisfaction: float = 0.5  # Agent's satisfaction with this agent card


@dataclass
class AgentCardSelection:
    """
    Agent's decision about which agent cards to use for evaluation.
    
    Contains the agent's strategic choice of agent cards along with reasoning
    and cost analysis for transparency and debugging.
    """
    selected_agent_cards: List[str]
    reasoning: str
    expected_cost: Decimal
    question_type: str


@dataclass
class BudgetTracker:
    """
    Tracks agent spending and budget decisions.
    
    Provides comprehensive budget management including affordability checks,
    spending tracking per agent card, and cost analysis over time.
    """
    initial_budget: Decimal
    current_balance: Decimal
    total_spent: Decimal
    spending_per_agent_card: Dict[str, Decimal]
    evaluations_count: int
    average_cost_per_eval: Decimal
    
    def can_afford(self, cost: Decimal) -> bool:
        """Check if the agent can afford a given cost."""
        return self.current_balance >= cost
    
    def spend(self, amount: Decimal, agent_card_name: str) -> None:
        """Record a spending transaction and update budget tracking."""
        self.current_balance -= amount
        self.total_spent += amount
        self.spending_per_agent_card[agent_card_name] = (
            self.spending_per_agent_card.get(agent_card_name, Decimal('0')) + amount
        )
        self.evaluations_count += 1
        if self.evaluations_count > 0:
            self.average_cost_per_eval = self.total_spent / self.evaluations_count


class PayToPlayConfig(BaseEnvConfig):
    """Configuration for the Pay-to-Play Environment."""
    
    testing_mode: bool = Field(
        default=False, 
        description="If True, simulates payments without real blockchain transactions"
    )
    initial_budget_usd: float = Field(
        default=1.0,
        description="Initial budget for the agent in USD"
    )


class PayToPlayEnv(BaseEnv):
    """
    Environment that requires crypto payments to multiple agent cards before LLM evaluation.
    
    The agent must select and pay agent cards before each evaluation, making strategic
    decisions about cost, quality, and agent card specialties based on budget constraints
    and past performance.
    
    Key Features:
    - Real USDC payments on Base blockchain (or simulated for testing)
    - Strategic agent card selection based on question analysis
    - Budget management and cost optimization
    - Performance tracking and learning from agent card feedback
    - Comprehensive logging and monitoring via Weights & Biases
    """

    name = "pay_to_play"
    env_config_cls = PayToPlayConfig

    def __init__(
        self,
        config: PayToPlayConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        """
        Initialize the Pay-to-Play environment.
        
        Args:
            config: Environment configuration
            server_configs: API server configurations for LLM inference
            slurm: Whether to use SLURM for distributed training
            testing: Override for testing mode
        """
        super().__init__(config, server_configs, slurm, testing)
        self.config: PayToPlayConfig
        self.percent_correct_buffer = []
        self.eval_metrics = []
        self.payment_logs = []
        self.agent_card_selection_history = []
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(BASE_RPC_URL))
        self.usdc_contract = self.w3.eth.contract(
            address=USDC_CONTRACT_ADDRESS,
            abi=USDC_ABI
        )
        
        # Load wallet configuration and initialize agent cards
        wallet_config = self._load_wallet_config()
        self.agent_cards = self._initialize_agent_cards(wallet_config)
        
        # Agent wallet setup
        self.agent_account = Account.from_key(wallet_config["agent"]["private_key"])
        
        # Initialize budget tracking
        initial_budget = Decimal(str(self.config.initial_budget_usd))
        self.budget_tracker = BudgetTracker(
            initial_budget=initial_budget,
            current_balance=initial_budget,
            total_spent=Decimal('0'),
            spending_per_agent_card={},
            evaluations_count=0,
            average_cost_per_eval=Decimal('0')
        )
        
        # Testing mode override
        self.testing_mode = testing or self.config.testing_mode
        
        logging.info(f"PayToPlay Environment initialized with {len(self.agent_cards)} agent cards")
        logging.info(f"Agent wallet: {self.agent_account.address}")
        logging.info(f"Initial budget: ${initial_budget}")
        logging.info(f"Testing mode: {self.testing_mode}")

    def _load_wallet_config(self) -> Dict:
        """
        Load wallet configuration from JSON file.
        
        Returns:
            Dictionary containing wallet configuration
            
        Raises:
            FileNotFoundError: If secrets.json is not found
            ValueError: If required wallet configuration is missing
        """
        wallet_file = Path(__file__).parent / "secrets.json"
        if not wallet_file.exists():
            raise FileNotFoundError(
                f"Secrets configuration not found: {wallet_file}\n"
                f"Please copy secrets.json.template to secrets.json and configure your wallets."
            )
        
        try:
            with open(wallet_file, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in secrets.json: {e}")
            
        # Validate required fields
        if "agent" not in config or "private_key" not in config["agent"]:
            raise ValueError("Missing agent private_key in wallet configuration")
                
        return config

    def _initialize_agent_cards(self, wallet_config: Dict) -> Dict[str, AgentCardMetadata]:
        """
        Initialize the panel of agent cards with different specialties and prices.
        
        Args:
            wallet_config: Wallet configuration containing agent card credentials
            
        Returns:
            Dictionary mapping agent card IDs to AgentCardMetadata instances
            
        Raises:
            ValueError: If agent card configuration or wallet credentials are missing
        """
        agent_cards = {}
        
        # Get agent card wallet info from config
        agent_cards_wallet_config = wallet_config.get("agent_cards", {})
        
        if not agent_cards_wallet_config:
            raise ValueError(
                "No agent cards configuration found in secrets.json. "
                "Please add agent card wallet addresses using the template."
            )
        
        # Load agent card configurations from separate config file
        all_agent_card_configs = get_all_agent_card_configs()
        
        for agent_card_id, agent_card_config in all_agent_card_configs.items():
            # Get wallet credentials for this agent card
            wallet_info = agent_cards_wallet_config.get(agent_card_id, {})
            if not wallet_info.get("address") or not wallet_info.get("private_key"):
                raise ValueError(
                    f"Missing wallet configuration for agent card '{agent_card_id}' in secrets.json. "
                    f"Please add address and private_key for this agent card."
                )
            
            # Combine agent card config with wallet credentials
            agent_cards[agent_card_id] = AgentCardMetadata(
                name=agent_card_config.name,
                price_usd=agent_card_config.price_usd,
                specialties=agent_card_config.specialties,
                description=agent_card_config.description,
                system_prompt=agent_card_config.system_prompt,
                model_name=agent_card_config.model_name,
                address=wallet_info["address"],
                private_key=wallet_info["private_key"]
            )
        
        return agent_cards

    def _get_agent_card_performance_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for each agent card."""
        stats = {}
        for agent_card_name, agent_card in self.agent_cards.items():
            stats[agent_card_name] = {
                'avg_score': agent_card.average_score_given,
                'consistency': agent_card.consistency_score,
                'satisfaction': agent_card.agent_satisfaction,
                'total_evals': agent_card.total_evaluations
            }
        return stats

    @classmethod
    def config_init(cls) -> Tuple[PayToPlayConfig, List[APIServerConfig]]:
        """Initialize default configuration for the Pay-to-Play environment."""
        env_config = PayToPlayConfig(
            tokenizer_name="microsoft/DialoGPT-small",
            group_size=2,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=8,
            steps_per_eval=50,
            max_token_length=2048,
            wandb_name="pay_to_play_mixture_judges",
            testing_mode=False,
            initial_budget_usd=1.0,
        )
        server_configs = [
            APIServerConfig(
                model_name="microsoft/DialoGPT-small",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=64,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        """Setup the environment with sample questions for training."""
        dataset = self._load_questions_dataset()
        self.questions = [q["text"] for q in dataset["training_questions"]]
        self.iter = 0
        await self._check_wallet_balances()

    async def _check_wallet_balances(self):
        """Check and log wallet balances for agent and all agent cards."""
        try:
            agent_balance = self.usdc_contract.functions.balanceOf(self.agent_account.address).call()
            agent_balance_usd = agent_balance / (10 ** USDC_DECIMALS)
            
            logging.info(f"Agent USDC balance: ${agent_balance_usd:.6f}")
            logging.info(f"Agent budget tracker balance: ${self.budget_tracker.current_balance:.6f}")
            
            for agent_card_name, agent_card in self.agent_cards.items():
                try:
                    agent_card_balance = self.usdc_contract.functions.balanceOf(agent_card.address).call()
                    agent_card_balance_usd = agent_card_balance / (10 ** USDC_DECIMALS)
                    logging.info(f"Agent card {agent_card_name} USDC balance: ${agent_card_balance_usd:.6f}")
                except Exception as e:
                    logging.warning(f"Could not check balance for agent card {agent_card_name}: {e}")
            
            if not self.testing_mode and self.budget_tracker.current_balance <= 0:
                logging.warning("Agent has no budget remaining!")
                
        except Exception as e:
            logging.error(f"Error checking balances: {e}")

    async def _make_payments_to_agent_cards(self, selected_agent_cards: List[str]) -> Tuple[bool, Dict[str, Optional[str]]]:
        """
        Make USDC payments to selected agent cards.
        
        Returns:
            Tuple of (all_payments_successful, transaction_hashes_by_agent_card)
        """
        logging.info(f"💰 Starting payment process to {len(selected_agent_cards)} agent cards: {selected_agent_cards}")
        
        # Log balances before payment
        try:
            agent_balance_before = self.usdc_contract.functions.balanceOf(self.agent_account.address).call()
            agent_balance_usd_before = agent_balance_before / (10 ** USDC_DECIMALS)
            logging.info(f"💳 Agent USDC balance before payments: ${agent_balance_usd_before:.6f}")
        except Exception as e:
            logging.warning(f"Could not check agent balance: {e}")
            agent_balance_usd_before = 0
        
        if self.testing_mode:
            total_cost = sum(self.agent_cards[agent_card_name].price_usd for agent_card_name in selected_agent_cards)
            logging.info(f"🧪 SIMULATED payments totaling ${total_cost} to agent cards: {selected_agent_cards}")
            return True, {agent_card_name: None for agent_card_name in selected_agent_cards}
        
        tx_hashes = {}
        successful_payments = 0
        total_paid = Decimal('0')
        
        for agent_card_name in selected_agent_cards:
            agent_card = self.agent_cards[agent_card_name]
            payment_amount_usdc = int(agent_card.price_usd * (10 ** USDC_DECIMALS))
            
            try:
                logging.info(f"💸 Making REAL payment of ${agent_card.price_usd} to {agent_card_name} ({agent_card.address})")
                
                # Check balance
                balance = self.usdc_contract.functions.balanceOf(self.agent_account.address).call()
                if balance < payment_amount_usdc:
                    balance_usd = balance / (10 ** USDC_DECIMALS)
                    logging.error(f"❌ Insufficient USDC balance for {agent_card_name}: ${balance_usd:.6f} < ${agent_card.price_usd}")
                    tx_hashes[agent_card_name] = None
                    continue
                
                # Build and send transaction
                transfer_function = self.usdc_contract.functions.transfer(
                    agent_card.address,
                    payment_amount_usdc
                )
                
                gas_price = self.w3.eth.gas_price
                nonce = self.w3.eth.get_transaction_count(self.agent_account.address)
                gas_estimate = transfer_function.estimate_gas({'from': self.agent_account.address})
                
                transaction = transfer_function.build_transaction({
                    'from': self.agent_account.address,
                    'gas': gas_estimate,
                    'gasPrice': gas_price,
                    'nonce': nonce,
                })
                
                signed_txn = self.w3.eth.account.sign_transaction(transaction, self.agent_account.key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
                
                logging.info(f"📡 Transaction sent, waiting for confirmation...")
                
                # Wait for confirmation
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                
                if receipt.status == 1:
                    tx_hash_hex = tx_hash.hex()
                    logging.info(f"✅ Payment to {agent_card_name} successful: ${agent_card.price_usd}")
                    logging.info(f"🔗 Transaction: https://basescan.org/tx/{tx_hash_hex}")
                    tx_hashes[agent_card_name] = tx_hash_hex
                    successful_payments += 1
                    total_paid += agent_card.price_usd
                else:
                    logging.error(f"❌ Payment to {agent_card_name} failed - transaction reverted")
                    tx_hashes[agent_card_name] = None
                    
            except Exception as e:
                logging.error(f"❌ Payment to {agent_card_name} failed: {e}")
                tx_hashes[agent_card_name] = None
        
        # Log balances after payment
        try:
            agent_balance_after = self.usdc_contract.functions.balanceOf(self.agent_account.address).call()
            agent_balance_usd_after = agent_balance_after / (10 ** USDC_DECIMALS)
            logging.info(f"💳 Agent USDC balance after payments: ${agent_balance_usd_after:.6f}")
            logging.info(f"💰 Total paid: ${total_paid} | Balance change: ${agent_balance_usd_before - agent_balance_usd_after:.6f}")
        except Exception as e:
            logging.warning(f"Could not check agent balance after payment: {e}")
        
        all_successful = successful_payments == len(selected_agent_cards)
        logging.info(f"📊 Payment summary: {successful_payments}/{len(selected_agent_cards)} successful")
        return all_successful, tx_hashes

    async def collect_trajectories(self, item) -> Tuple[Optional[ScoredDataGroup], List]:
        """Collect trajectories and score them after strategic agent card selection and payment."""
        question = item
        
        # Agent selects agent cards strategically
        try:
            selection = await self._agent_select_agent_cards(question)
        except RuntimeError as e:
            logging.warning(f"⏭️ Skipping episode due to agent card selection failure: {e}")
            return None, []
        
        # Check budget
        if not self.budget_tracker.can_afford(selection.expected_cost):
            logging.error(f"⏭️ Skipping episode: Insufficient budget for evaluation. Need ${selection.expected_cost}, have ${self.budget_tracker.current_balance}")
            return None, []
        
        # Log selection decision
        self.agent_card_selection_history.append({
            "timestamp": time.time(),
            "question": question[:50] + "..." if len(question) > 50 else question,
            "selected_agent_cards": selection.selected_agent_cards,
            "reasoning": selection.reasoning,
            "expected_cost": float(selection.expected_cost),
            "question_type": selection.question_type
        })
        
        logging.info(f"Agent selected agent cards: {selection.selected_agent_cards}")
        logging.info(f"Selection reasoning: {selection.reasoning}")
        logging.info(f"Expected cost: ${selection.expected_cost}")
        
        # Generate responses
        logging.info(f"🤖 Generating {self.config.group_size} responses for question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        async def generate_responses():
            return await self.server.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."},
                    {"role": "user", "content": question}
                ],
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
            )
        
        try:
            chat_completions = await generate_responses()
        except Exception as e:
            logging.error(f"❌ Failed to generate responses: {e}")
            raise RuntimeError(f"Response generation failed: {e}")
        
        responses = []
        for i, completion in enumerate(chat_completions.choices):
            response_text = completion.message.content
            responses.append({
                "question": question,
                "response": response_text,
                "finish_reason": completion.finish_reason
            })
            # Log each agent generation
            logging.info(f"🤖 Agent Generation {i+1}: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
        
        logging.info(f"✅ Generated {len(responses)} responses for evaluation")
        
        # Make payments to selected agent cards
        payment_success, tx_hashes = await self._make_payments_to_agent_cards(selection.selected_agent_cards)
        
        # Log payment attempts
        for agent_card_name in selection.selected_agent_cards:
            agent_card_price = self.agent_cards[agent_card_name].price_usd
            success = tx_hashes.get(agent_card_name) is not None
            self.payment_logs.append({
                "timestamp": time.time(),
                "agent_card_name": agent_card_name,
                "success": success,
                "tx_hash": tx_hashes.get(agent_card_name),
                "amount_usd": float(agent_card_price),
                "question": question[:50] + "..." if len(question) > 50 else question
            })
            
            # Update budget tracker for successful payments
            if success:
                self.budget_tracker.spend(agent_card_price, agent_card_name)
        
        if not payment_success:
            logging.error("Some payments failed - STOPPING TRAINING")
            failed_agent_cards = [j for j, tx in tx_hashes.items() if tx is None]
            raise RuntimeError(f"Payments failed to agent cards: {failed_agent_cards}")
        
        # Evaluate responses with selected agent cards
        scored_data = await self._score_with_selected_agent_cards(responses, selection.selected_agent_cards)
        return scored_data, []

    async def _score_with_selected_agent_cards(self, responses, selected_agent_cards: List[str]) -> Optional[ScoredDataGroup]:
        """Score responses using the strategically selected agent cards."""
        all_scores = []
        agent_card_feedback = {}
        
        for agent_card_name in selected_agent_cards:
            agent_card = self.agent_cards[agent_card_name]
            
            logging.info(f"🧑‍⚖️ Agent card {agent_card_name} evaluating {len(responses)} responses...")
            
            # Evaluate each response with this agent card
            agent_card_scores = []
            for i, response_data in enumerate(responses):
                eval_prompt = f"""
Question: {response_data['question']}

Response to evaluate: {response_data['response']}

Please evaluate the quality, accuracy, and helpfulness of this response based on your expertise in {', '.join([s.value for s in agent_card.specialties])}.
Provide a score between 0.0 and 1.0, where 1.0 is excellent and 0.0 is poor.
End your evaluation with \\boxed{{score}} where score is your numerical rating.
"""
                
                # Get agent card evaluation
                async def get_agent_card_evaluation():
                    return await self.server.chat_completion(
                        messages=[
                            {"role": "system", "content": agent_card.system_prompt},
                            {"role": "user", "content": eval_prompt}
                        ],
                        n=1,
                        max_tokens=self.config.max_token_length,
                        split="eval"
                    )
                
                try:
                    agent_card_completion = await get_agent_card_evaluation()
                except Exception as e:
                    logging.error(f"❌ Agent card {agent_card_name} evaluation failed for response {i+1}: {e}")
                    # Use fallback score if agent card evaluation fails
                    score = 0.5
                    agent_card_response = f"Evaluation failed: {e}"
                else:
                    # Extract score from agent card response
                    agent_card_response = agent_card_completion.choices[0].message.content
                    score = self._extract_score_from_agent_card(agent_card_response)
                
                agent_card_scores.append(score)
                
                # Log detailed agent card feedback
                logging.info(f"  📝 Response {i+1} Score: {score:.3f}")
                logging.info(f"  💬 Agent card Feedback: {agent_card_response[:300]}{'...' if len(agent_card_response) > 300 else ''}")
                
                # Update agent card statistics
                agent_card.total_evaluations += 1
                agent_card.average_score_given = (agent_card.average_score_given * (agent_card.total_evaluations - 1) + score) / agent_card.total_evaluations
            
            logging.info(f"🧑‍⚖️ Agent card {agent_card_name} completed evaluation - Average score: {sum(agent_card_scores)/len(agent_card_scores):.3f}")
            
            all_scores.append(agent_card_scores)
            agent_card_feedback[agent_card_name] = {
                "scores": agent_card_scores,
                "average": sum(agent_card_scores) / len(agent_card_scores),
                "price": float(agent_card.price_usd)
            }
        
        # Aggregate scores from multiple agent cards (average)
        if not all_scores:
            return None
            
        num_responses = len(responses)
        aggregated_scores = []
        
        for i in range(num_responses):
            response_scores = [agent_card_scores[i] for agent_card_scores in all_scores]
            avg_score = sum(response_scores) / len(response_scores)
            aggregated_scores.append(avg_score)
            logging.info(f"📊 Response {i+1} Final Score: {avg_score:.3f} (from {len(response_scores)} agent cards)")
        
        logging.info(f"🎯 Evaluation Summary: Scores range {min(aggregated_scores):.3f} - {max(aggregated_scores):.3f}, Average: {sum(aggregated_scores)/len(aggregated_scores):.3f}")
        
        # Create scored data
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        
        for i, response_data in enumerate(responses):
            # Tokenize for trainer
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."},
                {"role": "user", "content": response_data['question']},
                {"role": "assistant", "content": response_data['response']}
            ]
            
            out_dict = tokenize_for_trainer(
                self.tokenizer, 
                messages, 
                response_data['finish_reason']
            )
            
            scores["tokens"].append(out_dict["tokens"])
            scores["masks"].append(out_dict["masks"])
            scores["scores"].append(aggregated_scores[i])
            
            # Track for metrics
            self.percent_correct_buffer.append(aggregated_scores[i])
        
        # Store agent card feedback for analysis
        if hasattr(self, 'last_agent_card_feedback'):
            self.last_agent_card_feedback = agent_card_feedback
        
        # Ensure we have different scores for training signal
        if len(set(scores["scores"])) == 1:
            return None
            
        return scores

    def _extract_score_from_agent_card(self, agent_card_response: str) -> float:
        """Extract numerical score from agent card response."""
        try:
            # Look for \boxed{score} pattern
            if "\\boxed{" in agent_card_response:
                start = agent_card_response.find("\\boxed{") + 7
                end = agent_card_response.find("}", start)
                if end != -1:
                    score_str = agent_card_response[start:end].strip()
                    score = float(score_str)
                    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
            # Fallback: look for decimal numbers
            import re
            numbers = re.findall(r'\b0\.\d+\b|\b1\.0\b', agent_card_response)
            if numbers:
                score = float(numbers[-1])
                return max(0.0, min(1.0, score))
                
        except (ValueError, IndexError):
            pass
        
        # If agent card can't provide a valid score, treat as failure
        return 0.0

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on test questions."""
        dataset = self._load_questions_dataset()
        eval_questions = [q["text"] for q in dataset["evaluation_questions"]]
        
        total_score = 0
        count = 0
        agent_card_performance = {agent_card_name: [] for agent_card_name in self.agent_cards.keys()}
        
        for question in eval_questions:
            completion = await self.server.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": question}
                ],
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,
                split="eval"
            )
            
            response = completion.choices[0].message.content
            
            # Evaluate with each agent card (no payment required for eval)
            question_scores = []
            for agent_card_name, agent_card in self.agent_cards.items():
                eval_prompt = f"Question: {question}\n\nResponse: {response}\n\nEvaluate this response based on your expertise:"
                
                agent_card_completion = await self.server.chat_completion(
                    messages=[
                        {"role": "system", "content": agent_card.system_prompt},
                        {"role": "user", "content": eval_prompt}
                    ],
                    n=1,
                    max_tokens=self.config.max_token_length,
                    split="eval"
                )
                
                score = self._extract_score_from_agent_card(agent_card_completion.choices[0].message.content)
                question_scores.append(score)
                agent_card_performance[agent_card_name].append(score)
            
            # Average score across all agent cards for this question
            avg_score = sum(question_scores) / len(question_scores)
            total_score += avg_score
            count += 1
        
        if count > 0:
            overall_avg_score = total_score / count
            self.eval_metrics.append(("eval/average_score", overall_avg_score))
            
            # Add per-agent card evaluation metrics
            for agent_card_name, scores in agent_card_performance.items():
                if scores:
                    agent_card_avg = sum(scores) / len(scores)
                    self.eval_metrics.append((f"eval/agent_card_{agent_card_name}_avg_score", agent_card_avg))

    async def get_next_item(self):
        """Get next question for training."""
        question = self.questions[self.iter % len(self.questions)]
        self.iter += 1
        return question

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to Weights & Biases."""
        if wandb_metrics is None:
            wandb_metrics = {}
            
        # Log budget and spending metrics
        wandb_metrics["budget/current_balance"] = float(self.budget_tracker.current_balance)
        wandb_metrics["budget/total_spent"] = float(self.budget_tracker.total_spent)
        wandb_metrics["budget/evaluations_count"] = self.budget_tracker.evaluations_count
        wandb_metrics["budget/average_cost_per_eval"] = float(self.budget_tracker.average_cost_per_eval)
        
        # Budget utilization percentage
        budget_utilization = float(self.budget_tracker.total_spent / self.budget_tracker.initial_budget) * 100
        wandb_metrics["budget/utilization_percent"] = budget_utilization
        
        # Per-agent card spending breakdown
        for agent_card_name, amount in self.budget_tracker.spending_per_agent_card.items():
            wandb_metrics[f"spending/agent_card_{agent_card_name}_total"] = float(amount)
            if self.budget_tracker.total_spent > 0:
                percentage = float(amount / self.budget_tracker.total_spent) * 100
                wandb_metrics[f"spending/agent_card_{agent_card_name}_percent"] = percentage
        
        # Agent card performance metrics
        for agent_card_name, agent_card in self.agent_cards.items():
            wandb_metrics[f"agent_card_performance/{agent_card_name}_avg_score"] = agent_card.average_score_given
            wandb_metrics[f"agent_card_performance/{agent_card_name}_total_evals"] = agent_card.total_evaluations
            wandb_metrics[f"agent_card_performance/{agent_card_name}_satisfaction"] = agent_card.agent_satisfaction
            wandb_metrics[f"agent_card_performance/{agent_card_name}_consistency"] = agent_card.consistency_score
            wandb_metrics[f"agent_card_performance/{agent_card_name}_price_usd"] = float(agent_card.price_usd)
        
        # Payment statistics
        if self.payment_logs:
            successful_payments = sum(1 for log in self.payment_logs if log['success'])
            total_payments = len(self.payment_logs)
            total_cost = sum(log['amount_usd'] for log in self.payment_logs if log['success'])
            
            wandb_metrics["payments/success_rate"] = successful_payments / total_payments if total_payments > 0 else 0
            wandb_metrics["payments/total_cost_usd"] = total_cost
            wandb_metrics["payments/total_attempts"] = total_payments
            
            # Agent card selection frequency
            agent_card_selections = {}
            for log in self.payment_logs:
                if log['success']:
                    agent_card_name = log['agent_card_name']
                    agent_card_selections[agent_card_name] = agent_card_selections.get(agent_card_name, 0) + 1
            
            for agent_card_name, selection_count in agent_card_selections.items():
                wandb_metrics[f"selection_frequency/{agent_card_name}"] = selection_count
                if successful_payments > 0:
                    wandb_metrics[f"selection_frequency/{agent_card_name}_percent"] = (selection_count / successful_payments) * 100
            
            # Create payment log table
            if len(self.payment_logs) > 0:
                table = wandb.Table(columns=["timestamp", "agent_card_name", "success", "tx_hash", "amount_usd"])
                for log in self.payment_logs[-10:]:  # Last 10 payments
                    table.add_data(
                        log['timestamp'],
                        log['agent_card_name'],
                        log['success'],
                        log.get('tx_hash', 'N/A'),
                        float(log['amount_usd']) if log['success'] else 0
                    )
                wandb_metrics["payments/recent_transactions"] = table
            
            self.payment_logs = []  # Clear logs
        
        # Agent card selection history
        if self.agent_card_selection_history:
            # Create selection history table
            selection_table = wandb.Table(columns=["timestamp", "question", "selected_agent_cards", "reasoning", "cost", "question_type"])
            for selection in self.agent_card_selection_history[-10:]:  # Last 10 selections
                selection_table.add_data(
                    selection['timestamp'],
                    selection['question'],
                    ', '.join(selection['selected_agent_cards']),
                    selection['reasoning'][:100] + "..." if len(selection['reasoning']) > 100 else selection['reasoning'],
                    selection['expected_cost'],
                    selection['question_type']
                )
            wandb_metrics["agent_decisions/agent_card_selections"] = selection_table
            
            # Selection strategy analysis
            question_types = {}
            for selection in self.agent_card_selection_history:
                q_type = selection['question_type']
                question_types[q_type] = question_types.get(q_type, 0) + 1
            
            for q_type, count in question_types.items():
                wandb_metrics[f"question_analysis/{q_type}_count"] = count
            
            self.agent_card_selection_history = []  # Clear history

        # Training performance metrics
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
            self.percent_correct_buffer = []

        # Evaluation metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []

        await super().wandb_log(wandb_metrics)

    def save_checkpoint(self, step, data=None):
        """Save checkpoint with environment state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        data["agent_address"] = self.agent_account.address
        data["agent_card_addresses"] = {agent_card_name: agent_card.address for agent_card_name, agent_card in self.agent_cards.items()}
        data["budget_tracker"] = {
            "current_balance": float(self.budget_tracker.current_balance),
            "total_spent": float(self.budget_tracker.total_spent),
            "evaluations_count": self.budget_tracker.evaluations_count,
            "spending_per_agent_card": {k: float(v) for k, v in self.budget_tracker.spending_per_agent_card.items()}
        }
        data["agent_card_performance"] = {
            agent_card_name: {
                "total_evaluations": agent_card.total_evaluations,
                "average_score_given": agent_card.average_score_given,
                "agent_satisfaction": agent_card.agent_satisfaction,
                "consistency_score": agent_card.consistency_score
            }
            for agent_card_name, agent_card in self.agent_cards.items()
        }
        super().save_checkpoint(step, data)

    async def _agent_select_agent_cards(self, question: str) -> AgentCardSelection:
        """Agent makes strategic decision about which agent cards to hire."""
        # Get agent card performance history
        agent_card_stats = self._get_agent_card_performance_stats()
        
        # Analyze the question to understand its requirements
        question_analysis = self._analyze_question_requirements(question)
        
        # Create a much simpler selection prompt
        agent_cards_list = []
        for agent_card_name, agent_card in self.agent_cards.items():
            agent_card_specialties = [s.value for s in agent_card.specialties]
            agent_cards_list.append(f"{agent_card_name}: ${agent_card.price_usd} ({', '.join(agent_card_specialties)})")
        
        selection_prompt = f"""Question: "{question}"

Budget: ${self.budget_tracker.current_balance:.2f}

Available agent cards:
{chr(10).join(agent_cards_list)}

Select 1-2 agent cards by ID. Respond with JSON:
{{"selected_agent_cards": ["agent_card_id"], "reasoning": "brief reason"}}"""

        # Get agent's decision
        async def get_agent_selection():
            return await self.server.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a strategic AI agent. Select agent cards for evaluation. Respond only with valid JSON."},
                    {"role": "user", "content": selection_prompt}
                ],
                n=1,
                max_tokens=200,  # Much smaller limit for simple response
                temperature=0.1
            )
        
        try:
            selection_response = await get_agent_selection()
        except Exception as e:
            logging.error(f"❌ Agent agent card selection failed: {e}")
            # Fallback to first available agent card
            fallback_agent_card = list(self.agent_cards.keys())[0]
            logging.info(f"🔄 Using fallback agent card: {fallback_agent_card}")
            return AgentCardSelection(
                selected_agent_cards=[fallback_agent_card],
                reasoning="Fallback selection due to agent failure",
                expected_cost=self.agent_cards[fallback_agent_card].price_usd,
                question_type=question_analysis.get('category', 'General')
            )
        
        # Parse response
        selection_text = selection_response.choices[0].message.content
        logging.info(f"🤖 Agent selection response: {selection_text}")
        
        try:
            import re
            import json
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', selection_text, re.DOTALL)
            if json_match:
                selection_data = json.loads(json_match.group())
                selected_names = selection_data.get("selected_agent_cards", [])
                
                # Validate selections
                valid_selections = [name for name in selected_names if name in self.agent_cards]
                
                if valid_selections:
                    total_cost = sum(self.agent_cards[name].price_usd for name in valid_selections)
                    
                    if self.budget_tracker.can_afford(total_cost):
                        logging.info(f"✅ Selected agent cards: {valid_selections} for ${total_cost}")
                        return AgentCardSelection(
                            selected_agent_cards=valid_selections,
                            reasoning=selection_data.get("reasoning", "Agent selection"),
                            expected_cost=total_cost,
                            question_type=question_analysis.get('category', 'General')
                        )
                    else:
                        logging.warning(f"⚠️ Selection too expensive: ${total_cost} > ${self.budget_tracker.current_balance}")
                        
        except Exception as e:
            logging.error(f"❌ Failed to parse agent response: {e}")
        
        # Fallback to cheapest agent card
        cheapest_agent_card = min(self.agent_cards.keys(), key=lambda j: self.agent_cards[j].price_usd)
        logging.info(f"🔄 Using cheapest agent card fallback: {cheapest_agent_card}")
        return AgentCardSelection(
            selected_agent_cards=[cheapest_agent_card],
            reasoning="Fallback to cheapest agent card",
            expected_cost=self.agent_cards[cheapest_agent_card].price_usd,
            question_type=question_analysis.get('category', 'General')
        )

    def _load_questions_dataset(self) -> Dict:
        """Load questions dataset from JSON file."""
        questions_file = Path(__file__).parent / "questions.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions dataset not found: {questions_file}")
        
        with open(questions_file, 'r') as f:
            return json.load(f)

    def _analyze_question_requirements(self, question_text: str) -> Dict[str, any]:
        """Analyze a question to determine what specialties might be needed."""
        dataset = self._load_questions_dataset()
        
        # Search for question in dataset
        question_data = None
        for q in dataset["training_questions"] + dataset["evaluation_questions"]:
            if q["text"] == question_text:
                question_data = q
                break
        
        if question_data:
            # Calculate complexity score based on specialties and difficulty
            difficulty_multiplier = {"basic": 1, "intermediate": 2, "advanced": 3, "expert": 4}
            complexity_score = len(question_data["expected_specialties"]) * difficulty_multiplier.get(question_data["difficulty"], 2)
            
            return {
                "category": question_data["category"],
                "difficulty": question_data["difficulty"],
                "expected_specialties": question_data["expected_specialties"],
                "description": question_data["description"],
                "requires_multiple_agent_cards": len(question_data["expected_specialties"]) > 1,
                "complexity_score": complexity_score
            }
        
        # Fallback for unknown questions
        return {
            "category": "unknown",
            "difficulty": "intermediate",
            "expected_specialties": ["general"],
            "description": "Unknown question type",
            "requires_multiple_agent_cards": False,
            "complexity_score": 2
        }


if __name__ == "__main__":
    PayToPlayEnv.cli()  