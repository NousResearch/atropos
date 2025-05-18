#!/usr/bin/env python3
"""
WikipediaArticleCreatorEnv: Environment for training an LLM to research and create Wikipedia-style articles

This environment uses web search and content extraction tools to enable multi-step research 
and article generation on arbitrary topics.
"""

import json
import logging
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union

import wandb
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.message_history_utils import truncate_thinking
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

from environments.hack0.wikipedia.tools.tavily_tools import TavilySearchTool, TavilyExtractTool

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# System prompt for the Wikipedia article creation task
SYSTEM_PROMPT = """
You are a skilled researcher and writer who creates accurate, neutral, and comprehensive Wikipedia-style articles.

Your task is to research the given topic using web search and content extraction tools, and then write a well-structured Wikipedia article based on your findings.

Follow these guidelines when creating your article:
1. Research thoroughly using the tools provided
2. Maintain a Neutral Point of View (NPOV) - present all significant viewpoints fairly
3. Structure your article with a clear introduction, organized sections, and a conclusion if appropriate
4. Cite reliable sources for factual claims
5. Use formal, encyclopedic language
6. Format your article in Markdown

During your work, you may:
1. Think through your research strategy and article planning
2. Search for information using web_search
3. Extract content from specific webpages using visit_page
4. Organize and synthesize information from multiple sources
5. Create a final Wikipedia-style article when you have sufficient information

You should enclose your thoughts and internal monologue inside <think> </think> tags, and then use tools or provide your final output.

IMPORTANT: When you have completed your research and are ready to provide the final article, format it as follows:
Final Step: ```markdown 
[Your complete article in markdown format]
```

For tool calls, use <tool_call> </tool_call> tags with the following JSON format:
<tool_call>
{"name": "web_search", "arguments": {"query": "example search query", "num_results": 5}}
</tool_call>

OR

<tool_call>
{"name": "visit_page", "arguments": {"url": "https://example.com/page"}}
</tool_call>
"""


class WikipediaArticleCreatorConfig(BaseEnvConfig):
    """Configuration for the WikipediaArticleCreator environment"""
    max_steps: int = 10  # Maximum research steps per article
    temperature: float = 0.7  # Sampling temperature
    thinking_active: bool = True  # Enable thinking tags
    eval_topics: int = 30  # Number of topics for evaluation
    tool_timeout: float = 15.0  # Timeout for tool execution (seconds)
    tavily_api_key: Optional[str] = None  # API key for Tavily (falls back to env var)
    min_article_sections: int = 3  # Minimum number of sections in final article
    max_article_tokens: int = 2048  # Maximum tokens in final article
    topics_file: str = "topics.json"  # File containing research topics
    logging_active: bool = True  # Enable detailed logging


class EpisodeState:
    """
    Maintains state for a single episode (article creation task)
    """
    def __init__(self, episode_id: int, topic: str):
        self.episode_id = episode_id
        self.topic = topic  # The research topic for this episode
        self.message_history: List[Dict] = []  # Stores all interactions
        self.tool_calls: List[Dict] = []  # Records tool calls made
        self.tool_results: List[Dict] = []  # Records tool results returned
        self.steps_taken: int = 0  # Number of steps in this episode
        self.is_terminal: bool = False  # Whether episode has terminated
        self.final_article: Optional[str] = None  # Final Wikipedia article in markdown
        self.research_facts: List[str] = []  # Important facts discovered during research
        self.score: float = 0.0  # Score for this episode


class WikipediaArticleCreatorEnv(BaseEnv):
    """
    Environment for training an LLM to research and create Wikipedia-style articles
    
    This environment:
    - Presents the model with a topic to research
    - Allows multi-step interactions using web_search and visit_page tools
    - Tracks research process and article quality
    - Rewards comprehensive, well-structured, and accurate articles
    """
    def __init__(
        self,
        config: WikipediaArticleCreatorConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        
        # Initialize environment
        self.config = config
        self.episodes: Dict[int, EpisodeState] = {}
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb = []
        
        # Set up tools
        tavily_key = config.tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not tavily_key:
            logger.warning("No Tavily API key provided - tools will not function properly")
            
        self.search_tool = TavilySearchTool(api_key=tavily_key)
        self.extract_tool = TavilyExtractTool(api_key=tavily_key)
        
        # Tool definitions for the LLM
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information on a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to perform."
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5, max: 10)",
                                "default": 5
                            },
                            "filter_year": {
                                "type": "string", 
                                "description": "Filter results to a specific year",
                                "nullable": True
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "visit_page",
                    "description": "Extract content from a specific webpage",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the webpage to visit"
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
        
        # Load topics if file exists
        self.topics = self._load_topics()
        self.iter = 0
        
        self.article_quality_metrics: List[Dict[str, float]] = []
        
    def _load_topics(self) -> List[str]:
        """Load research topics from wikipedia_articles.json or use defaults if file doesn't exist"""
        try:
            articles_path = os.path.join(os.path.dirname(__file__), "wikipedia_articles.json")
            if os.path.exists(articles_path):
                # The file is large, so we'll read it in chunks and extract just the titles
                topics = []
                with open(articles_path, 'r') as f:
                    # Read opening bracket
                    char = f.read(1)
                    if char != '[':
                        raise ValueError("Expected JSON array to start with '['")
                    
                    # Process articles one by one
                    count = 0
                    max_topics = 100  # Limit to 100 topics
                    
                    while count < max_topics:
                        article_json = ""
                        brace_count = 0
                        in_article = False
                        
                        # Read until we find a complete article JSON object
                        while True:
                            char = f.read(1)
                            if not char:  # End of file
                                break
                                
                            if char == '{' and not in_article:
                                in_article = True
                                brace_count = 1
                                article_json = '{'
                            elif in_article:
                                article_json += char
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        # Found complete article
                                        break
                        
                        if not article_json:
                            break
                            
                        try:
                            article = json.loads(article_json)
                            title = article.get("title", "")
                            if title and len(title) < 100:  # Skip very long titles
                                topics.append(title)
                                count += 1
                        except json.JSONDecodeError:
                            continue
                
                if topics:
                    logger.info(f"Loaded {len(topics)} topics from wikipedia_articles.json")
                    return topics
                    
        except Exception as e:
            logger.warning(f"Failed to load topics from wikipedia_articles.json: {e}")
        
        # Default topics if file doesn't exist or loading fails
        default_topics = [
            "History of artificial intelligence",
            "Climate change in the Arctic",
            "The Great Barrier Reef ecosystem",
            "Quantum computing principles",
            "Anti-black racism in the Arab World",
            "History of cryptography",
            "Renewable energy in developing countries",
            "Space exploration in the 21st century",
            "Traditional medicine systems around the world",
            "The evolution of human language"
        ]
        logger.info(f"Using {len(default_topics)} default topics")
        return default_topics
    
    @classmethod
    def config_init(cls) -> Tuple[WikipediaArticleCreatorConfig, List[APIServerConfig]]:
        """Initialize default configuration"""
        env_config = WikipediaArticleCreatorConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=512,
            steps_per_eval=20,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="wikipedia_article_creator",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            max_steps=10,
            temperature=0.7,
            thinking_active=True,
            eval_topics=5,
            tool_timeout=15.0,
            tavily_api_key=None,
            min_article_sections=3,
            max_article_tokens=2048,
            topics_file="topics.json",
            logging_active=True,
        )
        
        # Configure servers
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=8,
                num_requests_for_eval=64,
            ),
        ]
        
        return env_config, server_configs
    
    def _get_or_create_episode(self, episode_id: int, topic: Optional[str] = None) -> EpisodeState:
        """Get an existing episode or create a new one"""
        if episode_id not in self.episodes:
            if topic is None:
                topic = random.choice(self.topics)
            
            ep = EpisodeState(episode_id, topic)
            
            # Initialize with system prompt
            ep.message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add initial user prompt with the topic
            ep.message_history.append({
                "role": "user", 
                "content": f"Research and write a comprehensive Wikipedia-style article about: \"{topic}\""
            })
            
            self.episodes[episode_id] = ep
        
        return self.episodes[episode_id]
    
    def _parse_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from model response"""
        tool_calls = []
        
        for tool in self.tools:
            name = tool["function"]["name"]
            parsed_name, parsed_args, is_error = parse_tool_call(
                response, [tool], ["tool_call"]
            )
            
            if not is_error and parsed_name == name:
                tool_calls.append({
                    "name": name,
                    "arguments": parsed_args
                })
        
        return tool_calls
    
    def _extract_final_article(self, response: str) -> Optional[str]:
        """Extract final Wikipedia article markdown if present"""
        # Regular expression to match content between Final Step: ```markdown and ``` tags
        pattern = r"Final Step:\s*```markdown\s*(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        return None
    
    def _format_tool_results(self, tool_results: List[Dict]) -> str:
        """Format tool results as a user message"""
        if not tool_results:
            return "No results found."
        
        formatted_results = ["Tool Results:"]
        
        for result in tool_results:
            tool_name = result.get("name", "unknown_tool")
            args = result.get("arguments", {})
            data = result.get("data", [])
            
            if tool_name == "web_search":
                query = args.get("query", "")
                num_results = args.get("num_results", 5)
                formatted_results.append(f'web_search(query="{query}", num_results={num_results})\n')
                
                if isinstance(data, list):
                    formatted_results.append(json.dumps(data, indent=2))
                else:
                    formatted_results.append("No results found.")
            
            elif tool_name == "visit_page":
                url = args.get("url", "")
                formatted_results.append(f'visit_page(url="{url}")\n')
                
                if isinstance(data, dict):
                    content = data.get("content", "")
                    title = data.get("title", "")
                    success = data.get("success", False)
                    
                    if success:
                        formatted_results.append(f"Title: {title}")
                        formatted_results.append(f"Content:\n{content[:2000]}...")
                        if len(content) > 2000:
                            formatted_results.append("\n[Content truncated due to length]")
                    else:
                        error = data.get("error", "Unknown error")
                        formatted_results.append(f"Error: {error}")
                else:
                    formatted_results.append("Failed to retrieve page content.")
        
        return "\n\n".join(formatted_results)
    
    def _extract_research_facts(self, tool_results: List[Dict], facts: List[str]):
        """Extract important facts from tool results for later evaluation"""
        for result in tool_results:
            tool_name = result.get("name", "")
            data = result.get("data", None)
            
            if tool_name == "web_search" and isinstance(data, list):
                for item in data:
                    content = item.get("content", "")
                    if content:
                        # Simple sentence extraction - could be enhanced with NLP
                        sentences = re.split(r'(?<=[.!?])\s+', content)
                        for sentence in sentences:
                            if len(sentence) > 30 and sentence not in facts:
                                facts.append(sentence)
            
            elif tool_name == "visit_page" and isinstance(data, dict):
                content = data.get("content", "")
                if content:
                    paragraphs = content.split("\n\n")
                    for paragraph in paragraphs:
                        if len(paragraph) > 50 and paragraph not in facts:
                            facts.append(paragraph)
    
    async def _execute_tool_call(self, tool_call: Dict) -> Dict:
        """Execute a tool call and return the result"""
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        
        result = {
            "name": tool_name,
            "arguments": arguments,
            "data": None
        }
        
        try:
            if tool_name == "web_search":
                query = arguments.get("query", "")
                num_results = min(arguments.get("num_results", 5), 10)  # Limit to 10 max
                filter_year = arguments.get("filter_year", None)
                
                search_results = self.search_tool.forward(
                    query=query,
                    num_results=num_results,
                    filter_year=filter_year
                )
                result["data"] = search_results
                
            elif tool_name == "visit_page":
                url = arguments.get("url", "")
                page_data = self.extract_tool.forward(url=url)
                result["data"] = page_data
                
            else:
                logger.warning(f"Unknown tool: {tool_name}")
                result["data"] = {"error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            result["data"] = {"error": f"Tool execution failed: {str(e)}"}
            
        return result
    
    async def _get_model_response(self, messages: List[Dict]) -> str:
        """Get a response from the model for the current conversation state"""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        try:
            completion = await self.server.completion(
                prompt=prompt,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=self.config.temperature,
            )
            return completion.choices[0].text
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return ""

    async def _next_step(self, episode: EpisodeState) -> Tuple[bool, Dict]:
        """
        Process one step of article research interaction
        Returns (is_terminal, step_data)
        """
        # Get current conversation history
        messages = episode.message_history.copy()
        
        # Generate model response
        response = await self._get_model_response(messages)
        
        if not response:
            episode.is_terminal = True
            return True, {"response": "", "tool_calls": [], "tool_results": []}
        
        # Check for final article
        final_article = self._extract_final_article(response)
        if final_article:
            episode.is_terminal = True
            episode.final_article = final_article
            # Add response to history
            episode.message_history.append({"role": "assistant", "content": response})
            return True, {"response": response, "tool_calls": [], "tool_results": []}
        
        # Extract tool calls for research
        tool_calls = self._parse_tool_calls(response)
        
        # Execute research tool calls
        tool_results = []
        for tool_call in tool_calls:
            result = await self._execute_tool_call(tool_call)
            tool_results.append(result)
        
        # Add response and tool results to history
        episode.message_history.append({"role": "assistant", "content": response})
        
        # Format tool results as a user message
        tool_results_message = self._format_tool_results(tool_results)
        episode.message_history.append({"role": "user", "content": tool_results_message})
        
        # Update episode state
        episode.steps_taken += 1
        episode.tool_calls.extend(tool_calls)
        episode.tool_results.extend(tool_results)
        
        # Extract and store research facts for later evaluation
        self._extract_research_facts(tool_results, episode.research_facts)
        
        # Check if max steps reached
        if episode.steps_taken >= self.config.max_steps:
            episode.is_terminal = True
        
        return episode.is_terminal, {
            "response": response, 
            "tool_calls": tool_calls, 
            "tool_results": tool_results
        }
    
    def _assess_article_quality(self, final_article: str, research_facts: List[str]) -> Dict[str, float]:
        """
        Evaluate the quality of the final article
        Returns a dictionary of quality metrics
        """
        metrics = {
            "structure_score": 0.0,
            "comprehensiveness_score": 0.0,
            "fact_usage_score": 0.0,
            "overall_quality": 0.0,
        }
        
        # Basic structure analysis
        if not final_article:
            return metrics
            
        # Check for section headers
        sections = re.findall(r'^##?\s+.+$', final_article, re.MULTILINE)
        num_sections = len(sections)
        
        # Check for references
        references = re.findall(r'^##?\s*References', final_article, re.MULTILINE | re.IGNORECASE)
        has_references = len(references) > 0
        
        # Calculate structure score
        structure_score = 0.0
        if num_sections >= self.config.min_article_sections:
            structure_score += 0.7
        else:
            structure_score += 0.7 * (num_sections / self.config.min_article_sections)
            
        if has_references:
            structure_score += 0.3
            
        metrics["structure_score"] = structure_score
        
        # Calculate comprehensiveness score based on length and section count
        article_length = len(final_article)
        comp_score = min(1.0, article_length / 3000) * 0.7
        comp_score += min(1.0, num_sections / 5) * 0.3
        metrics["comprehensiveness_score"] = comp_score
        
        # Calculate fact usage score
        # This is a simplistic approach - could be enhanced with NLP/semantic matching
        fact_usage = 0.0
        if research_facts:
            facts_found = 0
            for fact in research_facts:
                # Check if key phrases from the fact appear in the article
                key_phrases = [p for p in fact.split() if len(p) > 5]
                if key_phrases:
                    for phrase in key_phrases[:5]:  # Use up to 5 phrases per fact
                        if phrase.lower() in final_article.lower():
                            facts_found += 1
                            break
            
            fact_usage = min(1.0, facts_found / len(research_facts))
        metrics["fact_usage_score"] = fact_usage
        
        # Calculate overall quality
        overall = (
            structure_score * 0.3 + 
            comp_score * 0.4 + 
            fact_usage * 0.3
        )
        metrics["overall_quality"] = overall
        
        return metrics
    
    async def collect_trajectories(self, item: Tuple[int, str]) -> Tuple[List[ScoredDataGroup], List]:
        """
        Manage full research trajectory collection
        
        Args:
            item: Tuple containing (episode_id, topic)
            
        Returns:
            Tuple of:
            - List of ScoredDataGroup objects: Scored data for training
            - List: Empty list (no backlog items)
        """
        episode_id, topic = item
        
        # Get or create episode state
        episode = self._get_or_create_episode(episode_id, topic)
        
        trajectory_data: List[ScoredDataGroup] = []
        
        # Run episode until terminal state
        while not episode.is_terminal:
            is_terminal, step_data = await self._next_step(episode)
            
            # Skip steps with no response or no tools used (unless terminal)
            response = step_data.get("response", "")
            if not response:
                continue
                
            # Create scored data for this step
            step_score = ScoredDataGroup()
            
            # Tokenize conversation up to this point
            tokenized = tokenize_for_trainer(self.tokenizer, episode.message_history)
            
            # Score based on tool usage (basic heuristic, improve in future)
            tool_calls = step_data.get("tool_calls", [])
            tool_results = step_data.get("tool_results", [])
            
            if is_terminal and episode.final_article:
                # Terminal step with article - score based on article quality
                quality_metrics = self._assess_article_quality(
                    episode.final_article, episode.research_facts
                )
                step_score["tokens"] = [tokenized["tokens"]]
                step_score["masks"] = [tokenized["masks"]]
                step_score["scores"] = [quality_metrics["overall_quality"] * 2 - 1]  # Scale to [-1, 1]
                
                # Record metrics for logging
                quality_metrics["topic"] = episode.topic
                quality_metrics["steps_taken"] = episode.steps_taken
                self.article_quality_metrics.append(quality_metrics)
                
            elif tool_calls:
                # Non-terminal step with tool usage - score based on usefulness
                step_score["tokens"] = [tokenized["tokens"]]
                step_score["masks"] = [tokenized["masks"]]
                
                # Simple usefulness heuristic:
                # - Higher score for visiting pages than generic searches
                # - Higher score if results were found than if errors
                usefulness = 0.0
                for result in tool_results:
                    name = result.get("name", "")
                    data = result.get("data", None)
                    
                    if name == "web_search" and isinstance(data, list) and data:
                        usefulness = max(usefulness, 0.6)
                    elif name == "visit_page" and isinstance(data, dict) and data.get("success", False):
                        usefulness = max(usefulness, 0.8)
                        
                step_score["scores"] = [usefulness * 2 - 1]  # Scale to [-1, 1]
            
            else:
                # Step with no tool usage - low score
                step_score["tokens"] = [tokenized["tokens"]]
                step_score["masks"] = [tokenized["masks"]]
                step_score["scores"] = [-0.5]  # Slight negative score
            
            trajectory_data.append(step_score)
        
        # Clean up episode data
        if episode_id in self.episodes:
            del self.episodes[episode_id]
        
        return trajectory_data, []
    
    async def score(
        self, rollout_group_data: List[ScoredDataGroup]
    ) -> List[ScoredDataGroup]:
        """
        Pass through the scored data from collect_trajectories
        
        This method is simpler than usual because scoring is done inline during collection,
        since the model's "actions" (tool use and article writing) are evaluated directly.
        """
        return rollout_group_data
    
    async def setup(self):
        """Set up the environment - load topics, etc."""
        pass
    
    async def get_next_item(self) -> Tuple[int, str]:
        """Get next episode ID and topic"""
        # Select a random topic
        topic = random.choice(self.topics)
        episode_id = self.iter
        self.iter += 1
        
        return (episode_id, topic)
    
    async def evaluate(self, *args, **kwargs):
        """Run evaluation on test set of topics"""
        if not self.config.use_wandb:
            logger.info("Skipping evaluation as wandb is not enabled")
            return
            
        num_eval = min(self.config.eval_topics, len(self.topics))
        eval_topics = random.sample(self.topics, num_eval)
        
        logger.info(f"Starting evaluation on {num_eval} topics")
        
        eval_metrics = {
            "avg_steps": 0.0,
            "avg_quality": 0.0,
            "avg_structure": 0.0,
            "avg_comprehensiveness": 0.0,
            "avg_fact_usage": 0.0,
            "completion_rate": 0.0
        }
        
        completed_count = 0
        total_steps = 0
        quality_scores = {"overall": [], "structure": [], "comprehensiveness": [], "fact_usage": []}
        
        # Run evaluation episodes
        for eval_idx, topic in enumerate(eval_topics):
            episode_id = 1000000 + eval_idx  # High range for eval episodes
            episode = self._get_or_create_episode(episode_id, topic)
            
            # Run episode until terminal
            while not episode.is_terminal:
                is_terminal, _ = await self._next_step(episode)
                if is_terminal:
                    break
            
            # Record metrics
            total_steps += episode.steps_taken
            
            if episode.final_article:
                completed_count += 1
                quality_metrics = self._assess_article_quality(
                    episode.final_article, episode.research_facts
                )
                
                quality_scores["overall"].append(quality_metrics["overall_quality"])
                quality_scores["structure"].append(quality_metrics["structure_score"])
                quality_scores["comprehensiveness"].append(quality_metrics["comprehensiveness_score"])
                quality_scores["fact_usage"].append(quality_metrics["fact_usage_score"])
            
            # Clean up episode
            if episode_id in self.episodes:
                del self.episodes[episode_id]
        
        # Calculate averages
        if num_eval > 0:
            eval_metrics["avg_steps"] = total_steps / num_eval
            eval_metrics["completion_rate"] = completed_count / num_eval
            
            if completed_count > 0:
                eval_metrics["avg_quality"] = sum(quality_scores["overall"]) / completed_count
                eval_metrics["avg_structure"] = sum(quality_scores["structure"]) / completed_count
                eval_metrics["avg_comprehensiveness"] = sum(quality_scores["comprehensiveness"]) / completed_count
                eval_metrics["avg_fact_usage"] = sum(quality_scores["fact_usage"]) / completed_count
        
        # Store metrics for wandb logging
        self.eval_metrics = [
            ("eval/avg_steps", eval_metrics["avg_steps"]),
            ("eval/completion_rate", eval_metrics["completion_rate"]),
            ("eval/avg_quality", eval_metrics["avg_quality"]),
            ("eval/avg_structure", eval_metrics["avg_structure"]),
            ("eval/avg_comprehensiveness", eval_metrics["avg_comprehensiveness"]),
            ("eval/avg_fact_usage", eval_metrics["avg_fact_usage"]),
        ]
    
    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb"""
        if wandb_metrics is None:
            wandb_metrics = {}
        
        # Add eval metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        
        # Clear metrics for next round
        self.eval_metrics = []
        
        # Add article quality metrics if available
        if self.article_quality_metrics:
            # Calculate averages
            avg_quality = sum(m["overall_quality"] for m in self.article_quality_metrics) / len(self.article_quality_metrics)
            avg_steps = sum(m["steps_taken"] for m in self.article_quality_metrics) / len(self.article_quality_metrics)
            
            wandb_metrics["train/avg_article_quality"] = avg_quality
            wandb_metrics["train/avg_steps_per_article"] = avg_steps
            
            # Create a table of article metrics
            if wandb.run is not None:
                table = wandb.Table(columns=["topic", "steps", "overall_quality", 
                                            "structure", "comprehensiveness", "fact_usage"])
                
                for metric in self.article_quality_metrics:
                    table.add_data(
                        metric["topic"],
                        metric["steps_taken"],
                        metric["overall_quality"],
                        metric["structure_score"],
                        metric["comprehensiveness_score"],
                        metric["fact_usage_score"]
                    )
                
                wandb_metrics["train/article_quality"] = table
            
            # Clear for next round
            self.article_quality_metrics = []
        
        await super().wandb_log(wandb_metrics)

    @classmethod
    def cli(cls):
        """Command-line interface entry point"""
        super().cli()


if __name__ == "__main__":
    WikipediaArticleCreatorEnv.cli()