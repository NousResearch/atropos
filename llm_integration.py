"""
LLM Integration for Society Simulator
Supports OpenAI, Anthropic with intelligent fallbacks
"""

import os
import time
import random
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class LLMProvider(Enum):
    NONE = "none"
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    MOCK = "mock"

@dataclass
class LLMRequest:
    agent_id: str
    prompt: str
    context: Dict[str, Any]
    max_tokens: int = 150
    temperature: float = 0.7

@dataclass
class LLMResponse:
    agent_id: str
    response: str
    success: bool
    provider: str
    latency: float
    error: Optional[str] = None

class LLMManager:
    """Manages LLM requests with fallbacks and caching"""
    
    def __init__(self, provider: LLMProvider = LLMProvider.NONE, 
                 api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
        self.model = model or self._get_default_model()
        
        # Response cache for efficiency
        self.response_cache = {}
        self.cache_hits = 0
        self.total_requests = 0
        
        # Fallback patterns
        self.fallback_responses = self._load_fallback_patterns()
        
        # Initialize provider
        self.client = None
        if self.provider != LLMProvider.NONE:
            self._initialize_client()
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment"""
        if self.provider == LLMProvider.OPENAI:
            return os.getenv('OPENAI_API_KEY')
        elif self.provider == LLMProvider.ANTHROPIC:
            return os.getenv('ANTHROPIC_API_KEY')
        return None
    
    def _get_default_model(self) -> str:
        """Get default model for provider"""
        if self.provider == LLMProvider.OPENAI:
            return "gpt-3.5-turbo"
        elif self.provider == LLMProvider.ANTHROPIC:
            return "claude-3-haiku-20240307"
        return "mock"
    
    def _initialize_client(self):
        """Initialize LLM client"""
        try:
            if self.provider == LLMProvider.OPENAI and self.api_key:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                print(f"✅ OpenAI client initialized with model {self.model}")
            elif self.provider == LLMProvider.ANTHROPIC and self.api_key:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print(f"✅ Anthropic client initialized with model {self.model}")
            else:
                print(f"⚠️  {self.provider.value} selected but no API key available")
                print(f"   Falling back to rule-based responses")
                self.provider = LLMProvider.NONE
        except ImportError as e:
            print(f"⚠️  {self.provider.value} library not installed: {e}")
            print(f"   pip install openai anthropic")
            print(f"   Falling back to rule-based responses")
            self.provider = LLMProvider.NONE
    
    def _load_fallback_patterns(self) -> Dict[str, List[str]]:
        """Load fallback response patterns"""
        return {
            "greeting": [
                "Hello there! Nice to meet you.",
                "Hi! How are you doing?",
                "Good day! What brings you here?",
                "Greetings, friend!",
                "Hey! Lovely weather today."
            ],
            "work": [
                "I should focus on my work today.",
                "Time to be productive and get things done.",
                "Let me work on improving my skills.",
                "I'll put in good effort on my tasks.",
                "Work keeps me motivated and busy."
            ],
            "trade": [
                "Would you be interested in trading some items?",
                "I have some resources I could exchange.",
                "Let's discuss a mutually beneficial trade.",
                "I'm looking to exchange some materials.",
                "Trading helps our community thrive."
            ],
            "social": [
                "It's great to connect with others.",
                "I enjoy meeting new people in our community.",
                "Social connections make life more meaningful.",
                "Let's share what we've been up to.",
                "Community bonds make us all stronger."
            ],
            "rest": [
                "I need to take some time to rest and recover.",
                "Rest is important for maintaining my energy.",
                "Time for a well-deserved break.",
                "I should recharge and take care of myself.",
                "Rest helps me be more effective later."
            ]
        }
    
    async def get_response(self, request: LLMRequest) -> LLMResponse:
        """Get LLM response with fallback handling"""
        self.total_requests += 1
        
        # Check cache first
        cache_key = self._make_cache_key(request)
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cached = self.response_cache[cache_key]
            return LLMResponse(
                agent_id=request.agent_id,
                response=cached,
                success=True,
                provider="cache",
                latency=0.001
            )
        
        start_time = time.time()
        
        # Try LLM provider
        if self.provider != LLMProvider.NONE and self.client:
            try:
                response = await self._call_llm_provider(request)
                latency = time.time() - start_time
                
                # Cache successful response
                self.response_cache[cache_key] = response
                
                return LLMResponse(
                    agent_id=request.agent_id,
                    response=response,
                    success=True,
                    provider=self.provider.value,
                    latency=latency
                )
            except Exception as e:
                print(f"⚠️  LLM request failed: {e}")
                # Fall through to rule-based response
        
        # Fallback to rule-based response
        response = self._generate_fallback_response(request)
        latency = time.time() - start_time
        
        return LLMResponse(
            agent_id=request.agent_id,
            response=response,
            success=True,
            provider="fallback",
            latency=latency
        )
    
    async def _call_llm_provider(self, request: LLMRequest) -> str:
        """Call the configured LLM provider"""
        if self.provider == LLMProvider.OPENAI:
            return await self._call_openai(request)
        elif self.provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(request)
        elif self.provider == LLMProvider.MOCK:
            return self._generate_mock_response(request)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    async def _call_openai(self, request: LLMRequest) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return response.choices[0].message.content.strip()
    
    async def _call_anthropic(self, request: LLMRequest) -> str:
        """Call Anthropic API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=[{"role": "user", "content": request.prompt}]
        )
        return response.content[0].text.strip()
    
    def _generate_mock_response(self, request: LLMRequest) -> str:
        """Generate mock LLM response for testing"""
        context = request.context
        agent_type = context.get('agent_type', 'citizen')
        
        responses = [
            f"As a {agent_type}, I think I should focus on my goals today.",
            f"The community needs more cooperation between different groups.",
            f"I'm considering what actions would benefit both myself and others.",
            f"Resource management is important for our society's prosperity.",
            f"Building relationships with others strengthens our community."
        ]
        
        return random.choice(responses)
    
    def _generate_fallback_response(self, request: LLMRequest) -> str:
        """Generate rule-based fallback response"""
        context = request.context
        
        # Determine response category based on context
        if "energy" in context and context.get("energy", 1.0) < 0.3:
            category = "rest"
        elif "trade" in request.prompt.lower() or context.get("resources", {}).get("currency", 0) > 500:
            category = "trade"
        elif "work" in request.prompt.lower() or context.get("state") == "working":
            category = "work"
        elif "social" in request.prompt.lower() or len(context.get("nearby_agents", [])) > 0:
            category = "social"
        else:
            category = "greeting"
        
        responses = self.fallback_responses.get(category, self.fallback_responses["greeting"])
        return random.choice(responses)
    
    def _make_cache_key(self, request: LLMRequest) -> str:
        """Create cache key for request"""
        # Simple cache key based on prompt and key context
        context_summary = {
            "agent_type": request.context.get("agent_type"),
            "energy_level": "low" if request.context.get("energy", 1.0) < 0.5 else "high",
            "has_neighbors": len(request.context.get("nearby_agents", [])) > 0
        }
        return f"{request.prompt[:50]}_{json.dumps(context_summary, sort_keys=True)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        cache_rate = (self.cache_hits / max(1, self.total_requests)) * 100
        return {
            "provider": self.provider.value,
            "model": self.model,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_rate": f"{cache_rate:.1f}%",
            "cached_responses": len(self.response_cache)
        }

def create_agent_prompt(agent_context: Dict[str, Any]) -> str:
    """Create LLM prompt for agent decision making"""
    agent_type = agent_context.get("agent_type", "citizen")
    energy = agent_context.get("energy", 1.0)
    happiness = agent_context.get("happiness", 0.5)
    resources = agent_context.get("resources", {})
    nearby_agents = agent_context.get("nearby_agents", [])
    
    prompt = f"""You are a {agent_type} in a virtual society. 

Current status:
- Energy: {energy:.2f} (0.0 = exhausted, 1.0 = full energy)
- Happiness: {happiness:.2f} (0.0 = miserable, 1.0 = ecstatic)
- Resources: {resources}
- Nearby agents: {len(nearby_agents)} others around you

Based on your current situation, what would you like to do? Choose one action and explain briefly:
- Work (to earn resources and feel productive)
- Socialize (to build relationships and boost happiness)
- Trade (to exchange resources with others)
- Rest (to recover energy)
- Move (to explore or find better opportunities)

Respond in character as a {agent_type} would, considering your current needs and situation."""

    return prompt

# Example usage
async def test_llm_integration():
    """Test LLM integration"""
    print("🧪 Testing LLM Integration")
    
    # Test different providers
    providers = [LLMProvider.MOCK, LLMProvider.NONE]
    
    for provider in providers:
        print(f"\n📡 Testing {provider.value}")
        llm = LLMManager(provider)
        
        # Create test request
        context = {
            "agent_type": "farmer",
            "energy": 0.3,
            "happiness": 0.7,
            "resources": {"food": 50, "currency": 200},
            "nearby_agents": [{"type": "trader"}]
        }
        
        prompt = create_agent_prompt(context)
        request = LLMRequest("test_agent", prompt, context)
        
        response = await llm.get_response(request)
        print(f"✅ Response: {response.response[:100]}...")
        print(f"   Provider: {response.provider}, Latency: {response.latency:.3f}s")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llm_integration())