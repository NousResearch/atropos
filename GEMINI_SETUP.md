# 🤖 Gemini API Setup Guide

## Quick Setup

Our LLM Society Simulation now uses Google's Gemini Pro as the default language model. Here's how to set it up:

### 1. Get a Free Gemini API Key

1. Go to **[Google AI Studio](https://makersuite.google.com/app/apikey)**
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated API key (starts with `AIzaSy...`)

### 2. Set Environment Variable

**On macOS/Linux:**
```bash
export GEMINI_API_KEY="your_api_key_here"
# Or add to your ~/.bashrc or ~/.zshrc for persistence
echo 'export GEMINI_API_KEY="your_api_key_here"' >> ~/.bashrc
```

**On Windows:**
```cmd
set GEMINI_API_KEY=your_api_key_here
# Or set permanently through System Properties
```

**Alternative Environment Variable:**
You can also use `GOOGLE_API_KEY` instead of `GEMINI_API_KEY`.

### 3. Test the Integration

```bash
# Test Gemini integration
python test_gemini_integration.py

# Run simulation with Gemini
python test_basic_simulation.py
```

## 🔄 Fallback Behavior

**No API Key Required for Testing!**

Our system automatically falls back to intelligent mock responses when:
- No API key is provided
- API key is invalid
- API rate limits are exceeded
- Network connectivity issues

This means you can run simulations immediately without any setup!

## 📊 API Usage & Costs

**Gemini Pro Pricing (as of 2024):**
- **Free tier**: 15 requests per minute
- **Text generation**: Very low cost per 1K tokens
- **Rate limits**: Generous for development

**Estimated Usage:**
- 10 agents, 50 steps: ~500 requests
- 500 agents, 1000 steps: ~500,000 requests
- Cost: Typically under $1-2 for large simulations

## 🎯 Benefits of Real Gemini vs Mock

| Feature | Mock Responses | Real Gemini |
|---------|----------------|-------------|
| **Consistency** | Rule-based patterns | Natural language understanding |
| **Creativity** | Limited variations | Rich, contextual responses |
| **Social Behavior** | Simple interactions | Complex social dynamics |
| **Emergent Behavior** | Predictable | Surprising and realistic |
| **Research Value** | Basic testing | Publication-worthy results |

## ⚙️ Configuration Options

You can customize Gemini behavior in `src/utils/config.py`:

```python
class LLMConfig(BaseModel):
    model_name: str = "gemini-pro"           # Model to use
    max_tokens: int = 150                    # Response length
    temperature: float = 0.7                 # Creativity (0.0-1.0)
    cache_responses: bool = True             # Cache for performance
```

Or via command line:
```bash
python src/main.py --model gemini-pro --agents 100 --steps 500
```

## 🚀 Next Steps

1. **Start Small**: Test with 10-50 agents first
2. **Monitor Usage**: Check API usage in Google Cloud Console
3. **Scale Up**: Gradually increase to 500+ agents
4. **Optimize**: Use caching and batching for efficiency

## 🔧 Troubleshooting

**Common Issues:**

1. **"API key not valid"**: Check environment variable setup
2. **Rate limiting**: Increase delays or get paid tier
3. **Network errors**: Check internet connection
4. **Import errors**: Run `pip install google-generativeai`

**Getting Help:**
- Check logs for detailed error messages
- Fallback to mock responses always works
- Test with `python test_gemini_integration.py`

---

## 🎉 Ready to Simulate!

With Gemini API configured, you're ready to run sophisticated LLM-driven agent simulations with realistic social dynamics and emergent behaviors!

```bash
# Quick start with Gemini
python src/main.py --agents 50 --steps 200 --model gemini-pro
```
