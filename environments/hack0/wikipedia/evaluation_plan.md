# Wikipedia Article Evaluation System Plan

## Overview

This document outlines the plan to implement an evaluation system that uses OpenAI models to assess AI-generated Wikipedia articles against reference articles from the actual Wikipedia. This system will provide detailed, line-by-line factual accuracy assessments and overall article quality metrics.

## Core Components

### 1. Data Collection Module

**Purpose**: Fetch both reference Wikipedia articles and AI-generated articles for comparison.

**Implementation Details**:
- Use Wikipedia's API to fetch reference articles based on titles
- Load AI-generated articles from the environment's output
- Support batch processing of multiple article pairs
- Handle caching of reference articles to reduce API calls

```python
class WikipediaReferenceCollector:
    """Fetches reference articles from Wikipedia's API"""
    
    def fetch_reference_article(self, title: str) -> Dict:
        """
        Fetches a Wikipedia article by title.
        Returns full article content and metadata.
        """
        pass
        
    def preprocess_reference(self, content: str) -> str:
        """
        Cleans and formats Wikipedia content for comparison.
        Removes irrelevant sections, templates, etc.
        """
        pass
```

### 2. Content Preparation Module

**Purpose**: Prepare both articles for fair comparison by standardizing formats, segmenting content, and implementing necessary preprocessing.

**Implementation Details**:
- Break AI-generated article into numbered lines for granular assessment
- Extract key sections from both articles (intro, main content, references)
- Normalize formatting differences between the two sources
- Implement content chunking for very large articles (to handle model context limitations)

```python
class ArticlePreprocessor:
    """Prepares articles for evaluation"""
    
    def segment_article(self, article_content: str) -> Dict[str, List[str]]:
        """
        Segments article into introduction, main sections, and references.
        Returns dict with all segments.
        """
        pass
        
    def number_lines(self, content: str) -> Tuple[str, List[str]]:
        """
        Numbers each line in the article for reference in the evaluation.
        Returns both the numbered version and a list of original lines.
        """
        pass
        
    def chunk_content(self, content: str, max_chunk_size: int = 8000) -> List[str]:
        """
        Splits content into chunks that fit within model context limits.
        Preserves paragraph and section boundaries where possible.
        """
        pass
```

### 3. Evaluation Engine

**Purpose**: Core system that compares the AI article against the reference, analyzing factual accuracy line by line.

**Implementation Details**:
- Design a robust prompt template for the OpenAI model
- Process articles in manageable chunks if necessary
- Generate YAML-formatted assessment of each line
- Categorize statements as CORRECT, INCORRECT, or UNKNOWN
- Include brief justification for each classification

```python
class ArticleEvaluator:
    """Core evaluation engine using OpenAI models"""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.2):
        """Initialize with preferred model and settings"""
        pass
        
    def create_evaluation_prompt(
        self, 
        reference_content: str, 
        generated_content: str
    ) -> str:
        """
        Creates the prompt for the evaluation model with both articles.
        Includes instructions for analysis and output format.
        """
        pass
        
    async def evaluate_chunk(self, prompt: str) -> Dict:
        """
        Sends a chunk of content to OpenAI for evaluation.
        Returns the parsed YAML response.
        """
        pass
        
    async def evaluate_full_article(
        self, 
        reference_article: str, 
        generated_article: str
    ) -> Dict:
        """
        Evaluates the entire article, potentially in chunks.
        Combines results into a complete evaluation report.
        """
        pass
```

### 4. Results Processing and Reporting

**Purpose**: Analyze evaluation results and generate insightful reports on article quality.

**Implementation Details**:
- Parse YAML results into structured data (pandas DataFrame)
- Calculate accuracy statistics (percentage of correct/incorrect/unknown statements)
- Generate visualizations of accuracy distribution
- Create summary reports with key metrics and recommendations
- Support for Weights & Biases integration for experiment tracking

```python
class EvaluationAnalyzer:
    """Processes and analyzes evaluation results"""
    
    def parse_results(self, evaluation_data: Dict) -> pd.DataFrame:
        """
        Converts raw evaluation data to a structured DataFrame.
        """
        pass
        
    def calculate_accuracy_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates accuracy metrics including:
        - Overall accuracy percentage
        - Section-by-section accuracy
        - Distribution of CORRECT/INCORRECT/UNKNOWN
        """
        pass
        
    def generate_report(
        self, 
        df: pd.DataFrame, 
        metrics: Dict[str, float], 
        format: str = "markdown"
    ) -> str:
        """
        Generates a comprehensive report with all metrics and sample issues.
        """
        pass
        
    def log_to_wandb(self, results: Dict, article_title: str):
        """
        Logs evaluation results to Weights & Biases.
        Creates tables, plots, and aggregated metrics.
        """
        pass
```

## Implementation Stages

### Phase 1: Core Functionality
1. Implement Wikipedia API integration for reference article collection
2. Build basic article preprocessing and line numbering
3. Create initial OpenAI prompt design and evaluation framework
4. Develop simple results parser and basic metrics

### Phase 2: Enhanced Evaluation
1. Improve prompt engineering based on initial results
2. Add support for chunking large articles
3. Implement more sophisticated content preprocessing
4. Expand evaluation categories and granularity

### Phase 3: Reporting and Integration
1. Develop comprehensive metrics and visualizations
2. Create formatted reports (Markdown, HTML, PDF)
3. Integrate with Weights & Biases for experiment tracking
4. Build batch processing for multiple articles

## Prompt Design (Initial Draft)

```
You are an expert fact-checker comparing an AI-generated article with a reference Wikipedia article.

# Classification Criteria
- CORRECT: The statement is accurate and verifiable in the reference article
- INCORRECT: The statement contradicts information in the reference article
- UNKNOWN: The reference doesn't mention this information or provides insufficient details to verify

# Output Format
You must produce valid YAML with this exact structure for each numbered line:
1:
  analysis: "Brief analysis of line 1"
  accuracy: "CORRECT|INCORRECT|UNKNOWN"
2:
  analysis: "Brief analysis of line 2"
  accuracy: "CORRECT|INCORRECT|UNKNOWN"
...

# REFERENCE ARTICLE:
{wiki_content}

# AI-GENERATED ARTICLE (NUMBERED LINES):
{numbered_ai_content}
```

## Integration with Existing Environment

This evaluation system will complement the existing `WikipediaArticleCreatorEnv` by:

1. Adding a new method to the environment for article evaluation
2. Extending the wandb logging to include evaluation metrics
3. Creating an evaluation pipeline that can be run as part of the training process or independently

```python
# Addition to WikipediaArticleCreatorEnv class
async def evaluate_article_quality(
    self, 
    topic: str,
    generated_article: str
) -> Dict:
    """
    Evaluates an AI-generated article against the Wikipedia reference.
    Returns detailed quality metrics and factual accuracy assessment.
    """
    # Implementation will use the modules described above
    pass
```

## Questions and Considerations

1. **API Rate Limiting**: How should we handle Wikipedia API rate limits for fetching reference articles?
2. **Cost Management**: What strategies can we implement to minimize OpenAI API costs while maintaining evaluation quality?
3. **Baseline Establishment**: Should we evaluate multiple AI models to establish a quality baseline?
4. **Evaluation Scope**: Should we evaluate only factual accuracy, or expand to other dimensions like clarity, structure, and neutrality?
5. **Handling Missing References**: How should we score statements that can't be verified because the reference article lacks coverage on that specific aspect?

## Next Steps

1. Create a prototype implementation of the Wikipedia reference collector
2. Develop and test the article preprocessing module
3. Design and test the initial evaluation prompt with a small sample of articles
4. Implement basic results processing and metrics calculation
5. Integrate with the existing environment for initial testing

## Resources Needed

1. OpenAI API access with sufficient quota for model calls
2. Wikipedia API access (no authentication required for basic access)
3. Development environment with required Python packages
   - openai
   - pandas
   - wandb
   - pyyaml
   - aiohttp
   - wikipediaapi