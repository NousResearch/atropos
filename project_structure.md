# Project Structure

This document outlines the organization and purpose of each file in the 24/7 AI Research Pipeline project.

## Root Directory

```
research-pipeline/
├── app.py                    # FastAPI application entry point
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container configuration
├── docker-compose.yml       # Docker services configuration
├── deploy.sh               # Deployment automation script
├── .env                    # Environment variables (not in git)
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
└── project_structure.md   # This file
```

## Core Components

```
research-pipeline/
├── production_research_pipeline.py  # Main pipeline orchestration
├── bigquery_manager.py             # BigQuery data management
├── paper_generator.py              # Research paper generation
└── simple_padres_research.py       # Padres API integration
```

## Testing

```
research-pipeline/
├── test_pipeline.py               # Main test suite
└── tests/                         # Additional test files
    ├── __init__.py
    ├── test_bigquery.py
    ├── test_paper_generator.py
    └── test_padres.py
```

## Generated Content

```
research-pipeline/
├── papers/                        # Generated research papers
│   └── .gitkeep
└── logs/                          # Application logs
    └── app.log
```

## File Descriptions

### Core Application Files

- **app.py**
  - FastAPI application setup
  - API endpoint definitions
  - Request/response models
  - Error handling

- **production_research_pipeline.py**
  - Main pipeline orchestration
  - Experiment execution logic
  - Integration with AI services
  - Error handling and retries

- **bigquery_manager.py**
  - BigQuery connection management
  - Data storage operations
  - Query execution
  - Schema management

- **paper_generator.py**
  - Research paper generation
  - Template management
  - Data formatting
  - File output handling

- **simple_padres_research.py**
  - Padres API integration
  - Spatial data collection
  - API authentication
  - Response parsing

### Configuration Files

- **requirements.txt**
  - Python package dependencies
  - Version specifications
  - Development dependencies

- **Dockerfile**
  - Container base image
  - Dependencies installation
  - Application setup
  - Environment configuration

- **docker-compose.yml**
  - Service definitions
  - Volume mappings
  - Environment variables
  - Network configuration

- **deploy.sh**
  - Deployment automation
  - Environment checks
  - Docker operations
  - Health verification

### Testing Files

- **test_pipeline.py**
  - Main test suite
  - API endpoint testing
  - Integration testing
  - BigQuery operations testing

- **tests/test_bigquery.py**
  - BigQuery manager tests
  - Data storage tests
  - Query execution tests

- **tests/test_paper_generator.py**
  - Paper generation tests
  - Template tests
  - Output validation

- **tests/test_padres.py**
  - Padres API integration tests
  - Authentication tests
  - Response parsing tests

### Generated Directories

- **papers/**
  - Directory for generated research papers
  - Papers are named with timestamps
  - Markdown format

- **logs/**
  - Application logs
  - Error logs
  - Debug information

## Key Dependencies

- **FastAPI**: Web framework
- **uvicorn**: ASGI server
- **google-cloud-bigquery**: BigQuery client
- **anthropic**: Claude API client
- **perplexity**: Perplexity API client
- **python-dotenv**: Environment management
- **pytest**: Testing framework

## Environment Variables

See `.env.example` for required environment variables and their descriptions.

## Development Workflow

1. Make changes in feature branch
2. Run tests: `python test_pipeline.py`
3. Build Docker image: `./deploy.sh build`
4. Test locally: `docker-compose up`
5. Deploy: `./deploy.sh deploy`

## Deployment Architecture

```
[Client] → [Load Balancer] → [Docker Container]
                              ├── FastAPI App
                              ├── Research Pipeline
                              └── BigQuery Client
```

## Monitoring Points

- Application logs in `logs/app.log`
- Docker container logs
- BigQuery query logs
- API endpoint health checks
- System metrics (CPU, Memory, Disk) 