# CloudVR-PerfGuard

**Automated Performance Regression Detection for VR Applications**

CloudVR-PerfGuard is a cloud-based service that automatically detects performance regressions in VR applications by running comprehensive performance tests across different GPU configurations and comparing results against baseline builds.

## 🎯 Key Features

- **Automated Performance Testing**: Run VR applications across multiple GPU types (T4, L4, A100) in containerized environments
- **Regression Detection**: Statistical analysis to identify performance regressions with configurable thresholds
- **VR-Specific Metrics**: Frame time consistency, VR comfort scores, motion-to-photon latency
- **Cloud-Native**: Built for scalable cloud deployment with container orchestration
- **RESTful API**: Easy integration into CI/CD pipelines
- **Detailed Reports**: Comprehensive HTML and JSON reports with actionable recommendations

## 🏗️ Architecture

CloudVR-PerfGuard is built on the proven AMIEN infrastructure, adapted for VR performance testing:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Performance     │    │  Container      │
│   Web Service   │───▶│  Tester          │───▶│  Manager        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Database      │    │  Regression      │    │  GPU Monitor    │
│   Manager       │    │  Detector        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

- **API Layer**: FastAPI-based REST API for build submission and report retrieval
- **Performance Tester**: Orchestrates parallel VR performance tests across GPU configurations
- **Regression Detector**: Statistical analysis engine for identifying performance regressions
- **Container Manager**: Docker-based isolation for VR application testing
- **GPU Monitor**: Real-time collection of GPU metrics during testing
- **Database Manager**: SQLite-based storage for test results and analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker with GPU support (nvidia-docker)
- NVIDIA GPU with recent drivers
- 8GB+ RAM recommended

### Installation

1. **Clone and setup the project:**
```bash
cd cloudvr_perfguard
pip install -r requirements.txt
```

2. **Run functionality tests:**
```bash
python main.py --test
```

3. **Start the API server:**
```bash
python main.py --dev  # Development mode
# or
python main.py        # Production mode
```

4. **Access the API documentation:**
Open http://localhost:8000/docs in your browser

## 📖 Usage

### 1. Submit a Baseline Build

First, establish a baseline for your VR application:

```bash
curl -X POST "http://localhost:8000/submit_build" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@my_vr_app.zip" \
  -F "app_name=MyVRGame" \
  -F "build_version=v1.0.0" \
  -F "platform=windows" \
  -F "submission_type=baseline"
```

### 2. Submit a Build for Regression Testing

Test a new build against the baseline:

```bash
curl -X POST "http://localhost:8000/submit_build" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@my_vr_app_v1.1.0.zip" \
  -F "app_name=MyVRGame" \
  -F "build_version=v1.1.0" \
  -F "platform=windows" \
  -F "submission_type=regression_test" \
  -F "baseline_version=v1.0.0"
```

### 3. Check Test Status

```bash
curl "http://localhost:8000/job_status/{job_id}"
```

### 4. Get Regression Report

```bash
# JSON report
curl "http://localhost:8000/regression_report/{job_id}"

# HTML report
curl "http://localhost:8000/regression_report/{job_id}/html"
```

## 🔧 Configuration

### Test Configuration

Customize performance tests by providing a test configuration:

```json
{
  "gpu_types": ["T4", "L4"],
  "test_duration_seconds": 60,
  "test_scenes": ["main_menu", "gameplay_scene"]
}
```

### Regression Thresholds

The system uses configurable thresholds for regression detection:

- **FPS Regression**: 5% decrease in average FPS
- **Frame Time Regression**: 10% increase in 99th percentile frame time
- **VR Comfort**: 10% decrease in comfort score
- **VRAM Usage**: 15% increase in peak VRAM

## 📊 Performance Metrics

CloudVR-PerfGuard tracks VR-specific performance metrics:

### Frame Rate Metrics
- Average, minimum, maximum FPS
- 1st and 99th percentile FPS
- FPS standard deviation

### Frame Time Metrics (Critical for VR)
- Average frame time
- 99th percentile frame time
- Frame time consistency

### VR-Specific Metrics
- Dropped frames count
- Reprojected frames count
- Motion-to-photon latency
- VR comfort score (0-100)

### System Metrics
- GPU utilization and temperature
- VRAM usage
- CPU utilization

## 🎯 VR Performance Grading

The system assigns performance grades based on VR standards:

- **Grade A**: ≥90 FPS average, ≥80 FPS minimum (Excellent VR)
- **Grade B**: ≥80 FPS average, ≥70 FPS minimum (Good VR)
- **Grade C**: ≥70 FPS average, ≥60 FPS minimum (Acceptable VR)
- **Grade D**: ≥60 FPS average, ≥45 FPS minimum (Poor VR)
- **Grade F**: <60 FPS average or <45 FPS minimum (Unacceptable VR)

## 🔍 Regression Analysis

### Statistical Analysis
- T-test for statistical significance
- Cohen's d for effect size measurement
- 95% confidence intervals

### Regression Severity Levels
- **Critical**: Performance below VR minimum thresholds
- **Major**: Significant performance degradation (>15%)
- **Minor**: Noticeable performance changes (5-15%)
- **Info**: Small changes within acceptable range

### Actionable Recommendations
The system provides specific recommendations based on detected regressions:
- GPU profiling suggestions
- Memory optimization guidance
- VR comfort improvement tips

## 🚀 Deployment

### Local Development
```bash
python main.py --dev --port 8000
```

### Docker Deployment
```bash
# Build container
docker build -t cloudvr-perfguard .

# Run with GPU support
docker run --gpus all -p 8000:8000 cloudvr-perfguard
```

### Cloud Deployment (Google Cloud)
```bash
# Deploy to Cloud Run with GPU support
gcloud run deploy cloudvr-perfguard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 4 \
  --gpu 1 \
  --gpu-type nvidia-tesla-t4
```

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/
```

### Run Integration Tests
```bash
python main.py --test
```

### Load Testing
```bash
# Test with multiple concurrent builds
python tests/load_test.py
```

## 📈 Monitoring and Observability

### Health Checks
- `/status` - API and service health
- `/metrics` - Prometheus-compatible metrics
- Container health monitoring

### Logging
- Structured logging with correlation IDs
- Performance test execution logs
- Regression analysis audit trail

## 🔮 Future Enhancements

### Phase 2: AI/RL Integration
- **Adversarial Workload Generation**: RL agents to discover performance bottlenecks
- **Predictive Regression Modeling**: ML models to predict regressions from code changes
- **Automated Root Cause Analysis**: AI-driven analysis of performance issues

### Phase 3: Advanced Features
- Multi-platform testing (Windows, Linux, Android)
- Real VR headset integration
- Advanced shader analysis
- Performance optimization suggestions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on the proven AMIEN infrastructure
- Inspired by the committee's guidance on practical VR performance testing
- Leverages cloud-native patterns for scalable testing

---

**CloudVR-PerfGuard**: Making VR performance regression testing as smooth as a 90 FPS experience! 🥽✨ 