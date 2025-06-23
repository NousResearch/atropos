# 🚀 AMIEN Production Deployment Checklist

## Pre-Deployment (Complete these first)

### 1. Environment Setup
- [ ] Update API keys in `production.env`
- [ ] Set GCP_PROJECT_ID: `amien-research-pipeline`
- [ ] Verify GCP billing is enabled
- [ ] Install gcloud CLI and authenticate

### 2. API Keys Required
- [ ] Google Gemini API key
- [ ] OpenAI API key (optional)
- [ ] Perplexity API key (optional)

### 3. GCP Services to Enable
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable scheduler.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable monitoring.googleapis.com
```

## Deployment Steps

### 1. Load Environment
```bash
source production.env
```

### 2. Run System Tests
```bash
python3 final_production_deployment.py --test-only
```

### 3. Deploy to GCP
```bash
./deploy_amien.sh
```

### 4. Verify Deployment
- [ ] Check Cloud Run services are running
- [ ] Verify API endpoints respond
- [ ] Test research generation
- [ ] Check scheduled jobs

### 5. Monitor System
- [ ] Set up monitoring dashboard
- [ ] Configure alerts
- [ ] Monitor costs
- [ ] Check logs

## Post-Deployment

### 1. First Research Cycle
- [ ] Trigger manual research generation
- [ ] Verify paper quality
- [ ] Check function discovery
- [ ] Monitor performance

### 2. Scaling Test
- [ ] Run massive scale experiment
- [ ] Monitor auto-scaling
- [ ] Check cost controls
- [ ] Verify data storage

### 3. Continuous Operation
- [ ] Daily research papers generating
- [ ] Weekly massive experiments running
- [ ] Monitoring alerts working
- [ ] Cost within budget

## Emergency Contacts
- GCP Support: https://cloud.google.com/support
- AMIEN Issues: Check logs in Cloud Logging
- Cost Alerts: Monitor billing dashboard

## Success Metrics
- ✅ Research papers: 1+ per day
- ✅ Experiments: 1000+ per week
- ✅ Uptime: >99.5%
- ✅ Cost: <$5000/month
- ✅ Quality: >85/100 average

Generated: 2025-05-27T23:10:19.003686
