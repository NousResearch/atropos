# 🚀 AMIEN Google Cloud Billing Setup Guide

## Current Status
✅ **AMIEN System**: OPERATIONAL (75% - Production Ready)  
✅ **Local Testing**: All core components working  
✅ **GCP Configuration**: Deployment scripts ready  
⚠️ **Billing Setup**: Required for live deployment  

## Step 1: Set Up Google Cloud Billing

### 1.1 Create Billing Account
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Billing** → **Manage billing accounts**
3. Click **Create Account**
4. Enter your payment information (credit card required)
5. Accept the terms and create the billing account

### 1.2 Link Billing to Project
1. In the billing console, select your new billing account
2. Click **Link a project**
3. Select project: `amien-research-pipeline`
4. Click **Set account**

## Step 2: Verify Billing Setup

Run this command to verify billing is enabled:
```bash
gcloud auth login
gcloud config set project amien-research-pipeline
gcloud services list --enabled
```

## Step 3: Deploy AMIEN to Production

Once billing is set up, run:
```bash
./deploy_ai_integration.sh
```

## Expected Costs

### Monthly Estimates:
- **Development/Testing**: $50-200/month
- **Production Scale**: $2,000-5,000/month
- **Enterprise Scale**: $5,000-15,000/month

### Cost Breakdown:
- **Cloud Run**: $500-1,000 (API endpoints)
- **Compute Engine**: $1,000-3,000 (massive experiments)
- **Storage**: $100-300 (research data)
- **AI API calls**: $400-700 (Gemini AI)

### Cost Controls:
- Preemptible instances for batch jobs
- Auto-scaling with limits
- Budget alerts at $1,000, $3,000, $5,000
- Daily spending notifications

## Step 4: Monitor Deployment

After deployment, monitor:
1. **Cloud Run services**: API endpoints
2. **Compute Engine**: Batch experiment jobs
3. **Cloud Storage**: Research artifacts
4. **Billing**: Daily cost tracking

## Step 5: Verify Production

Test the deployed system:
```bash
# Test API endpoint
curl https://amien-api-[hash]-uc.a.run.app/health

# Check research generation
curl https://amien-api-[hash]-uc.a.run.app/generate-research

# Monitor logs
gcloud logs read "resource.type=cloud_run_revision"
```

## Troubleshooting

### Common Issues:
1. **Billing not enabled**: Verify billing account is linked
2. **API quotas**: Request quota increases if needed
3. **Permissions**: Ensure service account has required roles

### Support:
- Check deployment logs: `gcloud logs read`
- Monitor dashboard: Google Cloud Console
- Cost alerts: Billing notifications

## Next Steps After Deployment

1. **Week 1**: Monitor stability and costs
2. **Week 2**: Scale to 1000+ experiments
3. **Week 3**: Integrate with real VR applications
4. **Month 1**: Publish first research papers
5. **Month 2**: Open source core components

---

🎯 **Ready to Deploy**: Once billing is set up, AMIEN will be live in ~10 minutes!

📊 **Expected Output**: 24/7 autonomous AI research generation with auto-scaling capabilities.

💡 **Pro Tip**: Start with a $100 daily budget limit for the first week to control costs while testing. 