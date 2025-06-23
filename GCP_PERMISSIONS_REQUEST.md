# 🔐 Google Cloud Permissions Request for AMIEN Deployment

## Request Summary
**Project**: `amien-research-pipeline`
**User**: `carlos@raxverse.com`
**Purpose**: Deploy AMIEN AI Research System to production

## Required Permissions

### 1. Cloud Scheduler API Access
**Service**: `scheduler.googleapis.com`
**Required Role**: `Cloud Scheduler Admin` or `Editor`
**Reason**: Needed for automated daily/weekly research generation cycles

### 2. Additional Roles Needed
- **Cloud Run Admin**: Deploy API services
- **Compute Engine Admin**: Run large-scale VR experiments
- **Storage Admin**: Store research artifacts
- **Service Account Admin**: Create service accounts for automation

## Current Status
✅ **Billing**: Linked and active (£402 credits)
✅ **Project**: Created (`amien-research-pipeline`)
✅ **Basic APIs**: Enabled (Cloud Run, Storage, Compute)
❌ **Cloud Scheduler**: Permission denied

## Business Justification
- **AMIEN System**: AI research generation platform for VR affordances
- **Expected Value**: Automated research papers and optimization functions
- **Cost**: $50-200/month for testing, scales to $2-5K for production
- **Timeline**: Ready to deploy immediately once permissions granted

## Technical Details
- **Current Error**: `PERMISSION_DENIED: Permission denied to enable service [scheduler.googleapis.com]`
- **Help Token**: `AeNz4PiEtHJa1kv6CkQdt1FdYin4DcmIaZog1-nrfTWNzqP651GYjtvAMXqxnRj74mowbsLk89gnzjNazJNh3BK8H9zA4ZZNXNKwSBXF_m72xlaz`
- **Account**: `carlos@raxverse.com`

## What Admin Needs to Do

### Option 1: Grant Specific Permissions
```bash
# Grant Cloud Scheduler permissions
gcloud projects add-iam-policy-binding amien-research-pipeline \
    --member="user:carlos@raxverse.com" \
    --role="roles/cloudscheduler.admin"

# Grant additional required roles
gcloud projects add-iam-policy-binding amien-research-pipeline \
    --member="user:carlos@raxverse.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding amien-research-pipeline \
    --member="user:carlos@raxverse.com" \
    --role="roles/compute.admin"
```

### Option 2: Grant Editor Role (Simpler)
```bash
gcloud projects add-iam-policy-binding amien-research-pipeline \
    --member="user:carlos@raxverse.com" \
    --role="roles/editor"
```

### Option 3: Enable API Directly
```bash
gcloud services enable scheduler.googleapis.com \
    --project=amien-research-pipeline
```

## Verification Commands
After permissions are granted, verify with:
```bash
gcloud services enable scheduler.googleapis.com
gcloud scheduler jobs list
```

## Security Considerations
- **Principle of Least Privilege**: Only requesting permissions needed for AMIEN
- **Project Scope**: Permissions limited to `amien-research-pipeline` project
- **Monitoring**: All actions logged in Cloud Audit Logs
- **Budget Controls**: Billing alerts set at £100, £300, £500

## Expected Deployment Timeline
- **Permissions granted**: 0-24 hours (admin action)
- **Deployment completion**: 10-15 minutes after permissions
- **First research generation**: Within 1 hour of deployment
- **Full production**: 24-48 hours for scaling verification

---

## Contact Information
**Requestor**: Carlos
**Email**: carlos@raxverse.com
**Project**: AMIEN AI Research System
**Urgency**: Medium (system ready, waiting on permissions)

## Next Steps
1. Admin grants permissions using commands above
2. Carlos runs: `./deploy_ai_integration.sh`
3. Verify deployment with health checks
4. Begin automated research generation

**Note**: System is 85% deployed and fully functional locally. This is the final step to go live.
