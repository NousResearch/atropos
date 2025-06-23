#!/bin/bash
set -e

echo "📊 Setting up monitoring and alerting for AMIEN"

# Enable required APIs
echo "📡 Enabling monitoring APIs..."
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable clouderrorreporting.googleapis.com

# Create notification channel (email)
echo "📧 Creating notification channel..."
NOTIFICATION_CHANNEL=$(gcloud alpha monitoring channels create \
    --display-name="AMIEN Admin Alerts" \
    --type=email \
    --channel-labels=email_address=carlos@raxverse.com \
    --description="Email notifications for AMIEN system alerts" \
    --format="value(name)" 2>/dev/null || echo "")

if [ -z "$NOTIFICATION_CHANNEL" ]; then
    echo "⚠️  Notification channel may already exist, continuing..."
    NOTIFICATION_CHANNEL=$(gcloud alpha monitoring channels list \
        --filter="displayName:'AMIEN Admin Alerts'" \
        --format="value(name)" | head -1)
fi

echo "📧 Notification channel: $NOTIFICATION_CHANNEL"

# Create alert policy for API errors
echo "🚨 Creating API error alert policy..."
cat > /tmp/api_error_policy.yaml << 'EOF'
displayName: "AMIEN API High Error Rate"
documentation:
  content: "AMIEN API is experiencing high error rates"
  mimeType: "text/markdown"
conditions:
- displayName: "High error rate condition"
  conditionThreshold:
    filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="amien-api-service"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 0.1
    duration: 300s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_RATE
      crossSeriesReducer: REDUCE_MEAN
      groupByFields:
      - resource.labels.service_name
combiner: OR
enabled: true
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/api_error_policy.yaml \
    --notification-channels="$NOTIFICATION_CHANNEL" || echo "API error policy may already exist"

# Create alert policy for research generation failures
echo "🔬 Creating research generation failure alert..."
cat > /tmp/research_failure_policy.yaml << 'EOF'
displayName: "AMIEN Research Generation Failures"
documentation:
  content: "AMIEN research generation is failing"
  mimeType: "text/markdown"
conditions:
- displayName: "Research generation failure condition"
  conditionThreshold:
    filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="amien-api-service" AND textPayload:"Research generation failed"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 0
    duration: 60s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_COUNT
      crossSeriesReducer: REDUCE_SUM
combiner: OR
enabled: true
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/research_failure_policy.yaml \
    --notification-channels="$NOTIFICATION_CHANNEL" || echo "Research failure policy may already exist"

# Create alert policy for high memory usage
echo "💾 Creating memory usage alert..."
cat > /tmp/memory_usage_policy.yaml << 'EOF'
displayName: "AMIEN High Memory Usage"
documentation:
  content: "AMIEN API is using high memory"
  mimeType: "text/markdown"
conditions:
- displayName: "High memory usage condition"
  conditionThreshold:
    filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="amien-api-service" AND metric.type="run.googleapis.com/container/memory/utilizations"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 0.85
    duration: 300s
    aggregations:
    - alignmentPeriod: 60s
      perSeriesAligner: ALIGN_MEAN
      crossSeriesReducer: REDUCE_MEAN
      groupByFields:
      - resource.labels.service_name
combiner: OR
enabled: true
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/memory_usage_policy.yaml \
    --notification-channels="$NOTIFICATION_CHANNEL" || echo "Memory usage policy may already exist"

# Create alert policy for scheduler job failures
echo "⏰ Creating scheduler failure alert..."
cat > /tmp/scheduler_failure_policy.yaml << 'EOF'
displayName: "AMIEN Scheduler Job Failures"
documentation:
  content: "AMIEN scheduled jobs are failing"
  mimeType: "text/markdown"
conditions:
- displayName: "Scheduler failure condition"
  conditionThreshold:
    filter: 'resource.type="cloud_scheduler_job" AND metric.type="cloudscheduler.googleapis.com/job/num_failed_attempts"'
    comparison: COMPARISON_GREATER_THAN
    thresholdValue: 2
    duration: 300s
    aggregations:
    - alignmentPeriod: 300s
      perSeriesAligner: ALIGN_SUM
      crossSeriesReducer: REDUCE_SUM
      groupByFields:
      - resource.labels.job_id
combiner: OR
enabled: true
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/scheduler_failure_policy.yaml \
    --notification-channels="$NOTIFICATION_CHANNEL" || echo "Scheduler failure policy may already exist"

# Create custom dashboard
echo "📈 Creating AMIEN monitoring dashboard..."
cat > /tmp/amien_dashboard.json << 'EOF'
{
  "displayName": "AMIEN Research Pipeline Dashboard",
  "mosaicLayout": {
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "API Request Rate",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"amien-api-service\" AND metric.type=\"run.googleapis.com/request_count\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_RATE",
                      "crossSeriesReducer": "REDUCE_SUM"
                    }
                  }
                },
                "plotType": "LINE"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Requests/sec",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "widget": {
          "title": "API Response Latency",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"amien-api-service\" AND metric.type=\"run.googleapis.com/request_latencies\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_DELTA",
                      "crossSeriesReducer": "REDUCE_PERCENTILE_95"
                    }
                  }
                },
                "plotType": "LINE"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Latency (ms)",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "yPos": 4,
        "widget": {
          "title": "Memory Utilization",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"amien-api-service\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN",
                      "crossSeriesReducer": "REDUCE_MEAN"
                    }
                  }
                },
                "plotType": "LINE"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Memory %",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 6,
        "height": 4,
        "xPos": 6,
        "yPos": 4,
        "widget": {
          "title": "CPU Utilization",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"amien-api-service\" AND metric.type=\"run.googleapis.com/container/cpu/utilizations\"",
                    "aggregation": {
                      "alignmentPeriod": "60s",
                      "perSeriesAligner": "ALIGN_MEAN",
                      "crossSeriesReducer": "REDUCE_MEAN"
                    }
                  }
                },
                "plotType": "LINE"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "CPU %",
              "scale": "LINEAR"
            }
          }
        }
      },
      {
        "width": 12,
        "height": 4,
        "yPos": 8,
        "widget": {
          "title": "Scheduler Job Success Rate",
          "xyChart": {
            "dataSets": [
              {
                "timeSeriesQuery": {
                  "timeSeriesFilter": {
                    "filter": "resource.type=\"cloud_scheduler_job\" AND metric.type=\"cloudscheduler.googleapis.com/job/num_attempts\"",
                    "aggregation": {
                      "alignmentPeriod": "3600s",
                      "perSeriesAligner": "ALIGN_SUM",
                      "crossSeriesReducer": "REDUCE_SUM",
                      "groupByFields": ["resource.labels.job_id"]
                    }
                  }
                },
                "plotType": "STACKED_BAR"
              }
            ],
            "timeshiftDuration": "0s",
            "yAxis": {
              "label": "Job Attempts",
              "scale": "LINEAR"
            }
          }
        }
      }
    ]
  }
}
EOF

gcloud monitoring dashboards create --config-from-file=/tmp/amien_dashboard.json || echo "Dashboard may already exist"

# Clean up temporary files
rm -f /tmp/*_policy.yaml /tmp/amien_dashboard.json

echo "✅ Monitoring setup complete!"
echo ""
echo "📊 Monitoring Features Configured:"
echo "  • API error rate alerts (>10% error rate)"
echo "  • Research generation failure alerts"
echo "  • High memory usage alerts (>85%)"
echo "  • Scheduler job failure alerts (>2 failures)"
echo "  • Custom AMIEN dashboard with key metrics"
echo ""
echo "📧 Alerts will be sent to: carlos@raxverse.com"
echo ""
echo "🔗 Access your dashboard:"
echo "  https://console.cloud.google.com/monitoring/dashboards"
echo ""
echo "📋 View alert policies:"
echo "  gcloud alpha monitoring policies list"
