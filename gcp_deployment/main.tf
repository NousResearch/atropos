
# AMIEN Infrastructure Configuration
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = "amien-research-pipeline"
  region  = "us-central1"
}

# Cloud Storage buckets
resource "google_storage_bucket" "research_data" {
  name     = "amien-research-data"
  location = "us-central1"
}

resource "google_storage_bucket" "experiment_results" {
  name     = "amien-experiment-results"
  location = "us-central1"
}

resource "google_storage_bucket" "ai_models" {
  name     = "amien-ai-models"
  location = "us-central1"
}

resource "google_storage_bucket" "user_data" {
  name     = "amien-user-data"
  location = "us-central1"
}

# Secret Manager secrets
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"
}

resource "google_secret_manager_secret" "openai_api_key" {
  secret_id = "openai-api-key"
}

# Cloud Run services
resource "google_cloud_run_service" "api_service" {
  name     = "amien-api-service"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "gcr.io/amien-research-pipeline/amien-api-service:latest"

        env {
          name  = "PROJECT_ID"
          value = "amien-research-pipeline"
        }

        env {
          name = "GEMINI_API_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.gemini_api_key.secret_id
              key  = "latest"
            }
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# IAM policy for Cloud Run
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.api_service.name
  location = google_cloud_run_service.api_service.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Compute Engine instance template for massive scale
resource "google_compute_instance_template" "massive_scale" {
  name_prefix  = "amien-massive-scale-"
  machine_type = "c2-standard-16"

  disk {
    source_image = "debian-cloud/debian-11"
    auto_delete  = true
    boot         = true
    disk_size_gb = 100
  }

  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = file("startup-script.sh")

  lifecycle {
    create_before_destroy = true
  }
}

# Managed instance group for auto-scaling
resource "google_compute_region_instance_group_manager" "massive_scale" {
  name   = "amien-massive-scale-group"
  region = "us-central1"

  version {
    instance_template = google_compute_instance_template.massive_scale.id
  }

  base_instance_name = "amien-massive-scale"
  target_size        = 0  # Start with 0, scale up when needed

  auto_healing_policies {
    health_check      = google_compute_health_check.massive_scale.id
    initial_delay_sec = 300
  }
}

# Health check
resource "google_compute_health_check" "massive_scale" {
  name = "amien-massive-scale-health-check"

  timeout_sec        = 5
  check_interval_sec = 10

  tcp_health_check {
    port = "22"
  }
}

# Auto-scaler
resource "google_compute_region_autoscaler" "massive_scale" {
  name   = "amien-massive-scale-autoscaler"
  region = "us-central1"
  target = google_compute_region_instance_group_manager.massive_scale.id

  autoscaling_policy {
    max_replicas    = 100
    min_replicas    = 0
    cooldown_period = 300

    cpu_utilization {
      target = 0.8
    }
  }
}

# Cloud Scheduler jobs
resource "google_cloud_scheduler_job" "daily_research" {
  name     = "daily-research"
  schedule = "0 2 * * *"
  region   = "us-central1"

  http_target {
    uri         = "${google_cloud_run_service.api_service.status[0].url}/research/generate"
    http_method = "POST"
  }
}

resource "google_cloud_scheduler_job" "weekly_massive_experiments" {
  name     = "weekly-massive-experiments"
  schedule = "0 0 * * 1"
  region   = "us-central1"

  http_target {
    uri         = "${google_cloud_run_service.api_service.status[0].url}/experiments/massive"
    http_method = "POST"
    body        = base64encode(jsonencode({sample_size = 5000}))
  }
}

# Outputs
output "api_service_url" {
  value = google_cloud_run_service.api_service.status[0].url
}

output "storage_buckets" {
  value = {
    research_data      = google_storage_bucket.research_data.name
    experiment_results = google_storage_bucket.experiment_results.name
    ai_models         = google_storage_bucket.ai_models.name
    user_data         = google_storage_bucket.user_data.name
  }
}
