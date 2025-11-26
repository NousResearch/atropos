variable "anthropic_api_key" {
  type = string
}

variable "anthropic_model" {
  type    = string
  default = "claude-sonnet-4-5-20250929"
}

variable "profile_key" {
  type    = string
  default = "rust"
}

variable "bootstrap_script" {
  type = string
}

variable "workspace_root" {
  type = string
}

variable "cline_src_dir" {
  type    = string
  default = "/tmp/nous-cline-worker"
}

variable "task_env_json" {
  type    = string
  default = "{}"
  description = "JSON-encoded environment variables for the task"
}

variable "job_name" {
  type    = string
  default = "cline-worker"
  description = "Unique job name for this worker instance"
}

variable "worker_id" {
  type    = string
  default = ""
  description = "Unique identifier for this worker instance"
}

variable "profile_dir" {
  type    = string
  default = ""
  description = "Path to Nix profile flake directory"
}

variable "atropos_root" {
  type    = string
  default = "/Users/shannon/Workspace/Nous/atropos"
}

job "cline-worker" {
  name        = var.job_name
  datacenters = ["dc1"]
  type        = "batch"

  meta {
    profile_key = var.profile_key
    worker_id   = var.worker_id
  }

  group "worker" {
    count = 1

    network {
      # Dynamic port allocation - Nomad assigns from ephemeral range
      port "protobus" {}
      port "hostbridge" {}
    }

    task "cline" {
      driver = "raw_exec"

      env = {
        CLINE_SRC_DIR          = var.cline_src_dir
        WORKSPACE_ROOT         = var.workspace_root
        TASK_BOOTSTRAP_SCRIPT  = var.bootstrap_script
        PROTOBUS_PORT          = "${NOMAD_PORT_protobus}"
        HOSTBRIDGE_PORT        = "${NOMAD_PORT_hostbridge}"
        ANTHROPIC_API_KEY      = var.anthropic_api_key
        ANTHROPIC_MODEL        = var.anthropic_model
        NODE_OPTIONS           = "--max-old-space-size=4096"
        CLINE_PROFILE_KEY      = var.profile_key
        # Task-specific environment will be injected via wrapper script
        TASK_ENV_JSON          = var.task_env_json
        PROFILE_DIR            = var.profile_dir
        ATROPOS_ROOT           = var.atropos_root
      }

      config {
        command = "${var.atropos_root}/environments/cline_env/cline_dev/nomad_worker_wrapper.sh"
        args    = []
      }

      resources {
        cpu    = 2000
        memory = 8192
      }
    }
  }
}
