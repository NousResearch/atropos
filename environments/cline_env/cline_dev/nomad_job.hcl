variable "anthropic_api_key" {
  type = string
}

variable "anthropic_model" {
  type    = string
  default = "claude-sonnet-4-5-20250929"
}

job "cline-smoke" {
  datacenters = ["dc1"]
  type        = "batch"

  group "worker" {
    count = 1

    network {
      port "protobus" {
        static = 46040
      }
      port "hostbridge" {
        static = 46041
      }
    }

    task "cline" {
      driver = "raw_exec"

      env = {
        CLINE_SRC_DIR         = "/tmp/nous-cline"
        WORKSPACE_ROOT        = "/tmp/ratatui-workspace"
        TASK_BOOTSTRAP_SCRIPT = "/Users/shannon/Workspace/Nous/atropos/environments/cline_env/cline_dev/examples/ratatui_vertical_gauge/bootstrap.sh"
        PROTOBUS_PORT         = "${NOMAD_PORT_protobus}"
        HOSTBRIDGE_PORT       = "${NOMAD_PORT_hostbridge}"
        ANTHROPIC_API_KEY     = "${var.anthropic_api_key}"
        ANTHROPIC_MODEL       = "${var.anthropic_model}"
        NODE_OPTIONS          = "--max-old-space-size=4096"
      }

      config {
        command = "/Users/shannon/Workspace/Nous/atropos/environments/cline_env/cline_dev/bootstrap_cline_worker.sh"
        args    = []
      }

      resources {
        cpu    = 2000
        memory = 8192
      }
    }
  }
}
