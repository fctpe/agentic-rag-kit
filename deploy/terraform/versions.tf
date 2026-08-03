terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

# No provider block, and no backend block. This is a module: the caller
# configures the provider (region, assumed role) and where state lives, which
# is also why `terraform init -backend=false && terraform validate` works here
# with no credentials — the CI check in .github/workflows/k8s.yml.
