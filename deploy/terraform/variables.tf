variable "name_prefix" {
  description = "Prefix for every resource name in this module."
  type        = string
  default     = "ragkit"
}

variable "tags" {
  description = "Tags applied to every taggable resource."
  type        = map(string)
  default     = {}
}

# --- Network placement -------------------------------------------------------
# Required, not created here. A module that builds its own VPC is a module you
# cannot drop into an existing account, and the network is not one of "the
# managed pieces" this owns.

variable "db_subnet_group_name" {
  description = "Existing DB subnet group. Use private subnets — this instance is not publicly accessible."
  type        = string
}

variable "vpc_security_group_ids" {
  description = "Security groups for the instance. They decide who reaches 5432; nothing here opens it."
  type        = list(string)
}

# --- Postgres ----------------------------------------------------------------

variable "engine_version" {
  description = "Major version only, so AWS applies minor upgrades in the maintenance window. pgvector needs 15 or newer."
  type        = string
  default     = "16"

  validation {
    condition     = tonumber(split(".", var.engine_version)[0]) >= 15
    error_message = "pgvector ships with RDS PostgreSQL 15 and newer; the first migration runs CREATE EXTENSION vector."
  }
}

variable "instance_class" {
  description = "Starting point, not a sizing recommendation — this repo ships no load test. HNSW index builds are the memory-hungry part."
  type        = string
  default     = "db.t4g.medium"
}

variable "allocated_storage" {
  description = "GiB. The corpus is small; checkpoints and the audit log are what grow."
  type        = number
  default     = 50
}

variable "max_allocated_storage" {
  description = "GiB ceiling for storage autoscaling. The audit log is append-only and never truncates — a full volume is an outage."
  type        = number
  default     = 200
}

variable "database_name" {
  description = "Initial database name."
  type        = string
  default     = "ragkit"
}

variable "master_username" {
  description = "Master user. The password is generated and rotated by RDS, never set here — see main.tf."
  type        = string
  default     = "rag"
}

variable "multi_az" {
  description = "Standby in a second AZ. The failover this survives is the one the liveness/readiness split exists for."
  type        = bool
  default     = true
}

variable "backup_retention_period" {
  description = "Days of automated backups. The audit trail and the approval checkpoints live in this database."
  type        = number
  default     = 14

  validation {
    condition     = var.backup_retention_period > 0
    error_message = "0 disables automated backups; docs/deployment.md's production checklist requires them."
  }
}

variable "deletion_protection" {
  description = "Refuse `terraform destroy` on the instance holding the audit log."
  type        = bool
  default     = true
}

variable "kms_key_id" {
  description = "Customer-managed KMS key for storage and secrets. Null uses the AWS-managed keys."
  type        = string
  default     = null
}

# --- DNS ---------------------------------------------------------------------

variable "route53_zone_id" {
  description = "Hosted zone for the two hostnames below."
  type        = string
}

variable "app_hostname" {
  description = "FQDN for the frontend. Must match the Ingress host in the overlay."
  type        = string
}

variable "api_hostname" {
  description = "FQDN for the backend. Must also be the NEXT_PUBLIC_API_BASE the frontend image was BUILT with — it is inlined into the browser bundle."
  type        = string
}

variable "ingress_hostname" {
  description = "DNS name of the load balancer the ingress controller provisioned, e.g. the ELB hostname from `kubectl get ingress`."
  type        = string
}

variable "dns_ttl" {
  description = "Seconds. Short enough that repointing at a new load balancer is not an afternoon."
  type        = number
  default     = 60
}
