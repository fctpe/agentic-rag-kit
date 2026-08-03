output "db_endpoint" {
  description = "host:port of the instance."
  value       = aws_db_instance.postgres.endpoint
}

output "db_name" {
  description = "Initial database name."
  value       = aws_db_instance.postgres.db_name
}

output "db_master_user_secret_arn" {
  description = "The RDS-managed secret holding the master password. Read it to assemble DATABASE_URL; Terraform state never has the password."
  value       = aws_db_instance.postgres.master_user_secret[0].secret_arn
}

output "app_secret_arn" {
  description = "Container for JWT_SECRET and OPENAI_API_KEY. Empty until someone puts a value in it — see main.tf."
  value       = aws_secretsmanager_secret.app.arn
}

output "database_url_template" {
  description = "DATABASE_URL with the password left out. The +asyncpg dialect is not optional: app/db.py builds an async engine, and a bare postgresql:// URL fails at startup."
  value       = "postgresql+asyncpg://${aws_db_instance.postgres.username}:PASSWORD@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
}

output "app_url" {
  description = "Frontend URL. Must match the Ingress host."
  value       = "https://${var.app_hostname}"
}

output "api_url" {
  description = "Backend URL. The frontend image must have been BUILT with NEXT_PUBLIC_API_BASE set to this."
  value       = "https://${var.api_hostname}"
}
