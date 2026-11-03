output "static_ip" {
  value = google_compute_address.demo.address
}

output "name_servers" {
  description = "Set these at the registrar when manage_dns is true."
  value       = var.manage_dns ? google_dns_managed_zone.demo[0].name_servers : []
}

output "earth_engine_service_account" {
  value = google_service_account.earth_engine.email
}

output "ssh" {
  value = "gcloud compute ssh ${var.vm_name} --zone ${var.zone} --tunnel-through-iap"
}

output "github_actions_variables" {
  description = "Repository variables for .github/workflows/deploy.yml (Settings > Secrets and variables > Actions > Variables)."
  value = var.github_repository != "" ? {
    GCP_PROJECT_ID   = var.project
    GCP_WIF_PROVIDER = google_iam_workload_identity_pool_provider.github[0].name
    GCP_DEPLOYER_SA  = google_service_account.deployer[0].email
    GCP_ZONE         = var.zone
    GCP_VM           = var.vm_name
    DEMO_DOMAIN      = var.domain
  } : {}
}
