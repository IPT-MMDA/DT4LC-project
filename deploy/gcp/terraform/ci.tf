# The CI identity. GitHub Actions authenticates through Workload Identity Federation, so no key is
# stored in GitHub: the deployer service account may open an IAP tunnel to the VM and log in with
# OS Login to run deploy/gcp/update.sh. With ci_terraform_apply it may also apply this configuration.

locals {
  ci = var.github_repository != ""
}

resource "google_iam_workload_identity_pool" "github" {
  count                     = local.ci ? 1 : 0
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  count                              = local.ci ? 1 : 0
  workload_identity_pool_id          = google_iam_workload_identity_pool.github[0].workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub"
  # Only jobs that declare the deploy environment get a token: a token without the environment
  # claim fails the mapping, and the environment's branch rule on GitHub admits master and release
  # tags only. A workflow on any other branch therefore cannot impersonate the deployer.
  attribute_mapping = {
    "google.subject"        = "assertion.sub"
    "attribute.repository"  = "assertion.repository"
    "attribute.environment" = "assertion.environment"
  }
  attribute_condition = "assertion.repository == \"${var.github_repository}\" && assertion.environment == \"${var.github_environment}\""
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "deployer" {
  count        = local.ci ? 1 : 0
  account_id   = "dt4lc-deployer"
  display_name = "DT4LC GitHub Actions deployer"
  depends_on   = [google_project_service.apis]
}

resource "google_service_account_iam_member" "deployer_wif" {
  count              = local.ci ? 1 : 0
  service_account_id = google_service_account.deployer[0].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github[0].name}/attribute.repository/${var.github_repository}"
}

# Container deploys: the IAP tunnel, OS Login with sudo, reading the instance.
# Terraform from CI (opt-in): project-wide rights, including over this identity's own resources.
resource "google_project_iam_member" "deployer" {
  for_each = local.ci ? toset(concat(
    ["roles/iap.tunnelResourceAccessor", "roles/compute.osAdminLogin", "roles/compute.viewer"],
    var.ci_terraform_apply ? ["roles/editor", "roles/resourcemanager.projectIamAdmin", "roles/iam.workloadIdentityPoolAdmin"] : [],
  )) : toset([])
  project = var.project
  role    = each.value
  member  = "serviceAccount:${google_service_account.deployer[0].email}"
}

# Logging in to a VM that carries a service account needs the right to act as that account.
resource "google_service_account_iam_member" "deployer_uses_vm_account" {
  count              = local.ci ? 1 : 0
  service_account_id = google_service_account.earth_engine.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer[0].email}"
}
