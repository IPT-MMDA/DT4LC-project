# The DT4LC public demo on one Compute Engine VM: docs/DEPLOY_DEMO.md behind Caddy.
# No secret lives here: the Earth Engine key is created with gcloud (README), the .env on the host
# is written by ../startup.sh and completed by a person.

locals {
  services = toset(concat(
    ["compute.googleapis.com", "iam.googleapis.com", "earthengine.googleapis.com", "serviceusage.googleapis.com"],
    var.manage_dns ? ["dns.googleapis.com"] : [],
    var.github_repository != "" ? ["iamcredentials.googleapis.com", "sts.googleapis.com", "iap.googleapis.com", "oslogin.googleapis.com"] : [],
  ))
}

resource "google_project_service" "apis" {
  for_each           = local.services
  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_address" "demo" {
  name       = "${var.vm_name}-ip"
  region     = var.region
  depends_on = [google_project_service.apis]
}

resource "google_compute_firewall" "web" {
  name          = "dt4lc-web"
  network       = "default"
  target_tags   = ["dt4lc-web"]
  source_ranges = ["0.0.0.0/0"]
  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }
  depends_on = [google_project_service.apis]
}

# SSH only through Identity-Aware Proxy: gcloud compute ssh --tunnel-through-iap
resource "google_compute_firewall" "ssh_iap" {
  name          = "dt4lc-ssh-iap"
  network       = "default"
  target_tags   = ["dt4lc-web"]
  source_ranges = ["35.235.240.0/20"]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  depends_on = [google_project_service.apis]
}

resource "google_service_account" "earth_engine" {
  account_id   = "dt4lc-earth-engine"
  display_name = "DT4LC Earth Engine"
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "earth_engine" {
  for_each = toset(["roles/earthengine.writer", "roles/serviceusage.serviceUsageConsumer"])
  project  = var.project
  role     = each.value
  member   = "serviceAccount:${google_service_account.earth_engine.email}"
}

data "google_compute_image" "debian" {
  family  = "debian-12"
  project = "debian-cloud"
}

resource "google_compute_instance" "demo" {
  name                      = var.vm_name
  machine_type              = var.machine_type
  zone                      = var.zone
  tags                      = ["dt4lc-web"]
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = data.google_compute_image.debian.self_link
      size  = var.disk_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.demo.address
    }
  }

  # The VM runs as the Earth Engine service account: the backend uses its default credentials
  # (GEE_USE_DEFAULT_CREDENTIALS in startup.sh), so there is no key file anywhere.
  service_account {
    email  = google_service_account.earth_engine.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    enable-oslogin = "TRUE"
    dt4lc-domain   = var.domain
    dt4lc-ref      = var.ref
  }
  metadata_startup_script = file("${path.module}/../startup.sh")

  lifecycle {
    # startup.sh runs on first boot only (it leaves /var/lib/dt4lc/provisioned); a later edit
    # of it must not replace the host. A new host always gets the current script. The boot image
    # is the latest of the debian-12 family at creation; the family moves on every few weeks, and
    # that must not replace the host either (the disk is upgraded with apt, as any Debian).
    ignore_changes = [metadata_startup_script, boot_disk[0].initialize_params[0].image]
  }

  depends_on = [google_project_service.apis]
}
