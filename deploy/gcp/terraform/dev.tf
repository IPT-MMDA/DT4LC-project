# The dev host: a second VM that serves the dev branch at <dev_subdomain>.<domain>, created only
# with dev_vm = true. Same size, image, firewall tags and service account as the demo, so a result
# on dev reproduces on the demo. deploy.yml updates it after every image build of dev, through the
# dev.dt4lc.org environment; the deployer's project-wide roles already cover it.

resource "google_compute_address" "dev" {
  count      = var.dev_vm ? 1 : 0
  name       = "${var.dev_vm_name}-ip"
  region     = var.region
  depends_on = [google_project_service.apis]
}

resource "google_compute_instance" "dev" {
  count                     = var.dev_vm ? 1 : 0
  name                      = var.dev_vm_name
  machine_type              = var.dev_machine_type
  zone                      = var.zone
  tags                      = ["dt4lc-web"]
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = data.google_compute_image.debian.self_link
      size  = var.dev_disk_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.dev[0].address
    }
  }

  service_account {
    email  = google_service_account.earth_engine.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    enable-oslogin  = "TRUE"
    dt4lc-domain    = "${var.dev_subdomain}.${var.domain}"
    dt4lc-ref       = var.dev_ref
    dt4lc-image-tag = "dev"
  }
  metadata_startup_script = file("${path.module}/../startup.sh")

  lifecycle {
    # As for the demo: startup.sh runs on first boot only, and a later edit must not replace the host.
    ignore_changes = [metadata_startup_script]
  }

  depends_on = [google_project_service.apis]
}

# Only the dev name: a www record would make Caddy ask for a certificate nobody needs.
resource "google_dns_record_set" "dev" {
  count        = var.dev_vm && var.manage_dns ? 1 : 0
  managed_zone = google_dns_managed_zone.demo[0].name
  name         = "${var.dev_subdomain}.${var.domain}."
  type         = "A"
  ttl          = 300
  rrdatas      = [google_compute_address.dev[0].address]
}
