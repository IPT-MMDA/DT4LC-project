# Optional Cloud DNS zone for the domain. After `apply`, set the registrar's nameservers to the
# `name_servers` output; until then, the same two A records can be created at the registrar.

resource "google_dns_managed_zone" "demo" {
  count       = var.manage_dns ? 1 : 0
  name        = replace(var.domain, ".", "-")
  dns_name    = "${var.domain}."
  description = "DT4LC public demo"
  depends_on  = [google_project_service.apis]
}

resource "google_dns_record_set" "apex" {
  count        = var.manage_dns ? 1 : 0
  managed_zone = google_dns_managed_zone.demo[0].name
  name         = "${var.domain}."
  type         = "A"
  ttl          = 300
  rrdatas      = [google_compute_address.demo.address]
}

resource "google_dns_record_set" "www" {
  count        = var.manage_dns ? 1 : 0
  managed_zone = google_dns_managed_zone.demo[0].name
  name         = "www.${var.domain}."
  type         = "A"
  ttl          = 300
  rrdatas      = [google_compute_address.demo.address]
}
