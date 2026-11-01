variable "project" {
  description = "The Google Cloud project id (not the number)."
  type        = string
}

variable "region" {
  description = "Region of the static address and the VM."
  type        = string
  default     = "europe-west1"
}

variable "zone" {
  description = "Zone of the VM."
  type        = string
  default     = "europe-west1-b"
}

variable "domain" {
  description = "The demo's domain; www is redirected to it."
  type        = string
  default     = "dt4lc.org"
}

variable "ref" {
  description = "The git ref the host checks out (a release tag)."
  type        = string
  default     = "v0.6.0"
}

variable "vm_name" {
  type    = string
  default = "dt4lc-demo"
}

variable "machine_type" {
  description = "e2-standard-4 runs the models on CPU in a few minutes; e2-standard-2 takes about twice as long."
  type        = string
  default     = "e2-standard-4"
}

variable "disk_gb" {
  description = "Boot disk; the backend image alone is about 12 GB."
  type        = number
  default     = 50
}

variable "manage_dns" {
  description = "Create a Cloud DNS zone for the domain with the A records. Point the registrar's nameservers at the zone's (see the name_servers output)."
  type        = bool
  default     = true
}

variable "github_repository" {
  description = "owner/name of the GitHub repository whose Actions may deploy to the host (Workload Identity Federation, no stored key). Empty disables the CI identity."
  type        = string
  default     = "IPT-MMDA/DT4LC-project"
}

variable "ci_terraform_apply" {
  description = "Also let the CI identity apply this configuration (it then needs project-wide roles). Off: CI only updates the containers."
  type        = bool
  default     = false
}
