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
  default     = "v0.6.2"
}

variable "vm_name" {
  type    = string
  default = "dt4lc-demo"
}

variable "machine_type" {
  description = "e2-standard-2 (2 vCPU, 8 GB) runs the models on CPU in some minutes; e2-standard-4 halves that at twice the price. Below 8 GB the model steps can run out of memory."
  type        = string
  default     = "e2-standard-2"
}

variable "disk_gb" {
  description = "Boot disk: images about 1.2 GB, model weights 0.5 GB, the rest for uploads and results."
  type        = number
  default     = 30
}

variable "manage_dns" {
  description = "Create a Cloud DNS zone for the domain with the A records. Point the registrar's nameservers at the zone's (see the name_servers output)."
  type        = bool
  default     = true
}

variable "github_repository" {
  description = "owner/name of the GitHub repository whose Actions may deploy to the host (Workload Identity Federation, no stored key). Empty disables the CI identity."
  type        = string
  default     = "IPT-MMDA/dt4lc"
}

variable "github_environments" {
  description = "The GitHub environments whose jobs may impersonate the deployer, one per host; each environment's branch rule limits which branches and tags can use it."
  type        = list(string)
  default     = ["dt4lc.org", "dev.dt4lc.org"]
}

variable "ci_terraform_apply" {
  description = "Also let the CI identity apply this configuration (it then needs project-wide roles). Off: CI only updates the containers."
  type        = bool
  default     = false
}
