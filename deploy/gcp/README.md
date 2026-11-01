# The public demo on Google Cloud

One Compute Engine VM runs the demo stack of [docs/DEPLOY_DEMO.md](../../docs/DEPLOY_DEMO.md) behind Caddy. The scripts here create the host and provision it; the runbook stays the reference for the stack itself.

| file | where it runs | what it does |
|---|---|---|
| `terraform/` | your machine | enables the APIs, reserves a static address, opens 80 and 443 (SSH through IAP only), creates the Earth Engine service account, the VM, and optionally a Cloud DNS zone with the A records |
| `startup.sh` | the VM, first boot | installs Docker Engine with Compose v2 and Caddy, clones the repository at the requested tag, writes the Caddyfile and a `.env` with every non-secret value |
| `finish.sh` | the VM, after you complete `.env` | validates the compose configuration, pulls the published images (or builds), starts the stack, runs the section-3 checks |
| `update.sh` | the VM | moves the checkout to a ref and the stack to the matching published images; what `deploy.yml` runs |
| `Caddyfile` | the VM | the TLS proxy for the domain, with `www` redirected |

Sizing: `e2-standard-4` (4 vCPU, 16 GB) runs the models on CPU in a few minutes; `e2-standard-2` works at about half the price and twice the time. The disk is 50 GB because the backend image is 12 GB.

## Steps

1. **Sign in and pick the project.** `gcloud auth login`, then note the project id (not the number) and make sure billing is on for it.
2. **Create the host with Terraform** (no secret enters the configuration or the state):

   ```sh
   cd deploy/gcp/terraform
   cp terraform.tfvars.example terraform.tfvars   # set project; ref = dev until the release is tagged
   terraform init
   terraform plan
   terraform apply
   ```

   The outputs give the static address, the zone's nameservers, the service account and the exact `gcloud` commands for the next steps. Re-running `apply` is safe; `terraform destroy` removes everything except the APIs.
3. **DNS.** With `manage_dns = true` (the default) Terraform created a Cloud DNS zone: at the registrar, replace the nameservers with the four from the `name_servers` output and wait for the change to spread (minutes to a few hours). With `manage_dns = false`, create two A records at the registrar instead, `@` and `www`, to the static address. Caddy fetches the certificate as soon as the name resolves to the host.
4. **Earth Engine:** register the project for noncommercial use at <https://code.earthengine.google.com/register>, then create the service account's key with the `create_earth_engine_key` output command. The key is written to `~/.dt4lc/`, outside git.
5. **The GitHub OAuth app:** runbook section 5, with the callback `https://<domain>/v1/auth/github/callback`.
6. **On the VM:** copy the key, complete `.env`, start.

   ```sh
   # the copy_key and ssh outputs give these two with your values filled in
   gcloud compute scp --zone <zone> --tunnel-through-iap ~/.dt4lc/gee-key-<project>.json dt4lc-demo:gee-key.json
   gcloud compute ssh dt4lc-demo --zone <zone> --tunnel-through-iap
   sudo mv ~/gee-key.json /opt/dt4lc/resources/.cache/gee-key.json && sudo chmod 600 /opt/dt4lc/resources/.cache/gee-key.json
   sudo nano /opt/dt4lc/.env          # the OAuth app, the allowlists, the LLM key, GEE_PROJECT_ID
   sudo /opt/dt4lc/deploy/gcp/finish.sh
   ```

   `startup.sh` logs to the serial console; `sudo journalctl -u google-startup-scripts` shows it. It takes a few minutes after the VM boots.
7. **Smoke test from outside:** the landing page, "Try with the Kakhovka sample", a sixth guest analysis refused, sign-in as an operator, `curl -s https://<domain>/v1/health`.
8. **Upgrades and backups:** runbook section 10. `resources/.cache/` holds the job database, uploads, outputs, the settings overrides and the Earth Engine key.

The key file stays in `~/.dt4lc/` on your machine and under `resources/.cache/` on the host; both are outside git.

## Automatic deploys from GitHub Actions

`.github/workflows/deploy.yml` does two things, and both are no-ops until the repository variables exist, so forks and contributors are unaffected:

- **containers:** after "Docker Build & Push" has published the images of a release tag or of `master`, it opens an IAP tunnel to the host and runs `update.sh` with that ref and image tag. A merge into `master` therefore reaches the demo a few minutes later as `latest`; a tag reaches it as its version. `workflow_dispatch` deploys any ref by hand.
- **Terraform:** `plan` on pull requests that touch `deploy/gcp/terraform/`, `apply` on `master` when `TF_APPLY_IN_CI` is `true`. Until then, apply locally.

Authentication is Workload Identity Federation: Terraform creates a `dt4lc-deployer` service account that GitHub Actions of this repository, and no one else, may impersonate. No key is stored in GitHub, and no project id or secret is in the repository.

Set-up, once, by the project owner:

1. Apply locally (steps 1 and 2 above). `terraform output github_actions_variables` prints the values.
2. In the repository: Settings → Secrets and variables → Actions → Variables: `GCP_PROJECT_ID`, `GCP_WIF_PROVIDER`, `GCP_DEPLOYER_SA`, `GCP_ZONE`, `GCP_VM`, `DEMO_DOMAIN`. For Terraform in CI, also `TF_STATE_BUCKET` (a bucket you created, see `backend.gcs.tf.example`) and `TF_APPLY_IN_CI`.
3. Make the two packages public (the package's settings → Change visibility), or the host cannot pull them; the release notes already tell readers to pull them.

Anyone deploying their own instance uses the same Terraform with their own `terraform.tfvars`; with `github_repository = ""` no CI identity is created.
