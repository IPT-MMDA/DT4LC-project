# The public demo on Google Cloud

One Compute Engine VM runs the demo stack of [docs/DEPLOY_DEMO.md](../../docs/DEPLOY_DEMO.md) behind Caddy. The scripts here create the host and provision it; the runbook stays the reference for the stack itself.

| file | where it runs | what it does |
|---|---|---|
| `terraform/` | your machine | enables the APIs, reserves a static address, opens 80 and 443 (SSH through IAP only), creates the Earth Engine service account, the VM, and optionally a Cloud DNS zone with the A records |
| `startup.sh` | the VM, first boot | installs Docker Engine with Compose v2 and Caddy, clones the repository at the requested tag, writes the Caddyfile and a `.env` with every non-secret value |
| `finish.sh` | the VM, after you complete `.env` | validates the compose configuration, pulls the published images (or builds), starts the stack, runs the section-3 checks |
| `update.sh` | the VM | moves the checkout to a ref and the stack to the matching published images; what `deploy.yml` runs |
| `Caddyfile` | the VM | the TLS proxy for the domain, with `www` redirected |

Sizing: `e2-standard-2` (2 vCPU, 8 GB, about 50 USD a month) runs the models on CPU in some minutes; `e2-standard-4` halves that at twice the price. Below 8 GB the model steps can run out of memory. The disk is 30 GB: the images take about 1.2 GB, the model weights 0.5 GB, and the rest is for uploads and results.

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

   The outputs give the static address, the zone's nameservers, the Earth Engine service account and the `ssh` command. Re-running `apply` is safe; `terraform destroy` removes everything except the APIs.
3. **DNS.** With `manage_dns = true` (the default) Terraform created a Cloud DNS zone: at the registrar, replace the nameservers with the four from the `name_servers` output and wait for the change to spread (minutes to a few hours). With `manage_dns = false`, create two A records at the registrar instead, `@` and `www`, to the static address. Caddy fetches the certificate as soon as the name resolves to the host.
4. **Earth Engine:** register the project for noncommercial use at <https://code.earthengine.google.com/register>. The VM runs as the `earth_engine_service_account` output, so there is no key to create or copy: `startup.sh` sets `GEE_USE_DEFAULT_CREDENTIALS=true` and the client library takes the credentials from the instance metadata. (Organisation policy forbids service-account keys on new projects anyway.)
5. **The GitHub OAuth app:** runbook section 5, with the callback `https://<domain>/v1/auth/github/callback`.
6. **On the VM:** complete `.env`, start. With the GitHub environment of the next section this step is not needed: the first "Deploy demo" run writes those values and starts the stack.

   ```sh
   gcloud compute ssh dt4lc-demo --zone <zone> --tunnel-through-iap   # the ssh output
   sudo nano /opt/dt4lc/.env          # the OAuth app, the allowlists, the LLM key
   sudo /opt/dt4lc/deploy/gcp/finish.sh
   ```

   `startup.sh` logs to the serial console; `sudo journalctl -u google-startup-scripts` shows it. It takes a few minutes after the VM boots.
7. **Smoke test from outside:** the landing page, "Try with the Kakhovka sample", a sixth guest analysis refused, sign-in as an operator, `curl -s https://<domain>/v1/health`.
8. **Upgrades and backups:** runbook section 10. `resources/.cache/` holds the job database, uploads, outputs and the settings overrides.

## Automatic deploys from GitHub Actions

`.github/workflows/deploy.yml` does two things, and both are no-ops until the repository variables exist, so forks and contributors are unaffected:

- **containers:** after "Docker Build & Push" has published the images of a release tag or of `master`, it opens an IAP tunnel to the host, writes the `demo` environment's variables and secrets into the host's `.env` (`configure.sh`, names in the log, never values) and runs `update.sh` with that ref and image tag. A merge into `master` therefore reaches the demo a few minutes later as `latest`; a tag reaches it as its version. `workflow_dispatch` deploys any ref by hand.
- **Terraform:** `plan` on pull requests that touch `deploy/gcp/terraform/`, `apply` on `master` when `TF_APPLY_IN_CI` is `true`. This job stays off until `TF_STATE_BUCKET` is set: it needs the state in that bucket (`backend.gcs.tf.example`, then `terraform init -migrate-state` from the machine that applied) and a deployer that may manage every resource in the configuration, not only the three roles the app deploy needs (`ci.tf`). Until then, apply locally; the app deploys do not depend on it.

Authentication is Workload Identity Federation: Terraform creates a `dt4lc-deployer` service account that GitHub Actions of this repository, and no one else, may impersonate. No key is stored in GitHub, and no project id or secret is in the repository.

Set-up, once, by the project owner:

1. Apply locally (steps 1 and 2 above). `terraform output github_actions_variables` prints the values.
2. In the repository: Settings → Secrets and variables → Actions → Variables: `GCP_PROJECT_ID`, `GCP_WIF_PROVIDER`, `GCP_DEPLOYER_SA`, `GCP_ZONE`, `GCP_VM`, `DEMO_DOMAIN`. For Terraform in CI, also `TF_STATE_BUCKET` (a bucket you created, see `backend.gcs.tf.example`) and `TF_APPLY_IN_CI`.
3. Make the two packages public (the package's settings → Change visibility), or the host cannot pull them; the release notes already tell readers to pull them.
4. Settings → Environments → New environment → `demo`; under "Deployment branches and tags" choose "Selected branches and tags" and add `master` and `v*`, so no other branch's workflow can read what follows. Its variables: `OAUTH_CLIENT_ID`, `OPERATOR_EMAILS`, `ALLOWED_EMAILS`, `ALLOWED_DOMAINS`, `LLM_PROVIDER_ORDER`; its secrets: `OAUTH_CLIENT_SECRET` and the provider keys you use among `GEMINI_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`. GitHub reserves the `GITHUB_` prefix, hence the `OAUTH_` names; on the host they become `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET`. Every deploy writes the non-empty ones, so a value changed here reaches the host on the next deploy, "Run workflow" included. `SESSION_SECRET` stays on the host, generated at first boot.

Anyone deploying their own instance uses the same Terraform with their own `terraform.tfvars`; with `github_repository = ""` no CI identity is created.
