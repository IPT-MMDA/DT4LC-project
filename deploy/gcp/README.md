# The public demo on Google Cloud

One Compute Engine VM runs the demo stack of [docs/DEPLOY_DEMO.md](../../docs/DEPLOY_DEMO.md) behind Caddy. The scripts here create the host and provision it; the runbook stays the reference for the stack itself.

| file | where it runs | what it does |
|---|---|---|
| `create-vm.sh` | your machine | enables the APIs, reserves a static address, opens 80 and 443 (SSH through IAP only), creates the Earth Engine service account and its key, creates the VM |
| `startup.sh` | the VM, first boot | installs Docker Engine with Compose v2 and Caddy, clones the repository at the requested tag, writes the Caddyfile and a `.env` with every non-secret value |
| `finish.sh` | the VM, after you complete `.env` | validates the compose configuration, builds and starts the stack, runs the section-3 checks |
| `Caddyfile` | the VM | the TLS proxy for the domain, with `www` redirected |

Sizing: `e2-standard-4` (4 vCPU, 16 GB) runs the models on CPU in a few minutes; `e2-standard-2` works at about half the price and twice the time. The disk is 50 GB because the backend image is 12 GB.

## Steps

1. **Sign in and pick the project.** `gcloud auth login`, then note the project id (not the number) and make sure billing is on for it.
2. **Create the host:** `PROJECT=<project id> ./deploy/gcp/create-vm.sh`. It prints the static address and the path of the Earth Engine key. Re-running is safe. `REF=dev` before the release is tagged; the default is `v0.6.0`.
3. **DNS, at the registrar:** two A records, `@` and `www`, to the static address. Caddy fetches the certificate as soon as the name resolves.
4. **Earth Engine:** register the project for noncommercial use at <https://code.earthengine.google.com/register>. The service account is authorised through the project.
5. **The GitHub OAuth app:** runbook section 5, with the callback `https://<domain>/v1/auth/github/callback`.
6. **On the VM:** copy the key, complete `.env`, start.

   ```sh
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
