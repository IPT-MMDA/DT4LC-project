#!/usr/bin/env bash
# Create the Google Cloud host for the DT4LC public demo: the APIs, a static address, the firewall,
# an Earth Engine service account with a key, and one Compute Engine VM that provisions itself
# with deploy/gcp/startup.sh. Safe to re-run: what exists is kept.
#
#   gcloud auth login
#   PROJECT=<project id> ./deploy/gcp/create-vm.sh
#
# Optional: DOMAIN (dt4lc.org), REF (v0.6.0, the git ref the host checks out), REGION (europe-west1),
# ZONE (<region>-b), VM (dt4lc-demo), MACHINE (e2-standard-4), DISK_GB (50).
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT to the project id, e.g. PROJECT=dt4lc-demo ./deploy/gcp/create-vm.sh}
DOMAIN=${DOMAIN:-dt4lc.org}
REF=${REF:-v0.6.0}
REGION=${REGION:-europe-west1}
ZONE=${ZONE:-${REGION}-b}
VM=${VM:-dt4lc-demo}
MACHINE=${MACHINE:-e2-standard-4}
DISK_GB=${DISK_GB:-50}
SA=dt4lc-earth-engine
SA_EMAIL="$SA@$PROJECT.iam.gserviceaccount.com"
HERE=$(cd "$(dirname "$0")" && pwd)
KEY_DIR=${KEY_DIR:-$HOME/.dt4lc}
KEY="$KEY_DIR/gee-key-$PROJECT.json"

gcloud config set project "$PROJECT" >/dev/null

echo "== APIs"
gcloud services enable compute.googleapis.com iam.googleapis.com earthengine.googleapis.com serviceusage.googleapis.com

echo "== static address $VM-ip ($REGION)"
gcloud compute addresses describe "$VM-ip" --region "$REGION" >/dev/null 2>&1 \
  || gcloud compute addresses create "$VM-ip" --region "$REGION"
IP=$(gcloud compute addresses describe "$VM-ip" --region "$REGION" --format='value(address)')

echo "== firewall: 80 and 443 from the internet, 22 through IAP only"
gcloud compute firewall-rules describe dt4lc-web >/dev/null 2>&1 \
  || gcloud compute firewall-rules create dt4lc-web --allow tcp:80,tcp:443 --target-tags dt4lc-web --source-ranges 0.0.0.0/0
gcloud compute firewall-rules describe dt4lc-ssh-iap >/dev/null 2>&1 \
  || gcloud compute firewall-rules create dt4lc-ssh-iap --allow tcp:22 --target-tags dt4lc-web --source-ranges 35.235.240.0/20

echo "== Earth Engine service account $SA_EMAIL"
gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "$SA" --display-name "DT4LC Earth Engine"
for role in roles/earthengine.writer roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA_EMAIL" --role "$role" --condition=None >/dev/null
done
mkdir -p "$KEY_DIR" && chmod 700 "$KEY_DIR"
if [ ! -f "$KEY" ]; then
  gcloud iam service-accounts keys create "$KEY" --iam-account "$SA_EMAIL"
  chmod 600 "$KEY"
fi

echo "== VM $VM ($ZONE, $MACHINE, ${DISK_GB} GB, Debian 12)"
gcloud compute instances describe "$VM" --zone "$ZONE" >/dev/null 2>&1 \
  || gcloud compute instances create "$VM" --zone "$ZONE" --machine-type "$MACHINE" \
       --image-family debian-12 --image-project debian-cloud \
       --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
       --address "$IP" --tags dt4lc-web \
       --metadata "enable-oslogin=TRUE,dt4lc-domain=$DOMAIN,dt4lc-ref=$REF" \
       --metadata-from-file "startup-script=$HERE/startup.sh"

cat <<DONE

Created or confirmed:
  VM         $VM in $ZONE ($MACHINE, ${DISK_GB} GB); it installs Docker, Caddy and the code by itself (a few minutes)
  address    $IP   -> at the registrar: A record "@" -> $IP, A record "www" -> $IP
  EE key     $KEY  (keep it out of git)

Next (deploy/gcp/README.md):
  1. register the project for Earth Engine, noncommercial use: https://code.earthengine.google.com/register
  2. gcloud compute scp --zone $ZONE --tunnel-through-iap "$KEY" $VM:gee-key.json
  3. gcloud compute ssh $VM --zone $ZONE --tunnel-through-iap
     sudo mv ~/gee-key.json /opt/dt4lc/resources/.cache/gee-key.json && sudo chmod 600 /opt/dt4lc/resources/.cache/gee-key.json
     sudo nano /opt/dt4lc/.env        # the lines marked "fill in"
     sudo /opt/dt4lc/deploy/gcp/finish.sh
DONE
