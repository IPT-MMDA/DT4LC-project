#!/bin/bash
# First-boot provisioning of the DT4LC demo host (runs as root; a marker makes later boots a no-op):
# Docker Engine with the Compose v2 plugin, Caddy, the repository at the requested ref, and a .env
# with every non-secret value of docs/DEPLOY_DEMO.md section 2. Secrets are added by a person.
set -euo pipefail
MARK=/var/lib/dt4lc/provisioned
[ -f "$MARK" ] && exit 0
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
DOMAIN=$(curl -sf -H 'Metadata-Flavor: Google' "$MD/dt4lc-domain" || echo dt4lc.org)
REF=$(curl -sf -H 'Metadata-Flavor: Google' "$MD/dt4lc-ref" || echo v0.6.0)
# Image tag: the dt4lc-image-tag attribute, else derived from the ref (v0.6.0 -> 0.6.0, master -> latest).
TAG=$(curl -sf -H 'Metadata-Flavor: Google' "$MD/dt4lc-image-tag" || true)
if [ -z "$TAG" ]; then case "$REF" in v[0-9]*) TAG=${REF#v} ;; *) TAG=latest ;; esac; fi
export DEBIAN_FRONTEND=noninteractive

apt-get update -q
apt-get install -y -q ca-certificates curl git gnupg openssl python3
install -m 0755 -d /etc/apt/keyrings
# Docker's repository: Debian's own docker.io package has no Compose plugin.
curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
# Caddy's repository.
curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
apt-get update -q
apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-compose-plugin caddy
systemctl enable --now docker

[ -d /opt/dt4lc/.git ] || git clone -q https://github.com/IPT-MMDA/DT4LC-project.git /opt/dt4lc
git -C /opt/dt4lc fetch -q --tags
git -C /opt/dt4lc checkout -q "$REF"

sed "s/dt4lc\.org/$DOMAIN/g" /opt/dt4lc/deploy/gcp/Caddyfile > /etc/caddy/Caddyfile
systemctl enable --now caddy
systemctl reload caddy

# .env: the template, then each demo value set in place (never a second copy of a variable).
ENV=/opt/dt4lc/.env
if [ ! -f "$ENV" ]; then
  cp /opt/dt4lc/.env.example "$ENV"
  chmod 600 "$ENV"
  setvar() {  # setvar NAME VALUE: replace the (possibly commented) line, or append
    if grep -qE "^#? *$1=" "$ENV"; then
      python3 - "$ENV" "$1" "$2" <<'PY'
import re, sys
path, name, value = sys.argv[1:]
text = open(path).read()
text, n = re.subn(rf"^#? *{re.escape(name)}=.*$", f"{name}={value}", text, count=1, flags=re.M)
open(path, "w").write(text)
PY
    else
      printf '%s=%s\n' "$1" "$2" >> "$ENV"
    fi
  }
  # Published images (docker-compose.ghcr.yml); drop that file from COMPOSE_FILE to build from source.
  setvar COMPOSE_FILE docker-compose.yml:docker-compose.demo.yml:docker-compose.ghcr.yml
  setvar DT4LC_IMAGE_TAG "$TAG"
  setvar COMPOSE_PROFILES demo
  setvar SESSION_SECRET "$(openssl rand -hex 32)"
  setvar SESSION_COOKIE_SECURE true
  setvar GUEST_ANALYSES_PER_HOUR 5
  setvar GUEST_SESSION_HOURS 24
  setvar GUEST_CLEANUP_MINUTES 60
  setvar OAUTH_REDIRECT_BASE "https://$DOMAIN"
  setvar FRONTEND_PORT 127.0.0.1:8080
  setvar LLM_ENABLE_OLLAMA false
  setvar GEE_SERVICE_ACCOUNT_KEY /app/resources/.cache/gee-key.json
  cat >> "$ENV" <<'NOTE'

# ---- fill in before deploy/gcp/finish.sh (docs/DEPLOY_DEMO.md sections 2 and 5) ----
# GITHUB_CLIENT_ID=            GITHUB_CLIENT_SECRET=         (the OAuth app; callback https://<domain>/v1/auth/github/callback)
# OPERATOR_EMAILS=             ALLOWED_EMAILS=               (primary verified GitHub emails, comma-separated)
# LLM_PROVIDER_ORDER=gemini    GEMINI_API_KEY=               (any provider you have a key for)
# GEE_PROJECT_ID=              (the Google Cloud project; the key file is resources/.cache/gee-key.json)
NOTE
fi
mkdir -p /opt/dt4lc/resources/.cache
mkdir -p "$(dirname "$MARK")" && date -u +%FT%TZ > "$MARK"
echo "dt4lc: provisioned for $DOMAIN at $REF; complete /opt/dt4lc/.env and run deploy/gcp/finish.sh"
