#!/usr/bin/env bash
# On the demo host: move the checkout to a git ref and the stack to the matching published images,
# then restart what changed. Run with sudo, by a person or by .github/workflows/deploy.yml:
#   update.sh v0.6.1 0.6.1      # a release
#   update.sh master latest     # the master branch
set -euo pipefail
REF=${1:?git ref (a tag or master)}
TAG=${2:?image tag (0.6.1 or latest)}
cd /opt/dt4lc
git fetch -q --tags origin
git checkout -q "$REF"
[ "$REF" = master ] && git pull -q --ff-only origin master
if grep -qE '^DT4LC_IMAGE_TAG=' .env; then
  sed -i -E "s|^DT4LC_IMAGE_TAG=.*|DT4LC_IMAGE_TAG=$TAG|" .env
else
  printf 'DT4LC_IMAGE_TAG=%s\n' "$TAG" >> .env
fi
docker compose pull -q
docker compose up -d --remove-orphans
docker image prune -f >/dev/null
# The backend loads its models after the containers start; give it up to two minutes.
for _ in $(seq 1 24); do
  if health=$(curl -sf http://127.0.0.1:8080/v1/health); then
    echo "${health:0:200}"
    echo "deployed $REF (images :$TAG)"
    exit 0
  fi
  sleep 5
done
echo "update.sh: the backend did not answer /v1/health within two minutes" >&2
docker compose ps
exit 1
