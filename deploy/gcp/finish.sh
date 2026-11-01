#!/usr/bin/env bash
# On the demo host, once /opt/dt4lc/.env is complete: validate the stack, build and start it, and run
# the checks of docs/DEPLOY_DEMO.md section 3. Run with sudo.
set -euo pipefail
cd /opt/dt4lc
for v in GITHUB_CLIENT_ID GITHUB_CLIENT_SECRET OPERATOR_EMAILS LLM_PROVIDER_ORDER GEE_PROJECT_ID; do
  grep -qE "^$v=.+" .env || echo "note: $v is not set in .env"
done
docker compose config --quiet && echo "compose config: valid"
echo "services: $(docker compose config --services | tr '\n' ' ')"
docker compose up -d --build
docker compose ps -a
echo "--- health";       curl -s http://127.0.0.1:8080/v1/health | head -c 300; echo
echo "--- sign-in";      curl -s http://127.0.0.1:8080/v1/auth/status; echo
echo "--- capabilities"; curl -s http://127.0.0.1:8080/v1/capabilities | python3 -c 'import json,sys; print(json.load(sys.stdin)["summary"]["line"])'
echo "--- the backend must not warn about these:"; docker compose logs backend | grep -e SESSION_SECRET -e DEV_ROLE || echo "(none)"
