#!/usr/bin/env bash
# On the demo host: set .env values read as NAME=VALUE lines on stdin, one per line. Each name
# replaces its live line, else its commented-out line, else is appended; never a second copy. Empty values
# and malformed lines are skipped. Prints the names it set, never the values. Run with sudo, by a
# person or by .github/workflows/deploy.yml (the `demo` environment's variables and secrets):
#   printf 'OPERATOR_EMAILS=%s\n' "$emails" | sudo deploy/gcp/configure.sh
# A change takes effect when the containers are created again (update.sh, or docker compose up -d).
set -euo pipefail
ENV=${DT4LC_ENV:-/opt/dt4lc/.env}
[ -f "$ENV" ] || { echo "configure.sh: $ENV does not exist (startup.sh writes it on first boot)" >&2; exit 1; }
PAIRS=$(cat)   # the script below arrives on python's stdin, so the pairs travel in the environment
DT4LC_PAIRS="$PAIRS" python3 - "$ENV" <<'PY'
import os, re, sys

path = sys.argv[1]
text = open(path).read()
names: list[str] = []
for raw in os.environ["DT4LC_PAIRS"].splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    name, value = line.split("=", 1)
    name = name.strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or value == "":
        continue
    # The live line if there is one (a commented example may precede it), else the first
    # commented line, else append; never a second live copy.
    for pattern in (rf"^{re.escape(name)}=.*$", rf"^# *{re.escape(name)}=.*$"):
        if re.search(pattern, text, flags=re.M):
            text = re.sub(pattern, lambda _m: f"{name}={value}", text, count=1, flags=re.M)
            break
    else:
        text = text.rstrip("\n") + f"\n{name}={value}\n"
    names.append(name)
open(path, "w").write(text)
print("configure.sh: set " + (", ".join(names) if names else "nothing"))
PY
