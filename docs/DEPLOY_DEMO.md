# Public demo: deployment runbook

This runbook is for the operator who sets up the DT4LC public demo on a Linux host and runs it afterwards. Every command runs in the repository directory on the host.

The demo is the normal Docker stack plus an override, `docker-compose.demo.yml`:

| Part | What it does |
|------|--------------|
| `models-init` (profile `demo`) | Runs `deploy/models_init.py` at each `docker compose up` and installs the missing model weights into the `models_data` volume, so no visitor waits for a download |
| `backend` | Publishes no port. Runs with `DT4LC_DEPLOYMENT=demo` (the Capabilities line says "public demo"), lets only operators download models, and trusts `X-Forwarded-For` only from the demo network |
| `frontend` | Serves `deploy/nginx-demo.conf.template`: the only way in to the backend, with per-IP rate limits and the client address forwarded |
| network | A fixed subnet, `DEMO_SUBNET` (default `172.30.60.0/24`) |

HTTPS is left to a reverse proxy on the host (section 6). Visitors are guests with an hourly analysis quota. People on the allowlists sign in with GitHub and become members (no quota, their work is kept) or operators (who also get Settings).

The stack runs before the GitHub OAuth app exists: the account menu then says "Sign-in not configured on this server" and everyone is a guest. You can do sections 5 and 6 in either order, but you can only test sign-in once the HTTPS address works.

Contents:

1. [Prerequisites](#1-prerequisites)
2. [The demo `.env`](#2-the-demo-env)
3. [First start and checks](#3-first-start-and-checks)
4. [Model pre-install](#4-model-pre-install)
5. [GitHub sign-in: the OAuth app](#5-github-sign-in-the-oauth-app)
6. [Reverse proxy and TLS](#6-reverse-proxy-and-tls)
7. [Rate limits and quotas](#7-rate-limits-and-quotas)
8. [Guest cleanup](#8-guest-cleanup)
9. [Operator Settings](#9-operator-settings)
10. [Upgrades and backups](#10-upgrades-and-backups)
11. [Troubleshooting](#11-troubleshooting)

## 1. Prerequisites

- **A Linux host with Docker Engine and the Docker Compose v2 plugin.** The override uses Compose's `!reset` tag, which older Compose releases reject; the `config` check in section 3 tells you whether yours accepts it. Docker Desktop is not suitable, because it hides the client addresses (see [Troubleshooting](#11-troubleshooting)).
- **Disk:** at least 20 GB free. The backend image is about 12 GB and the model weights about 0.5 GB. The build cache, uploads and results come on top.
- **A DNS name** for the demo pointing at the host. This runbook uses `demo.example.org`. Ports 80 and 443 must be open to the internet for the TLS proxy and its certificate.
- **Outbound HTTPS:**
  - Docker Hub, PyPI and npm while building;
  - Hugging Face and GitHub on the first start, for the model weights and the Delineate-Anything code archive;
  - GitHub for sign-in;
  - the APIs of the LLM providers you use.
- **A GitHub account** (personal or organization) to own the OAuth app.
- **An API key for at least one LLM provider:** Gemini, Groq, Anthropic, Mistral or OpenAI. Without one the assistant is offline: analyses still run, but answers come from the catalog.

Get the code, and check out the release you deploy:

```bash
git clone https://github.com/IPT-MMDA/DT4LC-project.git
cd DT4LC-project
git checkout <release tag>
```

## 2. The demo `.env`

```bash
cp .env.example .env
chmod 600 .env    # it will hold the session secret, the OAuth client secret and the API keys
```

Set these values. Uncomment or edit the lines `.env.example` already has; don't add a second copy of a variable.

```dotenv
# The demo stack: plain `docker compose ...` commands then use the override and the demo profile
COMPOSE_FILE=docker-compose.yml:docker-compose.demo.yml
COMPOSE_PROFILES=demo

# Sessions
SESSION_SECRET=<64 hex characters>
SESSION_COOKIE_SECURE=true

# Guests
GUEST_ANALYSES_PER_HOUR=5
GUEST_SESSION_HOURS=24
GUEST_CLEANUP_MINUTES=60

# GitHub sign-in (section 5)
GITHUB_CLIENT_ID=<client id>
GITHUB_CLIENT_SECRET=<client secret>
OAUTH_REDIRECT_BASE=https://demo.example.org

# Who may sign in (comma-separated)
OPERATOR_EMAILS=you@example.org
ALLOWED_EMAILS=
ALLOWED_DOMAINS=

# The assistant: the providers you have keys for, in order
LLM_ENABLE_OLLAMA=false
LLM_PROVIDER_ORDER=gemini,groq
GEMINI_API_KEY=<key>
GROQ_API_KEY=<key>

# The frontend on the loopback interface only; the TLS proxy (section 6) faces the internet
FRONTEND_PORT=127.0.0.1:8080
```

What each group does:

- **`COMPOSE_FILE` and `COMPOSE_PROFILES`.** You need both. Without `COMPOSE_FILE`, Compose reads only `docker-compose.yml` and starts the default stack, which publishes the backend on port 8000, even with `--profile demo`. Instead of these two lines you can pass `-f docker-compose.yml -f docker-compose.demo.yml --profile demo` to every command; this runbook assumes the two lines.
- **`SESSION_SECRET`** signs the session cookie and the short-lived OAuth state cookie. Generate it once with `python3 -c "import secrets; print(secrets.token_hex(32))"` or `openssl rand -hex 32`. If it is unset, the backend picks a random secret at each start and logs a warning, so every session, signed in or not, ends when the backend restarts. Changing the secret also ends every session.
- **`SESSION_COOKIE_SECURE=true`** marks the cookies HTTPS-only, so the browser sends them over HTTPS only. Keep it `true` behind TLS.
- **Guests:**
  - `GUEST_ANALYSES_PER_HOUR` (default 5) is how many analyses a guest may start per rolling hour, counted per session and per client IP. With 0, guests cannot run analyses.
  - `GUEST_SESSION_HOURS` (default 24) is how long a session lasts, signed in or not. It is also the age at which the cleanup removes guest jobs and uploads.
  - `GUEST_CLEANUP_MINUTES` (default 60) is how often the cleanup runs (section 8).
  - Operators can change the first two at runtime in Settings (section 9).
- **GitHub:** section 5 explains these three. Sign-in is configured only when all three are set.
- **Allowlists** are comma-separated, ignore case, and are matched against the user's primary verified GitHub email (section 5.3). Put your own address in `OPERATOR_EMAILS`: only an operator can open Settings, so the first operator has to come from `.env`.
- **The assistant:**
  - Set the keys you have and list those providers in `LLM_PROVIDER_ORDER`. The backend tries them in order and falls back on failure.
  - Anthropic, Mistral and OpenAI work the same way, with `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY` and `OPENAI_API_KEY`, once they are listed in the order.
  - The model variables (`GEMINI_MODELS`, `GROQ_MODEL`, …) are described in `.env.example`.
  - Turn Ollama off. `.env.example` enables it, but the demo profile does not start it. If you start it anyway, with `COMPOSE_PROFILES=demo,ollama`, `docker-compose.demo.yml` publishes none of its ports; the backend reaches it at `http://ollama:11434` on the compose network. (The `ollama` profile of the default stack, without the override, publishes port 11434 on every interface of the host.)
- **`FRONTEND_PORT=127.0.0.1:8080`** publishes the frontend on the loopback interface only. The plain-HTTP port is then unreachable from outside, and every request comes through the TLS proxy. 8080 is an example: any free port works if the proxy uses the same one.
- **`DEMO_SUBNET`:** leave it unset unless `172.30.60.0/24` clashes with a network on the host ([Troubleshooting](#11-troubleshooting)). The backend trusts `X-Forwarded-For` only from this subnet.
- **Not needed:**
  - `DT4LC_DEPLOYMENT`, `MODEL_DOWNLOAD_ON_DEMAND` and `FORWARDED_ALLOW_IPS`. `docker-compose.demo.yml` sets them for the backend (`demo`, `operators`, the demo subnet), and a value in `.env` does not change them. An operator can still change the download policy in Settings (section 9).
  - `BACKEND_PORT` has no effect: the demo publishes no backend port.
  - `DEMO_SKIP_MODEL_DOWNLOAD` is for smoke tests only; leave it unset.
  - `GITHUB_OAUTH_BASE` and `GITHUB_API_BASE` point sign-in at a fake GitHub in tests; leave them unset.
- **Never set `DEV_ROLE` on the demo.** It gives every request that role, signed-in users included: `DEV_ROLE=operator` makes every visitor an operator. The backend logs a warning at start when it is set.

## 3. First start and checks

Check the configuration first:

```bash
docker compose config --services     # models-init, backend, frontend
docker compose config --quiet && echo valid
```

If `models-init` is missing from the list, `COMPOSE_FILE` or `COMPOSE_PROFILES` is not set in `.env`. If Compose rejects `!reset`, it is too old.

Start the stack:

```bash
docker compose up -d --build
```

The first build takes a while. Compose starts `models-init` first, the backend once `models-init` has exited successfully, and the frontend once the backend is healthy. On the first start the models download (about 0.5 GB); later starts skip them.

Check it:

```bash
docker compose ps -a                       # models-init: Exited (0); backend and frontend: Up (healthy)
docker compose logs models-init            # each model "installed" or "already installed at ..."
curl -s http://127.0.0.1:8080/v1/health    # {"ok":true, ... "assistant":{"connected":true, ...}}
curl -s http://127.0.0.1:8080/v1/capabilities \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["summary"]["line"])'
                                           # starts with "public demo · guest access"
curl -s http://127.0.0.1:8080/v1/auth/status    # "configured":true once section 5 is done
docker compose logs backend | grep -e SESSION_SECRET -e DEV_ROLE    # no output
```

`"connected":true` needs a working LLM key. `curl http://127.0.0.1:8000` is refused by design, because the demo publishes no backend port. Once the TLS proxy is in place (section 6), the demo is at `https://demo.example.org` and the app at `https://demo.example.org/app`.

## 4. Model pre-install

`models-init` runs `deploy/models_init.py` in the backend image. It installs every managed model into the `models_data` volume, which the backend mounts at `/app/models`:

- Prithvi EO v1 (100M), about 450 MB, from Hugging Face;
- Delineate-Anything (Small), about 85 MB, from Hugging Face, plus its code from GitHub.

A model that is already installed is skipped. So the step is quick when it runs again, and after an upgrade it installs only what is new.

If a download fails, `models-init` exits with code 1 and the backend does not start (`docker compose ps -a` shows `Exited (1)`). Read `docker compose logs models-init` and fix the cause, usually outbound access to Hugging Face or GitHub. Then retry:

```bash
docker compose run --rm models-init    # retry and watch the output
docker compose up -d
```

To check what is installed, run `docker compose exec backend du -sh /app/models` (about 0.5 GB).

Who may download later:

- The demo sets `MODEL_DOWNLOAD_ON_DEMAND=operators`, so guests and members never start a download. When an analysis needs a missing model, it is refused for them before anything downloads.
- An operator can install a missing model from Settings → Models → **Pre-download**.

The volume's name starts with the Compose project name, which is the directory name by default (for example `dt4lc-project_models_data`). `docker compose down -v` deletes it, and the next start downloads the weights again.

## 5. GitHub sign-in: the OAuth app

### 5.1 Create the app

1. Sign in to GitHub with the account that should own the app.
   - **Personal account:** profile picture (top right) → **Settings** → **Developer settings** (at the bottom of the left sidebar) → **OAuth Apps** → **New OAuth App**. The button says **Register a new application** if you have no app yet. The direct link is <https://github.com/settings/developers>.
   - **Organization:** the organization's **Settings** → **Developer settings** → **OAuth Apps**.
2. Fill in the form:

   | Field | Value |
   |-------|-------|
   | Application name | What users see on GitHub's consent page, e.g. `DT4LC demo` |
   | Homepage URL | `https://demo.example.org` |
   | Application description | Optional |
   | Authorization callback URL | `https://demo.example.org/v1/auth/github/callback` |
   | Enable Device Flow | Leave it unchecked |

   The server always sends `${OAUTH_REDIRECT_BASE}/v1/auth/github/callback` as the callback, so register exactly that. An OAuth app has one callback URL, so a second host (a staging copy, say) needs its own app.
3. Click **Register application**.
4. On the app's page, copy the **Client ID**. Under **Client secrets**, click **Generate a new client secret** and copy it straight away: GitHub shows it only once.

### 5.2 Configure the server

Put the three values in `.env`:

```dotenv
GITHUB_CLIENT_ID=<the Client ID>
GITHUB_CLIENT_SECRET=<the client secret>
OAUTH_REDIRECT_BASE=https://demo.example.org
```

`OAUTH_REDIRECT_BASE` is the public origin of the demo as the browser sees it: the scheme and host, with no path. A trailing slash is ignored.

Then create the backend again with the new environment:

```bash
docker compose up -d
```

`docker compose restart` is not enough: it restarts the containers with the environment they were created with.

Check:

```bash
curl -s http://127.0.0.1:8080/v1/auth/status
# {"configured":true,"user":null,"role":"guest"}
curl -s -o /dev/null -D - http://127.0.0.1:8080/v1/auth/github/login | grep -i '^location'
# location: https://github.com/login/oauth/authorize?client_id=...&redirect_uri=https%3A%2F%2Fdemo.example.org%2Fv1%2Fauth%2Fgithub%2Fcallback&scope=read%3Auser+user%3Aemail&state=...
```

The `redirect_uri` must decode to the callback URL you registered.

To sign in:

1. Open `https://demo.example.org/app`.
2. Open the account menu (the round button at the bottom of the left rail) and choose **Sign in with GitHub**.

The first time, GitHub asks the user to authorize the app to read their profile and email addresses (the scopes `read:user` and `user:email`). The server uses GitHub's token only during the callback, to read the profile and the email list, and does not keep it.

### 5.3 Roles

Sign-in looks only at the user's primary email on GitHub, and only if GitHub has verified it. Secondary addresses don't count.

| The primary verified email is | Role | What it means |
|-------------------------------|------|---------------|
| In `OPERATOR_EMAILS` | operator | No quota; sees every job and upload; **Settings** in the account menu |
| In `ALLOWED_EMAILS`, or at a domain in `ALLOWED_DOMAINS` | member | Member badge, no quota; jobs and uploads are kept, on any device |
| On none of the lists | guest | The app says "`<email>` is not on the allowlist for this server" |
| Not verified by GitHub, whatever the lists say | guest | The app says "Your primary GitHub email is not verified — verify it on GitHub, then sign in again" |

- **Matching:**
  - Case is ignored.
  - `ALLOWED_DOMAINS` matches the part after `@` exactly: `unige.ch` admits `ana@unige.ch` but not `ana@etu.unige.ch`. A leading `@` is allowed.
  - An address on two lists is an operator.
- **At sign-in,** the guest session's jobs and uploads move to the user's account.
- **The allowlists are checked on every request.** When you remove an address (in `.env` followed by `docker compose up -d`, or in Settings → Access), the change takes effect on that user's next request: they continue in a new guest session. Their jobs and uploads stay on the server.
- **A sign-in lasts `GUEST_SESSION_HOURS`,** like a guest session. After that the browser is a guest until the user signs in again.
- **Sign out** (account menu → **Sign out**) gives the browser a new guest session.

### 5.4 "Sign-in not configured on this server"

The account menu shows this text instead of **Sign in with GitHub** when the backend does not have all three of `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` and `OAUTH_REDIRECT_BASE`. An empty value counts as missing. Everyone is then a guest:

- `GET /v1/auth/status` answers `"configured":false`;
- `GET /v1/auth/github/login` answers 404 `SIGN_IN_NOT_CONFIGURED`;
- Settings → Access shows "Sign-in not configured".

To fix it, set the three values in `.env` and run `docker compose up -d`. They come from the environment only: Settings cannot set them.

## 6. Reverse proxy and TLS

The frontend listens on `http://127.0.0.1:8080`. Put a TLS-terminating proxy on the host in front of it.

A connection from the host to that loopback port reaches the frontend container from the demo network's gateway, inside `DEMO_SUBNET`. nginx in the container therefore trusts the proxy's `X-Forwarded-For` and takes the last address in it as the client.

The proxy must:

- send the client's address as the last (or only) entry of `X-Forwarded-For`. Otherwise every visitor shares one rate limit and one guest quota.
- accept request bodies up to 210 MB. GeoTIFF uploads can be up to 200 MB.
- pass event streams through without buffering. Job progress and chat replies are Server-Sent Events.
- allow at least 300 s between two reads, as the demo nginx does.

### Caddy

`/etc/caddy/Caddyfile`, for the Debian and Ubuntu packages:

```caddy
demo.example.org {
    reverse_proxy 127.0.0.1:8080
}
```

Then run `sudo systemctl reload caddy`. This is all Caddy needs:

- it gets and renews the certificate itself (ports 80 and 443 must reach it);
- it replaces any incoming `X-Forwarded-For` with the client's address;
- it has no request-size limit by default;
- it passes event streams through as they arrive.

### nginx and certbot

`/etc/nginx/sites-available/dt4lc-demo`:

```nginx
server {
    listen 80;
    server_name demo.example.org;

    client_max_body_size 210m;    # uploads: the backend accepts up to 200 MB per file

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;    # the client, as the only entry
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_buffering off;    # job and chat event streams
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/dt4lc-demo /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d demo.example.org --redirect
```

Certbot gets the certificate and adds the HTTPS lines to this server block. With `--redirect` it also redirects HTTP to HTTPS. The certbot package renews the certificate on a timer.

### Check the client address

From another machine:

```bash
curl -s https://demo.example.org/v1/health
```

Then, on the host:

```bash
docker compose logs --tail 5 frontend
```

The first field of each access-log line is the client address that nginx settled on. It must be the other machine's public address, not the gateway of `DEMO_SUBNET` (`172.30.60.1` by default).

## 7. Rate limits and quotas

There are two layers: nginx limits per IP, then the app's guest quota.

### nginx

The limits are in `deploy/nginx-demo.conf.template`. nginx applies them per client IP, whatever the role:

| Requests | Limit |
|----------|-------|
| `POST` to `/v1/jobs`, `/v1/chat`, `/v1/upload`, `/v1/sample`, `/v1/plan` or `/v1/execute` (these exact paths) | 10 per minute, burst 5: six at once, then one every 6 s |
| Any request to `/v1/…`, `/openapi.json`, `/docs` or `/redoc` | 20 per second, burst 200 |

Pages and static files are not limited. A request over a limit gets 429 with `Retry-After: 10` and the project's error body:

```json
{"ok":false,"error":{"code":"RATE_LIMITED","message":"Too many requests from your address. Wait a few seconds and try again.","details":{}}}
```

nginx does not know who is signed in, so members and operators get the same limits. People behind one shared address, such as an office network, share them.

To tune the limits:

1. Edit `rate=` in the `limit_req_zone` lines and `burst=` in the `limit_req` lines of `deploy/nginx-demo.conf.template`.
2. Run `docker compose restart frontend`. The nginx image renders the template at each start.

The file is tracked by git, so keep your change in a local commit or apply it again after upgrades.

### The guest quota

- **Who:** guests only. Members and operators have no quota.
- **The limit:** a guest may start `GUEST_ANALYSES_PER_HOUR` analyses per rolling hour. The app counts them per session and per client IP (the address nginx forwards), and both counts must have room.
- **Over the limit:** the answer is 429 `GUEST_QUOTA`, with `details: {limit, used, resets_at}` and `Retry-After`. The message says when the next analysis is available.
- **Proposals:** a proposal counts only once it is confirmed.
- **Restarts:** the counts live in memory, so a backend restart resets them.
- **To change it:** set `GUEST_ANALYSES_PER_HOUR` in `.env` and run `docker compose up -d`, or change it in Settings → Access, which applies at once.

## 8. Guest cleanup

The backend runs a cleanup every `GUEST_CLEANUP_MINUTES` minutes: 60 by default, at least 1. The first run comes one interval after the start. Each run removes:

- finished guest jobs submitted more than `GUEST_SESSION_HOURS` ago, with their output files in `resources/.cache/outputs/<job id>/`;
- guest proposals left unconfirmed for that long;
- guest uploads older than `GUEST_SESSION_HOURS`: the GeoTIFF, its preview and its record;
- idle quota counters.

Members' and operators' jobs and uploads are kept, including the guest work adopted at sign-in. Running jobs are never removed. When a run removes something, the backend logs it (`Cleaned up ... old jobs`, `Cleaned up ... expired guest uploads`).

`GUEST_CLEANUP_MINUTES` comes from the environment only. To change it, edit `.env` and run `docker compose up -d`.

If you change `GUEST_SESSION_HOURS` in Settings, the change applies to the next cleanup and to new sessions. Existing session cookies keep the expiry they were issued with.

## 9. Operator Settings

Operators open Settings from the account menu, or at `/app/settings`. Everyone else gets the not-found page there, and the API answers `/v1/settings` with 404, as it does for an unknown path.

| Section | What an operator can do |
|---------|-------------------------|
| Assistant | Set the provider order, turn each provider on or off, and pick its model or models. **Test** sends a short prompt to each enabled provider in the saved order. Keys show only as present or absent. |
| Models | **Pre-download** a model that is not installed. Choose who may download on first use: everyone, members and operators, operators only, or nobody. With "nobody", Pre-download is refused too. |
| Access | Edit the allowed emails, allowed domains and operator emails (you cannot remove your own address), the guest analyses per hour (0–1000) and the guest session hours (1–720). The section also shows whether sign-in is configured. |
| Data sources | Read-only: the Earth Engine status, the uploads directory, the bundled sample |
| Server | Read-only: the version, the deployment, the paths, the storage used, the jobs |

How saving works:

- **Save** writes the changed values to the overrides file and they apply at once, without a restart. The file is `DT4LC_SETTINGS_PATH`: by default `resources/.cache/settings.json` on the host, which is `/app/resources/.cache/settings.json` in the container. `.env` is never written, and no secret goes into the file.
- **An override wins over `.env`.** Each field shows where its value comes from:
  - "override": the overrides file;
  - "from .env": the server's environment. For the download policy on the demo, that is the `operators` value set in `docker-compose.demo.yml`;
  - "default".
- **While a field has an override,** editing the same variable in `.env` does not change it. **Reset to .env** drops the override when you save.
- **The file** is a flat JSON object from variable name to string, for example `{"GUEST_ANALYSES_PER_HOUR": "3", "LLM_PROVIDER_ORDER": "groq,gemini"}`.
  - The backend reads it again whenever it changes, so a hand edit applies too.
  - Unknown names are ignored, with a warning in the log.
  - Deleting the file drops every override.
- **Only these variables can be overridden:** `LLM_PROVIDER_ORDER`, `LLM_ENABLE_<provider>`, `GEMINI_MODELS`, `GROQ_MODEL`, `OLLAMA_MODEL`, `APERTUS_MODEL`, `ANTHROPIC_MODELS`, `MISTRAL_MODELS`, `OPENAI_MODELS`, `MODEL_DOWNLOAD_ON_DEMAND`, `ALLOWED_EMAILS`, `ALLOWED_DOMAINS`, `OPERATOR_EMAILS`, `GUEST_ANALYSES_PER_HOUR` and `GUEST_SESSION_HOURS`.
- **Everything else stays in `.env`:** API keys, the session and GitHub variables, `GUEST_CLEANUP_MINUTES`, ports, the subnet. After you change one of them, run `docker compose up -d`.

## 10. Upgrades and backups

### Upgrade

```bash
git pull                   # or: git fetch --tags && git checkout <new tag>
docker compose up -d --build
docker compose ps -a
```

Before you start the new version, compare `.env.example` with your `.env` for new variables.

`models-init` runs again and installs only missing models. The data in `resources/.cache/` is kept: the jobs database, the uploads, the results and the overrides file.

### Back up

| What | Where |
|------|-------|
| Jobs database (SQLite) | `resources/.cache/jobs.db`, with `jobs.db-wal` and `jobs.db-shm` next to it |
| Uploads | `resources/.cache/uploads/` |
| Results | `resources/.cache/outputs/` |
| Operator overrides | `resources/.cache/settings.json` |
| Layer metadata | `resources/.cache/layer_metadata/` |
| Configuration and secrets | `.env`: store it as carefully as the secrets in it |

```bash
docker compose stop backend
sudo tar -czf dt4lc-cache-$(date +%F).tar.gz resources/.cache
docker compose start backend
```

- **Why stop the backend:** it keeps a consistent copy of the SQLite database. The API is unavailable while it is stopped.
- **Why `sudo`:** the backend runs as root in its container, so the files belong to root on the host.
- **To restore:**

  ```bash
  docker compose stop backend
  sudo mv resources/.cache resources/.cache.old
  sudo tar -xzf dt4lc-cache-<date>.tar.gz
  docker compose start backend
  ```

The model weights in `models_data` need no backup: `models-init` downloads them again.

### Stop, remove, switch

- **`docker compose down`** stops and removes the containers and the network. `resources/.cache/` and the `models_data` volume stay.
- **`docker compose down -v`** also deletes the project's named volumes, `models_data` among them.
- **To switch one host between the default stack and the demo,** run `docker compose down` with the old configuration first, before you change `COMPOSE_FILE`: the demo pins the network's subnet. The two stacks use the same container names (`dt4lc-backend`, `dt4lc-frontend`), so they cannot run side by side.

## 11. Troubleshooting

**`Pool overlaps with other one on this address space` on `docker compose up`.**
Another network on the host uses `172.30.60.0/24`. Set `DEMO_SUBNET` in `.env` to a free private /24, for example `172.30.61.0/24`, then run `docker compose down` and `docker compose up -d`.

**Everyone shares one rate limit or one guest quota, and the frontend log shows the gateway (`172.30.60.1`) for every request.**
The proxy does not send `X-Forwarded-For`, or clients reach the frontend port without the proxy. Fix the proxy (section 6) and keep `FRONTEND_PORT` on `127.0.0.1`. On Docker Desktop every connection to a published port arrives from the gateway, so use Docker Engine on Linux.

**`413 Request Entity Too Large` on an upload.**
An HTML 413 comes from a proxy's body limit: usually the host proxy (nginx's default is 1 MB, so set `client_max_body_size 210m`), or the demo nginx for a file over 210 MB. A JSON 413 with `FILE_TOO_LARGE` comes from the backend: the file is over 200 MB.

**Job progress or chat replies arrive all at once, at the end.**
The proxy buffers the event streams. For nginx, add `proxy_buffering off`.

**The backend does not start, or the frontend never starts.**
Run `docker compose ps -a`. If `models-init` exited with a non-zero code, see section 4. Otherwise read `docker compose logs backend`. The frontend waits until the backend is healthy.

**The account menu says "Sign-in not configured on this server".**
See section 5.4.

**GitHub shows an error about the `redirect_uri`.**
The callback URL registered in the OAuth app does not match `${OAUTH_REDIRECT_BASE}/v1/auth/github/callback`. Compare it with the `redirect_uri` in the login redirect (section 5.2).

**After GitHub, the app says "Sign-in failed — try again".**
The backend logs the reason. Run `docker compose logs backend | grep "GitHub sign-in"`:

- `refused: missing code or state, or state mismatch`: the state cookie did not come back. Check three things:
  - the browser's address must be exactly the `OAUTH_REDIRECT_BASE` origin, so watch for `www.` and for `http` against `https`;
  - `SESSION_COOKIE_SECURE=true` needs HTTPS;
  - the user must finish on GitHub within 10 minutes.
- `refused: access_denied`: the user cancelled on GitHub.
- `failed: token exchange failed: ...`: the client secret is usually wrong or revoked. Generate a new one, put it in `.env` and run `docker compose up -d`.
- `failed:` with a connection error or an HTTP status: check that the host reaches `github.com` and `api.github.com`.

**"`<email>` is not on the allowlist for this server".**
The user's primary verified GitHub email is on no list. Add it in Settings → Access, or in `.env` followed by `docker compose up -d`, and the user signs in again. If the allowlisted address is a secondary one on GitHub, the user makes it primary.

**Settings is missing from the account menu.**
Only operators see it. The user's primary verified email must be in `OPERATOR_EMAILS`.

**Everyone is signed out after a restart.**
`SESSION_SECRET` is unset; the backend logs `SESSION_SECRET is not set`.

**Every visitor is a member or an operator.**
`DEV_ROLE` is set. Remove it from `.env` and run `docker compose up -d`.

**A change in `.env` did nothing.**
`docker compose restart` keeps the old environment, so run `docker compose up -d`. For a variable that Settings can override, check that the field does not say "override".

**`docker compose` commands act on the default stack** (`docker compose config --services` shows no `models-init`).
`COMPOSE_FILE` or `COMPOSE_PROFILES` is missing from `.env`, or the command runs in another directory.

**`curl http://127.0.0.1:8000` is refused.**
This is by design: the demo publishes no backend port. Use `http://127.0.0.1:8080` on the host.
