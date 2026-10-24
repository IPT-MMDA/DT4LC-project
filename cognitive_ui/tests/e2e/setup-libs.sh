#!/usr/bin/env bash
# Extract the shared libraries chrome-headless-shell needs (libnss3, libnspr4,
# libasound2) into tests/e2e/.libroot without root. run.mjs adds
# .libroot/usr/lib/x86_64-linux-gnu to LD_LIBRARY_PATH when it exists.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$here/.libroot"
debs="$root/.debs"

rm -rf "$debs"
mkdir -p "$debs"
cd "$debs"

apt-get download libnss3 libnspr4
apt-get download libasound2t64 || apt-get download libasound2

for deb in ./*.deb; do
  dpkg -x "$deb" "$root"
done
cd "$here"
rm -rf "$debs"

echo "Libraries extracted into $root/usr/lib/x86_64-linux-gnu"
