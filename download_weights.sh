#!/usr/bin/env bash
# Download the two v1.1 checkpoints from the GitHub Release (release assets,
# not Git LFS).  Verifies against CHECKSUMS.sha256.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
REPO="tualatint/hanoi_general_newcam_1.0"
TAG="v1.1"

declare -A ASSETS=(
  [bc_vservo_transport_gen_v2_uni_sem7.ckpt]="runs/bc_vservo_transport_gen_v2_uni_sem7/best.ckpt"
  [bc_vservo_insert_v2b.ckpt]="runs/bc_vservo_insert_v2b/best.ckpt"
)

verify() {
    local dest="$1"
    local expected
    expected="$(awk -v p="${dest}" '$2==p {print $1}' CHECKSUMS.sha256)"
    [ -n "$expected" ] || return 1
    [ -f "$dest" ] || return 1
    [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$expected" ]
}

for name in "${!ASSETS[@]}"; do
    dest="${ASSETS[$name]}"
    mkdir -p "$(dirname "$dest")"
    if verify "$dest"; then
        echo "[download_weights] verified ${dest}"
        continue
    fi
    url="https://github.com/${REPO}/releases/download/${TAG}/${name}"
    echo "[download_weights] ${name} -> ${dest}"
    if ! curl -fL --connect-timeout 20 --max-time 1200 --retry 5 --retry-delay 5 \
            -o "${dest}.part" "${url}"; then
        echo "[download_weights] direct URL failed; using the API asset endpoint"
        api_url="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/tags/${TAG}" \
            | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(next(a['url'] for a in d['assets'] if a['name']=='${name}'))")"
        if ! curl -fL --connect-timeout 20 --max-time 1200 --retry 5 --retry-delay 5 \
                -H "Accept: application/octet-stream" -o "${dest}.part" "${api_url}"; then
            wget -c --tries=10 --timeout=60 --waitretry=5 -O "${dest}.part" "${api_url}"
        fi
    fi
    mv "${dest}.part" "${dest}"
done

sha256sum -c CHECKSUMS.sha256
echo "[download_weights] both checkpoints verified"
