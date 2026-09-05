#!/usr/bin/env bash
# Provision the Linux backend used by the repository-repair consumer tests.
set -euo pipefail

if [[ $(uname -s) != Linux ]]; then
  echo "This setup script requires Linux; macOS uses sandbox-exec." >&2
  exit 1
fi
if ! command -v bwrap >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y bubblewrap
fi

probe() {
  bwrap --unshare-all --die-with-parent --ro-bind / / --proc /proc --dev /dev -- /usr/bin/true
}
if probe 2>/dev/null; then
  exit 0
fi

# Ubuntu 24.04 can restrict unprivileged user namespaces through AppArmor.
# Grant that capability only to bubblewrap; keep the global restriction intact.
if [[ -r /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]] &&
   [[ $(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns) == 1 ]] &&
   command -v apparmor_parser >/dev/null; then
  sudo tee /etc/apparmor.d/rig-ecs-consumer-bwrap >/dev/null <<'PROFILE'
abi <abi/4.0>,
include <tunables/global>
profile rig-ecs-consumer-bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
}
PROFILE
  sudo apparmor_parser -r /etc/apparmor.d/rig-ecs-consumer-bwrap
fi
# Fail visibly if namespaces are unavailable; never run model code unsandboxed.
probe
