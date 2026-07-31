#!/bin/bash

# [DONOTMERGE] diagnostic helper. Opens an interactive tmate session on the runner and
# exfils the SSH connection string to the PR as a comment, so the driving agent can attach
# from a tmux pane (it cannot scrape GitHub's live logs). The runner then blocks until the
# operator creates /tmp/continue (or the timeout elapses), keeping the built-from-main
# environment (venv with the freshly built wheel + checked-out test tree) alive for
# interactive lldb / pdb work on the vulkan f64 subgroup crash.

set -x

brew list tmate >/dev/null 2>&1 || brew install tmate

SOCK=/tmp/tmate.sock
rm -f "${SOCK}"
tmate -S "${SOCK}" new-session -d
tmate -S "${SOCK}" wait tmate-ready
SSH=$(tmate -S "${SOCK}" display -p '#{tmate_ssh}')
WEB=$(tmate -S "${SOCK}" display -p '#{tmate_web}')
echo "TMATE_SSH=${SSH}"
echo "TMATE_WEB=${WEB}"

# Preset an env-bootstrap for the interactive shell: cd into the workspace and export the
# runtime lib dir so `python tests/run_tests.py ...` works immediately after connecting.
cat > /tmp/diag_env.sh <<EOF
cd "${GITHUB_WORKSPACE}"
export QD_LIB_DIR="\$(python -c 'import quadrants as qd; print(qd.__path__[0])' | tail -n 1)/_lib/runtime"
export MVK_CONFIG_LOG_LEVEL=1
echo "diag env ready: cwd=\$(pwd) python=\$(command -v python)"
EOF

BODY=$(printf 'tmate ready (run %s). Attach with:\n```\n%s\n```\nweb: %s\n\nOn connect: `source /tmp/diag_env.sh`. End the session with `touch /tmp/continue`.' \
  "${GITHUB_RUN_ID}" "${SSH}" "${WEB}")
gh pr comment "${PR_NUMBER:-830}" --repo "${GITHUB_REPOSITORY}" --body "${BODY}" || echo "gh pr comment failed (check pull-requests: write permission)"

echo "Waiting up to 60 min for /tmp/continue ..."
for _ in $(seq 1 3600); do
  [ -f /tmp/continue ] && { echo "continue sentinel seen; ending session"; break; }
  sleep 1
done
