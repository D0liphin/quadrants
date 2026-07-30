#!/bin/bash
# Quiesce macOS "consumer" background daemons on the CI runner.
#
# Why: deep-capture of the slow Mac test legs showed the 3-vCPU runner's run queue
# fills with ~300 runnable macOS consumer/cloud daemons (nsurlsessiond, Spotlight
# mds, cloudd, locationd, FindMy, remindd, telemetry, ...). The CPU-bound pytest
# workers are only ~1% of the runnable threads but get preempted ~15-19M times, so
# per-test teardown balloons from ~1s to 150-380s. These daemons have no purpose on
# a headless CI VM (no signed-in Apple account, no user, no location).
#
# SIGSTOP removes a process from the run queue immediately and reliably without
# fighting launchd respawn (a stopped process stays alive, so launchd does not
# relaunch it). Only an explicit allowlist of consumer services is touched; core
# system daemons (launchd, WindowServer, configd, mDNSResponder, trustd, securityd,
# opendirectoryd, the Actions runner, ssh, python) are never touched.
#
# Usage: quiesce_daemons.sh [full|restop|resume]
#   full   - disable Spotlight + SIGSTOP the allowlist (call once, after pip installs)
#   restop - just (re-)SIGSTOP the allowlist (cheap; call periodically to catch respawns)
#   resume - SIGCONT the allowlist (call at end of the test step so uploads / post-job
#            cleanup run on a normal system; a SIGSTOP'd process stays stopped otherwise)
set -u
ACTION="${1:-full}"

CONSUMER_DAEMONS=(
  mds mds_stores mdworker mdworker_shared mdbworker mdwrite
  cloudd cloudpaird cloudphotod bird
  nsurlsessiond nsurlstoraged networkserviceproxy
  locationd geod
  remindd calaccessd dataaccessd
  searchpartyuseragent searchpartyd findmydeviced findmylocated
  callservicesd
  dasd systemstats coreduetd
  contextstored knowledge-agent
  appleaccountd accountsd amsaccountsd amsengagementd
  askpermissiond
  homed homened
  assistantd assistantd_service siriknowledged siriinferenced
  symptomsd symptomsd-diag
  useractivityd apsd triald tipsd suggestd parsecd commerce
  photoanalysisd mediaanalysisd analyticsd rapportd
  studentd familycircled screentimed newsd
)

if [ "${ACTION}" = "resume" ]; then
  resumed=0
  for d in "${CONSUMER_DAEMONS[@]}"; do
    if pgrep -x "$d" >/dev/null 2>&1; then
      sudo -n killall -CONT "$d" 2>/dev/null && resumed=$((resumed+1))
    fi
  done
  sudo -n mdutil -a -i on >/dev/null 2>&1 || true
  echo "quiesce(resume): SIGCONT'd ${resumed} consumer daemons at $(date -u +%H:%M:%SZ)"
  exit 0
fi

if [ "${ACTION}" = "full" ]; then
  sudo -n mdutil -a -i off >/dev/null 2>&1 || true
  sudo -n mdutil -a -d      >/dev/null 2>&1 || true
fi

stopped=0
for d in "${CONSUMER_DAEMONS[@]}"; do
  if pgrep -x "$d" >/dev/null 2>&1; then
    sudo -n killall -STOP "$d" 2>/dev/null && stopped=$((stopped+1))
  fi
done
echo "quiesce(${ACTION}): SIGSTOP'd ${stopped} consumer daemons at $(date -u +%H:%M:%SZ)"
