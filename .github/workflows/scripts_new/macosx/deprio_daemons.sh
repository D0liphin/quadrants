#!/bin/bash
# Deprioritize (do NOT stop) the macOS consumer-daemon swarm.
#
# Why not SIGSTOP: a stopped daemon still exists, so any synchronous XPC/Mach call to
# it blocks the caller until it is resumed. In the quadrants test phase that turned
# per-test teardown into multi-minute hangs (worse than the original problem). Here we
# leave the daemons fully alive and answering, we only move them to the Darwin
# background scheduling tier (throttled CPU + deferred I/O) and BSD nice +20, so on the
# 3-vCPU runner they always yield the cores to the CPU-bound pytest workers.
#
# Only an explicit allowlist of consumer services is touched; core system daemons
# (launchd, WindowServer, configd, mDNSResponder, trustd, securityd, opendirectoryd,
# the Actions runner, ssh, python) are never touched. Idempotent: safe to call
# repeatedly from the sampler to catch respawns / newly-woken daemons.
#
# Usage: deprio_daemons.sh
set -u

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

n=0
for d in "${CONSUMER_DAEMONS[@]}"; do
  for pid in $(pgrep -x "$d" 2>/dev/null); do
    # BSD nice: lowest user priority.
    sudo -n renice 20 -p "$pid" >/dev/null 2>&1 || true
    # Darwin background tier: throttled scheduling + deferred/throttled disk I/O.
    if sudo -n taskpolicy -b -p "$pid" >/dev/null 2>&1; then
      n=$((n+1))
    fi
  done
done
echo "deprio: throttled ${n} consumer-daemon pids to background tier at $(date -u +%H:%M:%SZ)"
