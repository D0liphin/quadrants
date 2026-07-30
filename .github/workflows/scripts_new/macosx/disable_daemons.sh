#!/bin/bash
# Fail-fast quiesce: DISABLE + BOOTOUT the macOS consumer-daemon launchd jobs.
#
# Why not SIGSTOP: a stopped daemon still exists, so a synchronous XPC/Mach call to it
# blocks the caller until it resumes (in the quadrants test phase this turned per-test
# teardown into multi-minute hangs). `launchctl bootout` unloads the job entirely, so a
# caller gets an immediate "service unavailable" instead of an indefinite hang, and
# `launchctl disable` stops launchd relaunching it for the rest of the boot. Anything
# the runner genuinely needs for artifact upload (DNS via mDNSResponder, raw TCP
# sockets from the .NET/Node runner) is untouched.
#
# Best-effort: every launchctl call is allowed to fail (SIP-protected or not present);
# we count and report how many jobs were actually booted out. Tried in the system and
# per-user (gui/user) domains since some of these run as user agents on the runner.
#
# Usage: disable_daemons.sh
set -u

UID_NUM="$(id -u)"

LABELS=(
  com.apple.metadata.mds com.apple.metadata.mds.spindump
  com.apple.cloudd com.apple.cloudpaird com.apple.cloudphotod com.apple.bird
  com.apple.nsurlsessiond com.apple.networkserviceproxy
  com.apple.locationd com.apple.geod
  com.apple.remindd com.apple.calaccessd com.apple.dataaccessd
  com.apple.searchpartyuseragent com.apple.searchpartyd com.apple.findmydeviced
  com.apple.callservicesd
  com.apple.dasd com.apple.systemstats com.apple.coreduetd
  com.apple.contextstored com.apple.knowledge-agent
  com.apple.appleaccountd com.apple.accountsd com.apple.amsaccountsd com.apple.amsengagementd
  com.apple.askpermissiond
  com.apple.homed com.apple.homened
  com.apple.assistantd com.apple.siriknowledged com.apple.siriinferenced
  com.apple.symptomsd
  com.apple.apsd com.apple.triald com.apple.tipsd com.apple.suggestd
  com.apple.parsecd com.apple.commerce
  com.apple.photoanalysisd com.apple.mediaanalysisd com.apple.analyticsd com.apple.rapportd
  com.apple.newsd com.apple.ScreenTimeAgent com.apple.familycircled
)

# Spotlight the reliable way (mds/mdworker storm).
sudo -n mdutil -a -i off >/dev/null 2>&1 || true
sudo -n mdutil -a -d      >/dev/null 2>&1 || true

n=0
for lbl in "${LABELS[@]}"; do
  for dom in "system" "gui/${UID_NUM}" "user/${UID_NUM}"; do
    sudo -n launchctl disable "${dom}/${lbl}" >/dev/null 2>&1 || true
    if sudo -n launchctl bootout "${dom}/${lbl}" >/dev/null 2>&1; then
      n=$((n+1))
    fi
  done
done
echo "disable: bootout'd ${n} consumer launchd jobs (disabled for boot) at $(date -u +%H:%M:%SZ)"
