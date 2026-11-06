#!/usr/bin/env bash
# Regression gate for state/cn360-tools/fastfail.so.
#
# The shim is load bearing: tools/capture/start_cn360_platform.sh and
# start_cn360_session.sh load it so the Game Center's login frame fails fast and
# the launcher surfaces its "在浏览器上登录" fallback (see AGENTS.md 2026-10-16).
# Three properties must hold, and every one of them is checked here against a
# real socket - the shim is a syscall interceptor, so a fake socket would prove
# nothing:
#
#   1. an address in FASTFAIL_IPS is refused immediately instead of hanging;
#   2. with FASTFAIL_PROCS set, a process whose comm is not listed is unaffected
#      (otherwise the external browser the launcher opens would be blocked too);
#   3. ordinary traffic that is not listed keeps working.
#
# The "must hang" probes use 171.8.167.89 (s.360.cn), which this network answers
# on ICMP but black-holes on TCP 80/443, so an unblocked connect takes seconds
# and a blocked one is immediate. Bounded by a timeout either way.
#
# Usage: tools/capture/test_fastfail_shim.sh
set -uo pipefail

WORK=$(cd "$(dirname "$0")/../.." && pwd)
SRC="$WORK/state/cn360-tools/fastfail.c"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

BLACKHOLE=171.8.167.89      # s.360.cn, ICMP ok / TCP black-holed here
PASS=0
FAIL=0

ok()   { echo "  PASS  $1"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL + 1)); }

if [ ! -f "$SRC" ]; then echo "missing $SRC" >&2; exit 1; fi
if ! gcc -shared -fPIC -O2 -o "$TMP/fastfail.so" "$SRC" -ldl -lpthread; then
  echo "shim build FAILED" >&2; exit 1
fi
echo "shim built from $SRC"

# connect_py <ip> -> prints "<seconds> <errno>" and exits 0 always.
cat > "$TMP/connect.py" <<'PY'
import socket, sys, time
ip = sys.argv[1]
t = time.time()
s = socket.socket()
s.settimeout(8)
try:
    s.connect((ip, 443))
    print("%.3f 0" % (time.time() - t))
except Exception as e:
    print("%.3f %s" % (time.time() - t, getattr(e, "errno", "n/a")))
finally:
    s.close()
PY

run_probe() {   # run_probe <proc-list-or-empty> <ip>  -> "<secs> <errno>"
  local procs="$1" ip="$2"
  if [ -n "$procs" ]; then
    LD_PRELOAD="$TMP/fastfail.so" FASTFAIL_PROCS="$procs" \
      FASTFAIL_IPS="$BLACKHOLE" FASTFAIL_LOG="$TMP/log" \
      timeout 20 /usr/bin/python3 "$TMP/connect.py" "$ip" 2>/dev/null | tail -1
  else
    LD_PRELOAD="$TMP/fastfail.so" \
      FASTFAIL_IPS="$BLACKHOLE" FASTFAIL_LOG="$TMP/log" \
      timeout 20 /usr/bin/python3 "$TMP/connect.py" "$ip" 2>/dev/null | tail -1
  fi
}

echo "== 1. listed address is refused immediately"
rm -f "$TMP/log"
res=$(run_probe "" "$BLACKHOLE")
secs=${res%% *}; errno=${res##* }
if awk -v s="${secs:-99}" 'BEGIN{exit !(s < 1.0)}' && [ "$errno" = "111" ]; then
  ok "refused in ${secs}s with ECONNREFUSED"
else
  bad "expected <1s and errno 111, got '${res:-<empty>}'"
fi

if grep -q "BLOCK $BLACKHOLE:443" "$TMP/log" 2>/dev/null; then
  ok "the refusal is recorded in FASTFAIL_LOG"
else
  bad "FASTFAIL_LOG has no BLOCK line for $BLACKHOLE"
fi

echo "== 2. FASTFAIL_PROCS allowlist keeps other processes out"
rm -f "$TMP/log"
res=$(run_probe "wgc.exe,CrUtilityMain" "$BLACKHOLE")
secs=${res%% *}
if awk -v s="${secs:-0}" 'BEGIN{exit !(s > 1.0)}'; then
  ok "unlisted process was NOT refused (took ${secs}s, as an unblocked connect does)"
else
  bad "unlisted process was refused in ${secs}s; the allowlist is not effective"
fi
if [ ! -s "$TMP/log" ] || ! grep -q "BLOCK" "$TMP/log" 2>/dev/null; then
  ok "nothing was blocked or logged for the unlisted process"
else
  bad "the unlisted process still produced block records"
fi

echo "== 3. ordinary traffic is untouched"
res=$(LD_PRELOAD="$TMP/fastfail.so" FASTFAIL_PROCS="wgc.exe" \
      FASTFAIL_IPS="$BLACKHOLE" FASTFAIL_LOG="$TMP/log3" \
      timeout 20 curl -sS -o /dev/null -m 12 -w '%{http_code}' https://www.baidu.com/ 2>/dev/null)
if [ "$res" = "200" ]; then
  ok "an unlisted host still answers (HTTP 200)"
else
  bad "unlisted host returned '${res:-<empty>}' instead of 200"
fi

echo
echo "passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ]
