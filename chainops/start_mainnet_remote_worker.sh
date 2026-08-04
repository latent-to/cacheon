#!/usr/bin/env bash
# Register one READY Lium/B300 worker epoch and start both durable tmux services.
#
# Run this once on the standing CPU validator.  It copies only the reviewed
# remote service, the reviewed fixed deployment adapter, and a non-secret
# registration record.  It does not copy the intake database, wallet, chain
# keys, object-store credentials, or arbitrary source roots.  After the final
# READY line, the operator terminal, laptop, and Codex process are irrelevant.
set -euo pipefail
umask 077

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd -P)

POD_HOST=
POD_PORT=
KNOWN_HOSTS=/root/cacheon-ops/state/lium-worker-known-hosts
WORKER_READINESS=
SERVICE_IDENTITY=
ADAPTER=$SCRIPT_DIR/cacheon_b300_evaluation_adapter.py
REMOTE_SERVICE=$SCRIPT_DIR/remote_worker_service.py
CREDENTIAL=/root/cacheon-ops/state/remote-worker-credential.secret
CREDENTIAL_ID=cacheon-mainnet-screen-v1
CPU_PYTHON=/root/miniconda3/envs/prod/bin/python
CACHEON_SOURCE=/root/cacheon-ops/source
SCREEN_DISPATCHER=$SCRIPT_DIR/mainnet_screen_dispatcher.py
SCREEN_DISPATCHER_TEMPLATE=/root/cacheon-ops/state/mainnet-screen-dispatcher-template.json
SPOOL_ROOT=/root/cacheon-ops/remote-worker/spool
STATE_ROOT=/root/cacheon-ops/remote-worker/state
LOG_ROOT=/root/cacheon-ops/logs
POLL_SECONDS=5
MAX_HEARTBEAT_AGE=45
COMMISSION_CURRENT_POD=0
POD_SOURCE_ROOT=
POD_SOURCE_REVISION=
POD_RUNTIME_ROOT=
POD_MODEL_ROOT=
POD_MODEL_RECEIPT=
POD_WORKER_IMAGE=
LANE_DEVICES=0,1,2,3
POD_AUTHORITY_ROOT=/data/cacheon-b300/launch-b300-v3-m4l/primary-authority-v3-m4l
POD_AUTHORITY_CONFIG=
POD_MEASUREMENT_CONFIG=
POD_CALIBRATION_PACKAGE=
POD_CALIBRATION_PROJECTION_RECEIPT=
POD_PROMPT_AUTHORITY=
REMOTE_COMMISSIONED_ROOT=/data/cacheon-b300/remote-worker/commissioned

REMOTE_ROOT=/data/cacheon-b300/remote-worker
REMOTE_BIN=/data/cacheon-b300/worker-bootstrap/bin
REMOTE_SERVICE_DEST=$REMOTE_BIN/remote_worker_service.py
REMOTE_ADAPTER_DEST=$REMOTE_BIN/cacheon-b300-evaluation-adapter
REMOTE_REGISTRATION=$REMOTE_ROOT/registration.json
REMOTE_CREDENTIAL=$REMOTE_ROOT/credential.secret
REMOTE_READY=/data/cacheon-b300/worker-bootstrap/ready-receipt.json
REMOTE_PYTHON=/data/cacheon-b300/venv/bin/python

usage() {
  cat <<'USAGE'
Usage: start_mainnet_remote_worker.sh \
  --pod-host HOST --pod-port PORT \
  --worker-readiness /absolute/path/worker-readiness.json \
  --service-identity ARENA_SERVICE_ID \
  [--known-hosts /root/cacheon-ops/state/lium-worker-known-hosts] \
  [--adapter /root/cacheon-ops/bin/cacheon_b300_evaluation_adapter.py] \
  [--credential /root/cacheon-ops/state/remote-worker-credential.secret] \
  [--credential-id cacheon-mainnet-screen-v1] \
  [--remote-service /root/cacheon-ops/bin/remote_worker_service.py] \
  [--cpu-python /root/miniconda3/envs/prod/bin/python] \
  [--cacheon-source /root/cacheon-ops/source] \
  [--screen-dispatcher /root/cacheon-ops/bin/mainnet_screen_dispatcher.py] \
  [--screen-dispatcher-template /root/cacheon-ops/state/mainnet-screen-dispatcher-template.json] \
  [--spool-root /root/cacheon-ops/remote-worker/spool]

If the existing pod predates the worker-bootstrap READY receipt, add:

  --commission-current-pod \
  --pod-source-root /absolute/current/source \
  --pod-source-revision 40_HEX_COMMIT \
  --pod-runtime-root /absolute/current/runtime-seeds \
  --pod-model-root /absolute/current/model \
  --pod-model-receipt /absolute/model-provision-sha256-DIGEST.json \
  --pod-worker-image repository@sha256:DIGEST
  [--lane-devices 0,1,2,3]
  [--pod-authority-root /data/cacheon-b300/launch-b300-v3-m4l/primary-authority-v3-m4l]

Commissioning runs only bounded identity reads: exact 8xB300 inventory/topology,
source/runtime tree hashing, read-only model-receipt inventory reopening, and
local immutable image inspection. It runs no evaluator, profiler, or kernel.

The pod must already have a verified cacheon-lium-worker-ready-v1 receipt.
The adapter is a reviewed deployment codec with a fixed CLI; miner bytes never
select a command, module, executable, or argv.
USAGE
}

log() {
  printf '[%s] [REMOTE-WORKER-START] %s\n' "$(date -u +%FT%TZ)" "$*"
}

fail() {
  printf '[%s] [REMOTE-WORKER-START-FAIL] %s\n' "$(date -u +%FT%TZ)" "$*" >&2
  exit 1
}

need() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

while (($#)); do
  case "$1" in
    --pod-host)
      (($# >= 2)) || fail "--pod-host requires a value"
      POD_HOST=$2
      shift 2
      ;;
    --pod-port)
      (($# >= 2)) || fail "--pod-port requires a value"
      POD_PORT=$2
      shift 2
      ;;
    --known-hosts)
      (($# >= 2)) || fail "--known-hosts requires a value"
      KNOWN_HOSTS=$2
      shift 2
      ;;
    --worker-readiness)
      (($# >= 2)) || fail "--worker-readiness requires a value"
      WORKER_READINESS=$2
      shift 2
      ;;
    --service-identity)
      (($# >= 2)) || fail "--service-identity requires a value"
      SERVICE_IDENTITY=$2
      shift 2
      ;;
    --adapter)
      (($# >= 2)) || fail "--adapter requires a value"
      ADAPTER=$2
      shift 2
      ;;
    --remote-service)
      (($# >= 2)) || fail "--remote-service requires a value"
      REMOTE_SERVICE=$2
      shift 2
      ;;
    --credential)
      (($# >= 2)) || fail "--credential requires a value"
      CREDENTIAL=$2
      shift 2
      ;;
    --credential-id)
      (($# >= 2)) || fail "--credential-id requires a value"
      CREDENTIAL_ID=$2
      shift 2
      ;;
    --cpu-python)
      (($# >= 2)) || fail "--cpu-python requires a value"
      CPU_PYTHON=$2
      shift 2
      ;;
    --cacheon-source)
      (($# >= 2)) || fail "--cacheon-source requires a value"
      CACHEON_SOURCE=$2
      shift 2
      ;;
    --screen-dispatcher)
      (($# >= 2)) || fail "--screen-dispatcher requires a value"
      SCREEN_DISPATCHER=$2
      shift 2
      ;;
    --screen-dispatcher-template)
      (($# >= 2)) || fail "--screen-dispatcher-template requires a value"
      SCREEN_DISPATCHER_TEMPLATE=$2
      shift 2
      ;;
    --spool-root)
      (($# >= 2)) || fail "--spool-root requires a value"
      SPOOL_ROOT=$2
      shift 2
      ;;
    --poll-seconds)
      (($# >= 2)) || fail "--poll-seconds requires a value"
      POLL_SECONDS=$2
      shift 2
      ;;
    --max-heartbeat-age)
      (($# >= 2)) || fail "--max-heartbeat-age requires a value"
      MAX_HEARTBEAT_AGE=$2
      shift 2
      ;;
    --commission-current-pod)
      COMMISSION_CURRENT_POD=1
      shift
      ;;
    --pod-source-root)
      (($# >= 2)) || fail "--pod-source-root requires a value"
      POD_SOURCE_ROOT=$2
      shift 2
      ;;
    --pod-source-revision)
      (($# >= 2)) || fail "--pod-source-revision requires a value"
      POD_SOURCE_REVISION=$2
      shift 2
      ;;
    --pod-runtime-root)
      (($# >= 2)) || fail "--pod-runtime-root requires a value"
      POD_RUNTIME_ROOT=$2
      shift 2
      ;;
    --pod-model-root)
      (($# >= 2)) || fail "--pod-model-root requires a value"
      POD_MODEL_ROOT=$2
      shift 2
      ;;
    --pod-model-receipt)
      (($# >= 2)) || fail "--pod-model-receipt requires a value"
      POD_MODEL_RECEIPT=$2
      shift 2
      ;;
    --pod-worker-image)
      (($# >= 2)) || fail "--pod-worker-image requires a value"
      POD_WORKER_IMAGE=$2
      shift 2
      ;;
    --lane-devices)
      (($# >= 2)) || fail "--lane-devices requires a value"
      LANE_DEVICES=$2
      shift 2
      ;;
    --pod-authority-root)
      (($# >= 2)) || fail "--pod-authority-root requires a value"
      POD_AUTHORITY_ROOT=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ "$(id -u)" -eq 0 ]] || fail "run on the CPU validator as root"
[[ "$POD_HOST" =~ ^[A-Za-z0-9.-]{1,253}$ ]] || fail "pod host is missing or malformed"
[[ "$POD_PORT" =~ ^[0-9]+$ ]] || fail "pod port is missing or malformed"
((POD_PORT >= 1 && POD_PORT <= 65535)) || fail "pod port is outside 1..65535"
if [[ "$COMMISSION_CURRENT_POD" == 0 ]]; then
  [[ -n "$WORKER_READINESS" && "$WORKER_READINESS" == /* ]] \
    || fail "worker readiness must be an absolute path"
  [[ "$SERVICE_IDENTITY" =~ ^[A-Za-z0-9._:@/+\-]{1,512}$ ]] \
    || fail "service identity is missing or malformed"
fi
[[ "$CREDENTIAL_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$ ]] \
  || fail "credential id is malformed"
[[ "$CPU_PYTHON" == /* && -x "$CPU_PYTHON" ]] \
  || fail "CPU Python must be an absolute executable regular path"
[[ "$CACHEON_SOURCE" == /* && -d "$CACHEON_SOURCE/cacheon" && ! -L "$CACHEON_SOURCE" ]] \
  || fail "Cacheon source must be an absolute checkout root"
[[ -f "$CACHEON_SOURCE/cacheon/chain/remote_evaluation_dispatcher.py" ]] \
  || fail "Cacheon source lacks the authenticated remote dispatcher"
[[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || fail "poll seconds is malformed"
((POLL_SECONDS >= 1 && POLL_SECONDS <= 60)) || fail "poll seconds is outside 1..60"
[[ "$MAX_HEARTBEAT_AGE" =~ ^[0-9]+$ ]] || fail "heartbeat age is malformed"
((MAX_HEARTBEAT_AGE >= 10 && MAX_HEARTBEAT_AGE <= 300)) \
  || fail "heartbeat age is outside 10..300"
if [[ "$COMMISSION_CURRENT_POD" == 1 ]]; then
  [[ "$LANE_DEVICES" =~ ^[0-7](,[0-7]){0,7}$ ]] \
    || fail "lane devices must be a comma-separated physical GPU list"
  for path_value in "$POD_RUNTIME_ROOT" "$POD_MODEL_ROOT" "$POD_MODEL_RECEIPT"; do
    [[ "$path_value" =~ ^/[A-Za-z0-9._/+\-]+$ ]] \
      || fail "commissioning paths must be absolute and shell-closed"
  done
  [[ "$POD_SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
    || fail "pod source revision must be exact lowercase 40-hex"
  [[ "$POD_WORKER_IMAGE" =~ ^[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}$ ]] \
    || fail "pod worker image must be an immutable repo digest"
  [[ "$POD_AUTHORITY_ROOT" =~ ^/[A-Za-z0-9._/+\-]+$ ]] \
    || fail "pod authority root must be absolute and shell-closed"
  POD_AUTHORITY_CONFIG=$POD_AUTHORITY_ROOT/authority-config.json
  POD_MEASUREMENT_CONFIG=$POD_AUTHORITY_ROOT/measurement-config.json
  POD_CALIBRATION_PACKAGE=$POD_AUTHORITY_ROOT/calibration-package.json
  POD_CALIBRATION_PROJECTION_RECEIPT=$POD_AUTHORITY_ROOT/calibration-projection-receipt.json
  POD_PROMPT_AUTHORITY=$POD_AUTHORITY_ROOT/prompt-authority.json
fi

for command_name in python3 ssh scp ssh-keygen sha256sum awk stat install mktemp tmux openssl grep seq sleep git; do
  need "$command_name"
done

install -d -m 0700 "$(dirname "$CREDENTIAL")"
if [[ ! -e "$CREDENTIAL" ]]; then
  credential_tmp=$(mktemp "$(dirname "$CREDENTIAL")/.remote-worker-credential.XXXXXX")
  openssl rand 48 >"$credential_tmp"
  chmod 0400 "$credential_tmp"
  mv -- "$credential_tmp" "$CREDENTIAL"
  log "created one root-only remote worker credential: $CREDENTIAL"
fi

for path in "$KNOWN_HOSTS" "$ADAPTER" "$REMOTE_SERVICE" "$CREDENTIAL" "$SCREEN_DISPATCHER" "$SCREEN_DISPATCHER_TEMPLATE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required file missing or symlink: $path"
done
if [[ "$COMMISSION_CURRENT_POD" == 0 ]]; then
  [[ -f "$WORKER_READINESS" && ! -L "$WORKER_READINESS" ]] \
    || fail "worker readiness is missing or symlinked"
fi
[[ "$KNOWN_HOSTS" == /* && "$ADAPTER" == /* && "$REMOTE_SERVICE" == /* && "$CREDENTIAL" == /* ]] \
  || fail "known_hosts, adapter, and service paths must be absolute"
known_mode=$(stat -c %a "$KNOWN_HOSTS")
[[ "$known_mode" == 600 || "$known_mode" == 400 ]] \
  || fail "known_hosts must have mode 0600 or 0400"
adapter_mode=$(stat -c %a "$ADAPTER")
((8#$adapter_mode & 0022)) && fail "adapter must not be group/world writable"
[[ "$(stat -c %u "$ADAPTER")" -eq 0 ]] || fail "adapter must be root-owned"
credential_mode=$(stat -c %a "$CREDENTIAL")
[[ "$credential_mode" == 600 || "$credential_mode" == 400 ]] \
  || fail "remote worker credential must have mode 0600 or 0400"
credential_size=$(stat -c %s "$CREDENTIAL")
((credential_size >= 32 && credential_size <= 4096)) \
  || fail "remote worker credential must contain 32..4096 bytes"

if ! ssh-keygen -F "[$POD_HOST]:$POD_PORT" -f "$KNOWN_HOSTS" >/dev/null 2>&1; then
  if [[ "$POD_PORT" == 22 ]] && ssh-keygen -F "$POD_HOST" -f "$KNOWN_HOSTS" >/dev/null 2>&1; then
    :
  else
    fail "pinned known_hosts has no entry for the exact pod endpoint"
  fi
fi

install -d -m 0700 "$STATE_ROOT/registrations" "$SPOOL_ROOT/outbox" \
  "$SPOOL_ROOT/results" "$SPOOL_ROOT/state" "$LOG_ROOT"
for guarded in "$STATE_ROOT" "$STATE_ROOT/registrations" "$SPOOL_ROOT" "$LOG_ROOT"; do
  [[ ! -L "$guarded" ]] || fail "refusing symlinked local state path: $guarded"
done

SSH=(
  ssh -p "$POD_PORT"
  -o "UserKnownHostsFile=$KNOWN_HOSTS"
  -o StrictHostKeyChecking=yes
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=2
  "root@$POD_HOST"
)
SCP=(
  scp -q -P "$POD_PORT"
  -o "UserKnownHostsFile=$KNOWN_HOSTS"
  -o StrictHostKeyChecking=yes
  -o BatchMode=yes
  -o ConnectTimeout=15
)

temporary_root=$(mktemp -d "$STATE_ROOT/.register.XXXXXX")
cleanup() {
  case "$temporary_root" in
    "$STATE_ROOT"/.register.*) rm -rf -- "$temporary_root" ;;
    *) fail "unsafe temporary registration path" ;;
  esac
}
trap cleanup EXIT

READY_COPY=$temporary_root/ready-receipt.json
REGISTRATION_BUILD=$temporary_root/registration.json
ARENA_MANIFEST_COPY=
SCREEN_DEPLOYMENT_COPY=

if [[ "$COMMISSION_CURRENT_POD" == 1 ]]; then
  [[ "$(git -C "$CACHEON_SOURCE" rev-parse HEAD)" == "$POD_SOURCE_REVISION" ]] \
    || fail "Cacheon checkout HEAD differs from pod source revision"
  [[ -z "$(git -C "$CACHEON_SOURCE" status --porcelain --untracked-files=no)" ]] \
    || fail "Cacheon checkout has tracked changes; commit the exact deployment first"
  POD_SOURCE_ROOT=/data/cacheon-b300/source-$POD_SOURCE_REVISION
  source_archive=$temporary_root/source-$POD_SOURCE_REVISION.tar
  git -C "$CACHEON_SOURCE" archive --format=tar --output "$source_archive" "$POD_SOURCE_REVISION"
  source_archive_sha=$(file_sha256 "$source_archive")
  remote_source_archive=/data/cacheon-b300/worker-bootstrap/incoming/source-$POD_SOURCE_REVISION-$source_archive_sha.tar
  service_sha=$(file_sha256 "$REMOTE_SERVICE")
  commission_part=$REMOTE_BIN/.remote_worker_service.commission.incoming
  log "installing the bounded current-pod commissioner"
  "${SSH[@]}" 'bash -s' -- "$REMOTE_PYTHON" <<'REMOTE'
set -euo pipefail
python=$1
    test "$(id -u)" -eq 0
    test -x "$python"
    install -d -m 0700 /data/cacheon-b300/worker-bootstrap/bin /data/cacheon-b300/worker-bootstrap/incoming /data/cacheon-b300/worker-bootstrap /data/cacheon-b300/remote-worker
    test ! -L /data/cacheon-b300/worker-bootstrap
    test ! -L /data/cacheon-b300/remote-worker
REMOTE
  "${SCP[@]}" -- "$REMOTE_SERVICE" "root@$POD_HOST:$commission_part"
  "${SSH[@]}" 'bash -s' -- "$commission_part" "$REMOTE_SERVICE_DEST" "$service_sha" <<'REMOTE'
set -euo pipefail
incoming=$1
destination=$2
expected=$3
[[ "$incoming" == /data/cacheon-b300/worker-bootstrap/bin/.remote_worker_service.commission.incoming ]]
[[ "$(sha256sum "$incoming" | awk '{print $1}')" == "$expected" ]]
chmod 0500 "$incoming"
mv -f -- "$incoming" "$destination"
REMOTE
  log "installing exact committed Cacheon source $POD_SOURCE_REVISION"
  "${SCP[@]}" -- "$source_archive" "root@$POD_HOST:$remote_source_archive"
  "${SSH[@]}" \
    "'$REMOTE_PYTHON' '$REMOTE_SERVICE_DEST' install-source --archive '$remote_source_archive' --archive-sha256 '$source_archive_sha' --source-revision '$POD_SOURCE_REVISION'"
  log "commissioning current pod identities without running GPU work"
  "${SSH[@]}" \
    "'$REMOTE_PYTHON' '$REMOTE_SERVICE_DEST' commission-current-pod --source-root '$POD_SOURCE_ROOT' --source-revision '$POD_SOURCE_REVISION' --runtime-root '$POD_RUNTIME_ROOT' --model-root '$POD_MODEL_ROOT' --model-receipt '$POD_MODEL_RECEIPT' --worker-image '$POD_WORKER_IMAGE' --python-executable '$REMOTE_PYTHON' --lane-devices '$LANE_DEVICES' --pod-endpoint '$POD_HOST:$POD_PORT' --output '$REMOTE_READY'"
fi

log "reading the exact READY receipt through the pinned endpoint"
"${SCP[@]}" -- "root@$POD_HOST:$REMOTE_READY" "$READY_COPY"

if [[ "$COMMISSION_CURRENT_POD" == 1 ]]; then
  log "materializing the exact sealed TP4 screen service identities"
  "${SSH[@]}" 'bash -s' -- \
    "$REMOTE_PYTHON" "$POD_SOURCE_ROOT" "$REMOTE_READY" \
    "$POD_AUTHORITY_CONFIG" "$POD_MEASUREMENT_CONFIG" \
    "$POD_CALIBRATION_PACKAGE" "$POD_CALIBRATION_PROJECTION_RECEIPT" \
    "$POD_PROMPT_AUTHORITY" "$REMOTE_COMMISSIONED_ROOT" <<'REMOTE'
set -euo pipefail
python=$1
source_root=$2
ready_receipt=$3
authority_config=$4
measurement_config=$5
calibration_package=$6
calibration_projection_receipt=$7
prompt_authority=$8
output_root=$9
test -x "$python"
test -f "$source_root/cacheon/eval/b300_screen_deployment.py"
for authority in "$ready_receipt" "$authority_config" "$measurement_config" "$calibration_package" "$calibration_projection_receipt" "$prompt_authority"; do
  test -f "$authority"
  test ! -L "$authority"
done
env PYTHONPATH="$source_root" PYTHONDONTWRITEBYTECODE=1 \
  "$python" -m cacheon.eval.b300_screen_deployment materialize \
  --ready-receipt "$ready_receipt" \
  --authority-config "$authority_config" \
  --measurement-config "$measurement_config" \
  --calibration-package "$calibration_package" \
  --calibration-projection-receipt "$calibration_projection_receipt" \
  --prompt-authority "$prompt_authority" \
  --output-root "$output_root"
for product in screen-deployment.json arena-service-manifest.json worker-readiness.json; do
  test -f "$output_root/$product"
  test ! -L "$output_root/$product"
done
REMOTE
  ARENA_MANIFEST_COPY=$temporary_root/arena-service-manifest.json
  WORKER_READINESS=$temporary_root/worker-readiness.json
  SCREEN_DEPLOYMENT_COPY=$temporary_root/screen-deployment.json
  "${SCP[@]}" -- \
    "root@$POD_HOST:$REMOTE_COMMISSIONED_ROOT/arena-service-manifest.json" \
    "$ARENA_MANIFEST_COPY"
  "${SCP[@]}" -- \
    "root@$POD_HOST:$REMOTE_COMMISSIONED_ROOT/worker-readiness.json" \
    "$WORKER_READINESS"
  "${SCP[@]}" -- \
    "root@$POD_HOST:$REMOTE_COMMISSIONED_ROOT/screen-deployment.json" \
    "$SCREEN_DEPLOYMENT_COPY"
  chmod 0400 "$ARENA_MANIFEST_COPY" "$WORKER_READINESS" "$SCREEN_DEPLOYMENT_COPY"
  SERVICE_IDENTITY=$(env PYTHONPATH="$CACHEON_SOURCE" "$CPU_PYTHON" - \
    "$SCREEN_DISPATCHER" "$ARENA_MANIFEST_COPY" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

dispatcher_path, manifest_path = sys.argv[1:]
specification = importlib.util.spec_from_file_location(
    "cacheon_fixed_mainnet_screen_dispatcher", dispatcher_path
)
if specification is None or specification.loader is None:
    raise SystemExit("standing screen dispatcher cannot be loaded")
module = importlib.util.module_from_spec(specification)
sys.path.insert(0, str(Path(dispatcher_path).parent))
specification.loader.exec_module(module)
with open(manifest_path, encoding="utf-8") as handle:
    manifest = module._manifest_from_dict(json.load(handle))
print(manifest.service_id)
PY
  )
  [[ "$SERVICE_IDENTITY" =~ ^[A-Za-z0-9._:@/+\-]{1,512}$ ]] \
    || fail "materialized service identity is malformed"
  log "materialized service identity: $SERVICE_IDENTITY"
fi

[[ -f "$WORKER_READINESS" && ! -L "$WORKER_READINESS" ]] \
  || fail "materialized worker readiness is missing or symlinked"

env PYTHONPATH="$CACHEON_SOURCE" "$CPU_PYTHON" "$REMOTE_SERVICE" make-registration \
  --ready-receipt "$READY_COPY" \
  --worker-readiness "$WORKER_READINESS" \
  --known-hosts "$KNOWN_HOSTS" \
  --pod-host "$POD_HOST" \
  --pod-port "$POD_PORT" \
  --service-identity "$SERVICE_IDENTITY" \
  --remote-service "$REMOTE_SERVICE" \
  --adapter "$ADAPTER" \
  --credential "$CREDENTIAL" \
  --credential-id "$CREDENTIAL_ID" \
  --python-executable "$REMOTE_PYTHON" \
  --lane-devices "$LANE_DEVICES" \
  --bind-ready-receipt \
  --output "$REGISTRATION_BUILD" >/dev/null

readarray -t registration_summary < <(
  "$CPU_PYTHON" - "$REGISTRATION_BUILD" <<'PY'
import json
import sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
print(row["worker_epoch"])
print(row["registration_digest"])
print(row["remote_service_sha256"])
print(row["adapter_sha256"])
print(row["credential_file_sha256"])
PY
)
WORKER_EPOCH=${registration_summary[0]}
REGISTRATION_DIGEST=${registration_summary[1]}
SERVICE_SHA=${registration_summary[2]}
ADAPTER_SHA=${registration_summary[3]}
CREDENTIAL_SHA=${registration_summary[4]}
[[ "$WORKER_EPOCH" =~ ^[0-9a-f]{32}$ ]] || fail "registration returned malformed worker epoch"
[[ "$REGISTRATION_DIGEST" =~ ^[0-9a-f]{64}$ ]] || fail "registration digest is malformed"
[[ "$SERVICE_SHA" == "$(file_sha256 "$REMOTE_SERVICE")" ]] || fail "service digest changed during registration"
[[ "$ADAPTER_SHA" == "$(file_sha256 "$ADAPTER")" ]] || fail "adapter digest changed during registration"
[[ "$CREDENTIAL_SHA" == "$(file_sha256 "$CREDENTIAL")" ]] || fail "credential digest changed during registration"

REGISTRATION=$STATE_ROOT/registrations/$WORKER_EPOCH.json
if [[ -e "$REGISTRATION" ]]; then
  [[ -f "$REGISTRATION" && ! -L "$REGISTRATION" ]] || fail "registration path collision"
  [[ "$(file_sha256 "$REGISTRATION")" == "$(file_sha256 "$REGISTRATION_BUILD")" ]] \
    || fail "worker epoch is already bound to a different registration"
else
  install -m 0400 "$REGISTRATION_BUILD" "$REGISTRATION"
fi

DISPATCHER_CONFIG=$STATE_ROOT/mainnet-screen-dispatcher-$WORKER_EPOCH.json
dispatcher_config_tmp=$(mktemp "$STATE_ROOT/.mainnet-screen-dispatcher.XXXXXX")
env PYTHONPATH="$CACHEON_SOURCE:$(dirname "$SCREEN_DISPATCHER")" \
  "$CPU_PYTHON" - \
  "$SCREEN_DISPATCHER_TEMPLATE" "$REGISTRATION" "$CREDENTIAL" \
  "$SPOOL_ROOT" "$dispatcher_config_tmp" "$ARENA_MANIFEST_COPY" <<'PY'
import json
import os
import sys

(
    template_path,
    registration_path,
    credential_path,
    spool_root,
    output_path,
    manifest_path,
) = sys.argv[1:]
with open(template_path, encoding="utf-8") as handle:
    config = json.load(handle)
with open(registration_path, encoding="utf-8") as handle:
    registration = json.load(handle)
manifest = None
if manifest_path:
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
expected = {
    "arena_service_manifest", "credential_digest", "credential_path",
    "heartbeat_interval_ms", "heartbeat_join_timeout_ms", "idle_poll_ms",
    "intake_db", "intake_policy", "intake_scope", "lease_blocks",
    "lock_attempts", "lock_retry_delay_ms", "owner", "registration_digest",
    "registration_path", "response_timeout_seconds",
    "restart_initial_backoff_ms", "restart_max_backoff_ms", "schema",
    "spool_root", "transport_identity_digest", "transport_poll_seconds",
    "worker_readiness",
}
if set(config) != expected or config.get("schema") != "cacheon-mainnet-screen-dispatcher-config-v1":
    raise SystemExit("screen dispatcher template fields/schema are not closed")
config.update({
    "credential_digest": registration["credential_digest"],
    "credential_path": credential_path,
    "registration_digest": registration["registration_digest"],
    "registration_path": registration_path,
    "spool_root": spool_root,
    "transport_identity_digest": registration["transport_identity_digest"],
    "worker_readiness": registration["worker_readiness"],
})
if manifest is not None:
    config["arena_service_manifest"] = manifest
encoded = json.dumps(config, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
with open(output_path, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(output_path, 0o400)
PY
mv -f -- "$dispatcher_config_tmp" "$DISPATCHER_CONFIG"
env PYTHONPATH="$CACHEON_SOURCE" \
  "$CPU_PYTHON" - "$SCREEN_DISPATCHER" "$DISPATCHER_CONFIG" <<'PY'
import importlib.util
import sys

dispatcher_path, config_path = sys.argv[1:]
specification = importlib.util.spec_from_file_location(
    "cacheon_fixed_mainnet_screen_dispatcher", dispatcher_path
)
if specification is None or specification.loader is None:
    raise SystemExit("standing screen dispatcher cannot be loaded")
module = importlib.util.module_from_spec(specification)
sys.path.insert(0, str(__import__("pathlib").Path(dispatcher_path).parent))
specification.loader.exec_module(module)
module.load_config(config_path)
PY
log "sealed standing screen dispatcher config: $DISPATCHER_CONFIG"

log "installing the closed service, fixed adapter, and registration on worker epoch $WORKER_EPOCH"
"${SSH[@]}" 'bash -s' -- "$REMOTE_PYTHON" <<'REMOTE'
set -euo pipefail
python=$1
  test "$(id -u)" -eq 0
  test -f /data/cacheon-b300/worker-bootstrap/ready-receipt.json
  test -x "$python"
  install -d -m 0700 /data/cacheon-b300/worker-bootstrap/bin /data/cacheon-b300/remote-worker /data/cacheon-b300/remote-worker/incoming /data/cacheon-b300/remote-worker/logs
  for path in /data /data/cacheon-b300 /data/cacheon-b300/worker-bootstrap /data/cacheon-b300/worker-bootstrap/bin /data/cacheon-b300/remote-worker; do
    test ! -L "$path"
  done
REMOTE

remote_service_part=$REMOTE_BIN/.remote_worker_service.py.$WORKER_EPOCH.incoming
remote_adapter_part=$REMOTE_BIN/.cacheon-b300-evaluation-adapter.$WORKER_EPOCH.incoming
remote_registration_part=$REMOTE_ROOT/.registration.$WORKER_EPOCH.incoming
remote_credential_part=$REMOTE_ROOT/.credential.$WORKER_EPOCH.incoming
"${SCP[@]}" -- "$REMOTE_SERVICE" "root@$POD_HOST:$remote_service_part"
"${SCP[@]}" -- "$ADAPTER" "root@$POD_HOST:$remote_adapter_part"
"${SCP[@]}" -- "$REGISTRATION" "root@$POD_HOST:$remote_registration_part"
"${SCP[@]}" -- "$CREDENTIAL" "root@$POD_HOST:$remote_credential_part"

"${SSH[@]}" 'bash -s' -- \
  "$remote_service_part" "$REMOTE_SERVICE_DEST" "$SERVICE_SHA" \
  "$remote_adapter_part" "$REMOTE_ADAPTER_DEST" "$ADAPTER_SHA" \
  "$remote_registration_part" "$REMOTE_REGISTRATION" \
  "$REGISTRATION_DIGEST" \
  "$remote_credential_part" "$REMOTE_CREDENTIAL" "$CREDENTIAL_SHA" \
  "$REMOTE_PYTHON" <<'REMOTE'
set -euo pipefail
service_part=$1
service_destination=$2
service_sha=$3
adapter_part=$4
adapter_destination=$5
adapter_sha=$6
registration_part=$7
registration_destination=$8
registration_digest=$9
credential_part=${10}
credential_destination=${11}
credential_sha=${12}
python=${13}
case "$service_part" in /data/cacheon-b300/worker-bootstrap/bin/.remote_worker_service.py.*.incoming) ;; *) exit 2;; esac
case "$adapter_part" in /data/cacheon-b300/worker-bootstrap/bin/.cacheon-b300-evaluation-adapter.*.incoming) ;; *) exit 2;; esac
case "$registration_part" in /data/cacheon-b300/remote-worker/.registration.*.incoming) ;; *) exit 2;; esac
case "$credential_part" in /data/cacheon-b300/remote-worker/.credential.*.incoming) ;; *) exit 2;; esac
[[ "$(sha256sum "$service_part" | awk '{print $1}')" == "$service_sha" ]]
[[ "$(sha256sum "$adapter_part" | awk '{print $1}')" == "$adapter_sha" ]]
chmod 0500 "$service_part" "$adapter_part"
chmod 0400 "$registration_part"
[[ "$(sha256sum "$credential_part" | awk '{print $1}')" == "$credential_sha" ]]
chmod 0400 "$credential_part"
mv -f -- "$service_part" "$service_destination"
mv -f -- "$adapter_part" "$adapter_destination"
mv -f -- "$registration_part" "$registration_destination"
mv -f -- "$credential_part" "$credential_destination"
"$python" "$service_destination" pod-serve --help >/dev/null
python3 - "$registration_destination" "$registration_digest" <<'PY'
import json
import sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
if row.get("registration_digest") != sys.argv[2]:
    raise SystemExit("registration digest changed in transfer")
PY
REMOTE

POD_SESSION=cacheon-remote-worker-${WORKER_EPOCH:0:12}
POD_LOG=$REMOTE_ROOT/logs/service-$WORKER_EPOCH.log
pod_session_state=$("${SSH[@]}" \
  "if tmux has-session -t '$POD_SESSION' 2>/dev/null; then tmux display-message -p -t '$POD_SESSION:0.0' '#{pane_dead}'; fi")
if [[ "$pod_session_state" == 1 ]]; then
  log "restarting dead pod tmux for the same immutable worker epoch"
  "${SSH[@]}" "tmux kill-session -t '$POD_SESSION'"
  pod_session_state=
fi
if [[ -z "$pod_session_state" ]]; then
  "${SSH[@]}" 'bash -s' -- "$POD_SESSION" "$REMOTE_PYTHON" "$REMOTE_SERVICE_DEST" "$POD_LOG" "$POLL_SECONDS" <<'REMOTE'
set -euo pipefail
session=$1
python=$2
service=$3
log_path=$4
poll_seconds=$5
quoted_python=$(printf '%q' "$python")
quoted_service=$(printf '%q' "$service")
quoted_log=$(printf '%q' "$log_path")
command="exec $quoted_python $quoted_service pod-serve --poll-seconds $poll_seconds >>$quoted_log 2>&1"
tmux new-session -d -s "$session" "$command"
tmux set-option -t "$session" remain-on-exit on >/dev/null
REMOTE
  log "started pod tmux $POD_SESSION"
else
  log "pod tmux already running: $POD_SESSION"
fi

log "waiting at most 60 seconds for the worker service heartbeat"
heartbeat_ready=0
for _ in $(seq 1 30); do
  if heartbeat_json=$("${SSH[@]}" "'$REMOTE_PYTHON' '$REMOTE_SERVICE_DEST' heartbeat-status" 2>/dev/null); then
    if [[ "$heartbeat_json" == *"\"worker_epoch\":\"$WORKER_EPOCH\""* ]]; then
      heartbeat_ready=1
      break
    fi
  fi
  sleep 2
done
if [[ "$heartbeat_ready" != 1 ]]; then
  fail "pod service did not publish its bound heartbeat; inspect $POD_LOG"
fi

CURRENT_REGISTRATION=$STATE_ROOT/current-registration.json
current_tmp=$(mktemp "$STATE_ROOT/.current-registration.XXXXXX")
install -m 0400 "$REGISTRATION" "$current_tmp"
mv -f -- "$current_tmp" "$CURRENT_REGISTRATION"

CPU_SESSION=cacheon-remote-dispatch-${WORKER_EPOCH:0:12}
CPU_LOG=$LOG_ROOT/remote-dispatch-$WORKER_EPOCH.log
cpu_session_state=
if tmux has-session -t "$CPU_SESSION" 2>/dev/null; then
  cpu_session_state=$(tmux display-message -p -t "$CPU_SESSION:0.0" '#{pane_dead}')
fi
if [[ "$cpu_session_state" == 1 ]]; then
  log "restarting dead CPU dispatcher tmux for the same immutable worker epoch"
  tmux kill-session -t "$CPU_SESSION"
  cpu_session_state=
fi
if [[ -z "$cpu_session_state" ]]; then
  quoted_service=$(printf '%q' "$REMOTE_SERVICE")
  quoted_registration=$(printf '%q' "$REGISTRATION")
  quoted_current=$(printf '%q' "$CURRENT_REGISTRATION")
  quoted_spool=$(printf '%q' "$SPOOL_ROOT")
  quoted_log=$(printf '%q' "$CPU_LOG")
  quoted_python=$(printf '%q' "$CPU_PYTHON")
  quoted_source=$(printf '%q' "$CACHEON_SOURCE")
  cpu_command="exec env PYTHONPATH=$quoted_source $quoted_python $quoted_service cpu-serve --registration $quoted_registration --current-registration $quoted_current --spool-root $quoted_spool --poll-seconds $POLL_SECONDS --max-heartbeat-age $MAX_HEARTBEAT_AGE >>$quoted_log 2>&1"
  tmux new-session -d -s "$CPU_SESSION" "$cpu_command"
  tmux set-option -t "$CPU_SESSION" remain-on-exit on >/dev/null
  log "started CPU dispatcher tmux $CPU_SESSION"
else
  log "CPU dispatcher tmux already running: $CPU_SESSION"
fi

log "waiting at most 30 seconds for the standing CPU transfer heartbeat"
cpu_heartbeat_ready=0
for _ in $(seq 1 30); do
  if [[ -f "$SPOOL_ROOT/state/heartbeat.json" ]] \
    && grep -Fq "\"worker_epoch\":\"$WORKER_EPOCH\"" "$SPOOL_ROOT/state/heartbeat.json"; then
    cpu_heartbeat_ready=1
    break
  fi
  sleep 1
done
[[ "$cpu_heartbeat_ready" == 1 ]] \
  || fail "CPU transfer service did not publish its bound heartbeat; inspect $CPU_LOG"

SCREEN_SESSION=cacheon-mainnet-screen-dispatcher
SCREEN_LOG=$LOG_ROOT/mainnet-screen-dispatcher-$WORKER_EPOCH.log
if tmux has-session -t "$SCREEN_SESSION" 2>/dev/null; then
  log "superseding the prior standing screen dispatcher before worker replacement"
  tmux send-keys -t "$SCREEN_SESSION" C-c >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    tmux has-session -t "$SCREEN_SESSION" 2>/dev/null || break
    sleep 1
  done
  if tmux has-session -t "$SCREEN_SESSION" 2>/dev/null; then
    tmux kill-session -t "$SCREEN_SESSION"
  fi
fi
quoted_python=$(printf '%q' "$CPU_PYTHON")
quoted_source=$(printf '%q' "$CACHEON_SOURCE")
quoted_dispatcher=$(printf '%q' "$SCREEN_DISPATCHER")
quoted_config=$(printf '%q' "$DISPATCHER_CONFIG")
quoted_screen_log=$(printf '%q' "$SCREEN_LOG")
screen_command="exec env PYTHONPATH=$quoted_source $quoted_python $quoted_dispatcher --config $quoted_config >>$quoted_screen_log 2>&1"
tmux new-session -d -s "$SCREEN_SESSION" "$screen_command"
tmux set-option -t "$SCREEN_SESSION" remain-on-exit on >/dev/null
sleep 2
screen_dead=$(tmux display-message -p -t "$SCREEN_SESSION:0.0" '#{pane_dead}')
[[ "$screen_dead" == 0 ]] \
  || fail "standing screen dispatcher exited during startup; inspect $SCREEN_LOG"
log "started standing FIFO screen dispatcher tmux $SCREEN_SESSION"

atomic_status=$STATE_ROOT/current-worker.env
status_tmp=$(mktemp "$STATE_ROOT/.current-worker.XXXXXX")
{
  printf 'schema=cacheon-remote-worker-current-v1\n'
  printf 'state=RUNNING\n'
  printf 'started_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'worker_epoch=%s\n' "$WORKER_EPOCH"
  printf 'registration_digest=%s\n' "$REGISTRATION_DIGEST"
  printf 'pod_host=%s\n' "$POD_HOST"
  printf 'pod_port=%s\n' "$POD_PORT"
  printf 'pod_session=%s\n' "$POD_SESSION"
  printf 'cpu_session=%s\n' "$CPU_SESSION"
  printf 'screen_session=%s\n' "$SCREEN_SESSION"
  printf 'screen_dispatcher_config=%s\n' "$DISPATCHER_CONFIG"
  printf 'spool_root=%s\n' "$SPOOL_ROOT"
} >"$status_tmp"
chmod 0400 "$status_tmp"
mv -f -- "$status_tmp" "$atomic_status"

log "REMOTE-WORKER-RUNNING worker_epoch=$WORKER_EPOCH registration=$REGISTRATION_DIGEST"
log "CPU log: $CPU_LOG"
log "screen dispatcher log: $SCREEN_LOG"
log "pod log: $POD_LOG"
log "results: $SPOOL_ROOT/results"
log "the CPU VM and pod tmux sessions now own the service; this terminal may close"
