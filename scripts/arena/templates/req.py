"""Off-chain request driver for a commissioned pair (from recipe adbf084e, 2026-09-27; the registration binds the pair's standalone service copy): production producers only, scratch intake DB.

Run as: PYTHONPATH=$SRC $PY -B req.py $STAGE $SRC <0700 bundle copy> <fresh 0700 work dir>
"""
import shutil, sys
from pathlib import Path
from cacheon.arena_service import ArenaCandidateBinding, ArenaService
from cacheon.bundle_hash import content_hash
from cacheon.copy_fingerprint import fingerprint_submitted_delta
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.chain.evaluation_coordinator import ClaimedQualificationEvaluation, EvaluationCoordinator, WorkerReadiness, _qualification_reservations
from cacheon.chain.intake import FinalizedArrival, FinalizedIntakeStore, IntakePolicy, IntakeScope
from cacheon.chain.mainnet_screen_dispatcher import RemoteOnlyArenaProvider, _manifest_from_dict, make_qualification_publication_resolver
from cacheon.chain.publication import publish_worker_bundle, reopen_worker_bundle
from cacheon.chain.qualification_request import qualification_request_body
from cacheon.chain.remote_evaluation_dispatcher import seal_remote_request
from cacheon.chain.remote_worker_pod_service import accept_request
from cacheon.chain.remote_worker_registration import PodPaths, registration_credential, registration_transport_identity, verify_pod_registration
from cacheon.chain.remote_worker_spool import load_json, pack_directory
from cacheon.chain.ssh_worker_transport import DurableSpoolAuthenticatedWorkerTransport

STAGE, SRC, BUNDLE, WORK = map(Path, sys.argv[1:5])
R, QA = STAGE / "remote-worker", STAGE / "qualification-authority"
paths = PodPaths(root=R, ready_receipt=STAGE / "ready-receipt.json", registration=STAGE / "registration.json",
                 service=STAGE / "worker-bootstrap/bin/remote_worker_service.py",
                 adapter=QA / "cacheon_b300_qualification_adapter.py", credential=R / "credential.secret")
registration = verify_pod_registration(paths)
identity, credential = registration_transport_identity(registration), registration_credential(registration, paths.credential)
manifest = _manifest_from_dict(load_json(STAGE / "commissioned/arena-service-manifest.json"))
service = ArenaService(manifest, RemoteOnlyArenaProvider(manifest.provider_digest))
readiness = WorkerReadiness(**registration["worker_readiness"])
readiness.validate(service)
capture = load_json(STAGE / "calibration-capture.json")
publication = publish_worker_bundle(BUNDLE, WORK / "publications", content_hash(BUNDLE))
SCOPE, POLICY, DB = IntakeScope("0x" + "0" * 64, 0), IntakePolicy(), WORK / "intake/intake.sqlite3"
BLOCK, HASH = 1, "0x" + "0" * 63 + "1"
with FinalizedIntakeStore(DB, POLICY, scope=SCOPE) as store:
    (row,) = store.reserve_finalized((FinalizedArrival("offchain-operator", publication.content_hash,
                                                       "offchain://" + publication.content_hash, BLOCK, HASH, 0),),
                                     finalized_block=BLOCK, finalized_block_hash=HASH)
    store.mark_fetching(row.reservation_id)
    store.mark_published(row.reservation_id, delta_fingerprint=fingerprint_submitted_delta(publication.root),
                         publication_digest=publication.digest, publication_root=publication.root)
    store.prepare_qualification_queue(service_digest=service.identity, closed_targets=manifest.closed_targets)
coordinator = EvaluationCoordinator(intake_db=DB, policy=POLICY, scope=SCOPE, service=service, readiness=readiness,
                                    owner="offchain-qualification", advance_finalized_cursor=lambda: (BLOCK, HASH),
                                    qualification_max_members=1)
store, point = coordinator._open_at_durable_cursor()
try:
    lease = store.claim_evaluation_lease(stage="qualification", owner=coordinator.owner, current_block=point[0],
                                         lease_blocks=coordinator.lease_blocks, max_members=1)
    rows = tuple(store.get(rid) for rid in lease.reservation_ids)
    attempts = tuple(store.qualification_attempts(r.reservation_id) + 1 for r in rows)
finally:
    store.close()
pubs = tuple(reopen_worker_bundle(r.publication_root, r.arrival.content_hash, expected_receipt_digest=r.publication_digest)
             for r in rows)
claim = ClaimedQualificationEvaluation(lease, rows, pubs, tuple(
    ArenaCandidateBinding(a, p, n) for p, a, n in zip(pubs, _qualification_reservations(rows, pubs), attempts, strict=True)))
body = qualification_request_body(coordinator, claim, incumbent_tree_digest=capture["incumbent_tree_digest"],
                                  incumbent_stack_digest=EvaluationStackManifest.from_dict(capture["incumbent_stack"]).digest)
request = seal_remote_request(claim.lease, readiness, registration["service_identity"], identity, credential, body)
transport = DurableSpoolAuthenticatedWorkerTransport(
    registration_path=paths.registration, spool_root=WORK / "spool", credential_path=Path(registration["credential_path"]),
    qualification_publication_resolver=make_qualification_publication_resolver(
        intake_db=DB, policy=POLICY, scope=SCOPE, store_factory=FinalizedIntakeStore))
plan = transport.plan_qualification_request(request)
transport.materialize_planned_qualification(plan, request)
carrier = transport.publish_planned_qualification(plan).carrier_path
(carrier / "transport").mkdir(mode=0o700, exist_ok=True)
archive = carrier / "transport" / f"{plan.request_id}.tar"
sha = pack_directory(carrier, "request.json", load_json(carrier / "request.json")["artifacts"], archive)
shutil.copyfile(archive, R / "incoming" / f".{plan.request_id}.{sha}.tar.part")
accept_request(paths, request_id=plan.request_id, archive_sha256=sha, archive_size=archive.stat().st_size)
print("REQUEST_ID=" + plan.request_id)
