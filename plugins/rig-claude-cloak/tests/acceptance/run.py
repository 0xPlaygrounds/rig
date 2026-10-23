#!/usr/bin/env python3
"""Offline plan/report by default. Live requires explicit, bounded authorization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import uuid
import re

from audit import Evidence, FACTS, PHASE_NAMES, PROBE_VARIANTS, PROBE_ORDERS, import_effects, private_json, report, tool_result_detail
from ledger import Ledger, AGGREGATE_REQUEST_LIMITS
from preflight import verify_build, verify_claude, verify_root_ttl, verify_probe, private_capture, private_probe_metadata, ROOT_TTL_TEST, TOOL_RESULT_TEST, build_record

MODEL = "claude-cloak/claude-opus-5-5"
PHASES = [
    ("cold_repeat", "Two fresh zero-tool sessions with the same synthetic prefix; record writes and reads without assuming cold."),
    ("append_tools", "In a dedicated TUI session, recall facts, read synthetic facts.txt, then read absent.txt; assert dispatch and cache reuse."),
    ("effort", "Change low to medium; recall without repeating values; compare cache counters."),
    ("manual", "Ask for only 'Ready', compact manually, recall named facts without values, then recall again; join summary effect by provider request id."),
    ("automatic", "Use isolated reduced-trigger settings, append a large newest message with new facts, answer neutrally, trigger compaction and recall old/new facts."),
    ("fork_resume", "Fork then resume isolated session; compare pseudonymous session/human identities and fact retention."),
    ("roster", "Restart with zero tools; repeat stable prefix in fresh sessions; compare marker topology and cache counters."),
]


def plan():
    return {"model": MODEL, "automated": ["cold_repeat"], "matrix_option": "--matrix drives tools, effort, manual/automatic compaction and zero-tool controls through a dedicated fux TUI; --fork-resume adds fork/resume. Live execution remains unverified until explicitly authorized.", "manual": [dict(phase=p, action=a) for p, a in PHASES[1:]],
            "authorization": "No credential access or network in plan/report modes. Live uses existing credentials normally, including refresh.",
            "caps": "Explicit Messages request cap, per-request advertised output token cap, elapsed seconds. Authentication calls are not counted as inference.",
            "scope": "Proxy-visible final wire with accept-encoding narrowed to gzip/deflate/identity. No checksum or Claude Code equivalence claim."}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def environment(scratch, certificate, port):
    # Allowlist, rather than inheriting API keys, runtime bindings or debug logs.
    env = {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C.UTF-8", "TERM": "dumb",
           "HOME": str(scratch / "home"), "RIG_HOME": str(scratch / "rig-home"),
           "RIG_SOURCE": str(scratch / "source"), "RUST_LOG": "off"}
    for key, directory in (("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data")):
        env[key] = str(scratch / "home" / directory)
    env.update(HTTPS_PROXY=f"http://127.0.0.1:{port}", HTTP_PROXY=f"http://127.0.0.1:{port}",
               NO_PROXY="", SSL_CERT_FILE=str(certificate))
    return env


def stop(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def live(args):
    if not args.confirm_stored_credentials or not all((args.binary, args.source, args.plugin_config, args.credentials, args.output, args.requests, args.output_tokens, args.seconds)):
        raise ValueError("live requires --confirm-stored-credentials, binary/source/plugin-config/credentials/output and all three caps")
    if args.fork_resume and not args.matrix:
        raise ValueError("--fork-resume requires --matrix")
    clients = (args.clients,) if args.clients in ("rig", "claude") else ("claude", "rig") if args.compare else ("rig",)
    args.compare = args.compare or "claude" in clients
    if args.claude_auth != "stored" and "claude" not in clients:
        raise ValueError("--claude-auth requires a Claude Code client run")
    if args.compare and (not args.matrix or not args.claude_binary or not args.claude_auth_home):
        raise ValueError("--compare requires --matrix, --claude-binary and --claude-auth-home")
    if not args.ledger or not args.build_record:
        raise ValueError("live requires --ledger and --build-record verified before inference")
    if (args.pause_after_seed or args.pause_after_phase or args.pause_on_failure) and (not args.raw_capture or not args.matrix):
        raise ValueError("inspection pauses require --raw-capture and --matrix")
    if args.startup_only and not args.matrix:
        raise ValueError("--startup-only requires --matrix")
    if args.continue_after_read_refusal and (not args.matrix or "rig" not in clients or args.scope != "full" or args.startup_only):
        raise ValueError("--continue-after-read-refusal requires a full Rig matrix")
    if args.read_refusal_next is not None and not args.continue_after_read_refusal:
        raise ValueError("--read-refusal-next requires --continue-after-read-refusal")
    if args.collect_failures and (not args.matrix or clients != ("rig",) or args.scope != "full" or args.startup_only or args.continue_after_read_refusal or args.read_refusal_next is not None):
        raise ValueError("--collect-failures requires a Rig-only full matrix without startup-only or other continuation modes")
    if args.scope == "auto-controls" and (not args.matrix or clients != ("rig",) or args.fork_resume or args.startup_only):
        raise ValueError("--scope auto-controls requires a Rig-only matrix without fork/resume or startup-only")
    if args.root_ttl_probe and (args.matrix or args.compare or args.startup_only or not Path(args.root_ttl_probe).is_file()):
        raise ValueError("--root-ttl-probe requires a compiled test binary and a separate non-matrix run")
    if args.tool_result_probe and (args.root_ttl_probe or args.matrix or args.compare or args.startup_only or not Path(args.tool_result_probe).is_file() or not args.captured_request):
        raise ValueError("--tool-result-probe requires a compiled test binary, --captured-request and a separate non-matrix run")
    if args.captured_request and not args.tool_result_probe:
        raise ValueError("--captured-request requires --tool-result-probe")
    if args.probe_variant != "tool_result_string" and not args.tool_result_probe:
        raise ValueError("--probe-variant requires --tool-result-probe")
    if args.probe_order != "baseline-first" and not args.tool_result_probe:
        raise ValueError("--probe-order requires --tool-result-probe")
    if bool(args.probe_metadata) != bool(args.tool_result_probe and args.probe_variant == "previous_request_id"):
        raise ValueError("--probe-metadata is required only for the previous_request_id probe")
    captured_request = private_capture(args.captured_request) if args.tool_result_probe else None
    probe_metadata = private_probe_metadata(args.probe_metadata) if args.probe_metadata else None
    if not (2 <= args.requests <= 24 and 1 <= args.output_tokens <= 128000 and 1 <= args.seconds <= 1800):
        raise ValueError("caps: requests 2..24, output-tokens 1..128000, seconds 1..1800")
    binary, source, plugins, credentials, output = [Path(p).resolve() for p in
                                                   (args.binary, args.source, args.plugin_config, args.credentials, args.output)]
    if output == source or source in output.parents:
        raise ValueError("keep artifacts outside the source tree")
    if not binary.is_file() or not plugins.is_file() or not credentials.is_dir():
        raise ValueError("binary, plugin manifest or credential directory missing")
    binding = verify_build(binary, source, plugins, args.build_record)
    claude_binding = verify_claude(Path(args.claude_binary).resolve()) if args.compare else None
    ttl_binding = verify_root_ttl(Path(args.root_ttl_probe).resolve()) if args.root_ttl_probe else None
    tool_binding = verify_probe(Path(args.tool_result_probe).resolve(), TOOL_RESULT_TEST) if args.tool_result_probe else None
    second_read_file = args.missing_file or ("probe-" + os.urandom(8).hex() + ".txt")
    if args.match_inputs:
        second_read_file = json.loads((Path(args.match_inputs) / "policy.json").read_text())["second_read_file"]
    if not isinstance(second_read_file, str) or not re.fullmatch(r"probe-[0-9a-f]{16}\.txt", second_read_file):
        raise ValueError("missing-file must be a neutral probe-<16 hex digits>.txt name")
    for command in ("bwrap", "mitmdump", "git") + (("fux",) if args.matrix else ()):
        if not shutil.which(command):
            raise ValueError(f"required external dependency missing: {command}; nothing installed")
    # Do not read or copy the auth file. Bind its original directory for normal atomic refresh.
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.umask(0o077)
    evidence = Evidence(output)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]))
    settings = "[compaction]\nkeep_recent = 1024\nkeep_asked = 0\nlimits = { reserve = 970000, summary_tokens = 1024 }\n"
    prompt = ("Synthetic acceptance ledger. Memorize these facts: " + json.dumps(FACTS) +
              "\n" + "The neutral filler sentence carries no additional facts.\n" * 700 +
              "\nReturn the five ledger values only. Do not use tools.")
    private_json(output / "provenance.json", {"schema": 1, "commit": commit, "source_dirty": dirty,
                 "binary_sha256": digest(binary), "plugin_config_sha256": digest(plugins),
                 "build_record": binding, "claude_build": claude_binding,
                 "checksum_verifier_sha256": digest(args.checksum_verifier) if args.checksum_verifier else None,
                 "root_ttl_probe": ttl_binding,
                 "tool_result_probe": {**tool_binding, "variant": args.probe_variant, "order": args.probe_order} if tool_binding else None,
                 "settings": settings, "model": MODEL, "effort": "low", "input": prompt,
                 "scenarios": ["fresh_prefix", "fresh_prefix_repeat"], "matrix": args.matrix, "fork_resume": args.fork_resume, "plan": plan(),
                 "caps": {"requests": args.requests, "aggregate_requests": args.aggregate_requests, "output_tokens": args.output_tokens, "seconds": args.seconds},
                 "clients": clients, "scope": args.scope or ("probes" if args.compare else "full"),
                 "claude_auth_mode": args.claude_auth,
                 "continue_after_read_refusal": args.continue_after_read_refusal,
                 "read_refusal_next": (args.read_refusal_next or "effort") if args.continue_after_read_refusal else None,
                 "collect_failures": args.collect_failures,
                 "claude_isolation": ["--safe-mode", "--strict-mcp-config", "--mcp-config empty", "--setting-sources empty", "--permission-mode dontAsk", "--prompt-suggestions false", "--name synthetic parity (exclude auxiliary title generation)", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", "CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1 (observer verifies auxiliary absence)"],
                 "build_binding": "Recorded successful build command and generated source are checked; hashes are not an independent compiler attestation."})
    proxy = child = None
    # Scratch contains ordinary Rig sessions, potentially including signed/internal content.
    # It is private, never exported, and removed when this owned run finishes.
    if not Path("/dev/shm").is_dir() or not os.access("/dev/shm", os.W_OK):
        raise ValueError("live requires writable /dev/shm for private temporary session state")
    with tempfile.TemporaryDirectory(prefix="cloak-acceptance-", dir="/dev/shm") as temporary:
        scratch = Path(temporary)
        for folder in ("home", "rig-home/anthropic-oauth", "source", "workspace", "ca", "raw"):
            (scratch / folder).mkdir(parents=True, exist_ok=True)
        (scratch / "rig-home/settings.toml").write_text(settings)
        private_json(scratch / "rig-home/defaults.json", {"model": MODEL, "effort": {"effort": "low"}})
        (scratch / "workspace/facts.txt").write_text(json.dumps(FACTS))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        ledger = Ledger(Path(args.ledger).absolute())
        run_id = uuid.uuid4().hex
        deadline = ledger.begin(run_id, args.seconds, args.aggregate_requests)
        private_json(output / "policy.json", {"requests": args.requests, "aggregate_requests": args.aggregate_requests, "output_tokens": args.output_tokens, "deadline": deadline, "matrix": args.matrix, "second_read_file": second_read_file,
                     "ledger": str(ledger.path), "run_id": run_id, "scope": args.scope or ("probes" if args.compare else "full"), "blocked_auxiliary_models": args.blocked_auxiliary_model, "startup_only": args.startup_only, "clients": clients, "root_ttl_probe": bool(args.root_ttl_probe), "tool_result_probe": bool(args.tool_result_probe), "probe_variant": args.probe_variant if args.tool_result_probe else None, "collect_failures": args.collect_failures,
                     **({"probe_order": args.probe_order} if args.tool_result_probe else {}),
                     "checksum_verifier": str(Path(args.checksum_verifier).resolve()) if args.checksum_verifier else None})
        (output / "phase.txt").write_text("unknown")
        (output / "client.txt").write_text(clients[0])
        proxy_env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(scratch / "home"), "CLOAK_ACCEPTANCE_ROOT": str(output)}
        proxy_env.update(CLOAK_WORKSPACE_RIG=str(scratch / "workspace"), CLOAK_WORKSPACE_CLAUDE=str(scratch / "claude/workspace"))
        if args.raw_capture:
            proxy_env["CLOAK_RAW_CAPTURE_DIR"] = str(scratch / "raw")
        try:
            proxy = subprocess.Popen(["mitmdump", "--quiet", "--listen-host", "127.0.0.1", "--listen-port", str(port),
                                      "--set", "confdir=" + str(scratch / "ca"), "--set", "connection_strategy=lazy",
                                      "-s", str(Path(__file__).with_name("proxy.py"))], env=proxy_env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            certificate = scratch / "ca/mitmproxy-ca-cert.pem"
            while not certificate.exists():
                if proxy.poll() is not None or time.time() >= deadline:
                    raise RuntimeError("private proxy failed to start (no raw proxy output persisted)")
                time.sleep(0.1)
            if args.root_ttl_probe or args.tool_result_probe:
                phase = "tool_result_probe" if args.tool_result_probe else "root_ttl_probe"
                (output / "phase.txt").write_text(phase)
                (output / "client.txt").write_text("rig")
                probe_env = environment(scratch, certificate, port)
                probe_env["CLAUDE_CLOAK_LIVE_MODEL"] = MODEL
                mounts = []
                if args.tool_result_probe:
                    target = scratch / "captured-request.json"
                    target.touch(mode=0o600)
                    mounts = ["--ro-bind", str(captured_request), str(target)]
                    probe_env.update(CLAUDE_CLOAK_LIVE_TOOL_RESULT="1", CLAUDE_CLOAK_CAPTURED_REQUEST=str(target),
                                     CLAUDE_CLOAK_PROBE_VARIANT=args.probe_variant,
                                     CLAUDE_CLOAK_PROBE_ORDER=args.probe_order,
                                     CLAUDE_CLOAK_TOOL_RESULT_REPORT=str(scratch / "tool-result-detail.json"))
                    if probe_metadata:
                        metadata_target = scratch / "probe-metadata.json"
                        metadata_target.touch(mode=0o600)
                        mounts.extend(["--ro-bind", str(probe_metadata), str(metadata_target)])
                        probe_env["CLAUDE_CLOAK_PROBE_METADATA"] = str(metadata_target)
                else:
                    probe_env["CLAUDE_CLOAK_LIVE_ROOT_TTL"] = "1"
                command = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                           "--bind", str(scratch), str(scratch), "--bind", str(credentials), str(scratch / "rig-home/anthropic-oauth"),
                           *mounts, "--chdir", str(scratch / "workspace"), "--", str(Path(args.tool_result_probe or args.root_ttl_probe).resolve()), "--exact",
                           TOOL_RESULT_TEST if args.tool_result_probe else ROOT_TTL_TEST, "--ignored", "--nocapture"]
                child = subprocess.Popen(command, env=probe_env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, start_new_session=True)
                try:
                    code = child.wait(timeout=max(0.001, deadline - time.time()))
                    outcome = {"exit_code": code, "timed_out": False}
                except subprocess.TimeoutExpired:
                    stop(child)
                    outcome = {"exit_code": child.returncode, "timed_out": True}
                private_json(output / ("tool-result-probe.json" if args.tool_result_probe else "root-ttl-probe.json"), outcome)
                if args.tool_result_probe:
                    detail = {}
                    try:
                        with (scratch / "tool-result-detail.json").open("rb") as stream:
                            data = stream.read(16385)
                        if len(data) <= 16384:
                            detail = tool_result_detail(json.loads(data))
                    except (OSError, ValueError, TypeError):
                        pass
                    private_json(output / "tool-result-detail.json", detail)
            if args.matrix:
                from matrix import Matrix, actions, APPEND
                private_json(output / "matrix-inputs.json", {"actions": actions(second_read_file), "auto_append": APPEND})
                scope = args.scope or ("probes" if args.compare else "full")
                options = {"scope": scope, "pause_after_seed": args.pause_after_seed, "raw_directory": scratch / "raw", "startup_only": args.startup_only,
                           "pause_after_phase": args.pause_after_phase, "pause_on_failure": args.pause_on_failure,
                           "continue_after_read_refusal": args.continue_after_read_refusal,
                           "read_refusal_next": args.read_refusal_next or "effort", "collect_failures": args.collect_failures}
                drivers = []
                for client in clients:
                    if client == "rig":
                        drivers.append(Matrix(scratch, output, binary, credentials, environment(scratch, certificate, port), deadline, args.fork_resume, **options))
                        continue
                    from comparison import ClaudeMatrix
                    cc_scratch = scratch / "claude"
                    for folder in ("home", "workspace", "rig-home", "ca"):
                        (cc_scratch / folder).mkdir(parents=True, exist_ok=True)
                    cc_certificate = cc_scratch / "ca/proxy.pem"
                    shutil.copyfile(certificate, cc_certificate)
                    (cc_scratch / "workspace/facts.txt").write_text(json.dumps(FACTS))
                    drivers.append(ClaudeMatrix(cc_scratch, output, Path(args.claude_binary).resolve(), credentials,
                                   environment(cc_scratch, cc_certificate, port), deadline, args.fork_resume,
                                   auth_home=Path(args.claude_auth_home).resolve(), claude_auth=args.claude_auth, **options))
                for driver in drivers:
                    try:
                        driver.run()
                    except (RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                        # Both matched clients are planned observations. A failed
                        # phase stops that client's later phases, never reruns it.
                        audit = output / "audit.jsonl"
                        guards = [json.loads(line).get("local_guard") for line in audit.read_text().splitlines()] if audit.exists() else []
                        if any(guard and guard != "auxiliary_model" for guard in guards) or time.time() >= deadline:
                            break
            for scenario in (() if args.matrix or args.root_ttl_probe or args.tool_result_probe else ("fresh_prefix", "fresh_prefix_repeat")):
                (output / "phase.txt").write_text(scenario)
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                command = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                           "--bind", str(scratch), str(scratch),
                           "--bind", str(credentials), str(scratch / "rig-home/anthropic-oauth"),
                           "--chdir", str(scratch / "workspace"),
                           "--", str(binary), "--tools=", "--print", "--", prompt]
                child = subprocess.Popen(command, cwd=scratch / "workspace", env=environment(scratch, certificate, port),
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         start_new_session=True)
                try:
                    code = child.wait(timeout=remaining)
                    outcome = {"scenario": scenario, "exit_code": code, "timed_out": False}
                except subprocess.TimeoutExpired:
                    stop(child)
                    outcome = {"scenario": scenario, "exit_code": child.returncode, "timed_out": True}
                private_json(output / (scenario + ".json"), outcome)
                if outcome["timed_out"] or outcome["exit_code"]:
                    break
        finally:
            stop(child)
            stop(proxy)
            try:
                import_effects(scratch / "rig-home/sessions", evidence)
                auth_path = output / "child-startup.json"
                if args.compare and auth_path.exists():
                    auth = json.loads(auth_path.read_text())
                    provenance = json.loads((output / "provenance.json").read_text())
                    provenance["claude_auth_identity"] = {key: auth[key] for key in ("stored_account_matches_rig", "selected_account_matches_rig") if type(auth.get(key)) is bool}
                    private_json(output / "provenance.json", provenance)
                result = report(output)
                if args.compare:
                    from comparison import compare_profiles
                    compare_profiles(output)
            finally:
                ledger.finish(run_id)
    return 0 if result["classification"] == "pass" or (args.tool_result_probe and result.get("diagnostic", {}).get("complete")) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--analyze", type=Path)
    mode.add_argument("--compare-reports", nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    parser.add_argument("--confirm-stored-credentials", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="drive the bounded TUI matrix through a dedicated fux server")
    parser.add_argument("--fork-resume", action="store_true", help="include fork/resume in --matrix")
    parser.add_argument("--compare", action="store_true", help="run matched Rig and Claude Code TUIs")
    parser.add_argument("--clients", choices=("rig", "claude", "both"), help="select separate bounded runs sharing --ledger; both runs Claude Code first")
    parser.add_argument("--scope", choices=("probes", "full", "auto-controls"))
    parser.add_argument("--raw-capture", action="store_true", help="retain decoded original requests only in owned tmpfs until cleanup")
    parser.add_argument("--pause-after-seed", action="store_true", help="wait for continue-CLIENT sentinel after seed for private checksum inspection")
    parser.add_argument("--pause-after-phase", choices=[phase for phase in PHASE_NAMES if phase not in ("root_ttl_probe", "tool_result_probe")], help="pause before cleanup after this phase, whether passing or failing")
    parser.add_argument("--pause-on-failure", action="store_true", help="pause before the first failed/unknown phase raises, preserving its failed result")
    parser.add_argument("--continue-after-read-refusal", action="store_true", help="observe later full Rig phases after one fully captured read_error refusal; preserve the failed phase and overall failure")
    parser.add_argument("--read-refusal-next", choices=("effort", "manual_compact"), help="choose the distinct next phase after a validated read refusal; default effort, requires continuation")
    parser.add_argument("--collect-failures", action="store_true", help="diagnostic full Rig run: advance only after complete correlated semantic failures to the next fixed distinct phase; retain all failures")
    parser.add_argument("--startup-only", action="store_true", help="validate child startup/input readiness, then quit without sending a model prompt")
    parser.add_argument("--claude-auth", choices=("stored", "rig-access-token"), default="stored", help="select the existing Claude Code token or match the existing Rig access token without copying a refresh grant")
    parser.add_argument("--aggregate-requests", type=int, choices=AGGREGATE_REQUEST_LIMITS, default=48, help="shared-ledger request ceiling; 96 or 200 requires explicit authorization to extend the existing budget")
    parser.add_argument("--probe-variant", choices=PROBE_VARIANTS, default="tool_result_string", help="declared single-variable diagnostic; requires --tool-result-probe")
    parser.add_argument("--probe-order", choices=PROBE_ORDERS, default="baseline-first", help="controlled-pair send order; requires --tool-result-probe")
    parser.add_argument("--blocked-auxiliary-model", action="append", default=["claude-haiku-5-5"])
    for option in ("claude-binary", "claude-auth-home", "ledger", "build-record", "checksum-verifier", "record-build", "generated-main", "build-command", "missing-file", "match-inputs", "root-ttl-probe", "tool-result-probe", "captured-request", "probe-metadata"):
        parser.add_argument("--" + option)
    for option in ("binary", "source", "plugin-config", "credentials", "output"):
        parser.add_argument("--" + option)
    for option in ("requests", "output-tokens", "seconds"):
        parser.add_argument("--" + option, type=int)
    args = parser.parse_args()
    if args.clients == "both":
        args.compare = True
    if args.record_build:
        if args.live or not all((args.binary, args.source, args.plugin_config, args.generated_main, args.build_command)):
            parser.error("--record-build is offline and requires binary/source/plugin-config/generated-main/build-command")
        private_json(Path(args.record_build), build_record(args.binary, args.source, args.plugin_config, args.generated_main, args.build_command))
        return 0
    if args.live:
        try:
            return live(args)
        except (ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
            parser.error(str(error))
    elif args.compare_reports:
        if not args.output:
            parser.error("--compare-reports requires --output outside the source trees")
        from comparison import compare_profiles
        destination = Path(args.output)
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
        print(json.dumps(compare_profiles(destination, args.compare_reports), indent=2))
        return 0
    elif args.analyze:
        result = report(args.analyze)
        print(json.dumps(result, indent=2))
        return 0 if result["classification"] == "pass" else 1
    else:
        print(json.dumps(plan(), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
