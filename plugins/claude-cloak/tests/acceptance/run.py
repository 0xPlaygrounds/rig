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

from audit import Evidence, FACTS, import_effects, private_json, report

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
    if not (2 <= args.requests <= 24 and 1 <= args.output_tokens <= 128000 and 1 <= args.seconds <= 1800):
        raise ValueError("caps: requests 2..24, output-tokens 1..128000, seconds 1..1800")
    binary, source, plugins, credentials, output = [Path(p).resolve() for p in
                                                   (args.binary, args.source, args.plugin_config, args.credentials, args.output)]
    if output == source or source in output.parents:
        raise ValueError("keep artifacts outside the source tree")
    if not binary.is_file() or not plugins.is_file() or not credentials.is_dir():
        raise ValueError("binary, plugin manifest or credential directory missing")
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
                 "settings": settings, "model": MODEL, "effort": "low", "input": prompt,
                 "scenarios": ["fresh_prefix", "fresh_prefix_repeat"], "matrix": args.matrix, "fork_resume": args.fork_resume, "plan": plan(),
                 "caps": {"requests": args.requests, "output_tokens": args.output_tokens, "seconds": args.seconds},
                 "build_binding": "Caller must verify binary was built from the recorded source/config; hashes alone do not establish that."})
    proxy = child = None
    # Scratch contains ordinary Rig sessions, potentially including signed/internal content.
    # It is private, never exported, and removed when this owned run finishes.
    if not Path("/dev/shm").is_dir() or not os.access("/dev/shm", os.W_OK):
        raise ValueError("live requires writable /dev/shm for private temporary session state")
    with tempfile.TemporaryDirectory(prefix="cloak-acceptance-", dir="/dev/shm") as temporary:
        scratch = Path(temporary)
        for folder in ("home", "rig-home/anthropic-oauth", "source", "workspace", "ca"):
            (scratch / folder).mkdir(parents=True, exist_ok=True)
        (scratch / "rig-home/settings.toml").write_text(settings)
        private_json(scratch / "rig-home/defaults.json", {"model": MODEL, "effort": {"effort": "low"}})
        (scratch / "workspace/facts.txt").write_text(json.dumps(FACTS))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        deadline = time.time() + args.seconds
        second_read_file = "probe-" + os.urandom(8).hex() + ".txt"
        private_json(output / "policy.json", {"requests": args.requests, "output_tokens": args.output_tokens, "deadline": deadline, "matrix": args.matrix, "second_read_file": second_read_file})
        proxy_env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(scratch / "home"), "CLOAK_ACCEPTANCE_ROOT": str(output)}
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
            if args.matrix:
                from matrix import Matrix, actions, APPEND
                private_json(output / "matrix-inputs.json", {"actions": actions(second_read_file), "auto_append": APPEND})
                Matrix(scratch, output, binary, credentials, environment(scratch, certificate, port), deadline, args.fork_resume).run()
            for scenario in (() if args.matrix else ("fresh_prefix", "fresh_prefix_repeat")):
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
            import_effects(scratch / "rig-home/sessions", evidence)
            result = report(output)
    return 0 if result["classification"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--analyze", type=Path)
    parser.add_argument("--confirm-stored-credentials", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="drive the bounded TUI matrix through a dedicated fux server")
    parser.add_argument("--fork-resume", action="store_true", help="include fork/resume in --matrix")
    for option in ("binary", "source", "plugin-config", "credentials", "output"):
        parser.add_argument("--" + option)
    for option in ("requests", "output-tokens", "seconds"):
        parser.add_argument("--" + option, type=int)
    args = parser.parse_args()
    if args.live:
        try:
            return live(args)
        except (ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
            parser.error(str(error))
    elif args.analyze:
        result = report(args.analyze)
        print(json.dumps(result, indent=2))
        return 0 if result["classification"] == "pass" else 1
    else:
        print(json.dumps(plan(), indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
