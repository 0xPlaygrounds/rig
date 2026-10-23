"""Offline build binding and CLI capability checks before inference is enabled."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib

REQUIRED = {
    "rig_claude_cloak::ClaudeCloakPlugin", "rig_tui::TuiPlugin",
    "rig_basics::print::PrintPlugin", "rig_basics::coding_tools::ReadTool",
    "rig_basics::compaction::CompactionPlugin", "rig_basics::sessions::SessionsPlugin",
    "rig_basics::sessions::agents::AgentsPlugin", "rig_basics::BasicCommandsPlugin",
    "rig_basics::telemetry::effect_log::EffectLogPlugin",
}
ROOT_TTL_TEST = "transport::live::tests::real_root_ttl_regression_and_control"
TOOL_RESULT_TEST = "transport::live::tests::tool_result::real_tool_result_representation_probe"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_identity(source):
    source = Path(source)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--binary"])
    names = subprocess.check_output(["git", "-C", str(source), "ls-files", "--others", "--exclude-standard", "-z"])
    untracked = hashlib.sha256()
    for name in sorted(value for value in names.split(b"\0") if value):
        path = source / os.fsdecode(name)
        untracked.update(name + b"\0")
        if path.is_symlink():
            untracked.update(b"symlink\0" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            untracked.update(bytes.fromhex(digest(path)))
        else:
            raise ValueError("untracked source changed while recording build identity")
    return {"head": head, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_sha256": untracked.hexdigest()}


def build_record(binary, source, plugins, generated_main, command):
    if not command:
        raise ValueError("record the exact successful build command")
    return {"schema": 1, "source": source_identity(source), "binary_sha256": digest(binary),
            "plugin_config_sha256": digest(plugins), "generated_main": str(Path(generated_main).resolve()),
            "generated_main_sha256": digest(generated_main), "build_command": command}


def verify_build(binary, source, plugins, record_path):
    record = json.loads(Path(record_path).read_text())
    expected = build_record(binary, source, plugins, record["generated_main"], record["build_command"])
    if record != expected:
        raise ValueError("binary/source/manifest changed since the successful build was recorded")
    manifest = tomllib.loads(Path(plugins).read_text())
    enabled = {p.get("plugin") for p in manifest.get("plugin", []) if p.get("enabled", True) and not p.get("disabled", False)}
    missing = REQUIRED - enabled
    generated = Path(record["generated_main"]).read_text()
    missing |= {name for name in REQUIRED if "load::<" + name + ">" not in generated}
    if missing:
        raise ValueError("acceptance build lacks required plugins: " + ", ".join(sorted(missing)))
    with tempfile.TemporaryDirectory(prefix="cloak-preflight-", dir="/dev/shm") as folder:
        path = Path(folder)
        env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": folder, "RIG_HOME": folder,
               "RIG_DESCRIBE": str(path / "description"), "RUST_LOG": "off"}
        result = subprocess.run([str(binary)], env=env, capture_output=True, timeout=30)
        report = (path / "description").read_text() if (path / "description").exists() else ""
        if result.returncode or not report.startswith("rig-describe 1\n") or "conflict\t" in report:
            raise ValueError("the supplied binary failed offline plugin description")
    return record


def verify_claude(binary):
    with tempfile.TemporaryDirectory(prefix="cloak-cli-preflight-", dir="/dev/shm") as folder:
        env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": folder}
        help_result = subprocess.run([str(binary), "--help"], env=env, capture_output=True, text=True, timeout=15)
        version = subprocess.check_output([str(binary), "--version"], env=env, text=True, timeout=15).strip()
    required = ("--safe-mode", "--strict-mcp-config", "--setting-sources", "--session-id", "--fork-session", "--effort", "--tools", "--name")
    if help_result.returncode or any(flag not in help_result.stdout for flag in required):
        raise ValueError("Claude Code lacks required isolation/session flags")
    return {"binary_sha256": digest(binary), "version": version,
            "help_sha256": hashlib.sha256(help_result.stdout.encode()).hexdigest()}


def verify_probe(binary, test):
    with tempfile.TemporaryDirectory(prefix="cloak-ttl-preflight-", dir="/dev/shm") as folder:
        result = subprocess.run([str(binary), "--list", "--format", "terse"],
                                env={"PATH": os.environ.get("PATH", os.defpath), "HOME": folder, "RIG_HOME": folder},
                                capture_output=True, text=True, timeout=15)
    if result.returncode or test + ": test" not in result.stdout.splitlines():
        raise ValueError("test binary does not contain the required ignored probe")
    return {"binary_sha256": digest(binary), "test": test}


def verify_root_ttl(binary):
    return verify_probe(binary, ROOT_TTL_TEST)


def private_capture(path):
    from audit import reject_symlinks
    path = Path(path).absolute()
    reject_symlinks(path)
    metadata = path.stat()
    if Path("/dev/shm") not in path.parents or not path.is_file() or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise ValueError("captured request requires an owned private file in tmpfs")
    for parent in path.parents:
        if parent == Path("/dev/shm"):
            break
        metadata = parent.stat()
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise ValueError("captured request requires private owned tmpfs directories")
    return path


def private_probe_metadata(path):
    import re
    path = private_capture(path)
    with path.open("rb") as stream:
        data = stream.read(4097)
    try:
        value = json.loads(data) if len(data) <= 4096 else None
    except (ValueError, UnicodeError):
        value = None
    if not isinstance(value, dict) or set(value) != {"previous_request_id"} or not isinstance(value["previous_request_id"], str) or not re.fullmatch(r"req_[A-Za-z0-9_-]{1,36}", value["previous_request_id"]):
        raise ValueError("probe metadata requires one valid previous request ID")
    return path
