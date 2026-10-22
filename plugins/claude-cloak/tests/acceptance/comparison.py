"""Claude Code TUI adapter and comparisons grouped by client, phase and call role."""
import json
import os
from pathlib import Path
import uuid
import subprocess
import math

from audit import PHASE_NAMES, private_json, reject_symlinks
from matrix import Matrix, actions, RECALL, SECOND_READ_RECALL, SEED, APPEND
from wire_profile import request_role


class ClaudeMatrix(Matrix):
    def __init__(self, *args, auth_home, claude_auth="stored", **kwargs):
        super().__init__(*args, client="claude", **kwargs)
        self.auth_home = Path(auth_home)
        self.claude_auth = claude_auth
        if self.scope == "full":
            self.checks["auto_compact"] = "unknown"
        self.fork_next = False

    def launch(self, tools="read", resume=None):
        self.session = resume or str(uuid.uuid4())
        child = Path(__file__).resolve().with_name("comparison.py")
        specification = {"binary": str(self.binary), "scratch": str(self.scratch), "auth_home": str(self.auth_home),
                         "auth_mode": self.claude_auth, "rig_credentials": str(self.credentials) if self.credentials else None,
                         "env": dict(self.env, TERM="xterm-256color", XDG_RUNTIME_DIR=self.controller_env["XDG_RUNTIME_DIR"]),
                         "session": self.session, "resume": bool(resume), "fork": self.fork_next,
                         "tools": "Read" if tools else "", "deadline": self.deadline,
                         "diagnostics": str(self.output / "child-startup.json")}
        self.fork_next = False
        spec = self.scratch / "claude-child.json"
        private_json(spec, specification)
        try:
            result = subprocess.run(["python3", str(child), "--validate-child", str(spec)], capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            self.status("child_failed")
            raise RuntimeError("Claude Code child preflight timed out") from None
        if result.returncode:
            self.status("child_failed", result.returncode)
            raise RuntimeError("Claude Code child preflight failed; see child-startup.json")
        # The access token is loaded by the child, never placed in fux argv/journals.
        self.launch_command(["python3", str(child), "--child", str(spec)])

    def settled(self):
        # Claude Code has no Rig effect log. Correlation is explicitly wire-only.
        return {r.get("request_id"): "compaction" if r.get("scenario") in ("manual_compact", "auto_compact") else "completion"
                for r in self.records() if r.get("kind") == "response" and r.get("terminal") and r.get("request_id")}

    @staticmethod
    def check(name, rows, settled):
        if name == "auto_trigger":
            # This client has a different threshold. Its matching large append is
            # followed by an explicit compact, reported as such in comparison.json.
            name = "neutral"
        if name == "auto_compact":
            name = "manual_compact"
        return Matrix.check(name, rows, settled)

    def fork_session_id(self):
        files = list((self.scratch / "home/.claude/projects").glob("*/*.jsonl"))
        candidates = [p for p in files if p.stem != self.session and len(p.stem) == 36]
        if not candidates:
            raise RuntimeError("Claude Code did not persist the fork session")
        return max(candidates, key=lambda p: p.stat().st_mtime_ns).stem

    def run(self):
        try:
            self.launch()
            if self.startup_only:
                self.send("/quit")
                self.status("startup_complete")
                return
            chosen = actions(self.second_read_file)[:4] if self.scope == "probes" else actions(self.second_read_file)
            for name, text in chosen:
                if name == "effort":
                    self.send("/effort medium")
                if name == "neutral":
                    self.send("/effort low")
                # Tool names are client-specific; synthetic content is identical.
                if name in ("read_success", "read_error"):
                    text = text.replace("Use read exactly", "Use Read exactly")
                self.phase(name, text)
            if self.scope == "probes":
                self.send("/quit")
                return
            if self.include_fork_resume:
                original = self.session
                self.send("/quit")
                self.fork_next = True
                self.launch(resume=original)
                self.phase("fork", RECALL + SECOND_READ_RECALL)
                fork = self.fork_session_id()
                self.send("/quit")
                self.launch(resume=fork)
                self.phase("resume", RECALL + SECOND_READ_RECALL)
            self.send("/quit")
            self.launch()
            self.phase("auto_seed", SEED)
            self.phase("auto_trigger", APPEND)
            self.phase("auto_compact", "/compact Preserve all five ledger values and the newest comet value. Omit padding.")
            self.phase("auto_recall", RECALL + " Also return the comet value from the newest large message.")
            self.phase("auto_rewarm", RECALL + " Also return the comet value.")
            self.send("/quit")
            for name in ("fresh_prefix", "fresh_prefix_repeat"):
                self.launch(tools="")
                self.phase(name, SEED)
                self.send("/quit")
        finally:
            self.close()
            private_json(self.checks_path, self.checks)


def rig_access(spec):
    try:
        value = json.loads((Path(spec["rig_credentials"]) / "credentials.json").read_text())
        expires = value.get("expires_at")
        if type(expires) not in (int, float):
            raise ValueError()
        selected = {"accessToken": value.get("access_token"), "expiresAt": expires * 1000,
                    "accountUuid": value.get("account_id"), "organizationUuid": value.get("org_id")}
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError("credential_store_invalid") from None
    if not isinstance(selected["accountUuid"], str) or not selected["accountUuid"]:
        raise RuntimeError("account_identity_missing")
    return selected


def access_token(spec):
    try:
        credentials = rig_access(spec) if spec.get("auth_mode", "stored") == "rig-access-token" else json.loads((Path(spec["auth_home"]) / ".claude/.credentials.json").read_text())["claudeAiOauth"]
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError("credential_store_invalid") from None
    if not isinstance(credentials.get("accessToken"), str) or not credentials["accessToken"]:
        raise RuntimeError("access_token_missing")
    if type(credentials.get("expiresAt")) not in (int, float) or not math.isfinite(credentials["expiresAt"]) or credentials["expiresAt"] / 1000 <= spec["deadline"] + 60:
        raise RuntimeError("access_token_expired")
    return credentials["accessToken"]


def claude_configuration(spec, config):
    minimal = {key: config[key] for key in ("machineID", "userID") if key in config}
    if spec.get("auth_mode", "stored") == "rig-access-token":
        selected = rig_access(spec)
        account = {"accountUuid": selected["accountUuid"]}
        if isinstance(selected.get("organizationUuid"), str) and selected["organizationUuid"]:
            account["organizationUuid"] = selected["organizationUuid"]
        minimal["oauthAccount"] = account
    elif "oauthAccount" in config:
        minimal["oauthAccount"] = config["oauthAccount"]
    minimal.update(hasCompletedOnboarding=True, theme="dark", projects={str(Path(spec["scratch"]) / "workspace"): {"hasTrustDialogAccepted": True}})
    return minimal


def validate_child(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    mode = spec.get("auth_mode", "stored")
    result = {"client": "claude", "state": "ready", "auth_mode": mode if mode in ("stored", "rig-access-token") else "invalid"}
    code = 0
    try:
        access_token(spec)
        if mode not in ("stored", "rig-access-token"):
            raise RuntimeError("configuration_invalid")
        for name in ("binary",):
            if not Path(spec[name]).is_file():
                raise RuntimeError("binary_missing")
        if not Path(spec["env"]["SSL_CERT_FILE"]).is_file():
            raise RuntimeError("certificate_missing")
        config = Path(spec["auth_home"]) / ".claude.json"
        if config.exists() and not isinstance(json.loads(config.read_text()), dict):
            raise RuntimeError("configuration_invalid")
        if mode == "rig-access-token":
            original_config = json.loads(config.read_text()) if config.exists() else {}
            account = rig_access(spec)["accountUuid"]
            stored_account = original_config.get("oauthAccount")
            result["stored_account_matches_rig"] = isinstance(stored_account, dict) and stored_account.get("accountUuid") == account
            result["selected_account_matches_rig"] = claude_configuration(spec, original_config).get("oauthAccount", {}).get("accountUuid") == account
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        known = {"credential_store_invalid", "access_token_missing", "access_token_expired", "account_identity_missing", "binary_missing", "certificate_missing", "configuration_invalid"}
        result = {"client": "claude", "auth_mode": mode if mode in ("stored", "rig-access-token") else "invalid", "state": str(error) if str(error) in known else "configuration_invalid"}
        code = 2
    private_json(Path(spec["diagnostics"]), result)
    return code


def child(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    scratch, auth_home = Path(spec["scratch"]), Path(spec["auth_home"])
    # Do not copy refresh grants into a disposable client store. A bounded run
    # requires a token that remains valid for its whole remaining duration.
    token = access_token(spec)
    config_path = auth_home / ".claude.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    minimal = claude_configuration(spec, config)
    private_json(scratch / "home/.claude.json", minimal)
    env = dict(spec["env"], CLAUDE_CODE_OAUTH_TOKEN=token, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", CLAUDE_CODE_DISABLE_TERMINAL_TITLE="1",
               NODE_EXTRA_CA_CERTS=spec["env"]["SSL_CERT_FILE"])
    command = ["bwrap", "--die-with-parent", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
               "--bind", str(scratch), str(scratch), "--chdir", str(scratch / "workspace"), "--", spec["binary"],
               "--safe-mode", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--tools", spec["tools"],
               "--allowedTools", spec["tools"], "--setting-sources", "", "--effort", "low", "--model", "claude-opus-5-5",
               "--permission-mode", "dontAsk", "--prompt-suggestions", "false"]
    command += ["--name", "synthetic parity"]
    if spec["resume"]:
        command += ["--resume", spec["session"]]
        if spec["fork"]:
            command += ["--fork-session"]
    else:
        command += ["--session-id", spec["session"]]
    os.execvpe(command[0], command, env)


def profile_equal(left, right, key):
    if key not in left or key not in right:
        return None
    if key == "messages" and left[key] != right[key]:
        def missing_optional(first, second):
            if isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
                return any(missing_optional(a, b) for a, b in zip(first, second))
            if isinstance(first, dict) and isinstance(second, dict):
                return ("is_error_present" in first) != ("is_error_present" in second) or any(missing_optional(first[name], second[name]) for name in first.keys() & second.keys())
            return False
        def without_optional(value):
            if isinstance(value, list):
                return [without_optional(item) for item in value]
            if isinstance(value, dict):
                return {name: without_optional(item) for name, item in value.items() if name != "is_error_present"}
            return value
        if missing_optional(left[key], right[key]) and without_optional(left[key]) == without_optional(right[key]):
            return None
    return left[key] == right[key]


def compare_profiles(root, inputs=None):
    root = Path(root)
    grouped = {}
    for source in inputs or [root]:
        path = Path(source) / "profiles.jsonl"
        reject_symlinks(path)
        for line in path.read_text().splitlines() if path.exists() else []:
            item = json.loads(line)
            role = request_role(item["messages"]) if isinstance(item.get("messages"), list) else item.get("call_role")
            if item.get("role") != "primary" or item.get("client") not in ("rig", "claude") or item.get("scenario") not in PHASE_NAMES or role not in ("prompt", "tool_continuation"):
                continue
            key = (item.get("scenario"), role)
            grouped.setdefault(key, {"rig": [], "claude": []})[item["client"]].append(item)
    comparisons = []
    for (phase, role), clients in grouped.items():
        for index in range(max(len(clients["rig"]), len(clients["claude"]))):
            left = clients["rig"][index] if index < len(clients["rig"]) else None
            right = clients["claude"][index] if index < len(clients["claude"]) else None
            row = {"phase": phase, "role": role, "within_role": index, "paired": left is not None and right is not None}
            if row["paired"]:
                row["equal"] = {key: profile_equal(left, right, key) for key in ("headers", "body_keys", "system", "messages", "tools", "thinking_type", "thinking_binding", "thinking_budget", "thinking_display", "context_management", "compaction", "metadata_user_id", "max_tokens", "effort", "tool_choice", "markers", "diagnostics_previous_message_present", "requested_fallback_models", "fallback_shape_valid")}
                row["equal"].update(correlation_header_structure=None, extra_header_count=None, session_header_matches_metadata=None)
                if all(isinstance(profile.get("correlation_headers"), dict) for profile in (left, right)):
                    # Per-run pseudonym keys differ; compare presence and known roles only.
                    structures = [{key: {field: value.get(field) for field in ("present", "role") if field in value} for key, value in profile["correlation_headers"].items() if isinstance(value, dict)} for profile in (left, right)]
                    mismatch = any(structures[0][name][field] != structures[1][name][field] for name in structures[0].keys() & structures[1].keys() for field in structures[0][name].keys() & structures[1][name].keys())
                    complete = structures[0].keys() == structures[1].keys() and all(structures[0][name].keys() == structures[1][name].keys() for name in structures[0])
                    row["equal"]["correlation_header_structure"] = False if mismatch else True if complete else None
                    row["equal"]["extra_header_count"] = profile_equal(left, right, "extra_header_count")
                    row["equal"]["session_header_matches_metadata"] = profile_equal(left, right, "session_header_matches_metadata")
                row["checksum"] = {name: value.get("checksum") if value.get("checksum") in ("match", "mismatch", "unknown") else "unknown" for name, value in (("rig", left), ("claude", right))}
            comparisons.append(row)
    result = {"schema": 1, "pairs": comparisons, "claude_large_append_compaction": "manual-after-append",
              "limits": ["Pairs share a phase and request role; differing tool turns remain visible.",
                         "Claude Code wire completion is not a Rig effect-log correlation.", "Application-specific prompts and tool definitions may differ."]}
    private_json(root / "comparison.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--child")
    mode.add_argument("--validate-child")
    args = parser.parse_args()
    if args.validate_child:
        raise SystemExit(validate_child(args.validate_child))
    child(args.child)
