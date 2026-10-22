"""Bounded fux driver for a dedicated actual Rig TUI. Never touches existing panes."""
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from audit import AUTO_INPUT, Evidence, FACTS, analyze, effect_records, private_json

RECALL = "Without tools or reopening files, return the exact harbor, garden, observatory, archive and station values from memory."
SEED = "Remember this synthetic ledger: " + json.dumps(FACTS) + "\n" + ("Neutral padding with no additional facts.\n" * 1800) + "\nReturn the five values only."
APPEND = AUTO_INPUT
SECOND_READ_RECALL = " Also recall the second file-read attempt from this conversation. Give its filename, outcome and number of attempts, formatted as filename | outcome | attempt count. Do not use tools."


def actions(second_read_file="probe-cobalt-619.txt"):
    return [
        ("seed", SEED), ("warm", RECALL),
        ("read_success", "Use read exactly once on facts.txt. Return the five ledger values. Do not use another tool."),
        ("read_error", f"Use read exactly once on {second_read_file}. Do not retry. Report its outcome, then return the five ledger values from memory."),
        ("effort", RECALL),
        ("neutral", "Reply exactly READY. Do not repeat any facts or use tools."),
        ("manual_compact", "/compact Preserve the five ledger values and the second file-read filename, outcome and attempt count. Omit padding. Keep at most 200 words."),
        ("manual_recall", RECALL + SECOND_READ_RECALL), ("rewarm", RECALL + SECOND_READ_RECALL),
    ]


class Matrix:
    def __init__(self, scratch, output, binary, credentials, env, deadline, include_fork_resume=False,
                 client="rig", scope="full", pause_after_seed=False, raw_directory=None, startup_only=False,
                 pause_after_phase=None, pause_on_failure=False, continue_after_read_refusal=False,
                 read_refusal_next="effort", collect_failures=False):
        self.scratch, self.output = Path(scratch), Path(output)
        self.binary, self.credentials, self.env = binary, credentials, env
        self.deadline = deadline
        self.include_fork_resume = include_fork_resume
        self.client, self.scope = client, scope
        self.pause_after_seed, self.raw_directory = pause_after_seed, raw_directory
        self.pause_after_phase, self.pause_on_failure = pause_after_phase, pause_on_failure
        self.continue_after_read_refusal = continue_after_read_refusal
        if read_refusal_next not in ("effort", "manual_compact") or (read_refusal_next != "effort" and not continue_after_read_refusal):
            raise ValueError("read refusal destination requires an enabled, known continuation")
        self.read_refusal_next = read_refusal_next
        self.pending_read_refusal = None
        self.read_refusal_continuation = None
        if collect_failures and (client != "rig" or scope != "full" or startup_only or continue_after_read_refusal):
            raise ValueError("failure collection requires a separate full Rig diagnostic")
        self.collect_failures = collect_failures
        self.pending_failed_phase = None
        self.startup_only = startup_only
        # fux's Unix socket also includes the private runtime directory.
        self.server = "ca-" + uuid.uuid4().hex[:16]
        self.pane = None
        self.launch_count = 0
        self.started = False
        runtime = self.scratch / "runtime"
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        runtime.chmod(0o700)
        self.controller_env = {"PATH": self.env.get("PATH", os.defpath),
                               "HOME": self.env.get("HOME", str(self.scratch / "home")),
                               "TERM": "xterm-256color", "XDG_RUNTIME_DIR": str(runtime)}
        self.observed = {}
        self.evidence = Evidence(output)
        policy_path = self.output / "policy.json"
        self.second_read_file = json.loads(policy_path.read_text()).get("second_read_file") if policy_path.exists() else "probe-cobalt-619.txt"
        self.checks = {name: "unknown" for name, _ in actions()}
        self.checks.update({name: "unknown" for name in ("auto_seed", "auto_trigger", "auto_recall", "auto_rewarm", "fresh_prefix", "fresh_prefix_repeat")})
        if include_fork_resume or scope == "auto-controls":
            self.checks.update(fork="unknown", resume="unknown")
        if scope == "probes":
            self.checks = {name: "unknown" for name, _ in actions()[:4]}
        self.checks_path = self.output / ("matrix.json" if client == "rig" else "claude-matrix.json")
        private_json(self.checks_path, self.checks)

    def status(self, event, code=None):
        allowed = ("server_failed", "socket_path_too_long", "fux_failed", "pane_created", "input_ready", "prompt_visible", "onboarding_required", "startup_timeout", "child_failed", "phase_failed", "startup_complete", "python_failed", "sandbox_failed", "cli_arguments_failed", "certificate_failed", "authentication_required", "workspace_trusted", "pane_unidentified", "fingerprint_reanchored")
        record = {"client": self.client, "event": event if event in allowed else "fux_failed", "launch": self.launch_count}
        if type(code) is int:
            record["exit_code"] = code
        with os.fdopen(os.open(self.output / "driver-events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def fux_command(server, words):
        if not words or (words[0] != "opts" and not re.fullmatch(r"%[0-9]+", words[0])):
            raise ValueError("fux action requires opts or an explicit pane")
        if words[-1] in ("get", "capture"):
            return ["fux", "str", server, "server", *words]
        # timeout transforms an existing request/options value on the stack.
        return ["fux", "str", server, "server", words[0], "1", "timeout", *words[1:]]

    def fux(self, words, allow_exited=False):
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("matrix elapsed-time cap reached")
        # An explicit private server selection is present on EVERY invocation.
        result = subprocess.run(self.fux_command(self.server, words),
                                capture_output=True, text=True, timeout=min(remaining, 10),
                                env=self.controller_env)
        # 124 means the action was accepted but the pane is still busy. The
        # phase's bounded wire/effect observer, not fux's settle timer, waits.
        if result.returncode not in ((0, 3, 124) if allow_exited else (0, 124)):
            self.status("fux_failed", result.returncode)
            raise RuntimeError("dedicated fux command failed; raw screen output intentionally not persisted")
        return result.stdout

    def launch(self, tools="read", resume=None):
        self.session = resume or str(time.time_ns())
        env = dict(self.env, TERM="xterm-256color", RIG_SESSION=self.session,
                   XDG_RUNTIME_DIR=self.controller_env["XDG_RUNTIME_DIR"])
        if resume:
            env.update(RIG_START="resumed")
        command = ["env", "-i", *[key + "=" + value for key, value in env.items()], "bwrap", "--die-with-parent",
                   "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                   "--bind", str(self.scratch), str(self.scratch),
                   "--bind", str(self.credentials), str(self.scratch / "rig-home/anthropic-oauth"),
                   "--chdir", str(self.scratch / "workspace"),
                   "--", str(self.binary), "--tools=" + tools]
        self.launch_command(command)

    def launch_command(self, command):
        (self.output / "client.txt").write_text(self.client)
        if not self.started:
            remaining = self.deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("matrix elapsed-time cap reached")
            started = subprocess.run(["fux", "str", self.server, "server", "start"],
                                     capture_output=True, text=True, timeout=min(remaining, 10),
                                     env=self.controller_env)
            if started.returncode:
                reason = "socket_path_too_long" if "socket" in started.stderr.lower() and "long" in started.stderr.lower() else "server_failed"
                self.status(reason, started.returncode)
                raise RuntimeError("private fux server could not start")
            self.started = True
        self.launch_count += 1
        words = ["opts", "str", f"cloak-acceptance-{self.launch_count}", "name", "140", "42", "size", "str", str(self.scratch / "workspace"), "cwd", "str", command[0], "program"]
        for argument in command[1:]:
            words += ["str", argument, "arg"]
        answer = self.fux(words + ["run"], allow_exited=True)
        pane = re.search(r"(?m)^%[0-9]+\b", answer)
        if not pane:
            self.status("pane_unidentified")
            raise RuntimeError("fux did not identify the newly created pane")
        self.pane = pane.group()
        self.status("pane_created")
        self.await_input_ready()

    def await_input_ready(self):
        end = min(self.deadline, time.time() + 30)
        while time.time() < end:
            # Read only the terminal's state. No screen text leaves private fux.
            answer = self.fux([self.pane, "str", "state", "get"], allow_exited=True)
            screen = self.fux([self.pane, "capture"], allow_exited=True).lower()
            if re.search(r"\bexited\b", answer):
                # Classify only known startup diagnostics. Screens and exception
                # text remain inside this process and private fux journals.
                code = next((name for needle, name in (("traceback (most recent call last)", "python_failed"),
                            ("python3: can't open", "python_failed"), ("bwrap:", "sandbox_failed"),
                            ("certificate", "certificate_failed"), ("not logged in", "authentication_required"),
                            ("error:", "cli_arguments_failed")) if needle in screen), "child_failed")
                self.status(code)
                raise RuntimeError("client exited before becoming ready for input")
            if self.client == "claude" and any(text in screen for text in ("yes, i trust this folder", "yes, i trust this directory")):
                self.status("workspace_trusted")
                self.fux([self.pane, "str", "Enter", "key", "send-keys"])
                continue
            if self.client == "claude" and any(text in screen for text in ("choose the text style", "choose a theme")):
                self.status("onboarding_required")
                raise RuntimeError("isolated client requires onboarding before model input")
            prompt_visible = self.client == "claude" and "❯" in screen
            if re.search(r"\breading\b", answer) or prompt_visible:
                if prompt_visible:
                    self.status("prompt_visible")
                self.status("input_ready")
                return
            self.fux([self.pane, "wait"], allow_exited=True)
        self.status("startup_timeout")
        raise RuntimeError("client did not become ready for input before startup timeout")

    def send(self, text):
        # paste-file handles large prompts without shell interpolation or argv limits.
        path = self.scratch / "input.txt"
        path.write_text(text)
        self.fux([self.pane, "str", str(path), "paste-file", "str", "Enter", "key", "send-keys"], allow_exited=text == "/quit")

    def records(self):
        path = self.output / "audit.jsonl"
        records = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                    if record.get("client", "rig") == self.client:
                        records.append(record)
                except ValueError:
                    # A writer may still be appending the last line. Retry on the next wait.
                    break
        return records

    def settled(self):
        outcomes = {}
        for path in (self.scratch / "rig-home/sessions").glob("*/files/effects.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    records = effect_records(json.loads(line))
                except ValueError:
                    break
                for record, result in records:
                    rid = self.evidence.alias(result.get("provider_request_id"))
                    if rid:
                        outcomes[rid] = record.get("purpose")
        return outcomes

    def phase(self, name, text):
        if self.pending_failed_phase is not None:
            previous, following, cause = self.pending_failed_phase
            if name != following or self.checks.get(previous) != "fail":
                raise RuntimeError("failure collection requires the next fixed script phase")
            with os.fdopen(os.open(self.output / "failure-continuations.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a") as output:
                output.write(json.dumps({"client": "rig", "from_phase": previous, "next_phase": name, "cause": cause,
                                         "failed_check_preserved": True, "retry": False,
                                         "effect_correlated": True, "tui_ready": True}, sort_keys=True) + "\n")
            self.pending_failed_phase = None
        if self.pending_read_refusal is not None:
            if name != self.read_refusal_next or text == self.pending_read_refusal or self.checks["read_error"] != "fail":
                raise RuntimeError("read refusal continuation requires the selected distinct phase")
            if name == "manual_compact" and text != dict(actions(self.second_read_file))["manual_compact"]:
                raise RuntimeError("read refusal compaction requires the unchanged compaction prompt")
            with os.fdopen(os.open(self.output / "continuations.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a") as output:
                output.write(json.dumps({"client": "rig", "from_phase": "read_error", "next_phase": name, "cause": "refusal",
                                         "failed_check_preserved": self.checks["read_error"] == "fail", "retry": False,
                                         "effect_correlated": True, "tui_ready": True}, sort_keys=True) + "\n")
            self.read_refusal_continuation = name
            self.pending_read_refusal = None
        (self.output / "client.txt").write_text(self.client)
        (self.output / "phase.txt").write_text(name)
        self.send(text)
        rows = []
        phase_rows = []
        continued = False
        while time.time() < self.deadline:
            phase_rows = [r for r in analyze(self.records())["calls"] if r.get("scenario") == name]
            rows = [r for r in phase_rows if r["classification"] != "blocked_auxiliary"]
            if any(r["classification"] == "fail" for r in rows):
                self.checks[name] = "fail"
                if self.continue_after_read_refusal and self.client == "rig" and name == "read_error" and self.read_refusal_wire(phase_rows, self.observed.get("read_success", {})):
                    settled = self.settled()
                    if not all(row.get("request_id") in settled for row in phase_rows):
                        time.sleep(0.1)
                        continue
                    self.await_input_ready()
                    phase_rows = [r for r in analyze(self.records())["calls"] if r.get("scenario") == name]
                    if self.read_refusal_wire(phase_rows, self.observed.get("read_success", {})) and all(row.get("request_id") in self.settled() for row in phase_rows):
                        self.observed[name] = phase_rows[-1]
                        self.pending_read_refusal = text
                        continued = True
                break
            if any(r.get("capture_unknown") for r in rows):
                break
            settled = self.settled()
            # Tool-use terminal is not a final answer. Wait for an end-turn response
            # and its real effect result, not an arbitrary sleep or a token count.
            if rows and rows[-1].get("final_answer") and rows[-1].get("request_id") in settled:
                if name == "auto_trigger" and self.client == "rig" and settled.get(rows[-1].get("request_id")) == "compaction":
                    time.sleep(0.1)
                    continue
                if name == "auto_trigger" and self.client == "rig" and not any(settled.get(r.get("request_id")) == "compaction" for r in rows):
                    self.checks[name] = "unknown"
                else:
                    self.checks[name] = self.check(name, rows, settled)
                last = rows[-1]
                if name == "read_error":
                    previous = self.observed.get("read_success", {})
                    if not all(key in previous and key in last for key in ("tool_results", "tool_errors")):
                        self.checks[name] = self.combine(self.checks[name], "unknown")
                    elif last["tool_results"] != previous["tool_results"] + 1 or last["tool_errors"] != previous["tool_errors"] + 1:
                        self.checks[name] = "fail"
                self.checks[name] = self.combine(self.checks[name], self.fingerprint_check(name, last))
                if self.collect_failures and name in ("manual_recall", "rewarm", "fork", "resume") and not self.failed_read_history():
                    self.checks[name] = "fail"
                if name == "manual_recall":
                    source = "read_error" if self.read_refusal_continuation == "manual_compact" else "neutral"
                    before = self.observed.get(source, {}).get("input_chars")
                    after = last.get("input_chars")
                    if type(before) is not int or type(after) is not int or not 0 < after < before:
                        self.checks[name] = "fail"
                if name == "auto_trigger" and not last.get("newest_probe_intact"):
                    self.checks[name] = "fail"
                if name == "fork":
                    self.checks[name] = self.combine(self.checks[name], self.identity(last, self.observed.get("seed", {}), "session", different=True))
                if name == "resume":
                    self.checks[name] = self.combine(self.checks[name], self.identity(last, self.observed.get("fork", {}), "session"))
                self.observed[name] = last
                break
            time.sleep(0.1)
        if self.collect_failures and self.checks[name] == "pass" and not self.collectable_failure(name, phase_rows, self.settled()):
            self.checks[name] = "fail"
        if self.collect_failures and self.checks[name] == "fail" and self.collectable_failure(name, phase_rows):
            while time.time() < self.deadline and not all(row.get("request_id") in self.settled() for row in phase_rows):
                time.sleep(0.1)
            if time.time() < self.deadline:
                self.await_input_ready()
                phase_rows = [r for r in analyze(self.records())["calls"] if r.get("scenario") == name]
                settled = self.settled()
                if self.collectable_failure(name, phase_rows, settled) and all(row.get("request_id") in settled for row in phase_rows):
                    following = self.next_fixed_phase(name)
                    if following is not None:
                        self.observed[name] = phase_rows[-1]
                        cause = "refusal" if phase_rows[-1].get("stop_reason") == "refusal" else "semantic_assertion"
                        self.pending_failed_phase = (name, following, cause)
                        continued = True
        private_json(self.checks_path, self.checks)
        if self.checks[name] != "pass":
            self.status("phase_failed")
        if (name == "seed" and self.pause_after_seed) or name == self.pause_after_phase or (self.pause_on_failure and self.checks[name] != "pass"):
            self.pause(name)
        if self.checks[name] != "pass" and not continued:
            raise RuntimeError("matrix phase did not pass: " + name)
        # Let the terminal's command/event processing settle before a local slash command.
        self.fux([self.pane, "wait"])
        return rows

    def skip_after_read_refusal(self, name):
        return self.pending_read_refusal is not None and self.read_refusal_next == "manual_compact" and name in ("effort", "neutral")

    def next_fixed_phase(self, name):
        phases = [phase for phase, _ in actions(self.second_read_file)]
        if self.include_fork_resume:
            phases += ["fork", "resume"]
        phases += ["auto_seed", "auto_trigger", "auto_recall", "auto_rewarm", "fresh_prefix", "fresh_prefix_repeat"]
        return next((following for current, following in zip(phases, phases[1:]) if current == name), None)

    def failed_read_history(self):
        rows = [row for row in analyze(self.records())["calls"] if row.get("scenario") == "read_error"]
        settled = self.settled()
        if len(rows) != 2 or not self.collectable_failure("read_error", rows, settled) or not all(row.get("request_id") in settled for row in rows):
            return False
        previous = self.observed.get("read_success", {})
        return all(type(previous.get(key)) is int and type(rows[-1].get(key)) is int and rows[-1][key] == previous[key] + 1 for key in ("tool_results", "tool_errors"))

    @staticmethod
    def collectable_failure(name, rows, settled=None):
        if not rows or rows[-1].get("stop_reason") not in ("end_turn", "refusal"):
            return False
        if rows[:-1]:
            if name in ("read_success", "read_error"):
                if any(row.get("stop_reason") != "tool_use" for row in rows[:-1]):
                    return False
            elif name == "auto_trigger":
                # The first pass checks framing before waiting for effects;
                # the second proves these completed calls were compaction.
                if any(row.get("stop_reason") != "end_turn" or (settled is not None and settled.get(row.get("request_id")) != "compaction") for row in rows[:-1]):
                    return False
            else:
                return False
        for row in rows:
            known_refusal = row.get("classification") == "fail" and row.get("stop_reason") == "refusal" and row.get("successful_stop") is False
            cancelled_after_terminal = row.get("close") == "cancel_after_terminal" and (row.get("classification") == "pass" or known_refusal) and row.get("pending_frame") is False
            if not (row.get("kind") == "response" and row.get("status") == 200 and row.get("terminal") is True
                    and row.get("message_stop") is True and (row.get("close") == "complete" or cancelled_after_terminal) and row.get("compression_complete") is True
                    and row.get("model_ok") is True and row.get("requested_model_ok") is True and row.get("tools_read_only") is True
                    and row.get("fallback_shape_valid") is True and isinstance(row.get("requested_fallback_models"), list)
                    and all(model == "claude-opus-5-5" for model in row["requested_fallback_models"])
                    and row.get("markers_valid") is True and row.get("request_id")
                    and row.get("classification") in ("pass", "fail")
                    and all(type(row.get("usage", {}).get(key)) is int and row["usage"][key] >= 0 for key in ("cache_creation_input_tokens", "cache_read_input_tokens"))):
                return False
            if any(row.get(key) for key in ("pending_frame", "parse_error", "provider_error", "local_guard", "capture_unknown")) or (row.get("transport_error") and not cancelled_after_terminal):
                return False
            if type(row.get("tool_invocations")) is not int or row["tool_invocations"] < 0:
                return False
        if len({row["request_id"] for row in rows}) != len(rows):
            return False
        if rows[-1]["tool_invocations"] != 0:
            return False
        invocations = sum(row["tool_invocations"] for row in rows)
        if name in ("read_success", "read_error"):
            initial_refusal = len(rows) == 1 and invocations == 0 and rows[-1]["stop_reason"] == "refusal"
            one_read = len(rows) == 2 and rows[0]["tool_invocations"] == 1
            return all(row.get("tool_target_match") is True for row in rows) and (initial_refusal or one_read)
        return invocations == 0

    @staticmethod
    def read_refusal_wire(rows, previous):
        if len(rows) != 2 or rows[0].get("classification") != "pass" or rows[-1].get("classification") != "fail" or rows[0].get("stop_reason") != "tool_use" or rows[-1].get("stop_reason") != "refusal":
            return False
        for row in rows:
            if not (row.get("kind") == "response" and row.get("status") == 200 and row.get("terminal") is True
                    and row.get("message_stop") is True and row.get("close") == "complete" and row.get("compression_complete") is True
                    and row.get("model_ok") is True and row.get("requested_model_ok") is True and row.get("tools_read_only") is True
                    and row.get("fallback_shape_valid") is True and not any(model != "claude-opus-5-5" for model in row.get("requested_fallback_models", []))
                    and row.get("markers_valid") is True and row.get("tool_target_match") is True and row.get("request_id")
                    and all(key in row.get("usage", {}) for key in ("cache_creation_input_tokens", "cache_read_input_tokens"))):
                return False
            if any(row.get(key) for key in ("pending_frame", "parse_error", "provider_error", "transport_error", "local_guard", "capture_unknown")):
                return False
        if rows[0].get("tool_invocations") != 1 or rows[-1].get("tool_invocations") != 0:
            return False
        return all(type(previous.get(key)) is int and type(rows[-1].get(key)) is int and rows[-1][key] == previous[key] + 1 for key in ("tool_results", "tool_errors"))

    def pause(self, name):
        record = {"client": self.client, "phase": name, "status": self.checks[name], "state": "waiting",
                  "raw_directory": str(self.raw_directory), "deadline": self.deadline}
        private_json(self.output / "paused.json", record)
        release = self.output / ("continue-" + self.client)
        while not release.exists():
            if time.time() >= self.deadline:
                raise TimeoutError("private request inspection pause reached the live deadline")
            time.sleep(0.1)
        release.unlink()
        private_json(self.output / "paused.json", {**record, "state": "released"})

    @staticmethod
    def combine(*statuses):
        return "fail" if "fail" in statuses else "unknown" if "unknown" in statuses else "pass"

    def fingerprint_check(self, name, current):
        if self.client == "rig":
            return self.identity(current, self.observed.get("seed", {}), "human_fingerprint") if name in ("manual_recall", "fork", "resume") else "pass"
        if name in ("manual_recall", "auto_recall"):
            previous = self.observed.get("seed" if name == "manual_recall" else "auto_seed", {})
            if not current.get("human_fingerprint") or not previous.get("human_fingerprint"):
                return "unknown"
            if current["human_fingerprint"] != previous["human_fingerprint"]:
                self.status("fingerprint_reanchored")
            return "pass"
        previous = {"rewarm": "manual_recall", "fork": "rewarm", "resume": "fork", "auto_rewarm": "auto_recall"}.get(name)
        return self.identity(current, self.observed.get(previous, {}), "human_fingerprint") if previous else "pass"

    @staticmethod
    def identity(current, previous, key, different=False):
        if not current.get(key) or not previous.get(key):
            return "unknown"
        return "pass" if (current[key] != previous[key]) == different else "fail"

    @staticmethod
    def check(name, rows, settled):
        if not rows:
            return "unknown"
        last = rows[-1]
        state = Matrix.combine(*(r["classification"] for r in rows))
        if state != "pass":
            return state
        if not all(key in last.get("usage", {}) for key in ("cache_creation_input_tokens", "cache_read_input_tokens")):
            return "unknown"
        if name == "manual_compact" and settled.get(last.get("request_id")) != "compaction":
            return "unknown"
        ok = True
        if "nonempty_text" in last:
            ok = ok and last["nonempty_text"]
        if name in ("neutral", "auto_trigger") and "ready_reply" in last:
            ok = ok and last["ready_reply"]
        if name not in ("neutral", "manual_compact", "auto_trigger"):
            ok = ok and all(last.get("facts", {}).get(key) for key in FACTS)
        if name in ("warm", "effort", "rewarm", "auto_rewarm", "fresh_prefix_repeat"):
            ok = ok and last.get("usage", {}).get("cache_read_input_tokens", 0) > 0
        if name == "read_success":
            ok = ok and last.get("tool_results", 0) > 0 and last.get("tool_errors", 0) == 0
        if name == "read_error":
            ok = ok and last.get("tool_errors", 0) > 0
        if name in ("read_success", "read_error"):
            if not all("tool_invocations" in row and "tool_target_match" in row for row in rows):
                return "unknown"
            ok = ok and sum(row["tool_invocations"] for row in rows) == 1 and all(row["tool_target_match"] for row in rows)
        if name == "effort":
            ok = ok and last.get("effort") == "medium"
        if name == "manual_compact":
            ok = ok and settled.get(last.get("request_id")) == "compaction"
        if name in ("manual_recall", "rewarm", "fork", "resume"):
            if "second_read" not in last:
                return "unknown"
            ok = ok and last["second_read"].get("filename") and last["second_read"].get("failed_once")
        if name in ("auto_recall", "auto_rewarm"):
            ok = ok and last.get("auto_fact")
        if name in ("fresh_prefix", "fresh_prefix_repeat"):
            ok = ok and last.get("tool_count") == 0
        ok = ok and all(key in last.get("usage", {}) for key in ("cache_creation_input_tokens", "cache_read_input_tokens"))
        return "pass" if ok else "fail"

    def close(self):
        # The server, its sockets and its journals share the same private runtime.
        subprocess.run(["fux", "str", self.server, "server", "kill-server"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                       env=self.controller_env)

    def run(self):
        try:
            if self.scope == "auto-controls":
                self.auto_controls()
                return
            # Manual run uses the normal context policy. The reduced policy is
            # installed only before the separate automatic-compaction session.
            (self.scratch / "rig-home/settings.toml").write_text("")
            self.launch()
            if self.startup_only:
                self.send("/quit")
                self.status("startup_complete")
                return
            chosen = actions(self.second_read_file)[:4] if self.scope == "probes" else actions(self.second_read_file)
            for name, text in chosen:
                if self.skip_after_read_refusal(name):
                    continue
                if name == "effort":
                    self.send("/effort medium")
                if name == "neutral":
                    self.send("/effort low")
                self.phase(name, text)
            if self.scope == "probes":
                self.send("/quit")
                return
            if self.include_fork_resume:
                self.send("/fork")
                self.phase("fork", RECALL + SECOND_READ_RECALL)
                # /fork creates an agent in this same session; ShownId is persisted.
                session = self.session
                self.send("/quit")
                self.launch(resume=session)
                self.phase("resume", RECALL + SECOND_READ_RECALL)
            self.send("/quit")
            self.auto_controls()
        finally:
            # Dedicated server only; never a default server or an existing user pane.
            self.close()
            private_json(self.checks_path, self.checks)

    def auto_controls(self):
        (self.scratch / "rig-home/settings.toml").write_text("[compaction]\nkeep_recent = 1024\nkeep_asked = 0\nlimits = { reserve = 970000, summary_tokens = 1024 }\n")
        self.launch()
        self.phase("auto_seed", SEED)
        self.phase("auto_trigger", APPEND)
        self.phase("auto_recall", RECALL + " Also return the comet value from the newest large message.")
        self.phase("auto_rewarm", RECALL + " Also return the comet value.")
        self.send("/quit")
        (self.scratch / "rig-home/settings.toml").write_text("")
        for name in ("fresh_prefix", "fresh_prefix_repeat"):
            self.launch(tools="")
            self.phase(name, SEED)
            self.send("/quit")
