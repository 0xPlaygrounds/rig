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
    def __init__(self, scratch, output, binary, credentials, env, deadline, include_fork_resume=False):
        self.scratch, self.output = Path(scratch), Path(output)
        self.binary, self.credentials, self.env = binary, credentials, env
        self.deadline = deadline
        self.include_fork_resume = include_fork_resume
        self.server = "cloak-acceptance-" + uuid.uuid4().hex
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
        if include_fork_resume:
            self.checks.update(fork="unknown", resume="unknown")
        private_json(self.output / "matrix.json", self.checks)

    @staticmethod
    def fux_command(server, words):
        if not words or (words[0] != "opts" and not re.fullmatch(r"%[0-9]+", words[0])):
            raise ValueError("fux action requires opts or an explicit pane")
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
            raise RuntimeError("dedicated fux command failed; raw screen output intentionally not persisted")
        return result.stdout

    def launch(self, tools="read", resume=None):
        if not self.started:
            remaining = self.deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("matrix elapsed-time cap reached")
            started = subprocess.run(["fux", "str", self.server, "server", "start"],
                                     capture_output=True, text=True, timeout=min(remaining, 10),
                                     env=self.controller_env)
            if started.returncode:
                raise RuntimeError("private fux server could not start")
            self.started = True
        self.launch_count += 1
        self.session = resume or str(time.time_ns())
        env = dict(self.env, TERM="xterm-256color", RIG_SESSION=self.session,
                   XDG_RUNTIME_DIR=self.controller_env["XDG_RUNTIME_DIR"])
        if resume:
            env.update(RIG_START="resumed")
        # env -i prevents fux's own runtime environment from leaking into Rig.
        command = ["env", "-i", *[key + "=" + value for key, value in env.items()], "bwrap", "--die-with-parent",
                   "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev",
                   "--bind", str(self.scratch), str(self.scratch),
                   "--bind", str(self.credentials), str(self.scratch / "rig-home/anthropic-oauth"),
                   "--chdir", str(self.scratch / "workspace"),
                   "--", str(self.binary), "--tools=" + tools]
        words = ["opts", "str", f"cloak-acceptance-{self.launch_count}", "name", "140", "42", "size", "str", str(self.scratch / "workspace"), "cwd", "str", command[0], "program"]
        for argument in command[1:]:
            words += ["str", argument, "arg"]
        answer = self.fux(words + ["run"])
        pane = re.search(r"(?m)^%[0-9]+\b", answer)
        if not pane:
            raise RuntimeError("fux did not identify the newly created pane")
        self.pane = pane.group()

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
                    records.append(json.loads(line))
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
        (self.output / "phase.txt").write_text(name)
        self.send(text)
        rows = []
        while time.time() < self.deadline:
            rows = [r for r in analyze(self.records())["calls"] if r.get("scenario") == name]
            if any(r["classification"] == "fail" for r in rows):
                self.checks[name] = "fail"
                break
            if any(r.get("capture_unknown") for r in rows):
                break
            settled = self.settled()
            # Tool-use terminal is not a final answer. Wait for an end-turn response
            # and its real effect result, not an arbitrary sleep or a token count.
            if rows and rows[-1].get("final_answer") and rows[-1].get("request_id") in settled:
                if name == "auto_trigger" and settled.get(rows[-1].get("request_id")) == "compaction":
                    time.sleep(0.1)
                    continue
                if name == "auto_trigger" and not any(settled.get(r.get("request_id")) == "compaction" for r in rows):
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
                if name in ("manual_recall", "fork", "resume"):
                    original = self.observed.get("seed", {})
                    self.checks[name] = self.combine(self.checks[name], self.identity(last, original, "human_fingerprint"))
                if name == "manual_recall" and last.get("input_chars", 0) >= self.observed.get("neutral", {}).get("input_chars", 0):
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
        private_json(self.output / "matrix.json", self.checks)
        if self.checks[name] != "pass":
            raise RuntimeError("matrix phase did not pass: " + name)
        # Let the terminal's command/event processing settle before a local slash command.
        self.fux([self.pane, "wait"])
        return rows

    @staticmethod
    def combine(*statuses):
        return "fail" if "fail" in statuses else "unknown" if "unknown" in statuses else "pass"

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
        if name not in ("neutral", "manual_compact", "auto_trigger"):
            ok = ok and all(last.get("facts", {}).get(key) for key in FACTS)
        if name in ("warm", "effort", "rewarm", "auto_rewarm", "fresh_prefix_repeat"):
            ok = ok and last.get("usage", {}).get("cache_read_input_tokens", 0) > 0
        if name == "read_success":
            ok = ok and last.get("tool_results", 0) > 0 and last.get("tool_errors", 0) == 0
        if name == "read_error":
            ok = ok and last.get("tool_errors", 0) > 0
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
            # Manual run uses the normal context policy. The reduced policy is
            # installed only before the separate automatic-compaction session.
            (self.scratch / "rig-home/settings.toml").write_text("")
            self.launch()
            for name, text in actions(self.second_read_file):
                if name == "effort":
                    self.send("/effort medium")
                if name == "neutral":
                    self.send("/effort low")
                self.phase(name, text)
            if self.include_fork_resume:
                self.send("/fork")
                self.phase("fork", RECALL + SECOND_READ_RECALL)
                # /fork creates an agent in this same session; ShownId is persisted.
                session = self.session
                self.send("/quit")
                self.launch(resume=session)
                self.phase("resume", RECALL + SECOND_READ_RECALL)
            self.send("/quit")
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
        finally:
            # Dedicated server only; never a default server or an existing user pane.
            self.close()
            private_json(self.output / "matrix.json", self.checks)
