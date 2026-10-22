"""Offline only. Uses fake identities and credentials, never the user's auth store."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zlib

from audit import Evidence, FACTS, SSE, analyze, effect_records, import_effects, report, request_record, markers, private_json
from run import environment, plan
from matrix import Matrix, actions, RECALL, SECOND_READ_RECALL


def stream_bytes():
    events = [
        {"type": "message_start", "message": {"id": "msg-fake", "model": "claude-opus-5-5", "usage": {"input_tokens": 12, "cache_read_input_tokens": 10, "cache_creation_input_tokens": 0}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "é " + " ".join(FACTS.values())}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 8}},
        {"type": "message_stop"},
    ]
    return b"".join(b"event: ignored\r\ndata: " + json.dumps(e, ensure_ascii=False).encode() + b"\r\n\r\n" for e in events)


class AuditTests(unittest.TestCase):
    def test_compression_and_every_byte_boundary(self):
        for encoding, content in (("identity", stream_bytes()), ("gzip", gzip.compress(stream_bytes())), ("deflate", zlib.compress(stream_bytes()))):
            with self.subTest(encoding=encoding):
                parser = SSE(encoding)
                for byte in content:
                    parser.feed(bytes([byte]))
                result = parser.finish()
                self.assertEqual(result["close"], "complete")
                self.assertTrue(result["compression_complete"])
                self.assertTrue(all(result["facts"].values()))
                self.assertEqual(result["usage"]["output_tokens"], 8)

    def test_terminal_then_cancel_is_not_truncation(self):
        parser = SSE()
        parser.feed(stream_bytes())
        self.assertEqual(parser.finish(cancelled=True)["close"], "cancel_after_terminal")
        partial = SSE()
        partial.feed(b'data: {"type":"message_start","message":{}}\n\n')
        self.assertEqual(partial.finish(cancelled=True)["close"], "truncated")

    def test_unknown_compression_rejected(self):
        for encoding in ("br", "zstd", "gzip, deflate"):
            with self.assertRaises(ValueError):
                SSE(encoding)

    def test_malformed_and_bounded_frames(self):
        with self.assertRaises(ValueError):
            SSE().feed(b"data: not-json\n\n")
        with self.assertRaises(ValueError):
            SSE(limit=10).feed(b"x" * 11)
        with self.assertRaises(ValueError):
            SSE("gzip", limit=10).feed(gzip.compress(b"x" * 1000))

    def test_truncated_compression_is_visible(self):
        parser = SSE("gzip")
        parser.feed(gzip.compress(stream_bytes())[:-5])
        self.assertFalse(parser.finish()["compression_complete"])

    def test_clean_eof_requires_complete_compression_and_frame(self):
        compressed = SSE("gzip")
        compressed.feed(gzip.compress(stream_bytes())[:-5])
        pending = SSE()
        pending.feed(stream_bytes() + b'data: {"type":')
        base = {"call": 1, "status": 200, "requested_model_ok": True}
        for parser in (compressed, pending):
            with self.subTest(parser=parser):
                self.assertEqual(analyze([{**base, **parser.finish()}])["classification"], "fail")
                self.assertEqual(analyze([{**base, **parser.finish(cancelled=True)}])["classification"], "pass")

    def test_structural_redaction_before_persistence(self):
        fake = "sk-ant-fake-secret-never-a-real-key"
        with tempfile.TemporaryDirectory() as root:
            evidence = Evidence(root)
            body = {"model": "claude-opus-5-5", "authorization": fake,
                    "metadata": {"user_id": json.dumps({"session_id": "fake-person@example.invalid", "device_id": fake})},
                    "system": [{"text": fake, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                    "messages": [{"content": [{"type": "thinking", "thinking": fake, "signature": fake},
                                               {"text": fake, "cache_control": {"type": "ephemeral", "ttl": fake}}]}],
                    "tools": [{"name": fake, "input_schema": {fake: fake}}]}
            evidence.save(request_record(body, evidence))
            persisted = (Path(root) / "audit.jsonl").read_text()
            self.assertNotIn(fake, persisted)
            self.assertNotIn("fake-person", persisted)
            self.assertNotIn("thinking", persisted)
            self.assertEqual(os.stat(Path(root) / "audit.jsonl").st_mode & 0o777, 0o600)

    def test_pseudonyms_survive_reload(self):
        with tempfile.TemporaryDirectory() as root:
            first = Evidence(root).alias("fake-identity")
            self.assertEqual(first, Evidence(root).alias("fake-identity"))
            self.assertNotEqual(first, Evidence(root).alias("other-identity"))

    def test_plain_and_delta_effect_outcomes(self):
        record = {"id": 1, "purpose": "summary", "outcome": {"Ok": {"outcome": "completion", "provider_request_id": "req-fake", "raw": {"usage": {"input_tokens": 20}}}}}
        wrapped = {"delta": 1, "base": 0, "record": record}
        self.assertEqual(effect_records(record), effect_records(wrapped))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            log = root / "sessions/a/files/effects.jsonl"
            log.parent.mkdir(parents=True)
            log.write_text(json.dumps(wrapped) + "\n")
            evidence = Evidence(root)
            self.assertEqual(import_effects(root / "sessions", evidence), 1)
            saved = json.loads((root / "audit.jsonl").read_text())
            self.assertEqual(saved["request_id"], evidence.alias("req-fake"))
            self.assertEqual(saved["purpose"], "summary")
            self.assertNotIn("req-fake", json.dumps(saved))

    def test_provider_and_local_guard_classification(self):
        base = {"call": 1, "kind": "response", "terminal": True, "successful_stop": True,
                "model_ok": True, "requested_model_ok": True, "status": 200, "close": "cancel_after_terminal", "usage": {"cache_read_input_tokens": 1, "cache_creation_input_tokens": 0}}
        self.assertEqual(analyze([base])["classification"], "pass")
        for override in ({"local_guard": "tool_roster"}, {"status": 500}, {"parse_error": True}, {"model_ok": False}, {"close": "truncated"}, {"successful_stop": False}):
            self.assertEqual(analyze([{**base, **override}])["classification"], "fail")
        self.assertEqual(analyze([{**base, "capture_unknown": "unsupported_encoding", "close": "truncated"}])["classification"], "unknown")
        self.assertEqual(analyze([{**base, "usage": {}}])["classification"], "unknown")
        self.assertEqual(analyze([])["classification"], "unknown")
        self.assertEqual(analyze([{"call": 1, "status": 200}])["classification"], "unknown")

    def test_marker_budget_and_ttl_order(self):
        base = {"call": 1, "terminal": True, "successful_stop": True, "model_ok": True, "requested_model_ok": True, "status": 200, "usage": {"cache_read_input_tokens": 1, "cache_creation_input_tokens": 0}}
        for ttl in (("1h", "5m"), ("5m", "1h"), ("1h",) * 5):
            result = analyze([{**base, "markers": [{"ttl": t, "type_valid": True} for t in ttl]}])
            self.assertEqual(result["classification"], "pass" if ttl == ("1h", "5m") else "fail")

    def test_markers_ignore_schema_input_nested_result_and_null(self):
        marker = {"type": "ephemeral", "ttl": "1h"}
        body = {"tools": [{"cache_control": marker, "input_schema": {"properties": {"cache_control": marker}}}],
                "system": [{"cache_control": marker}, {"cache_control": None}],
                "messages": [{"cache_control": marker, "content": [
                    {"type": "tool_use", "input": {"cache_control": marker}},
                    {"type": "tool_result", "content": [{"cache_control": marker}]},
                    {"type": "text", "cache_control": marker}]}], "cache_control": marker}
        self.assertEqual([m["path"] for m in markers(body)], ["/tools/0/cache_control", "/system/0/cache_control", "/messages/0/content/2/cache_control", "/cache_control"])
        self.assertEqual(markers({"cache_control": None}), [])

    def test_correlates_by_real_identifier_not_order(self):
        result = analyze([{"call": 2, "request_id": "private:" + "b" * 24},
                          {"kind": "effect", "request_id": "private:" + "a" * 24, "purpose": "compaction", "usage": {"input_tokens": 12}, "effect_id": "private:" + "c" * 24},
                          {"call": 1, "request_id": "private:" + "a" * 24}])
        self.assertFalse(result["calls"][0]["effect_correlated"])
        self.assertTrue(result["calls"][1]["effect_correlated"])
        self.assertEqual(result["calls"][1]["effect_purpose"], "compaction")
        self.assertEqual(result["calls"][1]["effect_usage"], {"input_tokens": 12})

    def test_untrusted_analysis_fields_do_not_reach_reports(self):
        secret = "fake-secret-do-not-export"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"call": 1, "headers": {"authorization": secret}, "body": secret,
                      "request_id": secret, "scenario": secret, "usage": {"input_tokens": secret, "authorization": secret},
                      "facts": {"harbor": secret}, "markers": [{"path": secret, "ttl": secret, "authorization": secret}]}
            (root / "audit.jsonl").write_text(json.dumps(record) + "\n")
            (root / "matrix.json").write_text(json.dumps({"seed": secret, secret: secret}))
            report(root)
            for name in ("summary.json", "REPORT.md"):
                output = (root / name).read_text()
                self.assertNotIn(secret, output)
                self.assertNotIn("authorization", output)
                self.assertNotIn("body", output)
                self.assertEqual((root / name).stat().st_mode & 0o777, 0o600)

    def test_analysis_rejects_symlink_roots_and_destinations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("untouched")
            evidence = root / "evidence"
            evidence.mkdir()
            linked_root = root / "linked"
            linked_root.symlink_to(evidence, target_is_directory=True)
            with self.assertRaises(ValueError):
                report(linked_root)
            for name in ("summary.json", "REPORT.md"):
                link = evidence / name
                link.symlink_to(target)
                with self.assertRaises(ValueError):
                    report(evidence)
                with self.assertRaises(ValueError):
                    private_json(link, {"replacement": True})
                self.assertEqual(target.read_text(), "untouched")
                link.unlink()

    def test_missing_live_phase_cannot_pass(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "policy.json").write_text("{}")
            result = report(root)
            self.assertEqual(result["classification"], "unknown")
            self.assertTrue((root / "REPORT.md").exists())


class RunnerTests(unittest.TestCase):
    def test_plan_is_offline_and_honest(self):
        self.assertEqual(plan()["automated"], ["cold_repeat"])
        self.assertEqual(len(plan()["manual"]), 6)
        command = [sys.executable, str(Path(__file__).with_name("run.py"))]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["model"], "claude-cloak/claude-opus-5-5")

    def test_live_needs_both_consent_and_caps(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("run.py")), "--live"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --confirm-stored-credentials", result.stderr)

    def test_child_environment_does_not_inherit_credentials(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-secret", "RIG_SESSION": "user-session", "RUST_LOG": "trace", "ALL_PROXY": "user-proxy"}):
            env = environment(Path("/isolated"), Path("/isolated/ca.pem"), 1234)
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("RIG_SESSION", env)
        self.assertNotIn("ALL_PROXY", env)
        self.assertEqual(env["RUST_LOG"], "off")
        self.assertEqual(env["HOME"], "/isolated/home")


class ProxyTests(unittest.TestCase):
    def test_request_and_output_caps_survive_addon_reload(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "policy.json").write_text(json.dumps({"requests": 1, "output_tokens": 100, "deadline": 99999999999}))
            (root / "phase.txt").write_text("fresh_prefix")
            (root / "client.txt").write_text("rig")
            fake_http = SimpleNamespace(Response=SimpleNamespace(make=lambda *args: args))
            with patch.dict(os.environ, {"CLOAK_ACCEPTANCE_ROOT": str(root)}), patch.dict(sys.modules, {"mitmproxy": SimpleNamespace(http=fake_http)}):
                spec = importlib.util.spec_from_file_location("acceptance_proxy_test", Path(__file__).with_name("proxy.py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                def flow(max_tokens, text=""):
                    body = json.dumps({"model": "claude-opus-5-5", "max_tokens": max_tokens, "tools": [], "messages": [{"role": "user", "content": [{"type": "tool_result", "content": text}]}]}).encode()
                    return SimpleNamespace(request=SimpleNamespace(host="api.anthropic.com", path="/v1/messages", get_content=lambda: body, headers={}), metadata={}, response=None)
                first = flow(100)
                module.addons[0].request(first)
                self.assertIsNone(first.response)
                second = flow(100)
                module.Audit().request(second)
                self.assertEqual(second.metadata["acceptance"]["local_guard"], "request_cap")
                (root / "policy.json").write_text(json.dumps({"requests": 10, "output_tokens": 100, "deadline": 99999999999}))
                third = flow(101)
                module.Audit().request(third)
                self.assertEqual(third.metadata["acceptance"]["local_guard"], "output_cap")
                self.assertEqual(first.request.headers["accept-encoding"], "gzip, deflate, identity")
                fake_token = "sk-ant-" + "synthetic_12345678"
                for encoded in (False, True):
                    leaked = flow(100, fake_token)
                    if encoded:
                        raw = leaked.request.get_content().replace(b"sk-ant-", b"sk-\\u0061nt-")
                        leaked.request.get_content = lambda: raw
                    module.Audit().request(leaked)
                    self.assertIsNotNone(leaked.response)
                    self.assertEqual(leaked.metadata["acceptance"]["local_guard"], "credential_like_input")
                authenticated = flow(100)
                authenticated.request.headers["authorization"] = "Bearer " + fake_token
                module.Audit().request(authenticated)
                self.assertIsNone(authenticated.response)
                self.assertNotIn(fake_token, (root / "audit.jsonl").read_text())


class MatrixTests(unittest.TestCase):
    def test_recall_does_not_supply_values_and_neutral_precedes_summary(self):
        self.assertTrue(all(value not in RECALL for value in FACTS.values()))
        names = [name for name, _ in actions()]
        self.assertEqual(names[names.index("manual_compact") - 1], "neutral")

    def test_full_action_sequence_without_processes_or_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "rig-home").mkdir()
            driver = Matrix(root, root / "out", "/fake/rig", "/fake/auth", {"PATH": os.defpath, "HOME": str(root)}, float("inf"), True)
            driver.session = "synthetic-session"
            phases = []
            with patch.object(driver, "launch") as launch, patch.object(driver, "send") as send, patch.object(driver, "phase", side_effect=lambda name, text: phases.append(name)), patch("matrix.subprocess.run") as run:
                driver.run()
            self.assertEqual(phases, [name for name, _ in actions()] + ["fork", "resume", "auto_seed", "auto_trigger", "auto_recall", "auto_rewarm", "fresh_prefix", "fresh_prefix_repeat"])
            self.assertEqual(launch.call_count, 5)
            send.assert_any_call("/effort medium")
            send.assert_any_call("/fork")
            self.assertIn(driver.server, run.call_args.args[0])
            self.assertIn("kill-server", run.call_args.args[0])

    def test_checks_do_not_mistake_summary_or_tool_error_for_success(self):
        row = {"classification": "pass", "facts": dict.fromkeys(FACTS, True), "usage": {"cache_read_input_tokens": 1, "cache_creation_input_tokens": 0}, "request_id": "alias", "effort": "medium", "tool_results": 1, "tool_errors": 0, "tool_invocations": 1, "tool_target_match": True}
        self.assertEqual(Matrix.check("read_success", [row], {}), "pass")
        self.assertEqual(Matrix.check("read_error", [row], {}), "fail")
        self.assertEqual(Matrix.check("manual_compact", [row], {}), "unknown")
        self.assertEqual(Matrix.check("manual_compact", [row], {"alias": "compaction"}), "pass")
        self.assertEqual(Matrix.check("auto_recall", [row], {}), "fail")

    def test_compaction_recall_requires_second_read_outcome(self):
        filename = "probe-86701c49.txt"
        parser = SSE(second_read_file=filename)
        parser.feed(stream_bytes())
        row = {**parser.finish(), "classification": "pass"}
        # All five facts survive, but a dropped file outcome must not pass.
        self.assertTrue(all(row["facts"].values()))
        self.assertEqual(Matrix.check("manual_recall", [row], {}), "fail")
        parser.consume({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "\n" + filename + " | failure | 1"}})
        self.assertEqual(Matrix.check("manual_recall", [{**parser.finish(), "classification": "pass"}], {}), "pass")
        self.assertNotIn(filename, SECOND_READ_RECALL)
        self.assertNotIn("failure", SECOND_READ_RECALL)
        self.assertNotIn("1", SECOND_READ_RECALL)
        self.assertIn(filename, dict(actions(filename))["read_error"])

    def test_consecutive_launches_have_unique_names_and_private_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            driver = Matrix(root, root / "out", "/fake/rig", "/fake/auth", {"PATH": os.defpath, "HOME": str(root)}, float("inf"))
            with patch("matrix.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="%2 reading")) as run:
                driver.launch()
                driver.launch()
                driver.close()
            commands = [call.args[0] for call in run.call_args_list]
            names = [command[command.index("name") - 1] for command in commands if command[-1] == "run"]
            self.assertEqual(sum(command[-1] == "start" for command in commands), 1)
            self.assertEqual(names, ["cloak-acceptance-1", "cloak-acceptance-2"])
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["env"]["XDG_RUNTIME_DIR"], str(root / "runtime"))
                self.assertIn(driver.server, call.args[0])
            self.assertEqual((root / "runtime").stat().st_mode & 0o777, 0o700)

    @unittest.skipUnless(os.environ.get("CLOAK_ACCEPTANCE_PROCESS_SMOKE") == "1" and shutil.which("fux") and shutil.which("bwrap") and Path("/dev/shm").is_dir(), "optional fake-process smoke requires explicit opt-in, fux, bwrap and /dev/shm")
    def test_fake_agent_two_launch_send_quit_cycles(self):
        probe = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--", "/bin/true"], capture_output=True, timeout=10)
        if probe.returncode:
            self.skipTest("bubblewrap user namespaces unavailable")
        with tempfile.TemporaryDirectory(prefix="cloak-fake-agent-") as binary_dir, tempfile.TemporaryDirectory(prefix="cloak-fux-smoke-", dir="/dev/shm") as directory:
            root = Path(directory)
            for folder in ("home", "workspace", "rig-home/anthropic-oauth", "fake-auth"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            binary = Path(binary_dir) / "fake-rig"
            binary.write_text("#!/bin/sh\nwhile IFS= read -r line; do\n  printf '%s\\n' \"$line\" >> seen\n  case \"$line\" in /quit) exit 0;; esac\ndone\n")
            binary.chmod(0o700)
            env = environment(root, root / "unused-ca", 1)
            driver = Matrix(root, root / "out", binary, root / "fake-auth", env, time.time() + 30)
            try:
                for index in (1, 2):
                    driver.launch()
                    driver.send(f"synthetic-smoke-{index}")
                    driver.send("/quit")
                    driver.fux([driver.pane, "wait-exit", "wait"])
                self.assertEqual((root / "workspace/seen").read_text().splitlines(), ["synthetic-smoke-1", "/quit", "synthetic-smoke-2", "/quit"])
                self.assertEqual(driver.controller_env["XDG_RUNTIME_DIR"], str(root / "runtime"))
            finally:
                driver.close()

    def test_fux_timeout_has_a_stack_receiver(self):
        for words in (["opts", "str", "/usr/bin/true", "program", "run"], ["%2", "wait"], ["%2", "str", "hello", "paste", "send-keys"]):
            command = Matrix.fux_command("private-test", words)
            self.assertEqual(command[:7], ["fux", "str", "private-test", "server", words[0], "1", "timeout"])
            self.assertEqual(command[7:], words[1:])
        with self.assertRaises(ValueError):
            Matrix.fux_command("private-test", ["1", "timeout", "wait"])

    @unittest.skipUnless(shutil.which("fux"), "optional native fux grammar validator unavailable")
    def test_native_fux_checks_actions_without_running_them(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "acceptance.fux"
            for words in (["opts", "str", "/usr/bin/true", "program", "run"], ["%2", "wait"], ["%2", "str", "hello", "paste", "str", "Enter", "key", "send-keys"]):
                # These fixed grammar probes have no whitespace-bearing tokens.
                command = Matrix.fux_command("private-acceptance-syntax-only", words)
                path.write_text(" ".join(command[1:]) + " drop\n")
                result = subprocess.run(["fux", "str", str(path), "check"], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_fux_busy_status_is_accepted_not_a_failed_action(self):
        with tempfile.TemporaryDirectory() as root:
            driver = Matrix(Path(root), Path(root), "/fake/rig", "/fake/auth", {"PATH": os.defpath, "HOME": root}, float("inf"))
            with patch("matrix.subprocess.run", return_value=SimpleNamespace(returncode=124, stdout="%2 busy")):
                self.assertEqual(driver.fux(["%2", "wait"]), "%2 busy")
            with patch("matrix.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="")):
                with self.assertRaises(RuntimeError):
                    driver.fux(["%2", "wait"])

    def test_missing_identity_and_usage_are_unknown(self):
        self.assertEqual(Matrix.identity({}, {}, "session"), "unknown")
        self.assertEqual(Matrix.identity({"session": "a"}, {}, "session"), "unknown")
        self.assertEqual(Matrix.identity({"session": "a"}, {"session": "a"}, "session"), "pass")
        self.assertEqual(Matrix.identity({"session": "a"}, {"session": "a"}, "session", different=True), "fail")
        self.assertEqual(Matrix.check("warm", [{"classification": "pass"}], {}), "unknown")

    def test_time_cap_stops_before_invoking_fux(self):
        with tempfile.TemporaryDirectory() as root:
            driver = Matrix(Path(root), Path(root), "/fake/rig", "/fake/auth", {}, 0)
            with patch("matrix.subprocess.run") as run:
                with self.assertRaises(TimeoutError):
                    driver.fux(["ls"])
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
