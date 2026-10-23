"""Offline regressions for budget enforcement and the two TUI adapters."""
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
import time
import unittest
import shutil
from types import SimpleNamespace
from unittest.mock import patch

from audit import Evidence, SSE, analyze, report, FACTS, structural_record, second_read_memory, tool_result_detail, fallback_record
from comparison import ClaudeMatrix, compare_profiles, profile_equal, access_token, child, validate_child
from ledger import Ledger
from matrix import Matrix, actions, SEED, APPEND
from preflight import REQUIRED, ROOT_TTL_TEST, build_record, verify_build, verify_root_ttl, private_capture, private_probe_metadata, source_identity
from wire_profile import checksum_record, shape, CORRELATION_HEADERS

LINUX_TMPFS = sys.platform.startswith("linux") and Path("/dev/shm").is_dir()


class ComparisonTests(unittest.TestCase):
    def test_stop_details_export_closed_categories_without_explanation_or_unknown_values(self):
        secret = "sk-ant-synthetic-private-explanation"
        for category, expected in (("cyber", "cyber"), ("bio", "bio"), (secret, "other"), (None, "absent")):
            with self.subTest(category=expected):
                parser = SSE()
                parser.consume({"type": "message_delta", "delta": {"stop_reason": "refusal", "stop_details": {
                    "type": "refusal", "category": category, "explanation": secret}}})
                safe = structural_record(parser.finish())
                self.assertTrue(safe["stop_details_present"])
                self.assertEqual(safe["stop_details_type"], "refusal")
                self.assertEqual(safe["refusal_category"], expected)
                self.assertTrue(safe["refusal_explanation_present"])
                self.assertFalse(safe["successful_stop"])
                self.assertNotIn(secret, json.dumps(safe))
        parser = SSE()
        parser.consume({"type": "message_start", "message": {"stop_details": {"type": secret, "category": secret, "explanation": secret}}})
        self.assertEqual(structural_record(parser.finish())["stop_details_type"], "other")
        self.assertNotIn(secret, json.dumps(structural_record(parser.finish())))
        absent = structural_record(SSE().finish())
        self.assertFalse(absent["stop_details_present"])
        self.assertEqual(absent["refusal_category"], "absent")
        self.assertNotIn(secret, json.dumps(structural_record({"stop_details_type": secret, "refusal_category": secret, "explanation": secret})))

    @unittest.skipUnless(LINUX_TMPFS, "private metadata test requires Linux and /dev/shm")
    def test_previous_request_metadata_stays_private_and_rejects_arbitrary_inputs(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            path = root / "metadata.json"
            path.touch(mode=0o600)
            valid = {"previous_request_id": "req_synthetic-test_1"}
            path.write_text(json.dumps(valid))
            before = path.read_bytes()
            self.assertEqual(private_probe_metadata(path), path)
            self.assertEqual(path.read_bytes(), before)
            for value in ({}, {**valid, "refresh_token": "sk-ant-synthetic-private"},
                          {"previous_request_id": "sk-ant-synthetic-private"}, {"previous_request_id": "req_"},
                          {"previous_request_id": "req_" + "a" * 37}, {"previous_request_id": "req_newline\n"},
                          {"previous_request_id": 1}, []):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError) as error:
                    private_probe_metadata(path)
                self.assertNotIn("sk-ant-synthetic-private", str(error.exception))
            path.write_bytes(b"x" * 4097)
            with self.assertRaises(ValueError):
                private_probe_metadata(path)
            path.write_text(json.dumps(valid))
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                private_probe_metadata(path)

    def test_collected_failure_journal_forces_failed_report_without_exporting_extra_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "matrix.json").write_text(json.dumps({"seed": "pass", "warm": "pass"}))
            row = {"kind": "response", "call": 1, "status": 200, "requested_model_ok": True, "model_ok": True,
                   "terminal": True, "successful_stop": True, "usage": {"cache_read_input_tokens": 1, "cache_creation_input_tokens": 0}}
            (root / "audit.jsonl").write_text(json.dumps(row) + "\n")
            self.assertEqual(report(root)["classification"], "pass")
            event = {"client": "rig", "from_phase": "seed", "next_phase": "warm", "cause": "semantic_assertion",
                     "failed_check_preserved": True, "retry": False, "effect_correlated": True, "tui_ready": True,
                     "raw_response": "private-content"}
            (root / "failure-continuations.jsonl").write_text(json.dumps(event) + "\n")
            result = report(root)
            self.assertEqual(result["classification"], "fail")
            self.assertEqual(result["failure_continuations"][0]["next_phase"], "warm")
            self.assertNotIn("private-content", json.dumps(result))
            (root / "failure-continuations.jsonl").write_text(json.dumps({**event, "next_phase": "private-content"}) + "\n")
            self.assertNotIn("private-content", json.dumps(report(root)))

    def test_optional_profile_fields_missing_from_old_capture_are_unknown_without_backfill(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for client in ("rig", "claude"):
                directory = root / client
                directory.mkdir()
                record = {"client": client, "role": "primary", "scenario": "seed", "call_role": "prompt"}
                if client == "rig":
                    record.update(requested_fallback_models=[], fallback_shape_valid=True,
                                  correlation_headers={"x-client-request-id": {"present": False}}, extra_header_count=0)
                (directory / "profiles.jsonl").write_text(json.dumps(record) + "\n")
                inputs.append(directory)
            before = [(path / "profiles.jsonl").read_bytes() for path in inputs]
            row = compare_profiles(root, inputs)["pairs"][0]
            for key in ("requested_fallback_models", "fallback_shape_valid", "correlation_header_structure", "extra_header_count", "session_header_matches_metadata"):
                self.assertIsNone(row["equal"][key])
            self.assertEqual(before, [(path / "profiles.jsonl").read_bytes() for path in inputs])
        old = {"messages": [{"blocks": [{"type": "tool_result", "is_error": False}]}]}
        new = {"messages": [{"blocks": [{"type": "tool_result", "is_error": False, "is_error_present": False}]}]}
        self.assertIsNone(profile_equal(old, new, "messages"))
        explicit_false = {"messages": [{"blocks": [{"type": "tool_result", "is_error": False, "is_error_present": True}]}]}
        self.assertFalse(profile_equal(new, explicit_false, "messages"))
        self.assertFalse(profile_equal(new, {"messages": [{"blocks": [{"type": "tool_result", "is_error": True, "is_error_present": True}]}]}, "messages"))
    def test_auto_controls_scope_reuses_the_unchanged_isolated_tail_and_keeps_other_checks_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rig-home").mkdir()
            driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 30, scope="auto-controls")
            phases = []
            settings = []
            def phase(name, text):
                phases.append((name, text))
                settings.append((root / "rig-home/settings.toml").read_text())
            with patch.object(driver, "launch") as launch, patch.object(driver, "phase", side_effect=phase), patch.object(driver, "send") as send, patch.object(driver, "close"):
                driver.run()
            self.assertEqual([name for name, _ in phases], ["auto_seed", "auto_trigger", "auto_recall", "auto_rewarm", "fresh_prefix", "fresh_prefix_repeat"])
            self.assertEqual(phases[0][1], SEED)
            self.assertEqual(phases[1][1], APPEND)
            self.assertEqual(phases[-1][1], SEED)
            self.assertIn("reserve = 970000", settings[0])
            self.assertEqual(settings[-1], "")
            self.assertEqual(launch.call_count, 3)
            self.assertEqual(launch.call_args.kwargs, {"tools": ""})
            self.assertEqual(send.call_count, 3)
            for name, _ in actions():
                self.assertEqual(driver.checks[name], "unknown")
            self.assertEqual(driver.checks["fork"], "unknown")
            self.assertEqual(driver.checks["resume"], "unknown")

    @staticmethod
    def read_refusal_rows():
        shared = {"kind": "response", "scenario": "read_error", "status": 200, "terminal": True, "message_stop": True,
                  "close": "complete", "compression_complete": True, "model_ok": True, "requested_model_ok": True,
                  "tools_read_only": True, "fallback_shape_valid": True, "requested_fallback_models": [],
                  "markers_valid": True, "tool_target_match": True,
                  "usage": {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 1}}
        return [{**shared, "classification": "pass", "stop_reason": "tool_use", "request_id": "first", "tool_invocations": 1},
                {**shared, "classification": "fail", "stop_reason": "refusal", "request_id": "last", "tool_invocations": 0, "tool_results": 2, "tool_errors": 1}]

    def test_read_refusal_continuation_rejects_every_transport_model_and_dispatch_gap(self):
        rows = self.read_refusal_rows()
        previous = {"tool_results": 1, "tool_errors": 0}
        self.assertTrue(Matrix.read_refusal_wire(rows, previous))
        changes = {"status": 400, "kind": "headers", "message_stop": False, "terminal": False, "compression_complete": False,
                   "close": "truncated", "model_ok": False, "requested_model_ok": False, "tools_read_only": False,
                   "fallback_shape_valid": False, "requested_fallback_models": ["claude-opus-4-8"], "markers_valid": False,
                   "tool_target_match": False, "request_id": None, "pending_frame": True, "parse_error": True,
                   "provider_error": True, "transport_error": True, "local_guard": "auxiliary_model", "capture_unknown": "unsupported_encoding",
                   "stop_reason": "max_tokens", "tool_invocations": 1, "tool_results": 3, "tool_errors": 0, "usage": {}}
        for key, value in changes.items():
            with self.subTest(key=key):
                changed = [dict(row) for row in rows]
                changed[-1][key] = value
                self.assertFalse(Matrix.read_refusal_wire(changed, previous))
        self.assertFalse(Matrix.read_refusal_wire(rows + [dict(rows[-1])], previous))
        self.assertFalse(Matrix.read_refusal_wire([{**rows[0], "tool_invocations": 2}, rows[-1]], previous))

    def test_explicit_continuation_waits_for_effects_and_advances_without_erasing_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, continue_after_read_refusal=True)
            driver.observed["read_success"] = {"tool_results": 1, "tool_errors": 0}
            calls = self.read_refusal_rows()
            settled_count = 0
            def settled():
                nonlocal settled_count
                settled_count += 1
                return {} if settled_count == 1 else {"first": "completion", "last": "interactive"}
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": calls}), patch.object(driver, "settled", side_effect=settled), patch.object(driver, "await_input_ready") as ready, patch.object(driver, "fux"), patch("matrix.time.sleep"):
                driver.phase("read_error", "failed read prompt")
                ready.assert_called_once()
            self.assertEqual(driver.checks["read_error"], "fail")
            self.assertEqual(driver.observed["read_error"]["stop_reason"], "refusal")
            row = {"scenario": "effort", "classification": "pass", "final_answer": True, "request_id": "next", "effort": "medium", "facts": dict.fromkeys(FACTS, True), "usage": {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 1}}
            with patch.object(driver, "send") as send, patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", return_value={"next": "completion"}), patch.object(driver, "fux"):
                driver.phase("effort", "different recall prompt")
                send.assert_called_once_with("different recall prompt")
            self.assertEqual(driver.checks["effort"], "pass")
            result = report(driver.output)
            self.assertEqual(result["classification"], "fail")
            self.assertFalse(result["continuations"][0]["retry"])
            self.assertNotIn("failed read prompt", json.dumps(result))
            driver.pending_read_refusal = "failed read prompt"
            with self.assertRaises(RuntimeError), patch.object(driver, "send") as send:
                driver.phase("read_error", "failed read prompt")
            send.assert_not_called()
            driver.pending_read_refusal = None
            driver.continue_after_read_refusal = False
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": calls}), patch.object(driver, "await_input_ready") as ready:
                with self.assertRaises(RuntimeError):
                    driver.phase("read_error", "failed read prompt")
                ready.assert_not_called()

    def test_fallback_shape_and_models_are_checked_before_forwarding(self):
        for fallback, expected, valid in (([{"model": "claude-opus-5-5"}], ["claude-opus-5-5"], True), ([{"model": "claude-opus-4-8"}], ["claude-opus-4-8"], True), ([{"model": "private-model"}], ["unexpected"], True), ("default", ["unexpected"], False), ({"model": "claude-opus-5-5"}, ["unexpected"], False), ([{"model": "claude-opus-5-5", "unexpected": True}], ["unexpected"], False)):
            with self.subTest(fallback=fallback), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                record = fallback_record({"fallbacks": fallback})
                self.assertEqual(record["requested_fallback_models"], expected)
                self.assertEqual(record["fallback_shape_valid"], valid)
                self.assertNotIn("private-model", json.dumps(structural_record(record)))
                (root / "policy.json").write_text(json.dumps({"matrix": True, "requests": 24, "output_tokens": 128000, "deadline": time.time() + 60}))
                (root / "phase.txt").write_text("seed")
                (root / "client.txt").write_text("rig")
                fake_http = SimpleNamespace(Response=SimpleNamespace(make=lambda *args: args))
                with patch.dict(os.environ, {"CLOAK_ACCEPTANCE_ROOT": str(root)}), patch.dict(__import__("sys").modules, {"mitmproxy": SimpleNamespace(http=fake_http)}):
                    spec = importlib.util.spec_from_file_location("fallback_proxy_test", Path(__file__).with_name("proxy.py"))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    body = {"model": "claude-opus-5-5", "max_tokens": 128000, "messages": [], "tools": [], "fallbacks": fallback}
                    raw = json.dumps(body).encode()
                    flow = SimpleNamespace(request=SimpleNamespace(host="api.anthropic.com", path="/v1/messages", headers={}, get_content=lambda: raw), metadata={}, response=None)
                    module.addons[0].request(flow)
                    if valid and expected == ["claude-opus-5-5"]:
                        self.assertIsNone(flow.response)
                    else:
                        self.assertEqual(flow.response[0], 400)
                        self.assertEqual(flow.metadata["acceptance"]["local_guard"], "fallback_model")
    def test_native_correlation_headers_export_only_presence_roles_and_pseudonyms(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Evidence(temporary)
            secret = "opaque-private-session-value"
            headers = {name: secret for name in CORRELATION_HEADERS}
            headers["x-claude-code-request-class"] = "main"
            headers["x-claude-code-agent-type"] = "custom"
            headers["X-Claude-Code-Session-Id"] = headers.pop("x-claude-code-session-id")
            headers.update({"authorization": secret, "cookie": secret, "x-api-key": secret, "content-length": "123", "unknown-private-name": secret})
            body = {"metadata": {"user_id": json.dumps({"session_id": secret})}, "messages": []}
            result = shape(body, headers, evidence)
            serialized = json.dumps(result)
            self.assertNotIn(secret, serialized)
            self.assertNotIn("unknown-private-name", serialized)
            self.assertNotIn("authorization", result["correlation_headers"])
            self.assertTrue(result["session_header_matches_metadata"])
            self.assertEqual(result["extra_header_count"], 1)
            self.assertEqual(set(result["correlation_headers"]), set(CORRELATION_HEADERS))
            self.assertTrue(all(item["present"] and item["alias"].startswith("private:") for item in result["correlation_headers"].values()))
            self.assertEqual(result["correlation_headers"]["x-claude-code-request-class"]["role"], "main")
            headers["x-claude-code-request-class"] = secret
            result = shape(body, headers, evidence)
            self.assertEqual(result["correlation_headers"]["x-claude-code-request-class"]["role"], "other")
            body["metadata"]["user_id"] = json.dumps({"session_id": "different"})
            self.assertFalse(shape(body, headers, evidence)["session_header_matches_metadata"])
            absent = shape({"messages": []}, {}, evidence)
            self.assertTrue(all(not item["present"] for item in absent["correlation_headers"].values()))
            self.assertNotIn("session_header_matches_metadata", absent)

    def test_tool_result_profile_distinguishes_missing_false_and_true_error_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            for field, present, error in (({}, False, False), ({"is_error": False}, True, False), ({"is_error": True}, True, True)):
                profile = shape({"messages": [{"role": "user", "content": [{"type": "tool_result", "content": "synthetic", **field}]}]}, {}, Evidence(temporary))
                block = profile["messages"][0]["blocks"][0]
                self.assertEqual(block["is_error_present"], present)
                self.assertEqual(block["is_error"], error)

    def test_cross_run_header_comparison_ignores_per_run_pseudonym_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for client in ("rig", "claude"):
                directory = root / client
                directory.mkdir()
                profile = shape({"messages": []}, {"x-client-request-id": "same-opaque-value"}, Evidence(directory))
                record = {"client": client, "role": "primary", "scenario": "seed", **profile}
                (directory / "profiles.jsonl").write_text(json.dumps(record) + "\n")
                inputs.append(directory)
            row = compare_profiles(root, inputs)["pairs"][0]
            self.assertTrue(row["equal"]["correlation_header_structure"])
            self.assertTrue(row["equal"]["extra_header_count"])
            path = inputs[1] / "profiles.jsonl"
            native = json.loads(path.read_text())
            native["correlation_headers"]["x-claude-code-request-class"] = {"present": True, "role": "main"}
            path.write_text(json.dumps(native) + "\n")
            self.assertFalse(compare_profiles(root, inputs)["pairs"][0]["equal"]["correlation_header_structure"])
            del native["correlation_headers"]["x-claude-code-request-class"]
            path.write_text(json.dumps(native) + "\n")
            self.assertIsNone(compare_profiles(root, inputs)["pairs"][0]["equal"]["correlation_header_structure"])

    def test_authorized_aggregate_extension_preserves_counts_time_and_legacy_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            started = time.time()
            legacy = {"schema": 1, "attempts": 47, "live_seconds": 238.0, "active": "old", "runs": {"old": {"attempts": 1, "started": started, "deadline": started + 60}}}
            path.write_text(json.dumps(legacy))
            ledger = Ledger(path)
            self.assertEqual(ledger.attempt("old", 24)[2], "aggregate_cap")
            ledger.finish("old")
            before = json.loads(path.read_text())
            with self.assertRaises(ValueError):
                ledger.begin("default-exhausted", 60)
            ledger.begin("extended", 60, 96)
            state = json.loads(path.read_text())
            self.assertEqual(state["attempts"], before["attempts"])
            self.assertEqual(state["live_seconds"], before["live_seconds"])
            self.assertEqual(state["runs"]["old"], before["runs"]["old"])
            self.assertEqual(state["aggregate_requests"], 96)
            self.assertEqual(state["runs"]["extended"]["aggregate_requests"], 96)
            self.assertIsNone(Ledger(path).attempt("extended", 24)[2])
            ledger.finish("extended")
            with self.assertRaises(ValueError):
                ledger.begin("default-still48", 60)
            with self.assertRaises(ValueError):
                ledger.begin("unbounded", 60, 1000)

    def test_aggregate_96_reserves_the_final_slot_and_keeps_per_run_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Ledger(Path(temporary) / "ledger.json")
            ledger.begin("one", 60, 96)
            ledger.update(lambda state: state.update(attempts=95))
            self.assertEqual(ledger.attempt("one", 24)[2], "aggregate_cap")
            ledger.finish("one")
            other = Ledger(Path(temporary) / "other.json")
            other.begin("two", 60, 96)
            for _ in range(23):
                self.assertIsNone(other.attempt("two", 1000)[2])
            self.assertEqual(other.attempt("two", 1000)[2], "request_cap")
            other.finish("two")

    def test_authorized_96_to_200_extension_preserves_all_spent_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            old = {"attempts": 23, "started": 1.0, "deadline": 61.0, "elapsed": 60.0, "aggregate_requests": 96}
            state = {"schema": 1, "aggregate_requests": 96, "attempts": 73, "live_seconds": 701.735,
                     "active": None, "runs": {"old": old}}
            path.write_text(json.dumps(state))
            ledger = Ledger(path)
            ledger.begin("raised", 60, 200)
            raised = json.loads(path.read_text())
            self.assertEqual(raised["attempts"], 73)
            self.assertEqual(raised["live_seconds"], 701.735)
            self.assertEqual(raised["runs"]["old"], old)
            self.assertEqual(raised["aggregate_requests"], 200)
            for _ in range(23):
                self.assertIsNone(ledger.attempt("raised", 1000)[2])
            self.assertEqual(ledger.attempt("raised", 1000)[2], "request_cap")
            ledger.finish("raised")
            ledger.begin("last", 60, 200)
            ledger.update(lambda value: value.update(attempts=199))
            self.assertEqual(ledger.attempt("last", 24)[2], "aggregate_cap")
            ledger.finish("last")

    def test_claude_child_matches_rig_access_and_account_without_refresh_or_host_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("rig", "auth/.claude", "scratch/home", "scratch/workspace"):
                (root / name).mkdir(parents=True, exist_ok=True)
            original = root / "auth/.claude.json"
            original.write_text(json.dumps({"machineID": "own-machine", "userID": "own-device", "oauthAccount": {"accountUuid": "other-account", "organizationUuid": "other-org", "emailAddress": "other@example.test"}}))
            cc_credentials = root / "auth/.claude/.credentials.json"
            cc_credentials.write_text(json.dumps({"claudeAiOauth": {"accessToken": "other-access", "refreshToken": "never-copy-cc-refresh", "expiresAt": (time.time() + 3600) * 1000}}))
            rig_credentials = root / "rig/credentials.json"
            rig_credentials.write_text(json.dumps({"access_token": "fake-rig-access", "refresh_token": "never-copy-rig-refresh", "expires_at": time.time() + 3600, "account_id": "rig-account", "org_id": "rig-org"}))
            snapshots = {path: path.read_bytes() for path in (original, cc_credentials, rig_credentials)}
            certificate = root / "certificate"
            certificate.touch()
            binary = root / "binary"
            binary.touch()
            spec = {"auth_mode": "rig-access-token", "auth_home": str(root / "auth"), "rig_credentials": str(root / "rig"), "scratch": str(root / "scratch"), "binary": str(binary), "env": {"SSL_CERT_FILE": str(certificate)}, "deadline": time.time() + 60, "diagnostics": str(root / "diagnostics.json"), "tools": "Read", "resume": False, "session": "synthetic-session"}
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            self.assertEqual(validate_child(path), 0)
            diagnostics = json.loads((root / "diagnostics.json").read_text())
            self.assertFalse(diagnostics["stored_account_matches_rig"])
            self.assertTrue(diagnostics["selected_account_matches_rig"])
            with patch("comparison.os.execvpe") as launch:
                child(path)
                self.assertEqual(launch.call_args.args[2]["CLAUDE_CODE_OAUTH_TOKEN"], "fake-rig-access")
            private_config = json.loads((root / "scratch/home/.claude.json").read_text())
            self.assertEqual(private_config["oauthAccount"], {"accountUuid": "rig-account", "organizationUuid": "rig-org"})
            self.assertEqual(private_config["machineID"], "own-machine")
            self.assertEqual(private_config["userID"], "own-device")
            self.assertNotIn("other@example.test", json.dumps(private_config))
            self.assertNotIn("refresh", json.dumps(private_config))
            self.assertFalse((root / "scratch/home/.claude/.credentials.json").exists())
            for stored, snapshot in snapshots.items():
                self.assertEqual(stored.read_bytes(), snapshot)
            for expiry in (time.time(), spec["deadline"] + 59, float("nan")):
                value = json.loads(rig_credentials.read_text())
                value["expires_at"] = expiry
                rig_credentials.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError, "expired"):
                    access_token(spec)

    @unittest.skipUnless(LINUX_TMPFS, "private capture test requires Linux and /dev/shm")
    def test_private_capture_requires_owned_tmpfs_without_symlinks_or_public_files(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temporary:
            root = Path(temporary)
            raw = root / "request.json"
            raw.touch(mode=0o600)
            self.assertEqual(private_capture(raw), raw)
            raw.chmod(0o644)
            with self.assertRaises(ValueError):
                private_capture(raw)
            raw.chmod(0o600)
            link = root / "link.json"
            link.symlink_to(raw)
            with self.assertRaises(ValueError):
                private_capture(link)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "request.json"
            raw.touch(mode=0o600)
            with self.assertRaises(ValueError):
                private_capture(raw)

    def test_two_call_tool_result_probe_is_diagnostic_and_preserves_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "policy.json").write_text(json.dumps({"tool_result_probe": True}))
            (root / "tool-result-probe.json").write_text(json.dumps({"exit_code": 0, "timed_out": False}))
            detail = {"baseline_byte_equal": True, "variant_only_content_and_cch": True, "advertised_max_tokens": 128000, "attempts": 2}
            (root / "tool-result-detail.json").write_text(json.dumps(detail))
            row = {"kind": "response", "scenario": "tool_result_probe", "status": 200, "requested_model_ok": True, "model_ok": True, "terminal": True, "message_stop": True, "compression_complete": True, "close": "complete", "request_id": "private:" + "a" * 24, "tools_read_only": True, "usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}
            records = [{**row, "call": 1, "stop_reason": "refusal", "successful_stop": False}, {**row, "call": 2, "stop_reason": "end_turn", "successful_stop": True}]
            (root / "audit.jsonl").write_text("\n".join(json.dumps(value) for value in records))
            result = report(root)
            self.assertTrue(result["diagnostic"]["complete"])
            self.assertFalse(result["diagnostic"]["acceptance"])
            self.assertEqual(result["classification"], "unknown")
            self.assertEqual(result["calls"][0]["classification"], "fail")
            self.assertEqual(result["diagnostic"]["controls"]["variant"], "tool_result_string")
            declared = {"baseline_byte_equal": True, "variant_only_declared_field_and_cch": True, "headers_equal": True, "variant": "thinking_binding_absent", "advertised_max_tokens": 128000, "attempts": 2}
            (root / "tool-result-detail.json").write_text(json.dumps(declared))
            self.assertEqual(report(root)["classification"], "fail")
            (root / "policy.json").write_text(json.dumps({"tool_result_probe": True, "probe_variant": "thinking_binding_absent"}))
            self.assertTrue(report(root)["diagnostic"]["complete"])
            (root / "tool-result-detail.json").write_text(json.dumps({**declared, "headers_equal": False}))
            self.assertFalse(report(root)["diagnostic"]["complete"])
            previous = {**declared, "variant": "previous_request_id", "previous_request_id": "req_private-test"}
            (root / "policy.json").write_text(json.dumps({"tool_result_probe": True, "probe_variant": "previous_request_id"}))
            (root / "tool-result-detail.json").write_text(json.dumps(previous))
            self.assertTrue(report(root)["diagnostic"]["complete"])
            self.assertNotIn("req_private-test", json.dumps(report(root)))
            identity = {**declared, "variant": "neutral_default_identity"}
            (root / "policy.json").write_text(json.dumps({"tool_result_probe": True, "probe_variant": "neutral_default_identity"}))
            (root / "tool-result-detail.json").write_text(json.dumps(identity))
            self.assertTrue(report(root)["diagnostic"]["complete"])
            (root / "tool-result-detail.json").write_text(json.dumps({**identity, "headers_equal": False}))
            self.assertFalse(report(root)["diagnostic"]["complete"])
            for order, labels in (("baseline-first", ["baseline", "variant"]), ("variant-first", ["variant", "baseline"])):
                (root / "policy.json").write_text(json.dumps({"tool_result_probe": True, "probe_variant": "neutral_default_identity", "probe_order": order}))
                ordered = {**identity, "order": order, "outcomes": [{"condition": label} for label in labels]}
                (root / "tool-result-detail.json").write_text(json.dumps(ordered))
                self.assertTrue(report(root)["diagnostic"]["complete"])
                for bad in ({**ordered, "order": "private-order"},
                            {**ordered, "outcomes": [{"condition": label} for label in reversed(labels)]},
                            {**ordered, "outcomes": [{"condition": "private-condition"}, {}]}):
                    (root / "tool-result-detail.json").write_text(json.dumps(bad))
                    result = report(root)
                    self.assertFalse(result["diagnostic"]["complete"])
                    self.assertNotIn("private-order", json.dumps(result))
                    self.assertNotIn("private-condition", json.dumps(result))
            (root / "tool-result-detail.json").write_text(json.dumps(ordered))
            unfinished = [{**records[0], "stop_reason": "tool_use", "successful_stop": True}, records[1]]
            (root / "audit.jsonl").write_text("\n".join(json.dumps(value) for value in unfinished))
            self.assertFalse(report(root)["diagnostic"]["complete"])
            (root / "audit.jsonl").write_text("\n".join(json.dumps(value) for value in records))
            header = {**declared, "variant": "request_class_main", "headers_equal": False, "body_equal": True, "headers_only_declared_change": True}
            (root / "policy.json").write_text(json.dumps({"tool_result_probe": True, "probe_variant": "request_class_main"}))
            (root / "tool-result-detail.json").write_text(json.dumps(header))
            self.assertTrue(report(root)["diagnostic"]["complete"])
            for key, value in (("headers_equal", True), ("body_equal", False), ("headers_only_declared_change", False)):
                (root / "tool-result-detail.json").write_text(json.dumps({**header, key: value}))
                self.assertFalse(report(root)["diagnostic"]["complete"])
            (root / "tool-result-detail.json").write_text(json.dumps(header))
            records[1]["message_stop"] = False
            (root / "audit.jsonl").write_text("\n".join(json.dumps(value) for value in records))
            self.assertEqual(report(root)["classification"], "fail")

    def test_tool_result_detail_rejects_arbitrary_fields_and_values(self):
        value = {"baseline_byte_equal": "private-text", "variant_only_content_and_cch": True, "advertised_max_tokens": 128001, "attempts": 3, "raw_body": "private-text", "outcomes": [{"status": 200, "served_model_ok": True, "terminal": True, "stop_reason": "private-text", "text": "private-text", "facts_all": "private-text"}]}
        safe = tool_result_detail(value)
        self.assertNotIn("private-text", json.dumps(safe))
        self.assertNotIn("baseline_byte_equal", safe)
        self.assertEqual(safe["outcomes"][0]["stop_reason"], "other")
        self.assertEqual(safe["variant"], "tool_result_string")
        unknown = tool_result_detail({"variant": "private-text", "variant_only_content_and_cch": True, "variant_only_declared_field_and_cch": True})
        self.assertNotIn("variant", unknown)
        self.assertNotIn("private-text", json.dumps(unknown))

    def test_tool_result_probe_keeps_the_read_only_proxy_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "policy.json").write_text(json.dumps({"tool_result_probe": True, "requests": 3, "output_tokens": 128000, "deadline": time.time() + 60}))
            (root / "phase.txt").write_text("tool_result_probe")
            (root / "client.txt").write_text("rig")
            fake_http = SimpleNamespace(Response=SimpleNamespace(make=lambda *args: args))
            with patch.dict(os.environ, {"CLOAK_ACCEPTANCE_ROOT": str(root)}), patch.dict(__import__("sys").modules, {"mitmproxy": SimpleNamespace(http=fake_http)}):
                spec = importlib.util.spec_from_file_location("tool_probe_proxy_test", Path(__file__).with_name("proxy.py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for name in ("_read", "write"):
                    raw = json.dumps({"model": "claude-opus-5-5", "max_tokens": 128000, "messages": [], "tools": [{"name": name}]}).encode()
                    flow = SimpleNamespace(request=SimpleNamespace(host="api.anthropic.com", path="/v1/messages", headers={}, get_content=lambda: raw), metadata={}, response=None)
                    module.addons[0].request(flow)
                    if name == "_read":
                        self.assertIsNone(flow.response)
                    else:
                        self.assertEqual(flow.metadata["acceptance"]["local_guard"], "tool_roster")

    def test_private_pause_preserves_failed_result_and_waits_after_passing_read_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 30, scope="probes", pause_after_phase="read_error", pause_on_failure=True)
            failed = {"scenario": "read_error", "classification": "fail"}
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [failed]}), patch.object(driver, "pause") as pause:
                with self.assertRaisesRegex(RuntimeError, "did not pass"):
                    driver.phase("read_error", "synthetic input")
                pause.assert_called_once_with("read_error")
                self.assertEqual(driver.checks["read_error"], "fail")
            driver.observed["read_success"] = {"tool_results": 1, "tool_errors": 0}
            row = {"scenario": "read_error", "classification": "pass", "final_answer": True, "request_id": "alias", "facts": dict.fromkeys(FACTS, True), "usage": {"cache_read_input_tokens": 1, "cache_creation_input_tokens": 0}, "tool_results": 2, "tool_errors": 1, "tool_invocations": 1, "tool_target_match": True}
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", return_value={"alias": "completion"}), patch.object(driver, "fux"), patch.object(driver, "pause") as pause:
                driver.phase("read_error", "synthetic input")
                pause.assert_called_once_with("read_error")
                self.assertEqual(driver.checks["read_error"], "pass")
            driver.checks["read_error"] = "fail"
            (driver.output / "continue-rig").touch()
            driver.pause("read_error")
            saved = json.loads((driver.output / "paused.json").read_text())
            self.assertEqual(saved["state"], "released")
            self.assertEqual(saved["status"], "fail")
            self.assertFalse((driver.output / "continue-rig").exists())

    def test_native_system_tail_does_not_hide_a_tool_continuation_in_existing_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            messages = [{"role": "user", "content": [{"type": "tool_result", "content": "synthetic"}]}, {"role": "system", "content": [{"type": "tool_addition"}]}]
            profile = shape({"messages": messages}, {}, Evidence(root))
            self.assertEqual(profile["call_role"], "tool_continuation")
            inputs = []
            for client in ("rig", "claude"):
                directory = root / client
                directory.mkdir()
                record = {"client": client, "role": "primary", "scenario": "read_error", "call_role": "prompt", "messages": profile["messages"] if client == "claude" else profile["messages"][:1]}
                (directory / "profiles.jsonl").write_text(json.dumps(record) + "\n")
                inputs.append(directory)
            rows = compare_profiles(root, inputs)["pairs"]
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["paired"])
            self.assertEqual(rows[0]["role"], "tool_continuation")

    def test_memory_record_accepts_clear_failure_synonyms_and_markdown_only(self):
        filename = "probe-0123456789abcdef.txt"
        for text in (filename + " | failure | 1", "| `" + filename + "` | **File not found** | **1** |", filename + " | ENOENT | one attempt", "- '" + filename + "' | error: file not found | 1", filename + " | failed (ENOENT) | 1"):
            with self.subTest(text=text):
                result = second_read_memory(text, filename)
                self.assertTrue(result["failed_once"])
                self.assertEqual(result, structural_record({"second_read": result})["second_read"])
        for text in (filename + " | success | 1", filename + " | failure | 2", filename + " | failed then succeeded | 1", filename + " | failure | 1 or 2", "File " + filename + " failed once", filename + " | failure | 1\n" + filename + " | success | 2", "different-" + filename + " | failure | 1", filename + " | no such file or directory | 10", "Requested filename: " + filename + "\npro_be-0123456789abcdef.txt | failed | 1", filename + " | failed | 1 ||", filename + " | failed | 1.5", filename + " | failed | 1 | ignored"):
            with self.subTest(text=text):
                self.assertFalse(second_read_memory(text, filename)["failed_once"])
        diagnostic = second_read_memory(filename + " | file not found | 2", filename)
        self.assertTrue(diagnostic["failure_mention"])
        self.assertFalse(diagnostic["attempts_one"])

    def test_memory_record_accepts_short_missing_file_reason_without_recovery_or_retries(self):
        filename = "probe-0123456789abcdef.txt"
        row = filename + " | failed: no such file | 1"
        self.assertTrue(second_read_memory(row, filename)["failed_once"])
        for text in (filename + " | failed: no such file then succeeded | 1",
                     filename + " | success: no such file | 1",
                     filename + " | failed: no such file | 2",
                     filename + " | failed: no such file | 1.5",
                     filename + " | failed: no such file | 1 | ignored",
                     "different-" + row,
                     row + "\n" + filename + " | success | 1"):
            with self.subTest(text=text):
                self.assertFalse(second_read_memory(text, filename)["failed_once"])

    def test_missing_or_invalid_phase_files_still_count_and_block_startup_messages(self):
        for startup_only in (False, True):
            for files in ({}, {"phase.txt": "unknown", "client.txt": "claude"}, {"phase.txt": "seed", "client.txt": "invalid"}, {"phase.txt": b"\xff", "client.txt": "claude"}):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    ledger = Ledger(root / "ledger.json")
                    ledger.begin("test", 60)
                    (root / "policy.json").write_text(json.dumps({"requests": 24, "output_tokens": 128000, "deadline": time.time() + 60, "ledger": str(ledger.path), "run_id": "test", "startup_only": startup_only}))
                    for name, value in files.items():
                        (root / name).write_bytes(value if isinstance(value, bytes) else value.encode())
                    fake_http = SimpleNamespace(Response=SimpleNamespace(make=lambda *args: args))
                    with patch.dict(os.environ, {"CLOAK_ACCEPTANCE_ROOT": str(root)}), patch.dict(__import__("sys").modules, {"mitmproxy": SimpleNamespace(http=fake_http)}):
                        spec = importlib.util.spec_from_file_location("startup_proxy_test", Path(__file__).with_name("proxy.py"))
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        raw = json.dumps({"model": "claude-opus-5-5", "max_tokens": 16, "messages": [], "tools": []}).encode()
                        flow = SimpleNamespace(request=SimpleNamespace(host="api.anthropic.com", path="/v1/messages", headers={}, get_content=lambda: raw), metadata={}, response=None)
                        module.addons[0].request(flow)
                    self.assertEqual(flow.response[0], 400)
                    self.assertEqual(flow.metadata["acceptance"]["local_guard"], "startup_guard" if startup_only else "invalid_request")
                    self.assertEqual(json.loads(ledger.path.read_text())["attempts"], 1)
                    ledger.finish("test")

    def test_startup_report_passes_only_ready_clients_without_messages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "policy.json").write_text(json.dumps({"startup_only": True, "clients": ["claude"]}))
            self.assertEqual(report(root)["classification"], "fail")
            (root / "driver-events.jsonl").write_text(json.dumps({"client": "claude", "event": "startup_complete"}) + "\n")
            self.assertEqual(report(root)["classification"], "pass")
            (root / "audit.jsonl").write_text(json.dumps({"kind": "request", "call": 1, "client": "claude", "local_guard": "startup_guard"}) + "\n")
            self.assertEqual(report(root)["classification"], "fail")

    def test_ttl_report_requires_two_ready_wire_completions_and_test_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "policy.json").write_text(json.dumps({"root_ttl_probe": True}))
            (root / "root-ttl-probe.json").write_text(json.dumps({"exit_code": 0, "timed_out": False}))
            self.assertEqual(report(root)["classification"], "fail")
            row = {"kind": "response", "scenario": "root_ttl_probe", "status": 200, "requested_model_ok": True, "model_ok": True, "terminal": True, "successful_stop": True, "stop_reason": "end_turn", "ready_reply": True, "usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}
            (root / "audit.jsonl").write_text("\n".join(json.dumps({**row, "call": index}) for index in (1, 2)))
            self.assertEqual(report(root)["classification"], "pass")
            (root / "root-ttl-probe.json").write_text(json.dumps({"exit_code": 1, "timed_out": False}))
            self.assertEqual(report(root)["classification"], "fail")

    @unittest.skipUnless(LINUX_TMPFS, "live preflight test requires Linux and /dev/shm")
    def test_ttl_preflight_lists_exact_test_without_authentication(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "fake"
            binary.write_bytes(b"fake")
            with patch("preflight.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=ROOT_TTL_TEST + ": test\n")) as run:
                self.assertEqual(verify_root_ttl(binary)["test"], ROOT_TTL_TEST)
                self.assertIn("--list", run.call_args.args[0])
                self.assertNotIn("CLAUDE_CLOAK_LIVE_ROOT_TTL", run.call_args.kwargs["env"])
            with patch("preflight.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="")), self.assertRaises(ValueError):
                verify_root_ttl(binary)

    def test_read_target_is_reconstructed_across_sse_fragments_and_never_exported(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            for name, key, path in (("read", "path", "facts.txt"), ("Read", "file_path", str(workspace / "facts.txt"))):
                parser = SSE(expected_read_target="facts.txt", read_tool=(name,), workspace=workspace)
                parser.consume({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "name": name, "input": {}}})
                raw = json.dumps({key: path})
                for character in raw:
                    parser.consume({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": character}})
                parser.consume({"type": "content_block_stop", "index": 2})
                result = structural_record(parser.finish())
                self.assertEqual(result["tool_invocations"], 1)
                self.assertTrue(result["tool_target_match"])
                self.assertNotIn(path, json.dumps(result))

    def test_read_target_rejects_wrong_name_path_and_incomplete_tool_blocks(self):
        for name, path, stopped in (("Read", "/other/facts.txt", True), ("Read", "../facts.txt", True), ("Write", "facts.txt", True), ("Read", "facts.txt", False)):
            parser = SSE(expected_read_target="facts.txt", read_tool=("Read",), workspace="/owned/workspace")
            parser.consume({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "name": name, "input": {"file_path": path}}})
            if stopped:
                parser.consume({"type": "content_block_stop", "index": 0})
            self.assertFalse(parser.finish()["tool_target_match"])
        parser = SSE(limit=8)
        parser.consume({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "name": "Read", "input": {}}})
        with self.assertRaisesRegex(ValueError, "bound"):
            parser.consume({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "x" * 9}})

    def test_read_phase_requires_exactly_one_new_invocation_and_expected_target(self):
        row = {"classification": "pass", "facts": dict.fromkeys(FACTS, True), "usage": {"cache_read_input_tokens": 1, "cache_creation_input_tokens": 0}, "tool_results": 1, "tool_errors": 0, "tool_invocations": 0, "tool_target_match": True}
        self.assertEqual(Matrix.check("read_success", [row], {}), "fail")
        self.assertEqual(Matrix.check("read_success", [{**row, "tool_invocations": 1}, row], {}), "pass")
        self.assertEqual(Matrix.check("read_success", [{**row, "tool_invocations": 2}, row], {}), "fail")
        self.assertEqual(Matrix.check("read_success", [{**row, "tool_invocations": 1, "tool_target_match": False}, row], {}), "fail")
        missing = dict(row)
        del missing["tool_invocations"]
        self.assertEqual(Matrix.check("read_success", [missing], {}), "unknown")

    def test_claude_compaction_reanchors_fingerprint_then_requires_continuity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for cls in (Matrix, ClaudeMatrix):
                kwargs = {"auth_home": "/fake"} if cls is ClaudeMatrix else {}
                driver = cls(root, root / cls.__name__, "/fake", None, {}, float("inf"), **kwargs)
                driver.observed = {"seed": {"human_fingerprint": "old"}, "manual_recall": {"human_fingerprint": "new"}, "rewarm": {"human_fingerprint": "new"}, "fork": {"human_fingerprint": "new"}}
                with patch.object(driver, "status") as status:
                    self.assertEqual(driver.fingerprint_check("manual_recall", {"human_fingerprint": "new"}), "pass" if cls is ClaudeMatrix else "fail")
                    self.assertEqual(driver.fingerprint_check("fork", {"human_fingerprint": "new"}), "pass" if cls is ClaudeMatrix else "fail")
                    if cls is ClaudeMatrix:
                        status.assert_called_with("fingerprint_reanchored")
                        self.assertEqual(driver.fingerprint_check("rewarm", {"human_fingerprint": "old"}), "fail")
                        self.assertEqual(driver.fingerprint_check("resume", {"human_fingerprint": "new"}), "pass")
                        self.assertEqual(driver.fingerprint_check("manual_recall", {}), "unknown")

    @unittest.skipUnless(LINUX_TMPFS and os.environ.get("CLOAK_ACCEPTANCE_PROCESS_SMOKE") == "1" and shutil.which("fux") and shutil.which("bwrap"), "opt-in private process smoke requires Linux, /dev/shm, fux and bwrap")
    def test_nested_claude_runtime_launches_and_quits_with_synthetic_auth(self):
        with tempfile.TemporaryDirectory(prefix="cloak-acceptance-", dir="/dev/shm") as temporary:
            root = Path(temporary) / "claude"
            for folder in ("home", "workspace", "auth/.claude", "ca", "out"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            (root / "auth/.claude/.credentials.json").write_text(json.dumps({"claudeAiOauth": {"accessToken": "fake-only", "expiresAt": (time.time() + 3600) * 1000}}))
            certificate = root / "ca/proxy.pem"
            certificate.write_text("synthetic certificate; the fake program performs no TLS")
            binary = root / "fake-cli"
            binary.write_text('#!/bin/sh\nprintf "SYNTHETIC_READY\\n"\nwhile IFS= read -r line; do [ "$line" = "/quit" ] && exit 0; done\n')
            binary.chmod(0o700)
            driver = ClaudeMatrix(root, root / "out", binary, None,
                                  {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(root / "home"), "SSL_CERT_FILE": str(certificate)},
                                  time.time() + 30, auth_home=root / "auth", startup_only=True)
            try:
                driver.launch()
                driver.send("/quit")
            finally:
                driver.close()
            events = [json.loads(line)["event"] for line in (root / "out/driver-events.jsonl").read_text().splitlines()]
            self.assertIn("input_ready", events)

    @unittest.skipUnless(LINUX_TMPFS and os.environ.get("CLOAK_ACCEPTANCE_PROCESS_SMOKE") == "1" and shutil.which("fux") and shutil.which("bwrap"), "opt-in private process smoke requires Linux, /dev/shm, fux and bwrap")
    def test_nested_claude_runtime_uses_rig_access_without_a_claude_token_store(self):
        with tempfile.TemporaryDirectory(prefix="cloak-acceptance-", dir="/dev/shm") as temporary:
            root = Path(temporary) / "claude"
            for folder in ("home", "workspace", "auth", "rig", "ca", "out"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            (root / "rig/credentials.json").write_text(json.dumps({"access_token": "fake-only", "expires_at": time.time() + 3600, "account_id": "synthetic-account"}))
            certificate = root / "ca/proxy.pem"
            certificate.write_text("synthetic certificate; the fake program performs no TLS")
            binary = root / "fake-cli"
            binary.write_text('#!/bin/sh\nprintf "SYNTHETIC_READY\\n"\nwhile IFS= read -r line; do [ "$line" = "/quit" ] && exit 0; done\n')
            binary.chmod(0o700)
            driver = ClaudeMatrix(root, root / "out", binary, root / "rig",
                                  {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(root / "home"), "SSL_CERT_FILE": str(certificate)},
                                  time.time() + 30, auth_home=root / "auth", claude_auth="rig-access-token", startup_only=True)
            driver.run()
            diagnostics = json.loads((root / "out/child-startup.json").read_text())
            self.assertEqual(diagnostics["auth_mode"], "rig-access-token")
            self.assertTrue(diagnostics["selected_account_matches_rig"])
            self.assertFalse((root / "auth/.claude/.credentials.json").exists())

    def test_untracked_source_content_and_names_are_part_of_build_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "new-source.rs"
            path.write_text("first")
            def git(command, **kwargs):
                if "rev-parse" in command:
                    return "fake-head\n"
                return b"new-source.rs\0" if "ls-files" in command else b"tracked diff"
            with patch("preflight.subprocess.check_output", side_effect=git):
                first = source_identity(root)
                path.write_text("second")
                self.assertNotEqual(source_identity(root)["untracked_sha256"], first["untracked_sha256"])
            self.assertNotIn("first", json.dumps(first))

    def test_exact_stop_reasons_and_unknown_values_are_redacted(self):
        for reason in ("refusal", "max_tokens", "end_turn", "tool_use", "secret-not-a-protocol-reason"):
            parser = SSE()
            parser.consume({"type": "message_delta", "delta": {"stop_reason": reason}})
            result = parser.finish()
            self.assertEqual(result["stop_reason"], "other" if reason.startswith("secret") else reason)
            self.assertEqual(result["final_answer"], reason == "end_turn")

    def test_ledger_is_shared_across_runs_and_refuses_unfinished_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Ledger(Path(temporary) / "ledger.json")
            ledger.begin("first", 60)
            with self.assertRaises(ValueError):
                ledger.begin("parallel", 60)
            self.assertIsNone(ledger.attempt("first", 24)[2])
            ledger.finish("first")
            other = Ledger(ledger.path)
            other.begin("second", 60)
            self.assertEqual(other.attempt("second", 24)[:2], (1, 2))
            other.finish("second")
            self.assertEqual(json.loads(ledger.path.read_text())["attempts"], 2)

    def test_final_budget_slot_is_a_nonforwarded_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Ledger(Path(temporary) / "ledger.json")
            ledger.begin("one", 60)
            self.assertIsNone(ledger.attempt("one", 2)[2])
            self.assertEqual(ledger.attempt("one", 2)[2], "request_cap")
            ledger.finish("one")

    def test_checksum_verifier_receives_the_original_bytes(self):
        raw = b'{ "system": ["literal"], "messages": [] }'
        seen = []
        result = checksum_record(raw, lambda value: seen.append(value) or True)
        self.assertEqual(seen, [raw])
        self.assertEqual(result["checksum"], "match")
        self.assertEqual(checksum_record(raw)["checksum"], "unknown")

    def test_wire_profile_never_exports_text_headers_or_schema_literals(self):
        with tempfile.TemporaryDirectory() as temporary:
            secret = "sk-ant-synthetic-never-real"
            body = {"system": [{"type": "text", "text": secret}], "messages": [], "tools": [{"name": "Read", "input_schema": {"description": secret}}], "metadata": {"user_id": secret}}
            value = shape(body, {"authorization": secret, "user-agent": secret}, Evidence(temporary))
            self.assertNotIn(secret, json.dumps(value))
            self.assertEqual(value["headers"]["user-agent"], "unexpected")
            self.assertEqual(value["tools"][0]["name"], "Read")

    def test_auxiliary_guard_is_nonretryable_and_does_not_pass_primary_scenarios(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "policy.json").write_text(json.dumps({"requests": 24, "output_tokens": 128000, "deadline": time.time() + 60, "matrix": True, "blocked_auxiliary_models": ["claude-haiku-5-5"]}))
            (root / "phase.txt").write_text("seed")
            (root / "client.txt").write_text("claude")
            fake_http = SimpleNamespace(Response=SimpleNamespace(make=lambda *args: args))
            with patch.dict(os.environ, {"CLOAK_ACCEPTANCE_ROOT": str(root)}), patch.dict(__import__("sys").modules, {"mitmproxy": SimpleNamespace(http=fake_http)}):
                spec = importlib.util.spec_from_file_location("comparison_proxy_test", Path(__file__).with_name("proxy.py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                raw = json.dumps({"model": "claude-haiku-5-5", "max_tokens": 16, "messages": [], "tools": []}).encode()
                flow = SimpleNamespace(request=SimpleNamespace(host="api.anthropic.com", path="/v1/messages", headers={}, get_content=lambda: raw), metadata={}, response=None)
                module.Audit().request(flow)
            self.assertEqual(flow.response[0], 400)
            self.assertEqual(flow.response[2]["x-should-retry"], "false")
            self.assertEqual(flow.metadata["acceptance"]["local_guard"], "auxiliary_model")
            self.assertEqual(analyze([flow.metadata["acceptance"]])["classification"], "unknown")

    def test_probe_scope_is_identical_except_tool_spelling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "rig-home").mkdir()
            observed = {}
            for cls in (Matrix, ClaudeMatrix):
                kwargs = {"auth_home": "/fake/home"} if cls is ClaudeMatrix else {}
                driver = cls(root, root / "out", "/fake/binary", "/fake/auth", {"PATH": os.defpath, "HOME": str(root)}, float("inf"), scope="probes", **kwargs)
                phases = []
                with patch.object(driver, "launch"), patch.object(driver, "send"), patch.object(driver, "phase", side_effect=lambda name, text: phases.append((name, text))), patch.object(driver, "close"):
                    driver.run()
                observed[driver.client] = [(name, text.replace("Use Read exactly", "Use read exactly")) for name, text in phases]
                self.assertEqual(list(driver.checks), [name for name, _ in actions()[:4]])
            self.assertEqual(observed["rig"], observed["claude"])

    def test_claude_only_report_and_adjacent_profile_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "claude-matrix.json").write_text(json.dumps({"seed": "fail", "warm": "unknown"}))
            result = report(root)
            self.assertEqual(result["clients"]["claude"]["seed"], "fail")
            self.assertEqual(result["classification"], "fail")
            inputs = []
            for client in ("rig", "claude"):
                folder = root / client
                folder.mkdir()
                (folder / "profiles.jsonl").write_text(json.dumps({"client": client, "role": "primary", "scenario": "seed", "call_role": "prompt", "checksum": "match"}) + "\n")
                inputs.append(folder)
            pairs = compare_profiles(root, inputs)["pairs"]
            self.assertTrue(pairs[0]["paired"])
            self.assertEqual(pairs[0]["role"], "prompt")

    def test_missing_effect_log_fails_build_preflight_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, plugins, main = root / "binary", root / "plugins.toml", root / "main.rs"
            binary.write_bytes(b"offline fake binary")
            enabled = REQUIRED - {"rig_basics::telemetry::effect_log::EffectLogPlugin"}
            plugins.write_text("\n".join('[[plugin]]\nplugin = "' + name + '"' for name in enabled))
            main.write_text("\n".join("load::<" + name + ">" for name in enabled))
            with patch("preflight.source_identity", return_value={"head": "fake", "tracked_diff_sha256": "fake"}):
                record = build_record(binary, root, plugins, main, "successful offline build")
                path = root / "record.json"
                path.write_text(json.dumps(record))
                with patch("preflight.subprocess.run") as launch, self.assertRaisesRegex(ValueError, "EffectLogPlugin"):
                    verify_build(binary, root, plugins, path)
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
