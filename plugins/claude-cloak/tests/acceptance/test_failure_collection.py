"""Offline bounds for observing fixed phases after known diagnostic failures."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from audit import FACTS, report
from matrix import Matrix


class FailureCollectionTests(unittest.TestCase):
    @staticmethod
    def response(**changes):
        return {"kind": "response", "scenario": "read_error", "status": 200, "terminal": True,
                "message_stop": True, "close": "complete", "compression_complete": True,
                "model_ok": True, "requested_model_ok": True, "tools_read_only": True,
                "fallback_shape_valid": True, "requested_fallback_models": [], "markers_valid": True,
                "request_id": "first", "classification": "fail", "stop_reason": "refusal",
                "tool_invocations": 0, "tool_target_match": True,
                "usage": {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 1}, **changes}

    def test_known_refusal_and_semantic_failure_only_with_complete_wire_proof(self):
        row = self.response()
        self.assertTrue(Matrix.collectable_failure("read_error", [row]))
        normal = self.response(stop_reason="end_turn", classification="pass")
        self.assertTrue(Matrix.collectable_failure("effort", [normal]))
        self.assertFalse(Matrix.collectable_failure("read_error", [normal]))
        changes = {"status": 400, "kind": "headers", "terminal": False, "message_stop": False,
                   "close": "truncated", "compression_complete": False, "model_ok": False,
                   "requested_model_ok": False, "tools_read_only": False, "fallback_shape_valid": False,
                   "requested_fallback_models": ["claude-opus-4-8"], "markers_valid": False,
                   "request_id": None, "classification": "unknown", "stop_reason": "max_tokens",
                   "pending_frame": True, "parse_error": True, "provider_error": True,
                   "transport_error": True, "local_guard": "output_cap", "capture_unknown": "unsupported",
                   "tool_invocations": -1, "tool_target_match": False, "usage": {}}
        for key, value in changes.items():
            with self.subTest(key=key):
                self.assertFalse(Matrix.collectable_failure("read_error", [{**row, key: value}]))
        for key in ("requested_fallback_models", "tool_invocations", "tool_target_match"):
            missing = dict(row)
            del missing[key]
            self.assertFalse(Matrix.collectable_failure("read_error", [missing]))

    def test_dispatch_and_terminal_bounds_are_not_semantic_failures(self):
        tool = self.response(classification="pass", stop_reason="tool_use", tool_invocations=1)
        final = self.response(request_id="second")
        self.assertTrue(Matrix.collectable_failure("read_error", [tool, final]))
        self.assertFalse(Matrix.collectable_failure("effort", [tool, final]))
        self.assertFalse(Matrix.collectable_failure("read_error", [tool]))
        self.assertFalse(Matrix.collectable_failure("read_error", [tool, {**final, "tool_invocations": 1}]))
        self.assertFalse(Matrix.collectable_failure("read_error", [tool, {**final, "request_id": "first"}]))
        self.assertFalse(Matrix.collectable_failure("read_error", [{**tool, "tool_invocations": 0}, final]))
        self.assertFalse(Matrix.collectable_failure("read_error", [self.response(), final]))
        self.assertFalse(Matrix.collectable_failure("neutral", [self.response(tool_invocations=1)]))
        self.assertFalse(Matrix.collectable_failure("neutral", [{**tool, "tool_invocations": 0}, final]))
        summary = self.response(stop_reason="end_turn", classification="pass")
        self.assertTrue(Matrix.collectable_failure("auto_trigger", [summary, final], {"first": "compaction", "second": "completion"}))
        self.assertFalse(Matrix.collectable_failure("auto_trigger", [summary, final], {"first": "completion", "second": "completion"}))
        self.assertFalse(Matrix.collectable_failure("auto_trigger", [summary, final], {}))

    def test_collection_waits_for_effects_and_only_advances_fixed_next_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, collect_failures=True)
            row = self.response()
            attempts = 0
            def settled():
                nonlocal attempts
                attempts += 1
                return {} if attempts == 1 else {"first": "completion"}
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", side_effect=settled), patch.object(driver, "await_input_ready") as ready, patch.object(driver, "fux"), patch("matrix.time.sleep"):
                driver.phase("read_error", "missing file request")
                ready.assert_called_once()
            self.assertEqual(driver.checks["read_error"], "fail")
            self.assertEqual(driver.pending_failed_phase, ("read_error", "effort", "refusal"))
            with self.assertRaises(RuntimeError), patch.object(driver, "send") as send:
                driver.phase("read_error", "missing file request")
            send.assert_not_called()
            next_row = self.response(scenario="effort", request_id="next", stop_reason="end_turn", classification="pass",
                                     final_answer=True, effort="medium", facts=dict.fromkeys(FACTS, True))
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [next_row]}), patch.object(driver, "settled", return_value={"next": "completion"}), patch.object(driver, "fux"):
                driver.phase("effort", "fixed recall")
            self.assertEqual(driver.checks["effort"], "pass")
            journal = json.loads((driver.output / "failure-continuations.jsonl").read_text())
            self.assertEqual(journal["next_phase"], "effort")
            self.assertFalse(journal["retry"])
            self.assertNotIn("missing file request", json.dumps(journal))
            self.assertEqual(report(driver.output)["classification"], "fail")

    def test_semantic_failure_stays_failed_and_transport_unknown_never_advances(self):
        for changes, collect in (({}, True), ({"model_ok": False}, False), ({"capture_unknown": "unsupported"}, False)):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, collect_failures=True)
                row = self.response(scenario="effort", stop_reason="end_turn", classification="pass" if not changes else "fail",
                                    final_answer=True, facts={}, effort="medium", **changes)
                with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", return_value={"first": "completion"}), patch.object(driver, "await_input_ready") as ready, patch.object(driver, "fux"):
                    if collect:
                        driver.phase("effort", "fixed recall")
                        ready.assert_called_once()
                        self.assertEqual(driver.pending_failed_phase, ("effort", "neutral", "semantic_assertion"))
                    else:
                        with self.assertRaises(RuntimeError):
                            driver.phase("effort", "fixed recall")
                        ready.assert_not_called()
                self.assertEqual(driver.checks["effort"], "fail")

    def test_scope_and_fixed_phase_sequence_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for options in ({"client": "claude"}, {"scope": "probes"}, {"scope": "auto-controls"}, {"continue_after_read_refusal": True}, {"startup_only": True}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, collect_failures=True, **options)
            driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, True, collect_failures=True)
            for first, second in (("neutral", "manual_compact"), ("manual_compact", "manual_recall"),
                                  ("rewarm", "fork"), ("fork", "resume"), ("resume", "auto_seed"),
                                  ("auto_rewarm", "fresh_prefix"), ("fresh_prefix", "fresh_prefix_repeat")):
                self.assertEqual(driver.next_fixed_phase(first), second)
            self.assertIsNone(driver.next_fixed_phase("fresh_prefix_repeat"))

    def test_correct_facts_do_not_hide_unexpected_tool_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, collect_failures=True)
            row = self.response(scenario="effort", stop_reason="end_turn", classification="pass", final_answer=True,
                                effort="medium", facts=dict.fromkeys(FACTS, True), tool_invocations=1)
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", return_value={"first": "completion"}), patch.object(driver, "await_input_ready") as ready:
                with self.assertRaises(RuntimeError):
                    driver.phase("effort", "fixed recall")
                ready.assert_not_called()
            self.assertEqual(driver.checks["effort"], "fail")
            self.assertIsNone(driver.pending_failed_phase)

    def test_passing_terminal_cancellation_requires_every_completion_proof(self):
        row = self.response(scenario="warm", classification="pass", stop_reason="end_turn", close="cancel_after_terminal", transport_error=True, pending_frame=False)
        self.assertTrue(Matrix.collectable_failure("warm", [row]))
        for key, value in (("terminal", False), ("message_stop", False), ("compression_complete", False),
                           ("pending_frame", True), ("pending_frame", None), ("parse_error", True), ("provider_error", True),
                           ("close", "truncated"), ("model_ok", False), ("classification", "fail"), ("usage", {})):
            with self.subTest(key=key):
                self.assertFalse(Matrix.collectable_failure("warm", [{**row, key: value}]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, collect_failures=True)
            row.update(final_answer=True, facts=dict.fromkeys(FACTS, True))
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", return_value={"first": "completion"}), patch.object(driver, "fux"):
                driver.phase("warm", "fixed recall")
            self.assertEqual(driver.checks["warm"], "pass")
            self.assertIsNone(driver.pending_failed_phase)
            row["facts"] = {}
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", return_value={"first": "completion"}), patch.object(driver, "await_input_ready"), patch.object(driver, "fux"):
                driver.phase("warm", "fixed recall")
            self.assertEqual(driver.checks["warm"], "fail")
            self.assertEqual(driver.pending_failed_phase, ("warm", "read_success", "semantic_assertion"))

    def test_retention_requires_real_failed_read_even_when_model_claims_it(self):
        for phase in ("manual_recall", "rewarm", "fork", "resume"):
            for real_read in (False, True):
                with self.subTest(phase=phase, real_read=real_read), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, True, collect_failures=True)
                    driver.checks["read_error"] = "fail"
                    driver.observed = {"read_success": {"tool_results": 1, "tool_errors": 0},
                                       "neutral": {"input_chars": 1000}, "seed": {"human_fingerprint": "same", "session": "original"},
                                       "fork": {"session": "forked"}}
                    history = [self.response(request_id="read-request", classification="pass", stop_reason="tool_use", tool_invocations=1),
                               self.response(request_id="read-refused", tool_results=2, tool_errors=1)] if real_read else [self.response(tool_results=1, tool_errors=0)]
                    current = self.response(scenario=phase, request_id="current", classification="pass", stop_reason="end_turn", final_answer=True,
                                            input_chars=100, facts=dict.fromkeys(FACTS, True), second_read={"filename": True, "failed_once": True},
                                            human_fingerprint="same", session="forked")
                    effects = {row["request_id"]: "completion" for row in [*history, current]}
                    with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [*history, current]}), patch.object(driver, "settled", return_value=effects), patch.object(driver, "await_input_ready"), patch.object(driver, "fux"):
                        driver.phase(phase, "fixed retention prompt")
                    self.assertEqual(driver.checks[phase], "pass" if real_read else "fail")
                    self.assertEqual(driver.checks["read_error"], "fail")

    def test_failed_read_history_requires_exact_target_delta_and_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, collect_failures=True)
            driver.observed["read_success"] = {"tool_results": 1, "tool_errors": 0}
            first = self.response(classification="pass", stop_reason="tool_use", tool_invocations=1)
            last = self.response(request_id="last", tool_results=2, tool_errors=1)
            for change, effects in (({"tool_target_match": False}, {"first": "completion", "last": "completion"}),
                                    ({"tool_results": 3}, {"first": "completion", "last": "completion"}),
                                    ({"tool_errors": 0}, {"first": "completion", "last": "completion"}), ({}, {"last": "completion"})):
                with self.subTest(change=change, effects=effects), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [first, {**last, **change}]}), patch.object(driver, "settled", return_value=effects):
                    self.assertFalse(driver.failed_read_history())

    def test_completed_refusal_cancellation_is_collected_without_erasing_failure(self):
        row = self.response(close="cancel_after_terminal", transport_error=True, pending_frame=False, successful_stop=False)
        self.assertTrue(Matrix.collectable_failure("read_error", [row]))
        for key, value in (("classification", "unknown"), ("successful_stop", None), ("successful_stop", True),
                           ("stop_reason", "max_tokens"), ("terminal", False), ("message_stop", False),
                           ("compression_complete", False), ("pending_frame", True), ("pending_frame", None),
                           ("parse_error", True), ("provider_error", True), ("local_guard", "output_cap"),
                           ("model_ok", False), ("markers_valid", False), ("tool_target_match", False),
                           ("tool_invocations", 1), ("close", "truncated")):
            with self.subTest(key=key):
                self.assertFalse(Matrix.collectable_failure("read_error", [{**row, key: value}]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, collect_failures=True)
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", return_value={"first": "completion"}), patch.object(driver, "await_input_ready") as ready, patch.object(driver, "fux"):
                driver.phase("read_error", "fixed missing read")
                ready.assert_called_once()
            self.assertEqual(driver.checks["read_error"], "fail")
            self.assertEqual(driver.pending_failed_phase, ("read_error", "effort", "refusal"))
            self.assertFalse(driver.failed_read_history())
