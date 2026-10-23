"""Offline checks for explicit compaction after an observed read refusal."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from audit import FACTS, report
from matrix import Matrix, actions


class RefusalPathTests(unittest.TestCase):
    def driver(self, root, **options):
        return Matrix(root, root / "out", "/fake", None, {}, time.time() + 5,
                      continue_after_read_refusal=True, read_refusal_next="manual_compact", **options)

    @staticmethod
    def answer(name, **changes):
        return {"scenario": name, "classification": "pass", "final_answer": True,
                "request_id": "next", "input_chars": 100, "human_fingerprint": "same",
                "facts": dict.fromkeys(FACTS, True), "second_read": {"filename": True, "failed_once": True},
                "usage": {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 1}, **changes}

    def test_selected_path_requires_opt_in_and_skips_only_pending_middle_phases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                Matrix(root, root / "out", "/fake", None, {}, time.time() + 5, read_refusal_next="manual_compact")
            driver = self.driver(root)
            self.assertFalse(driver.skip_after_read_refusal("effort"))
            driver.pending_read_refusal = "failed read prompt"
            self.assertTrue(driver.skip_after_read_refusal("effort"))
            self.assertTrue(driver.skip_after_read_refusal("neutral"))
            self.assertFalse(driver.skip_after_read_refusal("manual_compact"))
            self.assertFalse(driver.skip_after_read_refusal("read_error"))
            driver.read_refusal_next = "effort"
            self.assertFalse(driver.skip_after_read_refusal("neutral"))

    def test_compaction_uses_unchanged_prompt_and_keeps_failure_and_skipped_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            driver = self.driver(root)
            driver.checks["read_error"] = "fail"
            driver.pending_read_refusal = "failed read prompt"
            prompt = dict(actions(driver.second_read_file))["manual_compact"]
            for name, text in (("effort", "new prompt"), ("manual_compact", prompt + " altered")):
                with self.assertRaises(RuntimeError), patch.object(driver, "send") as send:
                    driver.phase(name, text)
                send.assert_not_called()
            row = self.answer("manual_compact")
            with patch.object(driver, "send") as send, patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", return_value={"next": "compaction"}), patch.object(driver, "fux"):
                driver.phase("manual_compact", prompt)
                send.assert_called_once_with(prompt)
            self.assertEqual(driver.checks["manual_compact"], "pass")
            self.assertEqual(driver.checks["read_error"], "fail")
            self.assertEqual(driver.checks["effort"], "unknown")
            self.assertEqual(driver.checks["neutral"], "unknown")
            journal = json.loads((driver.output / "continuations.jsonl").read_text())
            self.assertEqual(journal["next_phase"], "manual_compact")
            self.assertFalse(journal["retry"])
            self.assertNotIn("failed read prompt", json.dumps(journal))
            result = report(driver.output)
            self.assertEqual(result["classification"], "fail")
            self.assertEqual(result["continuations"][0]["next_phase"], "manual_compact")

    def test_compaction_refusal_stops_and_cannot_continue_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver = self.driver(Path(temporary))
            driver.checks["read_error"] = "fail"
            driver.pending_read_refusal = "failed read prompt"
            row = self.answer("manual_compact", classification="fail", final_answer=False, stop_reason="refusal")
            with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "fux"):
                with self.assertRaises(RuntimeError):
                    driver.phase("manual_compact", dict(actions(driver.second_read_file))["manual_compact"])
            self.assertEqual(driver.checks["manual_compact"], "fail")
            self.assertEqual(driver.checks["manual_recall"], "unknown")
            self.assertIsNone(driver.pending_read_refusal)

    def test_run_skips_middle_prompts_only_after_a_proven_refusal(self):
        for refused in (False, True):
            with self.subTest(refused=refused), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "rig-home").mkdir()
                driver = self.driver(root)
                phases = []
                def phase(name, text):
                    phases.append((name, text))
                    if name == "read_error" and refused:
                        driver.pending_read_refusal = text
                        driver.checks[name] = "fail"
                    if name == "manual_compact":
                        driver.pending_read_refusal = None
                with patch.object(driver, "launch"), patch.object(driver, "phase", side_effect=phase), patch.object(driver, "send") as send, patch.object(driver, "auto_controls"), patch.object(driver, "close"):
                    driver.run()
                expected = [(name, text) for name, text in actions(driver.second_read_file) if not refused or name not in ("effort", "neutral")]
                self.assertEqual(phases, expected)
                if refused:
                    self.assertEqual(driver.checks["effort"], "unknown")
                    self.assertEqual(driver.checks["neutral"], "unknown")
                    self.assertEqual([call.args[0] for call in send.call_args_list], ["/quit"])

    def test_recall_still_requires_actual_shrink_from_failed_read_history(self):
        for before, after, passed in ((1000, 100, True), (1000, 1000, False), (None, 100, False), (1000, 0, False)):
            with self.subTest(before=before, after=after), tempfile.TemporaryDirectory() as temporary:
                driver = self.driver(Path(temporary))
                driver.read_refusal_continuation = "manual_compact"
                driver.observed = {"read_error": {"input_chars": before}, "seed": {"human_fingerprint": "same"}}
                row = self.answer("manual_recall", input_chars=after)
                with patch.object(driver, "send"), patch.object(driver, "records", return_value=[]), patch("matrix.analyze", return_value={"calls": [row]}), patch.object(driver, "settled", return_value={"next": "completion"}), patch.object(driver, "fux"):
                    if passed:
                        driver.phase("manual_recall", "recall")
                    else:
                        with self.assertRaises(RuntimeError):
                            driver.phase("manual_recall", "recall")
                self.assertEqual(driver.checks["manual_recall"], "pass" if passed else "fail")
