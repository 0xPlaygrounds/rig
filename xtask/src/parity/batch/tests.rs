use super::*;

fn expected() -> BTreeSet<String> {
    BTreeSet::from(["rig::a$a::test".into()])
}

#[test]
fn missing_or_duplicate_terminal_outcomes_fail_closed() {
    assert!(outcomes("", &expected()).is_err());
    let event = json!({"type":"test", "event":"ok", "name":"rig::a$a::test"}).to_string();
    assert!(outcomes(&format!("{event}\n{event}"), &expected()).is_err());
    assert!(outcomes("{incomplete", &expected()).is_err());
}

#[test]
fn unselected_execution_fails_but_ignored_noise_does_not_count() {
    let mut event = json!({"type":"test", "event":"ok", "name":"other"});
    assert!(outcomes(&event.to_string(), &expected()).is_err());
    event["event"] = json!("ignored");
    assert!(outcomes(&event.to_string(), &expected()).is_err());
}

#[test]
fn failing_outcomes_are_preserved() {
    let event = json!({"type":"test", "event":"failed", "name":"rig::a$a::test"});
    assert_eq!(
        outcomes(&event.to_string(), &expected()).expect("parse")["rig::a$a::test"],
        event
    );
}

#[test]
fn original_comparison_permits_only_sibling_visibility_and_formatting() {
    let original = normalized_source(
        "const CAP: u64 = 3; fn check() { assert_eq!(CAP, 3); }",
        false,
    )
    .expect("parse");
    let visible = normalized_source(
        "pub(super) const CAP: u64 = 3;\npub(super) fn check() { assert_eq!(CAP, 3); }",
        true,
    )
    .expect("parse");
    assert_eq!(original, visible);
    assert_ne!(
        original,
        normalized_source(
            "const CAP: u64 = 3; fn check() { assert_eq!(CAP, 4); }",
            true
        )
        .expect("parse")
    );
}

#[test]
fn wrapped_signature_keeps_every_argument_and_body_obligation() {
    let original =
        normalized_source("fn check(value: u64) { assert_eq!(value, 3); }", false).expect("parse");
    let formatted = "pub(super) fn check(\n value: u64,\n) { assert_eq!(value, 3); }";
    assert_eq!(original, normalized_source(formatted, true).expect("parse"));
    for changed in [
        formatted.replace("u64", "u32"),
        formatted.replace("value: u64,", "value: u64, extra: bool,"),
        formatted.replace("value, 3", "value, 4"),
    ] {
        assert_ne!(original, normalized_source(&changed, true).expect("parse"));
    }
}

#[test]
fn shared_tool_visibility_keeps_fields_and_implementation_intact() {
    let original = "struct Tool(State); pub(super) fn existing() {} impl Tool { fn run(&self) { assert!(true); } }";
    let visible = original.replace(
        "struct Tool(State)",
        "pub(super) struct Tool(pub(super) State)",
    );
    let normalized = normalized_source(original, true).expect("parse");
    assert_eq!(
        normalized,
        normalized_source(&visible, true).expect("parse")
    );
    for changed in [
        visible.replace("State)", "DifferentState)"),
        visible.replace("assert!(true)", "assert!(false)"),
        visible.replace("pub(super) struct", "pub(crate) struct"),
    ] {
        assert_ne!(
            normalized,
            normalized_source(&changed, true).expect("parse")
        );
    }
}

#[test]
fn empty_and_duplicate_selections_are_rejected() {
    let mut spec = json!({"cells": []});
    assert!(options(&spec, "native", Path::new("target")).is_err());
    spec["cells"] = json!([{"native":"rig::a$a::test"}, {"native":"rig::a$a::test"}]);
    assert!(options(&spec, "native", Path::new("target")).is_err());
}

#[test]
fn shared_enum_visibility_preserves_variants_payloads_and_attributes() {
    let original = "#[derive(Clone)] #[repr(u8)] enum Shape { Zero, Nested(u64), Parallel = 3 }";
    let visible = original.replace("enum Shape", "pub(super) enum Shape");
    let expected = normalized_source(original, true).expect("parse");
    assert_eq!(expected, normalized_source(&visible, true).expect("parse"));
    for changed in [
        visible.replace("Nested(u64)", "Nested(u32)"),
        visible.replace("Parallel = 3", "Parallel = 4"),
        visible.replace("Zero, ", ""),
        visible.replace("#[derive(Clone)]", "#[derive(Debug)]"),
        visible.replace("pub(super)", "pub(crate)"),
    ] {
        assert_ne!(expected, normalized_source(&changed, true).expect("parse"));
    }
}

#[test]
fn logs_are_content_addressed() {
    assert_ne!(hash(b"first attempt"), hash(b"second attempt"));
}

#[test]
fn evidence_replay_cannot_inherit_golden_regeneration() {
    let command = command(Path::new("."), &["cargo".into()]).expect("command");
    assert!(
        command
            .get_envs()
            .any(|(name, value)| name == "RIG_REGENERATE_GOLDEN" && value.is_none())
    );
    assert!(
        command
            .get_envs()
            .any(|(name, value)| name == "RIG_PROVIDER_TEST_MODE"
                && value == Some(std::ffi::OsStr::new("replay")))
    );
}

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("rig-parity-{name}-{}", std::process::id()));
        fs::create_dir(&path).expect("new scratch directory");
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn failed_source_inspection_retains_run_and_log_references() {
    let scratch = Scratch::new("inspection-failure");
    let log = store_raw(&scratch.0.join("logs"), b"actual execution output", "log").expect("log");
    let artifact = json!({"logs":{"stdout":log}, "source_index":"before-index", "results":[]});
    let (name, passed) = finish_artifact(
        artifact,
        &json!({}),
        Err("escaping source symlink".into()),
        Ok(()),
        &scratch.0,
    )
    .expect("retain evidence");
    assert!(!passed);
    let saved = read(&scratch.0.join("runs").join(name)).expect("saved artifact");
    assert_eq!(saved["status"], "incomplete");
    assert_eq!(saved["logs"]["stdout"], log);
    assert_eq!(saved["source_index"], "before-index");
    assert!(
        saved["input_error"]
            .as_str()
            .expect("error")
            .contains("escaping source symlink")
    );
}

#[test]
fn failed_attempt_replaces_latest_success_and_preserves_history() {
    let scratch = Scratch::new("latest-report");
    let evidence = scratch.0.join("tests/ecs_parity/evidence");
    fs::create_dir_all(&evidence).expect("evidence directory");
    let prior = json!({"status":"executed", "runs":{"native":"historical"}});
    write(&evidence.join("regression-report.json"), &prior).expect("prior report");
    let spec = scratch.0.join("batch.json");
    write(&spec, &json!({"name":"regression", "cells":[]})).expect("spec");
    assert!(
        run(
            &scratch.0,
            vec![
                spec.to_string_lossy().into_owned(),
                scratch
                    .0
                    .join("missing-baseline")
                    .to_string_lossy()
                    .into_owned()
            ]
        )
        .is_err()
    );
    assert_eq!(
        read(&evidence.join("regression-report.json")).expect("latest")["status"],
        "incomplete"
    );
    let historical = evidence
        .join("reports")
        .join(format!("{}.json", hash(&bytes(&prior).expect("bytes"))));
    assert_eq!(read(&historical).expect("history"), prior);
}

#[test]
fn shared_observation_methods_only_permit_sibling_visibility() {
    let original = "struct Observed; impl Observed { fn check(&self) { assert!(true); } }";
    let expected = normalized_source(original, true).expect("parse");
    assert_eq!(
        expected,
        normalized_source(&original.replace("fn check", "pub(super) fn check"), true)
            .expect("parse")
    );
    for changed in [
        original.replace("fn check", "pub fn check"),
        original.replace("assert!(true)", "assert!(false)"),
        original.replace("&self", "&mut self"),
    ] {
        assert_ne!(expected, normalized_source(&changed, true).expect("parse"));
    }
}
