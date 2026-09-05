use super::*;

const REGRESSION: &str = "#[test]\nfn excessive_offset_is_empty() {\n    assert!(page_window::page(&[1, 2, 3], 5, 1).is_empty());\n}\n";
const CORRECT: &str = "pub fn page<T>(items: &[T], offset: usize, limit: usize) -> &[T] {\n    let start = offset.min(items.len());\n    let count = limit.min(items.len() - start);\n    &items[start..start + count]\n}\n";

fn observe_phase(state: &mut State, phase: Phase) -> Report {
    let report = state.validate(phase).unwrap();
    assert!(report.accepted, "{report:?}");
    state.observe_report(report.clone()).unwrap();
    report
}

fn proposal(state: &mut State, path: &str, content: &str) -> Patch {
    let patch = state
        .propose(path, content, "Preserve the documented boundary behavior.")
        .unwrap();
    state.observe_proposal(patch.clone()).unwrap();
    patch
}

fn regression_ready() -> State {
    let mut state = State::new().unwrap();
    observe_phase(&mut state, Phase::Initial);
    let patch = proposal(&mut state, "tests/regression.rs", REGRESSION);
    state.approve(&patch.operation, Approval::Approve).unwrap();
    state.apply(&patch.operation).unwrap().unwrap();
    observe_phase(&mut state, Phase::Regression);
    state
}

#[test]
fn only_observed_failure_unlocks_each_approved_edit() {
    let mut state = State::new().unwrap();
    assert!(
        state
            .propose("tests/regression.rs", REGRESSION, "regression")
            .is_err()
    );
    let report = state.validate(Phase::Initial).unwrap();
    assert!(
        state
            .propose("tests/regression.rs", REGRESSION, "regression")
            .is_err()
    );
    state.observe_report(report).unwrap();
    assert!(state.propose("src/lib.rs", CORRECT, "repair").is_err());
    let patch = proposal(&mut state, "tests/regression.rs", REGRESSION);
    assert!(state.apply(&patch.operation).is_err());
    state.approve(&patch.operation, Approval::Approve).unwrap();
    state.apply(&patch.operation).unwrap().unwrap();
    assert!(state.propose("src/lib.rs", CORRECT, "repair").is_err());
    observe_phase(&mut state, Phase::Regression);
    assert!(state.propose("src/lib.rs", CORRECT, "repair").is_ok());
}

#[test]
fn changed_diff_source_or_regression_invalidates_approval() {
    let mut state = regression_ready();
    let patch = proposal(&mut state, "src/lib.rs", CORRECT);
    state.approve(&patch.operation, Approval::Approve).unwrap();
    let changed = proposal(
        &mut state,
        "src/lib.rs",
        &format!("// Another proposal\n{CORRECT}"),
    );
    assert_ne!(patch.operation, changed.operation);
    assert!(state.apply(&patch.operation).is_err());
    assert!(state.apply(&changed.operation).is_err());
    state
        .approve(&changed.operation, Approval::Approve)
        .unwrap();
    std::fs::write(
        state.project.root().join("tests/regression.rs"),
        "// replaced regression\n",
    )
    .unwrap();
    assert!(state.apply(&changed.operation).is_err());
    assert_eq!(
        state.project.read("src/lib.rs").unwrap(),
        project::initial()["src/lib.rs"]
    );
}

#[test]
fn host_denial_and_cancellation_leave_production_unchanged() {
    for decision in [Approval::Deny, Approval::Cancel] {
        let mut state = regression_ready();
        let patch = proposal(&mut state, "src/lib.rs", CORRECT);
        state.approve(&patch.operation, decision).unwrap();
        state.approve(&patch.operation, decision).unwrap();
        assert!(state.approve(&patch.operation, Approval::Approve).is_err());
        assert!(state.apply(&patch.operation).unwrap().is_none());
        assert_eq!(state.project.writes(), 1);
        assert_eq!(
            state.project.read("src/lib.rs").unwrap(),
            project::initial()["src/lib.rs"]
        );
        assert!(!state.validated());
    }
}

#[test]
fn replay_and_restored_ledger_project_edits_once_and_detect_lost_outcomes() {
    let state = regression_ready();
    let snapshot: Snapshot =
        serde_json::from_str(&serde_json::to_string(&state.snapshot().unwrap()).unwrap()).unwrap();
    let mut live = State::restore(snapshot.clone()).unwrap();
    let mut replay = State::restore(snapshot).unwrap();
    let patch = proposal(&mut live, "src/lib.rs", CORRECT);
    replay.observe_proposal(patch.clone()).unwrap();
    live.approve(&patch.operation, Approval::Approve).unwrap();
    replay.approve(&patch.operation, Approval::Approve).unwrap();
    let receipt = live.apply(&patch.operation).unwrap().unwrap();
    live.observe_receipt(&receipt, false).unwrap();
    replay.observe_receipt(&receipt, true).unwrap();
    assert_eq!(
        live.project.image().unwrap(),
        replay.project.image().unwrap()
    );
    let cut = live.snapshot().unwrap();
    let mut restored =
        State::restore(serde_json::from_value(serde_json::to_value(&cut).unwrap()).unwrap())
            .unwrap();
    let error = restored.apply(&patch.operation).unwrap_err();
    assert!(error.to_string().contains("reconcile"));
    assert_eq!(restored.project.writes(), 2);
    let report = observe_phase(&mut restored, Phase::Final);
    replay.observe_report(report).unwrap();
    assert!(restored.validated() && replay.validated());
    let mut bad = cut.clone();
    bad.writes += 1;
    assert!(State::restore(bad).is_err());
    let mut bad = cut;
    bad.approvals.remove(&patch.operation);
    assert!(State::restore(bad).is_err());
}

#[test]
fn checkpoint_refuses_mislabelled_or_impossible_phase_evidence() {
    let state = regression_ready();
    let mut saved = state.snapshot().unwrap();
    saved.regression.as_mut().unwrap().phase = Phase::Initial;
    assert!(State::restore(saved).is_err());
    let mut saved = State::new().unwrap().snapshot().unwrap();
    saved.regression = state.regression.clone();
    assert!(State::restore(saved).is_err());
}

#[test]
fn a_mismatched_replay_receipt_is_refused_before_project_mutation() {
    let mut live = regression_ready();
    let mut replay = State::restore(live.snapshot().unwrap()).unwrap();
    let patch = proposal(&mut live, "src/lib.rs", CORRECT);
    replay.observe_proposal(patch.clone()).unwrap();
    live.approve(&patch.operation, Approval::Approve).unwrap();
    replay.approve(&patch.operation, Approval::Approve).unwrap();
    let mut receipt = live.apply(&patch.operation).unwrap().unwrap();
    receipt.after = "wrong recorded postimage".into();
    let before = replay.project.image().unwrap();
    assert!(replay.observe_receipt(&receipt, true).is_err());
    assert_eq!(replay.project.image().unwrap(), before);
    assert_eq!(replay.project.writes(), 1);
    assert_eq!(replay.ledger.len(), 1);
}

#[test]
fn saved_acceptance_cannot_override_failed_process_evidence() {
    let mut state = regression_ready();
    let wrong = "pub fn page<T>(items: &[T], offset: usize, limit: usize) -> &[T] {\n    let offset = offset.min(items.len());\n    let end = (offset + limit).min(items.len());\n    &items[offset..end]\n}\n";
    let patch = proposal(&mut state, "src/lib.rs", wrong);
    state.approve(&patch.operation, Approval::Approve).unwrap();
    state.apply(&patch.operation).unwrap().unwrap();
    let report = state.validate(Phase::Final).unwrap();
    assert!(!report.accepted);
    assert_eq!(report.reason, "independent contract validation failed");
    state.observe_report(report).unwrap();
    let mut snapshot = state.snapshot().unwrap();
    snapshot.final_report.as_mut().unwrap().accepted = true;
    assert!(State::restore(snapshot).is_err());
}
