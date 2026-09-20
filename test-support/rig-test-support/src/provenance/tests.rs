use super::*;

/// The declaration, not the directory listing, decides what may be
/// re-recorded: the hand-corrupted cells of the malformed-tool matrix sit
/// beside their live control in the same fixture directory.
#[test]
fn derived_cells_and_their_live_control_are_declared_apart() {
    assert_eq!(
        scenario_provenance(
            "anthropic",
            "malformed_tool_args_matrix/streaming_healthy_control"
        ),
        Provenance::Live
    );
    assert_eq!(
        scenario_provenance(
            "anthropic",
            "malformed_tool_args_matrix/streaming_malformed_fails_by_default"
        ),
        Provenance::Derived
    );
}

/// A derived scenario reaches the engine carrying its provenance, which is
/// what makes the recording refusal possible.
#[test]
fn a_declared_spec_carries_its_provenance() {
    let spec = declared_spec(
        "anthropic",
        "malformed_tool_args_matrix/streaming_malformed_fails_by_default",
    );
    assert_eq!(spec.provenance(), Some(Provenance::Derived));
    let synthetic = declared_spec(
        "openai",
        "openai_compatible/reasoning_content_tool_roundtrip",
    );
    assert_eq!(synthetic.provenance(), Some(Provenance::Scripted));
}

#[test]
fn unrecorded_scenarios_are_authorized_for_their_first_live_capture() {
    let manifest = manifest::Manifest::load(&crate::cassettes::workspace_root())
        .unwrap_or_else(|error| panic!("cassette scenario declarations: {error}"));
    let mut checked = 0;
    for provider in &manifest.providers {
        for entry in &provider.unrecorded {
            assert_eq!(
                scenario_provenance(&provider.provider, &entry.scenario),
                Provenance::Live,
                "{}/{} must allow its first capture",
                provider.provider,
                entry.scenario
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "the registry must exercise unrecorded scenarios"
    );
    let spec = declared_spec("gemini", "stream_faults/multi_frame_stream");
    assert_eq!(spec.provenance(), Some(Provenance::Live));
}

#[test]
#[should_panic(expected = "has no entry in")]
fn an_undeclared_scenario_has_no_provenance() {
    scenario_provenance("anthropic", "not_a_scenario/never_declared");
}

#[test]
fn a_recording_plan_cannot_touch_another_live_scenario() {
    let scope = Some(std::ffi::OsStr::new("openai/agent/completion_smoke"));
    assert!(recording_scope_allows(
        scope,
        "openai",
        "agent/completion_smoke"
    ));
    assert!(!recording_scope_allows(
        scope,
        "openai",
        "streaming/streaming_smoke"
    ));
}

#[test]
fn a_scripted_family_retains_its_declared_sources() {
    let family = ScriptedFamily::new("openai", "stream_faults");
    assert_eq!(family.provider(), "openai");
    family.assert_declared_source("streaming/streaming_smoke");
}

#[test]
#[should_panic(expected = "not one of its declared sources")]
fn a_scripted_family_cannot_borrow_an_undeclared_source() {
    ScriptedFamily::new("anthropic", "ecs_matrix_long_loop")
        .recorded_statuses_and_bodies("not_a_scenario/never_declared");
}

#[test]
#[should_panic(expected = "is not declared in")]
fn an_undeclared_scripted_family_is_refused() {
    ScriptedFamily::new("anthropic", "not_a_scripted_family");
}
