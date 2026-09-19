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
}

#[test]
#[should_panic(expected = "has no entry in")]
fn an_undeclared_scenario_has_no_provenance() {
    scenario_provenance("anthropic", "not_a_scenario/never_declared");
}

#[test]
#[should_panic(expected = "is not declared in")]
fn an_undeclared_scripted_family_is_refused() {
    ScriptedFamily::new("anthropic", "not_a_scripted_family");
}
