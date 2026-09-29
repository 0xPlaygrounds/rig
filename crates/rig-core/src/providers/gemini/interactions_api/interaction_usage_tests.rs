use super::interactions_api_types::InteractionUsage;
use crate::completion::Usage;

/// Shape taken verbatim from a committed cassette
/// (`crates/rig-cassette/fixtures/cassettes/gemini/interactions_api/basic_interaction_returns_id.yaml`).
/// The API reports the 222 thinking tokens beside input and output (14 + 34),
/// and its own total, 270, counts all three.
fn recorded() -> InteractionUsage {
    serde_json::from_value(serde_json::json!({
        "total_input_tokens": 14,
        "total_output_tokens": 34,
        "total_tokens": 270,
        "total_cached_tokens": 0,
        "total_thought_tokens": 222,
        "total_tool_use_tokens": 0,
    }))
    .expect("recorded usage should deserialize")
}

/// Thinking is output, so input plus output is the provider's own total.
#[test]
fn thinking_tokens_survive_the_interactions_mapping() {
    let usage = Usage::from(&recorded());
    assert_eq!(usage.input_tokens, Some(14));
    assert_eq!(usage.output_tokens, Some(34 + 222));
    assert_eq!(usage.total_tokens, Some(270));
    assert_eq!(usage.reasoning_tokens, Some(222));
    assert_eq!(
        usage.total_tokens,
        usage
            .input_tokens
            .zip(usage.output_tokens)
            .map(|(input, output)| input + output),
        "the total is input plus output"
    );
}

/// The Interactions wire reports `total_cached_tokens`; rig had no field for
/// it, so this surface reported zero cached tokens no matter what Gemini
/// said.
#[test]
fn cached_tokens_survive_the_interactions_mapping() {
    let mut wire = recorded();
    wire.total_cached_tokens = Some(9_000);
    assert_eq!(Usage::from(&wire).cached_input_tokens, Some(9_000));
}

/// Tool-use tokens are input.
#[test]
fn tool_use_tokens_survive_the_interactions_mapping() {
    let mut wire = recorded();
    wire.total_tool_use_tokens = Some(77);
    let usage = Usage::from(&wire);
    assert_eq!(usage.tool_use_prompt_tokens, Some(77));
    assert_eq!(usage.input_tokens, Some(14 + 77));
}

/// The total is input plus output, with or without a provider total, so it
/// counts every component: thinking and tool use included.
#[test]
fn the_total_counts_thinking_and_tool_use_without_a_provider_total() {
    let mut wire = recorded();
    wire.total_tokens = None;
    wire.total_tool_use_tokens = Some(5);
    assert_eq!(Usage::from(&wire).total_tokens, Some(14 + 34 + 222 + 5));
}

/// A wire that omits the new fields entirely must still map, so an older
/// recorded interaction keeps replaying; the counters it never sent are
/// absent rather than zero.
#[test]
fn the_older_three_field_shape_still_maps() {
    let wire: InteractionUsage = serde_json::from_value(serde_json::json!({
        "total_input_tokens": 3,
        "total_output_tokens": 4,
        "total_tokens": 7,
    }))
    .expect("the three-field shape should still deserialize");
    let usage = Usage::from(&wire);
    assert_eq!(usage.total_tokens, Some(7));
    assert_eq!(usage.reasoning_tokens, None);
    assert_eq!(usage.cached_input_tokens, None);
}
