use super::interactions_api_types::InteractionUsage;
use crate::completion::Usage;

/// Shape taken verbatim from a committed cassette
/// (`tests/cassettes/gemini/interactions_api/basic_interaction_returns_id.yaml`).
/// Note that input + output is 48 while the provider's own total is 270 —
/// the missing 222 is thinking, reported beside the pair rather than inside
/// it.
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

/// Before this mapping existed, `reasoning_tokens` was 0 here and the
/// normalized triple could not explain its own total.
#[test]
fn thinking_tokens_survive_the_interactions_mapping() {
    let usage = Usage::from(&recorded());
    assert_eq!(usage.input_tokens, 14);
    assert_eq!(usage.output_tokens, 34);
    assert_eq!(usage.total_tokens, 270);
    assert_eq!(usage.reasoning_tokens, 222);
    assert_eq!(
        usage.input_tokens + usage.output_tokens + usage.reasoning_tokens,
        usage.total_tokens,
        "the components should account for the provider's total"
    );
}

/// The Interactions wire reports `total_cached_tokens`; rig had no field for
/// it, so this surface reported zero cached tokens no matter what Gemini
/// said.
#[test]
fn cached_tokens_survive_the_interactions_mapping() {
    let mut wire = recorded();
    wire.total_cached_tokens = Some(9_000);
    assert_eq!(Usage::from(&wire).cached_input_tokens, 9_000);
}

#[test]
fn tool_use_tokens_survive_the_interactions_mapping() {
    let mut wire = recorded();
    wire.total_tool_use_tokens = Some(77);
    assert_eq!(Usage::from(&wire).tool_use_prompt_tokens, 77);
}

/// With no provider total, the fallback must count every component. Summing
/// only input+output understated the total by the whole thinking spend.
#[test]
fn the_total_fallback_counts_thinking_and_tool_use() {
    let mut wire = recorded();
    wire.total_tokens = None;
    wire.total_tool_use_tokens = Some(5);
    assert_eq!(Usage::from(&wire).total_tokens, 14 + 34 + 222 + 5);
}

/// A wire that omits the new fields entirely must still map, so an older
/// recorded interaction keeps replaying.
#[test]
fn the_older_three_field_shape_still_maps() {
    let wire: InteractionUsage = serde_json::from_value(serde_json::json!({
        "total_input_tokens": 3,
        "total_output_tokens": 4,
        "total_tokens": 7,
    }))
    .expect("the three-field shape should still deserialize");
    let usage = Usage::from(&wire);
    assert_eq!(usage.total_tokens, 7);
    assert_eq!(usage.reasoning_tokens, 0);
    assert_eq!(usage.cached_input_tokens, 0);
}
