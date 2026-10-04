use super::streaming::usage_of;
use serde_json::{Value, json};

/// Shape taken verbatim from a committed cassette
/// (`crates/rig-cassette/fixtures/cassettes/gemini/interactions_api/basic_interaction_returns_id.yaml`).
/// The API reports the 222 thinking tokens beside input and output (14 + 34),
/// and its own total, 270, counts all three.
fn recorded() -> Value {
    json!({
        "total_input_tokens": 14,
        "total_output_tokens": 34,
        "total_tokens": 270,
        "total_cached_tokens": 0,
        "total_thought_tokens": 222,
        "total_tool_use_tokens": 0,
    })
}

/// `recorded` with `key` set to `value`.
fn with(key: &str, value: Value) -> Value {
    let mut usage = recorded();
    usage[key] = value;
    usage
}

/// Thinking is output, so input plus output is the provider's own total.
#[test]
fn thinking_tokens_survive_the_interactions_mapping() {
    let usage = usage_of(&recorded());
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
    let usage = usage_of(&with("total_cached_tokens", json!(9_000)));
    assert_eq!(usage.cached_input_tokens, Some(9_000));
}

/// Tool-use tokens are input.
#[test]
fn tool_use_tokens_survive_the_interactions_mapping() {
    let usage = usage_of(&with("total_tool_use_tokens", json!(77)));
    assert_eq!(usage.tool_use_prompt_tokens, Some(77));
    assert_eq!(usage.input_tokens, Some(14 + 77));
}

/// The total is input plus output, with or without a provider total, so it
/// counts every component: thinking and tool use included.
#[test]
fn the_total_counts_thinking_and_tool_use_without_a_provider_total() {
    let mut wire = with("total_tool_use_tokens", json!(5));
    if let Some(fields) = wire.as_object_mut() {
        fields.shift_remove("total_tokens");
    }
    assert_eq!(usage_of(&wire).total_tokens, Some(14 + 34 + 222 + 5));
}

/// A wire that omits the newer fields entirely must still map, so an older
/// recorded interaction keeps replaying; the counters it never sent are
/// absent rather than zero.
#[test]
fn the_older_three_field_shape_still_maps() {
    let usage = usage_of(&json!({
        "total_input_tokens": 3,
        "total_output_tokens": 4,
        "total_tokens": 7,
    }));
    assert_eq!(usage.total_tokens, Some(7));
    assert_eq!(usage.reasoning_tokens, None);
    assert_eq!(usage.cached_input_tokens, None);
}

/// A count of the wrong type is unreported rather than failing the reply,
/// and a count spelled as a numeric string, as proto3 JSON spells 64-bit
/// integers, is read.
#[test]
fn a_mistyped_count_is_unreported() {
    let usage = usage_of(&with("total_input_tokens", json!(true)));
    assert_eq!(usage.input_tokens, None);
    assert_eq!(usage.output_tokens, Some(34 + 222));
    assert_eq!(usage.total_tokens, None);
    let usage = usage_of(&with("total_input_tokens", json!("14")));
    assert_eq!(usage.input_tokens, Some(14));
    assert_eq!(usage.total_tokens, Some(270));
}
