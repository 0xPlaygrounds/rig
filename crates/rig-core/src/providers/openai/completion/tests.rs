use super::*;
use serde_json::json;

/// The tool choice the OpenAI wires share reads and writes OpenAI's
/// spelling: a mode string, or a function object.
#[test]
fn a_tool_choice_round_trips_in_openai_spelling() {
    for (choice, wire) in [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::None, json!("none")),
        (ToolChoice::Required, json!("required")),
        (
            ToolChoice::function("add"),
            json!({"type": "function", "function": {"name": "add"}}),
        ),
    ] {
        assert_eq!(serde_json::to_value(&choice).expect("serializes"), wire);
        assert_eq!(
            serde_json::from_value::<ToolChoice>(wire).expect("deserializes"),
            choice
        );
    }
    assert!(serde_json::from_value::<ToolChoice>(json!("sometimes")).is_err());
}

/// The gate itself, over every family whose behavior was measured against
/// the live endpoint: the reasoning models reject the legacy field, and
/// everything else, including OpenAI's own older models and any
/// compatible server's model names, still gets the bytes it always got.
#[test]
fn modern_output_cap_covers_exactly_the_reasoning_families() {
    for model in [
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5-nano",
        "gpt-5-2025-08-07",
        "gpt-6",
        GPT_6_ASTRA,
        GPT_6_1_SOL,
        GPT_6_SOL,
        GPT_6_LUNA,
        GPT_5_4,
        GPT_5_4_MINI,
        GPT_5_4_NANO,
        GPT_5_2_PRO,
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
        "o4-mini-2025-04-16",
    ] {
        assert!(
            is_openai_reasoning_model(model),
            "{model} rejects `max_tokens` and must get the modern spelling"
        );
    }

    for model in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-nano",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "chatgpt-4o-latest",
        "Qwen/Qwen3-4B",
        "openai/gpt-oss-20b",
        "gpt-oss-120b",
        "llama-3.1-8b-instruct",
        "gpt-45",
        "gpt-",
        "o",
        "opus",
        "o5x",
        "",
    ] {
        assert!(
            !is_openai_reasoning_model(model),
            "{model:?} still takes `max_tokens`; changing its request would be a regression"
        );
    }
}
