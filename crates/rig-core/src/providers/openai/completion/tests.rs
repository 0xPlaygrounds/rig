use super::*;
use serde_json::{Value, json};

fn usage(body: Value) -> crate::completion::Usage {
    serde_json::from_value::<Usage>(body)
        .expect("chat-completions usage should deserialize")
        .to_normalized()
}

/// A gateway fronting an upstream that bills cache writes (OpenRouter over
/// Anthropic) reports them as `prompt_tokens_details.cache_write_tokens`;
/// they are the normalized `cache_creation_input_tokens`.
#[test]
fn usage_maps_cache_token_accounting() {
    let converted = usage(json!({
        "prompt_tokens": 500,
        "completion_tokens": 10,
        "total_tokens": 510,
        "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 50}
    }));

    assert_eq!(converted.input_tokens, Some(500));
    assert_eq!(converted.output_tokens, Some(10));
    assert_eq!(converted.cached_input_tokens, Some(400));
    assert_eq!(converted.cache_creation_input_tokens, Some(50));
}

/// A counter the provider did not send stays `None`: OpenAI's own detail
/// block has no `cache_write_tokens`, and a reply with no detail block has
/// neither cache counter.
#[test]
fn usage_cache_counters_absent_are_unreported() {
    let openai = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}
    }));
    assert_eq!(openai.cached_input_tokens, Some(0));
    assert_eq!(openai.cache_creation_input_tokens, None);

    let bare = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110
    }));
    assert_eq!(bare.cached_input_tokens, None);
    assert_eq!(bare.cache_creation_input_tokens, None);
}

/// Mistral reports cache hits as the structured block and as a top-level
/// `num_cached_tokens`, sometimes on its own. The structured count wins when
/// both are present. Its embeddings reply also carries the singular
/// `prompt_token_details` (`null`) beside the plural key, which must still
/// decode.
#[test]
fn usage_prefers_structured_cached_tokens_and_falls_back() {
    let structured = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "num_cached_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 7}
    }));
    assert_eq!(structured.cached_input_tokens, Some(7));

    let fallback = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "num_cached_tokens": 2
    }));
    assert_eq!(fallback.cached_input_tokens, Some(2));

    let both_spellings = usage(json!({
        "completion_tokens": 0,
        "prompt_token_details": null,
        "prompt_tokens": 42,
        "prompt_tokens_details": null,
        "total_tokens": 42
    }));
    assert_eq!(both_spellings.input_tokens, Some(42));
    assert_eq!(both_spellings.cached_input_tokens, None);
}

/// Mistral's Voxtral models report audio beside `prompt_tokens`, so the
/// input count is the sum; a text turn's detail block is unaffected. The
/// numbers are a live Voxtral turn's, quoted verbatim.
#[test]
fn usage_counts_audio_tokens_reported_beside_the_prompt_as_input() {
    let voxtral = usage(json!({
        "prompt_audio_seconds": 0,
        "prompt_tokens": 6,
        "completion_tokens": 2,
        "total_tokens": 383,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 375}
    }));
    assert_eq!(voxtral.input_tokens, Some(381));
    assert_eq!(voxtral.output_tokens, Some(2));
    assert_eq!(voxtral.total_tokens, Some(381 + 2));

    let text = usage(json!({
        "prompt_tokens": 19,
        "completion_tokens": 2,
        "total_tokens": 21,
        "prompt_tokens_details": {"cached_tokens": 0}
    }));
    assert_eq!(text.input_tokens, Some(19));
}

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
