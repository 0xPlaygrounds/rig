//! Guarantee 5 over every recorded pair: one prompt answered unary and
//! streamed, on one API.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::{comparable, raw_pair};
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::providers::gemini::GeminiConfig;
use crate::providers::openai::wire::{
    DEEPSEEK, DOUBLEWORD, GROQ, LLAMACPP, MISTRAL, OPENAI, OPENROUTER, OpenAIConfig, PERPLEXITY,
    VENICE,
};

/// The cassette corpus, beside this crate.
fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../rig-cassette/fixtures/cassettes")
}

/// The reply body of the one interaction `relative` records: a
/// single-quoted or double-quoted scalar, or a `|+` literal block for an
/// event stream.
/// Read without a YAML dependency, and never written.
fn recorded(relative: &str) -> String {
    let file = corpus().join(relative);
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|error| panic!("{} is readable: {error}", file.display()));
    let (_, then) = text
        .split_once("\nthen:\n")
        .unwrap_or_else(|| panic!("{relative} records a reply"));
    let mut lines = then.lines();
    let body = lines
        .by_ref()
        .find_map(|line| line.trim_start().strip_prefix("body:"))
        .map(str::trim)
        .unwrap_or_else(|| panic!("{relative} records a reply body"));
    if let Some(quoted) = body.strip_prefix('\'').and_then(|b| b.strip_suffix('\'')) {
        return quoted.replace("''", "'");
    }
    // A double-quoted scalar escapes as JSON does for what the recorder
    // writes.
    if body.starts_with('"') {
        return serde_json::from_str(body)
            .unwrap_or_else(|error| panic!("{relative} records a quoted body: {error}"));
    }
    let mut block = String::new();
    for line in lines {
        if !line.is_empty() && !line.starts_with("    ") {
            break;
        }
        block.push_str(line.get(4..).unwrap_or_default());
        block.push('\n');
    }
    block
}

/// One prompt answered both ways on one API.
struct Pair {
    provider: &'static str,
    unary: &'static str,
    streamed: &'static str,
    /// JSON pointers to what the two answers of the prompt cannot share
    /// (generated text, signatures, counts), dropped before comparing.
    minted: &'static [&'static str],
    /// Whether the API's reassembler rebuilds its unary document, so the
    /// pair agrees. `false` while the API runs the interim reassembler
    /// that keeps the old terminal record, so the pair disagrees: the
    /// family that replaces it sets `true`, and the row fails until it
    /// does.
    rebuilt: bool,
}

const PAIRS: &[Pair] = &[
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/raw_exposes_stop_sequence.yaml",
        streamed: "raw_stream_capture_matrix/raw_exposes_stop_sequence.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/raw_exposes_thinking_block_and_signature.yaml",
        streamed: "raw_stream_capture_matrix/terminal_raw_round_trips_for_thinking_stream.yaml",
        minted: &["/content/0/signature"],
        rebuilt: true,
    },
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/raw_exposes_tool_use_block.yaml",
        streamed: "raw_stream_capture_matrix/terminal_raw_round_trips_for_tool_use_stream.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/raw_round_trips_into_provider_type.yaml",
        streamed: "raw_stream_capture_matrix/terminal_raw_round_trips_into_provider_type.yaml",
        // The two recordings ask for different words.
        minted: &["/content/0/text", "/usage"],
        rebuilt: true,
    },
    Pair {
        provider: "anthropic",
        unary: "raw_capture_matrix/normalized_fields_match_raw_renormalized.yaml",
        streamed: "raw_stream_capture_matrix/normalized_terminal_matches_raw_renormalized.yaml",
        // The two recordings ask for different words.
        minted: &["/content/0/text", "/usage"],
        rebuilt: true,
    },
    Pair {
        provider: "cohere",
        unary: "raw_capture_matrix/raw_exposes_envelope_fields.yaml",
        streamed: "raw_stream_capture_matrix/stream_terminal_reproduces_the_usage_chunk.yaml",
        // The two recordings ask for different words.
        minted: &["/choices/0/message/content", "/usage"],
        rebuilt: true,
    },
    Pair {
        provider: "gemini",
        unary: "raw_capture_matrix/raw_exposes_forced_function_call.yaml",
        streamed: "raw_stream_capture_matrix/raw_terminal_keeps_stop_on_forced_function_call.yaml",
        minted: &["/responseId"],
        rebuilt: false,
    },
    Pair {
        provider: "ollama",
        unary: "raw_capture_matrix/raw_exposes_envelope_fields.yaml",
        streamed: "raw_stream_capture_matrix/stream_terminal_reproduces_the_usage_chunk.yaml",
        // The two recordings ask for different words.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning",
            "/usage",
        ],
        rebuilt: true,
    },
    Pair {
        provider: "openai",
        unary: "raw_capture_matrix/chat_raw_round_trips_typed.yaml",
        streamed: "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "openai",
        unary: "raw_capture_matrix/chat_tool_call_raw_round_trips_typed.yaml",
        streamed: "raw_stream_capture_matrix/chat_tool_call_stream_raw_round_trips_typed.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "openai",
        unary: "raw_capture_matrix/responses_raw_exposes_service_tier_and_store.yaml",
        streamed: "raw_stream_capture_matrix/responses_stream_raw_exposes_status.yaml",
        minted: &[],
        rebuilt: false,
    },
    Pair {
        provider: "openrouter",
        unary: "raw_capture_matrix/raw_round_trips_openrouter_type.yaml",
        streamed: "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost_and_provider.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "perplexity",
        unary: "raw_capture_matrix/normalized_fields_match_raw_renormalized.yaml",
        streamed: "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "deepseek",
        unary: "followup_hunt_matrix/blocking_stop_sequence_reaches_the_wire_and_stops_generation.yaml",
        streamed: "followup_hunt_matrix/streaming_stop_sequence_reaches_the_wire_and_stops_generation.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "deepseek",
        unary: "streaming_logprobs_matrix/blocking_disabled_length_top_absent.yaml",
        streamed: "streaming_logprobs_matrix/streaming_disabled_length_top_absent.yaml",
        // Each answer samples its own probabilities.
        minted: &["/choices/0/logprobs/content/0/logprob"],
        rebuilt: true,
    },
    Pair {
        provider: "deepseek",
        unary: "streaming_logprobs_matrix/blocking_low_length_top_absent.yaml",
        streamed: "streaming_logprobs_matrix/streaming_low_length_top_absent.yaml",
        // Each answer samples its own probabilities.
        minted: &["/choices/0/logprobs/reasoning_content/0/logprob"],
        rebuilt: true,
    },
    Pair {
        provider: "deepseek",
        unary: "streaming_logprobs_matrix/blocking_low_length_top_two.yaml",
        streamed: "streaming_logprobs_matrix/streaming_low_length_top_two.yaml",
        // Each answer samples its own probabilities.
        minted: &[
            "/choices/0/logprobs/reasoning_content/0/logprob",
            "/choices/0/logprobs/reasoning_content/0/top_logprobs/0/logprob",
            "/choices/0/logprobs/reasoning_content/0/top_logprobs/1/logprob",
        ],
        rebuilt: true,
    },
    Pair {
        provider: "deepseek",
        unary: "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap.yaml",
        streamed: "turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "doubleword",
        unary: "finish_reason_matrix/blocking_natural_stop.yaml",
        streamed: "finish_reason_matrix/streaming_natural_stop.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "doubleword",
        unary: "finish_reason_matrix/blocking_tool_calls.yaml",
        streamed: "finish_reason_matrix/streaming_tool_calls.yaml",
        // The unary answer reasoned before its call and the streamed one did
        // not.
        minted: &[
            "/choices/0/message/reasoning_content",
            "/choices/0/message/reasoning_details",
            "/choices/0/message/tool_calls/0/function/arguments",
            "/usage",
        ],
        rebuilt: true,
    },
    Pair {
        provider: "llamacpp",
        unary: "response_shape_matrix/two_candidates_blocking.yaml",
        streamed: "response_shape_matrix/two_candidates_streaming.yaml",
        // Each answer writes its own first candidate and times itself.
        minted: &["/choices/0/message/content", "/timings", "/usage"],
        rebuilt: true,
    },
    Pair {
        provider: "llamacpp",
        unary: "truncation_matrix/tool_call_cut_mid_arguments.yaml",
        streamed: "truncation_matrix/streaming_tool_call_cut_mid_arguments.yaml",
        // llama.cpp's unary body states an empty `content` beside its calls,
        // which its stream never sends; each answer times itself.
        minted: &["/choices/0/message/content", "/timings"],
        rebuilt: true,
    },
    Pair {
        provider: "openai",
        unary: "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_length_top_zero.yaml",
        streamed: "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_length_top_zero.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "openai",
        unary: "chat_streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_absent.yaml",
        streamed: "chat_streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_absent.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "openai",
        unary: "chat_terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_plain_two.yaml",
        streamed: "chat_terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_plain_two.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "openrouter",
        unary: "reasoning_tool_order_matrix/blocking_single.yaml",
        streamed: "reasoning_tool_order_matrix/streaming_single.yaml",
        // Each answer reasons in its own words, and OpenRouter's unary body
        // keeps the stream `index` on its calls.
        minted: &[
            "/choices/0/message/reasoning",
            "/choices/0/message/reasoning_details/0/text",
            "/choices/0/message/reasoning_details/0/signature",
            "/choices/0/message/tool_calls/0/index",
            "/usage",
        ],
        rebuilt: true,
    },
    Pair {
        provider: "openrouter",
        unary: "refusal_matrix/blocking_refusal_with_tools_in_request.yaml",
        streamed: "refusal_matrix/streaming_refusal_emits_no_tool_calls.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "openrouter",
        unary: "streaming_logprobs_matrix/blocking_gpt_4_1_mini_stop_top_absent.yaml",
        streamed: "streaming_logprobs_matrix/streaming_gpt_4_1_mini_stop_top_absent.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "openrouter",
        unary: "terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_plain_two.yaml",
        streamed: "terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_plain_two.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "openrouter",
        unary: "terminal_metadata_matrix/blocking_gpt_4_1_mini_tiny_tool.yaml",
        streamed: "terminal_metadata_matrix/streaming_gpt_4_1_mini_tiny_tool.yaml",
        // OpenRouter's unary body keeps the stream `index` on its calls.
        minted: &["/choices/0/message/tool_calls/0/index"],
        rebuilt: true,
    },
    Pair {
        provider: "openrouter",
        unary: "terminal_metadata_matrix/blocking_gpt_4o_mini_tiny_plain_one.yaml",
        streamed: "terminal_metadata_matrix/streaming_gpt_4o_mini_tiny_plain_one.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "venice",
        unary: "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap.yaml",
        streamed: "turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap.yaml",
        // Each answer is cut at its own words, and Venice states its
        // parameters in a unary body only.
        minted: &["/choices/0/message/content", "/venice_parameters"],
        rebuilt: true,
    },
    Pair {
        provider: "groq",
        unary: "history_survival_matrix/unary.yaml",
        streamed: "history_survival_matrix/streaming.yaml",
        // Each answer reasons and counts in its own words. A gpt-oss stream
        // names each delta's `channel`, which the unary message does not;
        // Groq states the seed in a unary body only and repeats the usage
        // under `x_groq` in a stream only.
        minted: &[
            "/choices/0/message/channel",
            "/choices/0/message/reasoning",
            "/system_fingerprint",
            "/usage",
            "/x_groq/seed",
            "/x_groq/usage",
        ],
        rebuilt: true,
    },
    Pair {
        provider: "groq",
        unary: "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming.yaml",
        streamed: "agent_tool_sessions/parallel_tool_calls_single_turn_streaming.yaml",
        // Each answer times itself. Groq states the seed in a unary body only
        // and repeats the usage under `x_groq` in a stream only.
        minted: &[
            "/usage/completion_time",
            "/usage/prompt_time",
            "/usage/queue_time",
            "/usage/total_time",
            "/x_groq/seed",
            "/x_groq/usage",
        ],
        rebuilt: true,
    },
    Pair {
        provider: "mistral",
        unary: "history_survival_matrix/unary.yaml",
        streamed: "history_survival_matrix/streaming.yaml",
        // Mistral's unary body states an empty `content` beside its calls and
        // keeps their stream `index`, which its stream does not; the cache
        // was cold for one answer only.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/tool_calls/0/index",
            "/choices/0/message/tool_calls/1/index",
            "/usage/prompt_tokens_details/cached_tokens",
        ],
        rebuilt: true,
    },
    Pair {
        provider: "openai",
        unary: "long_task_matrix/chat_repair.yaml",
        streamed: "long_task_matrix/chat_repair_streamed.yaml",
        minted: &[],
        rebuilt: true,
    },
    Pair {
        provider: "openrouter",
        unary: "reasoning_roundtrip/nonstreaming.yaml",
        streamed: "reasoning_roundtrip/streaming.yaml",
        // Each answer reasons and answers in its own words.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning",
            "/choices/0/message/reasoning_details/0/summary",
            "/choices/0/message/reasoning_details/1/data",
            "/usage",
        ],
        rebuilt: true,
    },
    Pair {
        provider: "openrouter",
        unary: "agent_tool_sessions/sequential_complex_tool_calls_nonstreaming.yaml",
        streamed: "agent_tool_sessions/sequential_complex_tool_calls_streaming.yaml",
        // OpenRouter's unary body keeps the stream `index` on its calls.
        minted: &["/choices/0/message/tool_calls/0/index"],
        rebuilt: true,
    },
    Pair {
        provider: "venice",
        unary: "reasoning_matrix/tool_unary.yaml",
        streamed: "reasoning_matrix/tool_streamed.yaml",
        // Each answer reasons in its own words. Venice's unary body states an
        // empty `content` beside its calls and its parameters, which its
        // stream does not.
        minted: &[
            "/choices/0/message/content",
            "/choices/0/message/reasoning_content",
            "/cost",
            "/usage",
            "/venice_parameters",
        ],
        rebuilt: true,
    },
    Pair {
        provider: "doubleword",
        unary: "history_survival_matrix/unary.yaml",
        streamed: "history_survival_matrix/streaming.yaml",
        // Each answer counts its own reasoning; Doubleword states the service
        // tier in a unary body only.
        minted: &["/service_tier", "/usage"],
        rebuilt: true,
    },
];

/// The `raw` pair `pair` decodes to on the wire that recorded it.
async fn decoded(pair: &Pair) -> (Value, Value) {
    let unary = recorded(&format!("{}/{}", pair.provider, pair.unary));
    let streamed = recorded(&format!("{}/{}", pair.provider, pair.streamed));
    let chat = |dialect| OpenAIConfig::new("test-key").with_dialect(dialect);
    let decoded = match (pair.provider, pair.unary) {
        ("anthropic", _) => {
            let wire = AnthropicConfig::new("test-key").completion("claude-haiku-4-5");
            raw_pair(wire, unary, streamed).await
        }
        ("gemini", _) => {
            let wire = GeminiConfig::new("test-key").completion("gemini-2.5-flash-lite");
            raw_pair(wire, unary, streamed).await
        }
        ("openai", unary_path) if unary_path.contains("responses") => {
            raw_pair(chat(&OPENAI).responses("gpt-4.1-nano"), unary, streamed).await
        }
        ("openai", _) => raw_pair(chat(&OPENAI).chat("gpt-4.1-nano"), unary, streamed).await,
        ("openrouter", _) => {
            raw_pair(
                chat(&OPENROUTER).chat("openai/gpt-4o-mini"),
                unary,
                streamed,
            )
            .await
        }
        ("perplexity", _) => raw_pair(chat(&PERPLEXITY).chat("sonar"), unary, streamed).await,
        ("deepseek", _) => raw_pair(chat(&DEEPSEEK).chat("deepseek-chat"), unary, streamed).await,
        ("doubleword", _) => {
            raw_pair(chat(&DOUBLEWORD).chat("Qwen/Qwen3.5-9B"), unary, streamed).await
        }
        ("llamacpp", _) => raw_pair(chat(&LLAMACPP).chat("model"), unary, streamed).await,
        ("groq", _) => raw_pair(chat(&GROQ).chat("model"), unary, streamed).await,
        ("mistral", _) => raw_pair(chat(&MISTRAL).chat("model"), unary, streamed).await,
        ("venice", _) => raw_pair(chat(&VENICE).chat("qwen3-5-9b"), unary, streamed).await,
        // Both recordings went to the OpenAI-compatible Chat route.
        ("cohere", _) => {
            let dialect = &crate::providers::openai::wire::COHERE;
            raw_pair(chat(dialect).chat("command-a-03-2025"), unary, streamed).await
        }
        ("ollama", _) => {
            let dialect = &crate::providers::openai::wire::OLLAMA;
            raw_pair(chat(dialect).chat("qwen3:4b"), unary, streamed).await
        }
        (provider, _) => panic!("no wire for {provider}"),
    };
    decoded.unwrap_or_else(|error| panic!("{} decodes: {error}", pair.streamed))
}

/// Every pair agrees whole once its API's reassembler rebuilds the unary
/// document, and a pair whose API is still interim does not.
#[tokio::test]
async fn every_recorded_pair_agrees_once_its_api_is_rebuilt() {
    let mut wrong = Vec::new();
    for pair in PAIRS {
        let (unary, streamed) = decoded(pair).await;
        assert!(
            !unary.is_null(),
            "{}: the unary reply has a document",
            pair.unary
        );
        let agrees = comparable(&unary, pair.minted) == comparable(&streamed, pair.minted);
        match (pair.rebuilt, agrees) {
            (true, false) => wrong.push(format!(
                "{}/{}: the streamed raw is not the unary document\n  unary:    {}\n  streamed: {}",
                pair.provider,
                pair.streamed,
                comparable(&unary, pair.minted),
                comparable(&streamed, pair.minted),
            )),
            (false, true) => wrong.push(format!(
                "{}/{}: the pair agrees; mark it rebuilt",
                pair.provider, pair.streamed
            )),
            _ => {}
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A provider that recorded both a unary and a streamed capture matrix has
/// a row, so a new pair cannot go unchecked.
#[test]
fn every_provider_with_both_capture_matrices_has_a_pair() {
    let listed: BTreeSet<&str> = PAIRS.iter().map(|pair| pair.provider).collect();
    let providers = std::fs::read_dir(corpus()).unwrap_or_else(|error| panic!("{error}"));
    let mut missing = Vec::new();
    for provider in providers.flatten() {
        let dir = provider.path();
        let both = ["raw_capture_matrix", "raw_stream_capture_matrix"]
            .iter()
            .all(|matrix| dir.join(matrix).is_dir());
        let name = provider.file_name().to_string_lossy().into_owned();
        if both && !listed.contains(name.as_str()) {
            missing.push(name);
        }
    }
    assert!(missing.is_empty(), "no parity pair for {missing:?}");
}

#[test]
fn comparable_drops_empties_minted_keys_and_named_pointers() {
    let document = json!({
        "id": "a",
        "created": 1,
        "object": "chat.completion",
        "choices": [{"message": {"content": "pong", "annotations": [], "refusal": null}}],
        "usage": {"total_tokens": 3},
        "extra": {}
    });
    assert_eq!(
        comparable(&document, &["/usage"]),
        json!({"object": "chat.completion", "choices": [{"message": {"content": "pong"}}]})
    );
}
