//! Guarantee 5 over every recorded pair: one prompt answered unary and
//! streamed, on one API.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::{comparable, raw_pair};
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::providers::gemini::GeminiConfig;
use crate::providers::openai::wire::{OPENAI, OPENROUTER, OpenAIConfig, PERPLEXITY};

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
        rebuilt: false,
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
        rebuilt: false,
    },
    Pair {
        provider: "openai",
        unary: "raw_capture_matrix/chat_raw_round_trips_typed.yaml",
        streamed: "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
        minted: &[],
        rebuilt: false,
    },
    Pair {
        provider: "openai",
        unary: "raw_capture_matrix/chat_tool_call_raw_round_trips_typed.yaml",
        streamed: "raw_stream_capture_matrix/chat_tool_call_stream_raw_round_trips_typed.yaml",
        minted: &[],
        rebuilt: false,
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
        rebuilt: false,
    },
    Pair {
        provider: "perplexity",
        unary: "raw_capture_matrix/normalized_fields_match_raw_renormalized.yaml",
        streamed: "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object.yaml",
        minted: &[],
        rebuilt: false,
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
