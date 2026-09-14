//! Request serialization checks for explicitly enabled providers.
// ---------------------------------------------------------------------------
// Serialization determinism
// ---------------------------------------------------------------------------
//
// The only one of this file's checks that can catch unstable map or tool
// ordering, because neither of the others can:
//
//   * the replay matcher compares *canonical* JSON (`canonical_json` sorts
//     object keys before comparing), so a reordered body still replays cleanly;
//   * `canonical_prefix_blocks` flattens `serde_json::Value`s, and a Value
//     round-trip normalizes key order too.
//
// So a `HashMap` leaking into a request body would bust every real provider
// cache on every turn while every piece of recorded evidence looked identical.
// A single recording cannot reveal it either — one recording has nothing to
// disagree with. Only serializing the *same* request more than once, in
// process, can.
//
// The check drives each provider's real conversion path end to end rather than
// asserting on a hand-built JSON literal: it builds the provider's own client
// against a recording HTTP transport, calls `completion`, and reads the bytes
// that were actually put on the wire. A literal would only prove that
// `serde_json` is deterministic, which was never in doubt.

use rig::client::CompletionClient as _;
use rig::completion::{CompletionModel as _, CompletionRequest, ToolDefinition};
use rig::message::{Message, UserContent};
use rig_core::test_utils::RecordingHttpClient;

/// How many times each provider's request is serialized before the bytes are
/// compared.
///
/// Rust's `HashMap` seeds its iteration order per instance, so an unstable map
/// usually — but not always — reorders between any two runs. Eight runs makes a
/// map that happens to hash two adjacent keys into a stable order overwhelmingly
/// likely to be caught anyway.
const DETERMINISM_RUNS: usize = 8;

/// A response body the providers' parsers will reject.
///
/// Deliberately not a valid completion: this check only reads the *request*
/// bytes, which the recording transport captures before any response is parsed.
/// Scripting a per-provider valid response would add a dozen fixtures that prove
/// nothing about determinism.
const IGNORED_RESPONSE: &str = "{}";

/// A request with enough moving parts to expose an unstable serializer.
///
/// Three tools rather than one, because a single-element collection cannot be
/// observably reordered; and `additional_params` with several keys, because that
/// is the map rig merges into the outbound body by hand and therefore the most
/// likely place for iteration order to leak.
fn determinism_probe_request() -> CompletionRequest {
    let tool = |name: &str, first: &str, second: &str| ToolDefinition {
        name: name.to_owned(),
        description: format!("Deterministic ordering probe tool {name}."),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                first: {"type": "string", "description": "first"},
                second: {"type": "number", "description": "second"},
            },
            "required": [first, second],
        }),
    };

    // Several metadata keys, because `Document::additional_props` is a
    // `HashMap` and a single-entry map cannot be observably reordered. The
    // `Display` rendering path already sorts these deliberately
    // (`crates/rig-core/src/completion/request.rs`), but the *serialization*
    // path that providers with a native document block use does not, so the
    // probe has to carry documents for those providers to be covered at all.
    let document = rig::completion::Document {
        id: "cache-determinism-doc".to_owned(),
        text: "Deterministic ordering probe document body.".to_owned(),
        additional_props: [
            ("author".to_owned(), "probe".to_owned()),
            ("source".to_owned(), "field-notes".to_owned()),
            ("revision".to_owned(), "3".to_owned()),
            ("locale".to_owned(), "en".to_owned()),
        ]
        .into_iter()
        .collect(),
    };

    CompletionRequest {
        chat_history: vec![
            Message::system("You are a deterministic serialization probe."),
            Message::User {
                content: vec![UserContent::text("probe")],
            },
        ],
        documents: vec![document],
        tools: vec![
            tool("alpha_probe", "alpha_first", "alpha_second"),
            tool("beta_probe", "beta_first", "beta_second"),
            tool("gamma_probe", "gamma_first", "gamma_second"),
        ],
        temperature: Some(0.0),
        max_tokens: Some(16),
        tool_choice: None,
        additional_params: Some(serde_json::json!({
            "seed": 7,
            "top_p": 0.5,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "user": "cache-determinism-probe",
        })),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// The single request body a recording transport captured.
fn captured_body(provider: &str, http: &RecordingHttpClient) -> String {
    let requests = http.requests();
    assert_eq!(
        requests.len(),
        1,
        "[{provider}] the determinism probe should put exactly one request on the wire, got {}",
        requests.len()
    );
    String::from_utf8(requests[0].body.to_vec())
        .unwrap_or_else(|error| panic!("[{provider}] request body should be UTF-8: {error}"))
}

/// Every serialization of the same request must produce identical bytes.
fn assert_identical(provider: &str, bodies: &[String]) {
    let first = &bodies[0];
    for (run, body) in bodies.iter().enumerate().skip(1) {
        if body == first {
            continue;
        }
        let diverges_at = first
            .char_indices()
            .zip(body.chars())
            .find(|((_, a), b)| a != b)
            .map_or_else(|| first.len().min(body.len()), |((index, _), _)| index);
        let window = |text: &str| {
            let start = diverges_at.saturating_sub(60);
            let end = (diverges_at + 120).min(text.len());
            text.get(start..end).unwrap_or(text).to_owned()
        };
        panic!(
            "[{provider}] serializing the same CompletionRequest twice produced different bytes \
             (run 0 vs run {run}, first divergence at byte {diverges_at}).\n\nThis busts prompt \
             caching on every request: the provider cache is a prefix match over the exact bytes, \
             so a body whose key or tool order moves between turns can never hit. Neither cassette \
             replay nor the recorded-prefix check can see this — replay compares key-sorted \
             canonical JSON — so this assertion is the only thing standing between an unstable map \
             and a silently uncacheable client.\n\n  run 0:   …{}…\n  run {run}:   …{}…",
            window(first),
            window(body),
        );
    }
}

/// Generate one determinism test per provider request builder.
///
/// The body of each arm builds that provider's own client over a recording
/// transport and returns the model to drive, so every arm exercises the real
/// `CompletionRequest` -> wire conversion rather than a shared stand-in.
macro_rules! determinism_test {
    ($name:ident, $provider:literal, |$http:ident| $build:block) => {
        #[tokio::test]
        async fn $name() {
            let mut bodies = Vec::with_capacity(DETERMINISM_RUNS);
            for _ in 0..DETERMINISM_RUNS {
                let $http = RecordingHttpClient::new(IGNORED_RESPONSE);
                let model = $build;
                // The response is intentionally unparseable; only the captured
                // request matters, and it is captured before parsing.
                let _ = model.completion(determinism_probe_request()).await;
                bodies.push(captured_body($provider, &$http));
            }
            assert_identical($provider, &bodies);
        }
    };
}

#[cfg(feature = "anthropic")]
determinism_test!(
    anthropic_request_serialization_is_deterministic,
    "anthropic",
    |http| {
        rig::providers::anthropic::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("anthropic client should build")
            .completion_model(rig::providers::anthropic::completion::CLAUDE_SONNET_4_6)
    }
);

#[cfg(feature = "openai")]
determinism_test!(
    openai_responses_request_serialization_is_deterministic,
    "openai/responses",
    |http| {
        rig::providers::openai::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("openai client should build")
            .completion_model(rig::providers::openai::GPT_4O)
    }
);

#[cfg(feature = "openai")]
determinism_test!(
    openai_chat_request_serialization_is_deterministic,
    "openai/chat-completions",
    |http| {
        rig::providers::openai::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("openai client should build")
            .completions_api()
            .completion_model(rig::providers::openai::GPT_4O)
    }
);

#[cfg(feature = "gemini")]
determinism_test!(
    gemini_request_serialization_is_deterministic,
    "gemini",
    |http| {
        rig::providers::gemini::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("gemini client should build")
            .completion_model(rig::providers::gemini::completion::GEMINI_2_5_FLASH)
    }
);

#[cfg(feature = "cohere")]
determinism_test!(
    cohere_request_serialization_is_deterministic,
    "cohere",
    |http| {
        rig::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("cohere client should build")
            .completion_model(rig::providers::cohere::COMMAND_A_03_2025)
    }
);

#[cfg(feature = "deepseek")]
determinism_test!(
    deepseek_request_serialization_is_deterministic,
    "deepseek",
    |http| {
        rig::providers::deepseek::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("deepseek client should build")
            .completion_model(rig::providers::deepseek::DEEPSEEK_V4_FLASH)
    }
);

#[cfg(feature = "mistral")]
determinism_test!(
    mistral_request_serialization_is_deterministic,
    "mistral",
    |http| {
        rig::providers::mistral::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("mistral client should build")
            .completion_model(rig::providers::mistral::MISTRAL_SMALL)
    }
);

#[cfg(feature = "openrouter")]
determinism_test!(
    openrouter_request_serialization_is_deterministic,
    "openrouter",
    |http| {
        rig::providers::openrouter::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("openrouter client should build")
            .completion_model("openai/gpt-4o-mini")
    }
);

#[cfg(feature = "groq")]
determinism_test!(
    groq_request_serialization_is_deterministic,
    "groq",
    |http| {
        rig::providers::groq::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("groq client should build")
            .completion_model(rig::providers::groq::LLAMA_3_1_8B_INSTANT)
    }
);

#[cfg(feature = "xai")]
determinism_test!(xai_request_serialization_is_deterministic, "xai", |http| {
    rig::providers::xai::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .expect("xai client should build")
        .completion_model(rig::providers::xai::GROK_3_MINI)
});

#[cfg(feature = "venice")]
determinism_test!(
    venice_request_serialization_is_deterministic,
    "venice",
    |http| {
        rig::providers::venice::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("venice client should build")
            .completion_model(rig::providers::venice::QWEN3_5_9B)
    }
);

#[cfg(feature = "doubleword")]
determinism_test!(
    doubleword_request_serialization_is_deterministic,
    "doubleword",
    |http| {
        rig::providers::doubleword::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("doubleword client should build")
            .completion_model(rig::providers::doubleword::QWEN3_5_9B)
    }
);

#[cfg(feature = "perplexity")]
determinism_test!(
    perplexity_request_serialization_is_deterministic,
    "perplexity",
    |http| {
        rig::providers::perplexity::Client::builder()
            .api_key("test-key")
            .http_client(http.clone())
            .build()
            .expect("perplexity client should build")
            .completion_model(rig::providers::perplexity::SONAR)
    }
);
