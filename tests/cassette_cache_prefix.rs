//! Structural guard: a multi-turn conversation's wire prefix must not move.
//!
//! Provider prompt caching is a **prefix match** — the cache key is derived from
//! the exact bytes up to each breakpoint, so any change to an earlier block
//! invalidates everything after it. A conversation whose turn-2 request rewrites,
//! reorders, or drops a block that turn-1 already sent therefore busts the cache
//! on every turn, and the cost shows up in production rather than in CI.
//!
//! Nothing else in the tree pins this. Each cassette test asserts on *its own*
//! turn's values; none compares turn N-1's request against turn N's. This check
//! is derived entirely from cassettes already recorded, so it costs no provider
//! traffic and applies retroactively to the whole corpus.
//!
//! Ported from `inspirations/pydantic-ai` (`tests/cassette_utils.py`
//! `check_cache_prefix_stability`), which guards the same invariant across a
//! 1313-cassette corpus. The invariant is imported; the plumbing is rig's. The
//! rule itself lives in `tests/common/cache_prefix.rs` so that the per-scenario
//! conformance harness (`tests/common/cache_conformance.rs`) applies the
//! identical flattening rather than a second copy that could drift.
//!
//! A test whose behavior is *deliberately* prefix-moving — compaction, history
//! rewriting, dynamic tool disclosure — records that fact in
//! `MOVES_CACHE_PREFIX`, with a reason. An entry with an empty reason is
//! rejected, and an entry that stops matching a real cassette is reported as
//! stale.
//!
//! # Three checks, three different blind spots
//!
//! 1. [`recorded_conversations_do_not_move_their_cache_prefix`] compares
//!    consecutive recorded requests. It can only see what was recorded.
//! 2. [`every_provider_is_covered_by_the_prefix_check`] asserts the first check
//!    actually *looked* at each provider. It previously failed open: any
//!    endpoint the table did not model was skipped silently, and the only global
//!    guard was `compared_pairs > 0` across the entire corpus — so a provider
//!    could have 100% of its traffic skipped while the suite stayed green. It
//!    did: Bedrock's `/model/<id>/converse` and Ollama's `/api/chat` were 100%
//!    unmodeled, which is 72 recorded requests that no prefix check ever saw.
//! 3. [`provider_request_serialization_is_deterministic`] serializes the same
//!    request twice in-process. This is the only one of the three that can catch
//!    unstable map or tool ordering, because *neither* of the others can: the
//!    replay matcher compares canonical (key-sorted) JSON, and `serde_json::Value`
//!    round-trips normalize key order too. A `HashMap` leaking into a request
//!    body would bust every real provider cache while all recorded evidence
//!    looked identical.

#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

#[path = "common/cache_prefix.rs"]
mod cache_prefix;

use cache_prefix::{EndpointKind, PrefixBlock, Violation};

/// Scenarios whose requests legitimately move the cache prefix.
///
/// `(cassette path suffix, reason)`. The reason is required: a bare exemption
/// records that someone silenced the check, not why it is correct.
const MOVES_CACHE_PREFIX: &[(&str, &str)] = &[
    (
        "anthropic/request_override/request_overridden_by_hook_blocking.yaml",
        "the scenario under test is a hook that rewrites the outbound request \
         between turns — moving the prefix is the behavior being recorded, not a \
         defect in it",
    ),
    (
        "anthropic/request_override/request_overridden_by_hook_streaming.yaml",
        "streaming twin of request_overridden_by_hook_blocking; same deliberate \
         hook rewrite",
    ),
    (
        "openai/streaming_grammar/three_turn_tool_session.yaml",
        "turns 2 and 3 are hand-built `completion_request`s that deliberately omit \
         `.tool(...)` (the prompt says \"Do not call any tools\"), so the tools \
         array disappears from the prefix. This is the test's own construction for \
         pinning rs_* id replay, not rig's agent loop — verified that the loop \
         re-advertises the full tool set on every turn in \
         anthropic/multi_turn_streaming/multi_turn_streaming_tools.yaml",
    ),
    (
        "openai/streaming_grammar/tool_then_followup_text.yaml",
        "same shape as three_turn_tool_session: a hand-built follow-up request \
         that intentionally does not re-advertise the tool",
    ),
];

/// Providers exempt from the per-provider coverage census.
///
/// `(provider directory, reason)`. Same contract as `MOVES_CACHE_PREFIX`: an
/// empty reason is rejected and a stale entry is reported. This list exists so
/// that a provider whose cache-bearing endpoint genuinely cannot be modeled is
/// recorded as a *decision* rather than disappearing into a silent skip.
///
/// It is deliberately empty. Every provider directory in the corpus now has a
/// modeled conversational endpoint; if that stops being true, the census failure
/// message says which provider drifted and adding an entry here is the explicit,
/// reasoned way to accept it.
const COVERAGE_EXEMPT_PROVIDERS: &[(&str, &str)] = &[];

/// A provider whose conversational requests are at least this fraction
/// unmodeled has effectively opted out of the prefix check without saying so.
///
/// The threshold is 95% rather than 100% so a provider that records one
/// experimental endpoint alongside a well-covered suite does not fail, while a
/// provider whose real chat traffic is invisible does.
const MAX_UNMODELED_FRACTION: f64 = 0.95;

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    #[serde(default)]
    path: String,
    #[serde(default)]
    body: Option<String>,
}

fn cassette_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let entry = entry.expect("cassette directory entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            cassette_files(&path, found);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "yaml")
        {
            found.push(path);
        }
    }
}

/// Ordered `(path, body)` for every recorded request in one cassette.
fn recorded_requests(contents: &str) -> Vec<(String, Value)> {
    serde_yaml::Deserializer::from_str(contents)
        .filter_map(|document| RecordedInteraction::deserialize(document).ok())
        .filter_map(|interaction| {
            let body = interaction.when.body?;
            let json = serde_json::from_str::<Value>(&body).ok()?;
            Some((interaction.when.path, json))
        })
        .collect()
}

fn cassette_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes");
    assert!(
        root.is_dir(),
        "cassette root moved or vanished: {}",
        root.display()
    );
    root
}

fn all_cassettes(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    cassette_files(root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no cassettes found under {}",
        root.display()
    );
    files
}

fn scenario_name(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .expect("cassette should live under the cassette root")
        .to_string_lossy()
        .replace('\\', "/")
}

fn provider_of(scenario: &str) -> &str {
    scenario.split('/').next().unwrap_or(scenario)
}

#[test]
fn recorded_conversations_do_not_move_their_cache_prefix() {
    let root = cassette_root();

    for (path, reason) in MOVES_CACHE_PREFIX {
        assert!(
            !reason.trim().is_empty(),
            "MOVES_CACHE_PREFIX entry `{path}` needs a reason explaining why moving the \
             cache prefix is correct for that scenario"
        );
    }

    let files = all_cassettes(&root);

    let mut violations: Vec<Violation> = Vec::new();
    let mut exempted = Vec::new();
    let mut compared_pairs = 0usize;

    for file in &files {
        let scenario = scenario_name(&root, file);

        if let Some((exempt, _)) = MOVES_CACHE_PREFIX
            .iter()
            .find(|(exempt, _)| scenario.ends_with(exempt))
        {
            exempted.push((*exempt).to_owned());
            continue;
        }

        let contents = std::fs::read_to_string(file).expect("cassette should be readable");
        let requests = recorded_requests(&contents);

        // Compare consecutive requests to the *same* endpoint. A scenario that
        // hits two endpoints has two independent conversations, and their blocks
        // are not comparable.
        let mut previous: Option<(String, Vec<PrefixBlock>)> = None;
        for (index, (path, body)) in requests.iter().enumerate() {
            let Some(blocks) = cache_prefix::canonical_prefix_blocks(path, body) else {
                previous = None;
                continue;
            };
            if let Some((previous_path, previous_blocks)) = previous.take()
                && previous_path == *path
                && cache_prefix::continues_the_same_conversation(&previous_blocks, &blocks)
            {
                compared_pairs += 1;
                if let Some(violation) =
                    cache_prefix::compare(&scenario, index, &previous_blocks, &blocks)
                {
                    violations.push(violation);
                }
            }
            previous = Some((path.clone(), blocks));
        }
    }

    // The check is only meaningful if it actually compared multi-turn traffic.
    assert!(
        compared_pairs > 0,
        "no consecutive same-endpoint request pairs were compared across {} cassettes — \
         the parser or the endpoint table has drifted and this test is now vacuous",
        files.len()
    );

    let stale = MOVES_CACHE_PREFIX
        .iter()
        .map(|(path, _)| *path)
        .filter(|path| !exempted.iter().any(|seen| seen == path))
        .collect::<Vec<_>>();
    assert!(
        stale.is_empty(),
        "stale MOVES_CACHE_PREFIX entries (the cassette moved or was deleted; delete the entry): {stale:?}"
    );

    assert!(
        violations.is_empty(),
        "a recorded conversation moves its provider-cache wire prefix, which busts the \
         prompt cache on every turn. If the behavior is deliberately prefix-moving \
         (compaction, history rewriting, dynamic tool disclosure), add the cassette to \
         MOVES_CACHE_PREFIX with a reason.\n\n{}",
        violations
            .iter()
            .map(Violation::to_string)
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

/// Per-provider census of what the prefix check actually looked at.
///
/// The check this guards used to fail open. `canonical_prefix_blocks` returns
/// `None` for any endpoint it does not model, the pair was skipped silently, and
/// the only global guard was `compared_pairs > 0` **across the entire corpus** —
/// so one well-covered provider kept the suite green while every other
/// provider's traffic went unexamined.
///
/// A request is now classified three ways rather than two (see
/// `cache_prefix::classify_endpoint`): modeled, non-conversational (embeddings,
/// image generation, audio, model listings — nothing with a prompt prefix to
/// protect), or *unmodeled*, which means a conversational endpoint the table
/// does not know about. Only the third is a finding, and it is a finding rather
/// than a skip.
#[derive(Default)]
struct Census {
    modeled: usize,
    non_conversational: usize,
    unmodeled: usize,
    unmodeled_paths: BTreeMap<String, usize>,
}

impl Census {
    fn conversational(&self) -> usize {
        self.modeled + self.unmodeled
    }

    fn unmodeled_fraction(&self) -> f64 {
        if self.conversational() == 0 {
            // No conversational traffic at all is itself a coverage failure:
            // every provider directory in this corpus records chat traffic.
            return 1.0;
        }
        self.unmodeled as f64 / self.conversational() as f64
    }
}

#[test]
fn every_provider_is_covered_by_the_prefix_check() {
    let root = cassette_root();

    for (provider, reason) in COVERAGE_EXEMPT_PROVIDERS {
        assert!(
            !reason.trim().is_empty(),
            "COVERAGE_EXEMPT_PROVIDERS entry `{provider}` needs a reason explaining why its \
             cache-bearing endpoint cannot be modeled"
        );
    }
    for (fragment, reason) in cache_prefix::NON_CONVERSATIONAL_ENDPOINTS {
        assert!(
            !reason.trim().is_empty(),
            "NON_CONVERSATIONAL_ENDPOINTS entry `{fragment}` needs a reason explaining why it \
             carries no cacheable prompt prefix"
        );
    }

    let mut census: BTreeMap<String, Census> = BTreeMap::new();

    for file in all_cassettes(&root) {
        let scenario = scenario_name(&root, &file);
        let provider = provider_of(&scenario).to_owned();
        let contents = std::fs::read_to_string(&file).expect("cassette should be readable");
        let entry = census.entry(provider).or_default();

        for (path, _) in recorded_requests(&contents) {
            match cache_prefix::classify_endpoint(&path) {
                EndpointKind::Modeled => entry.modeled += 1,
                EndpointKind::NotConversational => entry.non_conversational += 1,
                EndpointKind::Unmodeled => {
                    entry.unmodeled += 1;
                    *entry.unmodeled_paths.entry(path).or_default() += 1;
                }
            }
        }
    }

    assert!(
        !census.is_empty(),
        "no provider cassette directories were censused — the corpus layout has drifted"
    );

    let report = census
        .iter()
        .map(|(provider, stats)| {
            format!(
                "  {provider:<12} modeled={:<5} non-conversational={:<5} unmodeled={:<4} \
                 ({:.1}% of conversational traffic unmodeled)",
                stats.modeled,
                stats.non_conversational,
                stats.unmodeled,
                stats.unmodeled_fraction() * 100.0,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut failures = Vec::new();
    let mut used_exemptions = Vec::new();

    for (provider, stats) in &census {
        if let Some((exempt, _)) = COVERAGE_EXEMPT_PROVIDERS
            .iter()
            .find(|(exempt, _)| exempt == provider)
        {
            used_exemptions.push(*exempt);
            continue;
        }
        if stats.unmodeled_fraction() >= MAX_UNMODELED_FRACTION {
            failures.push(format!(
                "  {provider}: {:.1}% of its conversational requests speak an endpoint the cache \
                 prefix rule does not model, so this provider is effectively exempt from the \
                 check without saying so. Unmodeled endpoints: {:?}",
                stats.unmodeled_fraction() * 100.0,
                stats.unmodeled_paths.keys().collect::<Vec<_>>(),
            ));
        }
    }

    let stale = COVERAGE_EXEMPT_PROVIDERS
        .iter()
        .map(|(provider, _)| *provider)
        .filter(|provider| !used_exemptions.contains(provider))
        .collect::<Vec<_>>();
    assert!(
        stale.is_empty(),
        "stale COVERAGE_EXEMPT_PROVIDERS entries (the provider directory moved or was deleted; \
         delete the entry): {stale:?}\n\ncensus:\n{report}"
    );

    assert!(
        failures.is_empty(),
        "the cache prefix check does not actually cover every provider:\n{}\n\nAdd the endpoint \
         to `canonical_prefix_blocks` in tests/common/cache_prefix.rs (preferred — it is real \
         coverage), or, if it carries no cacheable prompt, to \
         `NON_CONVERSATIONAL_ENDPOINTS`, or, as a last resort, add the provider to \
         COVERAGE_EXEMPT_PROVIDERS with a reason.\n\nfull census:\n{report}",
        failures.join("\n")
    );
}

#[test]
fn prefix_blocks_do_not_iterate_a_single_object_field() {
    // Gemini's `systemInstruction` is one Content object, not a list. Iterating it
    // would reduce it to its keys and blind the check to content changes.
    let body = serde_json::json!({
        "systemInstruction": {"parts": [{"text": "be brief"}], "role": "model"},
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    });
    let blocks = cache_prefix::canonical_prefix_blocks(
        "/v1beta/models/gemini-2.5-flash:generateContent",
        &body,
    )
    .expect("generateContent should be modeled");

    assert_eq!(blocks.len(), 2, "{blocks:?}");
    assert_eq!(blocks[0].0, "systemInstruction");
    assert!(
        blocks[0].1.contains("be brief"),
        "the whole object must be one block: {blocks:?}"
    );
}

#[test]
fn a_moved_block_is_reported_and_an_appended_turn_is_not() {
    let turn_one = vec![("messages", "\"a\"".to_owned())];
    let appended = vec![
        ("messages", "\"a\"".to_owned()),
        ("messages", "\"b\"".to_owned()),
    ];
    assert!(cache_prefix::compare("s", 1, &turn_one, &appended).is_none());

    let rewritten = vec![("messages", "\"REWRITTEN\"".to_owned())];
    let violation = cache_prefix::compare("s", 1, &turn_one, &rewritten)
        .expect("a rewritten earlier block is a violation");
    assert_eq!(violation.block_index, 0);
    assert_eq!(violation.level, "messages");
}

#[test]
fn bedrock_converse_models_tools_system_and_messages_in_cache_order() {
    // Bedrock's Converse body was 100% unmodeled before this check was
    // hardened, so pin the shape rather than trusting the table by inspection.
    // `toolChoice` must stay *out*: it is a per-turn control, and folding it in
    // would report a legitimate tool-choice change as a prefix move.
    let body = serde_json::json!({
        "inferenceConfig": {"temperature": 0.0},
        "toolConfig": {
            "toolChoice": {"any": {}},
            "tools": [{"toolSpec": {"name": "add"}}],
        },
        "system": [{"text": "be brief"}],
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    });
    let blocks =
        cache_prefix::canonical_prefix_blocks("/model/amazon.nova-lite-v1%3A0/converse", &body)
            .expect("Bedrock Converse should be modeled");

    let levels = blocks.iter().map(|(level, _)| *level).collect::<Vec<_>>();
    assert_eq!(
        levels,
        vec!["toolConfig.tools", "system", "messages"],
        "{blocks:?}"
    );
    assert!(
        !blocks.iter().any(|(_, block)| block.contains("toolChoice")),
        "toolChoice is a per-turn control, not cached prompt content: {blocks:?}"
    );
}

#[test]
fn an_unknown_conversational_endpoint_is_a_finding_not_a_skip() {
    // The whole point of the coverage census: a shape nobody has modeled must
    // classify as a finding. If this ever returns `NotConversational`, the
    // non-conversational fragment list has grown too broad and the census has
    // gone back to failing open.
    assert_eq!(
        cache_prefix::classify_endpoint("/v3/some-future-chat-api"),
        EndpointKind::Unmodeled
    );
    assert_eq!(
        cache_prefix::classify_endpoint("/v1/embeddings"),
        EndpointKind::NotConversational
    );
    assert_eq!(
        cache_prefix::classify_endpoint("/v1/messages"),
        EndpointKind::Modeled
    );
}

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
        preamble: Some("You are a deterministic serialization probe.".to_owned()),
        chat_history: vec![Message::User {
            content: vec![UserContent::text("probe")],
        }],
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
            .map(|((index, _), _)| index)
            .unwrap_or_else(|| first.len().min(body.len()));
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

determinism_test!(xai_request_serialization_is_deterministic, "xai", |http| {
    rig::providers::xai::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .expect("xai client should build")
        .completion_model(rig::providers::xai::GROK_3_MINI)
});

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

#[test]
fn a_moving_cache_control_marker_is_not_a_prefix_move() {
    // Anthropic's documented incremental-caching pattern moves the conversation
    // breakpoint forward as the conversation grows. The marker is metadata
    // saying where to cache up to, not content being cached, and Anthropic's
    // prefix matching ignores it — measured: a probe whose marker moved exactly
    // this way still served over 80% of its grown turn from cache.
    //
    // Comparing raw bytes reported that correct behavior as a violation, and did
    // so *silently*: with the marker on `messages[0]`, the two turns looked like
    // two unrelated conversations and the pair was skipped entirely. Every
    // multi-turn Anthropic conversation using manual prompt caching was
    // invisible to this check.
    let turn_two = serde_json::json!({
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}],
        }],
    });
    let turn_three = serde_json::json!({
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "more", "cache_control": {"type": "ephemeral"}},
            ]},
        ],
    });

    let earlier = cache_prefix::canonical_prefix_blocks("/v1/messages", &turn_two)
        .expect("messages should be modeled");
    let later = cache_prefix::canonical_prefix_blocks("/v1/messages", &turn_three)
        .expect("messages should be modeled");

    assert!(
        cache_prefix::continues_the_same_conversation(&earlier, &later),
        "a moved breakpoint must not disguise a continuation as a new conversation: \
         {earlier:?} vs {later:?}"
    );
    assert!(
        cache_prefix::compare("s", 1, &earlier, &later).is_none(),
        "moving the breakpoint is not a content change: {earlier:?} vs {later:?}"
    );

    // The stripping must not blind the check to a real content change sitting
    // beside a marker.
    let rewritten = cache_prefix::canonical_prefix_blocks(
        "/v1/messages",
        &serde_json::json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "REWRITTEN"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ],
        }),
    )
    .expect("messages should be modeled");
    assert!(
        cache_prefix::compare("s", 1, &earlier, &rewritten).is_some(),
        "a rewritten earlier message is still a violation"
    );
}

/// Providers with a cassette directory but deliberately no cache scenario.
///
/// `(provider, reason)`, same contract as everything else in this file: an empty
/// reason is rejected, and an entry that stops matching a real provider
/// directory is reported as stale.
///
/// This list is the honest half of the matrix. Rig normalizes
/// `Usage::cached_input_tokens` for a dozen providers, and the point of the
/// conformance suite is that "we never checked" and "it does not cache" stop
/// being the same state. A provider that cannot be checked says why here.
const NO_CACHE_SUITE: &[(&str, &str)] = &[
    (
        "bedrock",
        "no usable credentials in this environment — the AWS_* variables are set but the session \
         token is rejected and AWS_PROFILE names a profile that does not exist, so the scenarios \
         cannot be recorded. Bedrock's Converse API does support prompt caching (`cachePoint` \
         blocks) and rig normalizes its usage, so this is an unrecorded gap rather than an \
         absence of caching",
    ),
    (
        "chatgpt",
        "OAuth-backed provider with no CHATGPT_ACCESS_TOKEN/CHATGPT_ACCOUNT_ID in this \
         environment, so its scenarios cannot be recorded",
    ),
    (
        "copilot",
        "OAuth-backed provider with no Copilot credentials in this environment, so its scenarios \
         cannot be recorded",
    ),
    (
        "mistralrs",
        "records against a local mistral.rs server that is not running in this environment",
    ),
    (
        "ollama",
        "Ollama's /api/chat usage payload carries no cached-token field of any kind, and rig's \
         Ollama provider therefore has no cache mapping to test — there is nothing for a cache \
         suite to assert",
    ),
    (
        "llamafile",
        "same as ollama: the local llama.cpp-compatible wire reports no cached-token field, so \
         rig has no cache mapping for it",
    ),
];

/// Every provider with cassettes must have a cache suite, or say why not.
#[test]
fn every_cassette_provider_has_a_cache_suite() {
    let root = cassette_root();

    for (provider, reason) in NO_CACHE_SUITE {
        assert!(
            !reason.trim().is_empty(),
            "NO_CACHE_SUITE entry `{provider}` needs a reason explaining why it has no cache \
             scenario"
        );
    }

    let mut providers: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&root).expect("cassette root should be readable") {
        let entry = entry.expect("cassette root entry should be readable");
        if entry.path().is_dir() {
            providers.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    providers.sort();
    assert!(
        !providers.is_empty(),
        "no provider cassette directories found"
    );

    let mut missing = Vec::new();
    let mut used_exemptions = Vec::new();

    for provider in &providers {
        let has_suite = root.join(provider).join("prompt_caching").is_dir();
        let exempt = NO_CACHE_SUITE
            .iter()
            .find(|(name, _)| name == provider)
            .map(|(name, _)| *name);

        match (has_suite, exempt) {
            (true, Some(name)) => {
                used_exemptions.push(name);
                missing.push(format!(
                    "  {provider}: has a prompt_caching/ directory *and* a NO_CACHE_SUITE entry — \
                     the suite exists now, so delete the entry"
                ));
            }
            (false, Some(name)) => used_exemptions.push(name),
            (true, None) => {}
            (false, None) => missing.push(format!(
                "  {provider}: has recorded cassettes but no tests/cassettes/{provider}/prompt_caching/ \
                 scenarios, so nothing has ever observed whether its prompt cache works"
            )),
        }
    }

    let stale = NO_CACHE_SUITE
        .iter()
        .map(|(provider, _)| *provider)
        .filter(|provider| !used_exemptions.contains(provider))
        .collect::<Vec<_>>();
    assert!(
        stale.is_empty(),
        "stale NO_CACHE_SUITE entries (the provider directory moved or was deleted; delete the \
         entry): {stale:?}"
    );

    assert!(
        missing.is_empty(),
        "a provider's prompt cache is unobserved:\n{}\n\nRecord a cache suite for it \
         (tests/common/cache_conformance.rs has the shared probe), or add it to NO_CACHE_SUITE \
         with a reason.",
        missing.join("\n")
    );
}
