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
//! # Two checks, two different blind spots
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
    let blocks =
        cache_prefix::canonical_prefix_blocks("/v1beta/models/gemini-2.5-flash:generateContent", &body)
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
