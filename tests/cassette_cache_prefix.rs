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
//! 1313-cassette corpus. The invariant is imported; the plumbing is rig's.
//!
//! A test whose behavior is *deliberately* prefix-moving — compaction, history
//! rewriting, dynamic tool disclosure — records that fact in
//! `MOVES_CACHE_PREFIX`, with a reason. An entry with an empty reason is
//! rejected, and an entry that stops matching a real cassette is reported as
//! stale.

#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

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

/// One cache-ordered block of a request, tagged with the field it came from so a
/// violation report says *where* the prefix moved.
type PrefixBlock = (&'static str, String);

/// Flatten a provider request body into cache-ordered blocks.
///
/// Returns `None` for endpoints this check does not model — an unmodeled shape
/// is skipped loudly at the call site rather than silently treated as compliant.
fn canonical_prefix_blocks(path: &str, body: &Value) -> Option<Vec<PrefixBlock>> {
    let mut blocks: Vec<PrefixBlock> = Vec::new();

    // A field may hold a list of blocks or a single one: Anthropic's `system` can
    // be a plain string, and Gemini's `systemInstruction` is one Content *object*.
    // Iterating that object would silently reduce it to its keys and blind the
    // check to content changes — pydantic-ai hit exactly this and documented it.
    let mut add = |level: &'static str, value: Option<&Value>| {
        let Some(value) = value else { return };
        if value.is_null() {
            return;
        }
        match value.as_array() {
            Some(items) => {
                for item in items {
                    blocks.push((level, item.to_string()));
                }
            }
            None => blocks.push((level, value.to_string())),
        }
    };

    if path.ends_with("/v1/messages") {
        // Anthropic Messages. `tools` renders before `system`, which renders
        // before `messages`.
        add("tools", body.get("tools"));
        add("system", body.get("system"));
        add("messages", body.get("messages"));
    } else if path.contains(":generateContent") || path.contains(":streamGenerateContent") {
        add("tools", body.get("tools"));
        add("systemInstruction", body.get("systemInstruction"));
        add("contents", body.get("contents"));
    } else if path.contains("/interactions") {
        add("tools", body.get("tools"));
        add("system_instruction", body.get("system_instruction"));
        add("input", body.get("input"));
    } else if path.ends_with("/chat/completions") {
        add("tools", body.get("tools"));
        add("messages", body.get("messages"));
    } else if path.ends_with("/responses") {
        add("tools", body.get("tools"));
        add("instructions", body.get("instructions"));
        add("input", body.get("input"));
    } else {
        return None;
    }

    Some(blocks)
}

/// The message-carrying level for each supported endpoint — the blocks that
/// identify *which conversation* a request belongs to.
const CONVERSATION_LEVELS: &[&str] = &["messages", "contents", "input"];

/// Whether `later` continues the conversation `earlier` started, rather than
/// being an unrelated request that happens to share the cassette and endpoint.
///
/// Many cassettes record several *independent* single-turn requests — a batch
/// extractor run over different texts, a document test asking about page 2 and
/// then page 1. Those share no prefix by design and comparing them is
/// meaningless: a different opening turn is a different cache entry, so there is
/// nothing for the provider to reuse and nothing for this check to protect.
///
/// Identity is the **first message-level block**. Tools and system prompts are
/// deliberately excluded from the identity test: a turn that changes its tool set
/// while continuing the same conversation is exactly the prefix move this check
/// exists to catch, so it must not be able to disguise itself as a new
/// conversation.
fn continues_the_same_conversation(earlier: &[PrefixBlock], later: &[PrefixBlock]) -> bool {
    // On OpenAI-compatible wires the system prompt is itself a `messages` entry,
    // and it is identical across unrelated runs that share a preamble (a batch
    // extractor over different texts). Keying identity on it would merge those
    // into one conversation, so identity is the first message that is *not* a
    // system/developer instruction — the conversation's opening turn.
    let first_message = |blocks: &[PrefixBlock]| {
        blocks
            .iter()
            .filter(|(level, _)| CONVERSATION_LEVELS.contains(level))
            .find(|(_, block)| {
                serde_json::from_str::<Value>(block)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("role")
                            .and_then(Value::as_str)
                            .map(|role| !matches!(role, "system" | "developer"))
                    })
                    // A block with no `role` (Gemini `contents` entries carry one;
                    // Responses `input` items may not) still identifies the turn.
                    .unwrap_or(true)
            })
            .map(|(_, block)| block.clone())
    };

    let message_count = |blocks: &[PrefixBlock]| {
        blocks
            .iter()
            .filter(|(level, _)| CONVERSATION_LEVELS.contains(level))
            .count()
    };

    match (first_message(earlier), first_message(later)) {
        (Some(earlier_first), Some(later_first)) => {
            // A continuation *grew*: the assistant turn and its tool result were
            // appended. An independent repeat — the same opening turn re-sent with
            // a different tool schema, as a batch extractor does — keeps the same
            // message count and shares no cache prefix worth protecting.
            earlier_first == later_first && message_count(later) > message_count(earlier)
        }
        // No message-level blocks at all: nothing to correlate on, so do not
        // invent a comparison.
        _ => false,
    }
}

struct Violation {
    scenario: String,
    pair: usize,
    level: &'static str,
    block_index: usize,
    earlier: String,
    later: String,
}

/// The earlier request's blocks must be a prefix of the later request's.
fn compare(
    scenario: &str,
    pair: usize,
    earlier: &[PrefixBlock],
    later: &[PrefixBlock],
) -> Option<Violation> {
    for (index, (earlier_level, earlier_block)) in earlier.iter().enumerate() {
        let Some((later_level, later_block)) = later.get(index) else {
            return Some(Violation {
                scenario: scenario.to_owned(),
                pair,
                level: earlier_level,
                block_index: index,
                earlier: truncate(earlier_block),
                later: "<dropped: the later request is shorter>".to_owned(),
            });
        };
        if earlier_level != later_level || earlier_block != later_block {
            return Some(Violation {
                scenario: scenario.to_owned(),
                pair,
                level: earlier_level,
                block_index: index,
                earlier: truncate(earlier_block),
                later: truncate(later_block),
            });
        }
    }
    None
}

fn truncate(block: &str) -> String {
    const LIMIT: usize = 220;
    if block.chars().count() <= LIMIT {
        return block.to_owned();
    }
    let head: String = block.chars().take(LIMIT).collect();
    format!("{head}…")
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

#[test]
fn recorded_conversations_do_not_move_their_cache_prefix() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/cassettes");
    assert!(
        root.is_dir(),
        "cassette root moved or vanished: {}",
        root.display()
    );

    for (path, reason) in MOVES_CACHE_PREFIX {
        assert!(
            !reason.trim().is_empty(),
            "MOVES_CACHE_PREFIX entry `{path}` needs a reason explaining why moving the \
             cache prefix is correct for that scenario"
        );
    }

    let mut files = Vec::new();
    cassette_files(&root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no cassettes found under {}",
        root.display()
    );

    let mut violations: Vec<Violation> = Vec::new();
    let mut exempted = Vec::new();
    let mut compared_pairs = 0usize;

    for file in &files {
        let scenario = file
            .strip_prefix(&root)
            .expect("cassette should live under the cassette root")
            .to_string_lossy()
            .replace('\\', "/");

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
            let Some(blocks) = canonical_prefix_blocks(path, body) else {
                previous = None;
                continue;
            };
            if let Some((previous_path, previous_blocks)) = previous.take()
                && previous_path == *path
                && continues_the_same_conversation(&previous_blocks, &blocks)
            {
                compared_pairs += 1;
                if let Some(violation) = compare(&scenario, index, &previous_blocks, &blocks) {
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
            .map(|violation| format!(
                "{} [{}] request pair {}, block {}:\n  earlier: {}\n  later:   {}",
                violation.scenario,
                violation.level,
                violation.pair,
                violation.block_index,
                violation.earlier,
                violation.later
            ))
            .collect::<Vec<_>>()
            .join("\n\n")
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
    let blocks = canonical_prefix_blocks("/v1beta/models/gemini-2.5-flash:generateContent", &body)
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
    assert!(compare("s", 1, &turn_one, &appended).is_none());

    let rewritten = vec![("messages", "\"REWRITTEN\"".to_owned())];
    let violation =
        compare("s", 1, &turn_one, &rewritten).expect("a rewritten earlier block is a violation");
    assert_eq!(violation.block_index, 0);
    assert_eq!(violation.level, "messages");
}
