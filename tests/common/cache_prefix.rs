//! The provider-cache wire-prefix rule, shared by the corpus scan and the
//! per-scenario conformance harness.
//!
//! Prompt caching is a **prefix match**: the cache key is derived from the exact
//! bytes up to each breakpoint, so any change to an earlier block invalidates
//! everything after it. A conversation whose turn-2 request rewrites, reorders,
//! or drops a block that turn-1 already sent therefore busts the cache on every
//! turn, and the cost shows up in production rather than in CI.
//!
//! Two callers apply the same rule to different inputs, and they must not drift:
//!
//! * [`tests/cassette_cache_prefix.rs`] sweeps every committed cassette, so the
//!   rule applies retroactively to the whole corpus at zero provider cost.
//! * [`crate::cache_conformance`] applies it to the three turns one cache probe
//!   just recorded, so a violation is reported by the scenario that caused it
//!   while the recording session is still open.
//!
//! A second, divergent copy of the flattening would be worse than no check —
//! the corpus scan would keep passing while the per-scenario check quietly
//! modeled a different shape. Hence this module.
//!
//! The invariant is ported from `inspirations/pydantic-ai`
//! (`tests/cassette_utils.py` `check_cache_prefix_stability`); the plumbing is
//! rig's.
#![allow(dead_code)]

use serde_json::Value;

/// One cache-ordered block of a request, tagged with the field it came from so a
/// violation report says *where* the prefix moved.
pub(crate) type PrefixBlock = (&'static str, String);

/// Flatten a provider request body into cache-ordered blocks.
///
/// Returns `None` for endpoints this check does not model — an unmodeled shape
/// is skipped loudly at the call site rather than silently treated as compliant.
///
/// The block order per endpoint is the order the provider renders the prompt in,
/// which is what the cache actually keys on. Getting that order wrong would make
/// the check compare blocks that never sit next to each other on the wire.
pub(crate) fn canonical_prefix_blocks(path: &str, body: &Value) -> Option<Vec<PrefixBlock>> {
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
    } else if path.ends_with("/v2/chat") {
        // Cohere Chat v2. Tools render ahead of the message list; Cohere carries
        // its system prompt as a `system`-role entry inside `messages`, so there
        // is no separate instruction field to model.
        add("tools", body.get("tools"));
        add("messages", body.get("messages"));
    } else if path.contains("/converse") {
        // Bedrock Converse (`/model/<id>/converse`, `/converse-stream`).
        //
        // Tools sit under `toolConfig.tools`, and only that sub-field is taken:
        // `toolConfig` also carries `toolChoice`, which is a per-turn control
        // rather than cached prompt content, and folding it in would report a
        // legitimate tool-choice change as a prefix move.
        add(
            "toolConfig.tools",
            body.get("toolConfig")
                .and_then(|config| config.get("tools")),
        );
        add("system", body.get("system"));
        add("messages", body.get("messages"));
    } else if path.ends_with("/api/chat") {
        // Ollama. Carries its system prompt as a `system`-role entry inside
        // `messages`, like the OpenAI-compatible wires.
        add("tools", body.get("tools"));
        add("messages", body.get("messages"));
    } else {
        return None;
    }

    Some(blocks)
}

/// Endpoints that carry no conversation, and therefore no cacheable prompt
/// prefix worth protecting.
///
/// This list is what lets the per-provider coverage census
/// (`tests/cassette_cache_prefix.rs`) **fail closed**. Without it, the census
/// could not tell "this provider's chat endpoint is invisible to the check" from
/// "this provider records a lot of embeddings calls", so every unmodeled shape
/// had to be tolerated. With it, an unmodeled endpoint that is *not* on this
/// list is a finding: either model it or record why it cannot be modeled.
///
/// `(path fragment, reason)`. The reason is required for the same purpose it is
/// required in `MOVES_CACHE_PREFIX`: a bare entry records that someone silenced
/// the census, not why that was correct.
pub(crate) const NON_CONVERSATIONAL_ENDPOINTS: &[(&str, &str)] = &[
    (
        // Covers both spellings in the corpus: OpenAI-style `/v1/embeddings`
        // and Cohere's `/v1/embed`.
        "/embed",
        "embedding requests carry a bare input array, not a conversation — there is no \
         multi-turn prefix for a provider to reuse",
    ),
    (
        "/images/generations",
        "image generation takes a single prompt string and returns pixels; no prompt cache \
         applies",
    ),
    (
        // Venice's spelling of the same thing.
        "/image/generate",
        "image generation takes a single prompt string and returns pixels; no prompt cache \
         applies",
    ),
    (
        "/audio/transcriptions",
        "audio transcription uploads a multipart audio file rather than a chat prefix",
    ),
    (
        "/audio/speech",
        "text-to-speech takes a single input string and returns audio bytes",
    ),
    (
        "/models",
        "model listing is metadata, carrying no prompt at all",
    ),
    (
        "/api/tags",
        "Ollama's local model inventory endpoint, carrying no prompt at all",
    ),
    (
        "/invoke",
        "Bedrock's `/model/<id>/invoke` is used here only for Titan embeddings, whose body is an \
         input string rather than a conversation",
    ),
    (
        "/files",
        "file upload/download carries document bytes, not a chat prefix",
    ),
];

/// How the prefix rule sees one recorded request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum EndpointKind {
    /// A conversational endpoint the table models.
    Modeled,
    /// A non-conversational endpoint — see [`NON_CONVERSATIONAL_ENDPOINTS`].
    NotConversational,
    /// A conversational endpoint the table does *not* model. A finding.
    Unmodeled,
}

/// Classify one recorded request path.
pub(crate) fn classify_endpoint(path: &str) -> EndpointKind {
    if endpoint_is_modeled(path) {
        return EndpointKind::Modeled;
    }
    if NON_CONVERSATIONAL_ENDPOINTS
        .iter()
        .any(|(fragment, _)| path.contains(fragment))
    {
        return EndpointKind::NotConversational;
    }
    EndpointKind::Unmodeled
}

/// Whether this check models the endpoint `path` speaks.
///
/// Layer 0's coverage census and the per-scenario harness both need to
/// distinguish "compared and clean" from "never looked at", which is the
/// distinction the original fail-open skip erased.
pub(crate) fn endpoint_is_modeled(path: &str) -> bool {
    canonical_prefix_blocks(path, &Value::Null).is_some()
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
pub(crate) fn continues_the_same_conversation(
    earlier: &[PrefixBlock],
    later: &[PrefixBlock],
) -> bool {
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

pub(crate) struct Violation {
    pub(crate) scenario: String,
    pub(crate) pair: usize,
    pub(crate) level: &'static str,
    pub(crate) block_index: usize,
    pub(crate) earlier: String,
    pub(crate) later: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} [{}] request pair {}, block {}:\n  earlier: {}\n  later:   {}",
            self.scenario, self.level, self.pair, self.block_index, self.earlier, self.later
        )
    }
}

/// The earlier request's blocks must be a prefix of the later request's.
pub(crate) fn compare(
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

pub(crate) fn truncate(block: &str) -> String {
    const LIMIT: usize = 220;
    if block.chars().count() <= LIMIT {
        return block.to_owned();
    }
    let head: String = block.chars().take(LIMIT).collect();
    format!("{head}…")
}
