//! Shared helpers for projecting between A2A `Part`s and Rig text.
//!
//! These are intentionally minimal: the client tool collapses agent responses
//! into a single string, so non-text parts (Data, Url) are stringified into
//! the joined buffer with a separator. Binary `Raw` parts are skipped with a
//! debug log because they are not routable to text-only Rig agents.

use a2a::{Part, PartContent};

use crate::error::A2AError;

/// Default maximum size for text projected from A2A parts.
pub(crate) const DEFAULT_TEXT_LIMIT: usize = 1024 * 1024;

/// Concatenate the text content of a sequence of `Part`s.
///
/// Each part contributes its content:
/// - `Text` writes its content verbatim.
/// - `Data` writes its JSON serialisation.
/// - `Url` writes the URL.
/// - `Raw` is skipped (binary payloads aren't routable to a text agent).
///
/// A single `\n` is inserted between non-adjacent text chunks. Adjacent
/// `Text` chunks concatenate without a separator (callers commonly split
/// a longer prompt across multiple Text parts), while every `Data` / `Url`
/// boundary inserts a newline so e.g. `{"a":1}{"b":2}` is not emitted as
/// one fused token.
#[cfg(test)]
pub(crate) fn parts_to_text(parts: &[Part]) -> String {
    match parts_to_text_inner(parts, None) {
        Ok(text) => text,
        Err(err) => panic!("unlimited projection failed: {err}"),
    }
}

/// Concatenate the text content of a sequence of `Part`s, failing before the
/// output exceeds `limit` bytes.
pub(crate) fn parts_to_text_limited(
    parts: &[Part],
    limit: usize,
    what: &'static str,
) -> Result<String, A2AError> {
    parts_to_text_inner(parts, Some((limit, what)))
}

fn parts_to_text_inner(
    parts: &[Part],
    limit: Option<(usize, &'static str)>,
) -> Result<String, A2AError> {
    #[derive(PartialEq, Eq, Clone, Copy)]
    enum Kind {
        Text,
        Data,
        Url,
    }
    let mut out = String::new();
    let mut prev_kind: Option<Kind> = None;
    for part in parts {
        let (kind, chunk): (Kind, std::borrow::Cow<'_, str>) = match &part.content {
            PartContent::Text(text) => (Kind::Text, text.as_str().into()),
            PartContent::Data(value) => (Kind::Data, value.to_string().into()),
            PartContent::Url(url) => (Kind::Url, url.as_str().into()),
            PartContent::Raw(raw) => {
                tracing::debug!(
                    target: "rig_a2a",
                    bytes = raw.len(),
                    "skipping binary A2A part while projecting to text"
                );
                continue;
            }
        };
        if chunk.is_empty() {
            continue;
        }
        if !matches!(
            (prev_kind, kind),
            (Some(Kind::Text), Kind::Text) | (None, _)
        ) && !out.is_empty()
        {
            push_limited(&mut out, "\n", limit)?;
        }
        push_limited(&mut out, &chunk, limit)?;
        prev_kind = Some(kind);
    }
    Ok(out)
}

pub(crate) fn push_limited(
    out: &mut String,
    chunk: &str,
    limit: Option<(usize, &'static str)>,
) -> Result<(), A2AError> {
    if let Some((limit, what)) = limit
        && out.len().saturating_add(chunk.len()) > limit
    {
        return Err(A2AError::PayloadTooLarge { what, limit });
    }
    out.push_str(chunk);
    Ok(())
}

/// Canonical slug used for Rig tool names.
///
/// Lowercase ASCII alphanumerics are preserved; `_` and `-` are preserved;
/// whitespace becomes `-`; everything else is dropped. The empty string
/// becomes `"agent"`.
pub(crate) fn slug(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if ch == '_' || ch == '-' {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('-');
        }
    }
    if out.is_empty() {
        out.push_str("agent");
    }
    out
}

/// Stable, short lowercase hex hash (8 chars) for lossy name projections.
///
/// This is standard 32-bit FNV-1a
/// (<https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function>).
/// Names derived from it must be stable across processes and Rust releases —
/// they are re-derived on every card fetch — which rules out
/// `std::hash::DefaultHasher`, and FNV-1a is the simplest well-known
/// dependency-free alternative. It is not security-sensitive: the hash only
/// disambiguates sibling skill ids after lossy slugging, where a handful of
/// names makes accidental 32-bit collisions vanishingly unlikely.
pub(crate) fn short_hash(input: &str) -> String {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in input.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    format!("{hash:08x}")
}

/// Provider-safe Rig tool name derived from remote A2A identity.
///
/// The name is a slug of the agent name and skill id plus a short stable hash
/// of the original skill id, so two distinct skill ids always produce distinct
/// tool names even after lossy slugging, and the result stays within
/// provider-safe length and character limits.
pub(crate) fn tool_name(agent_name: &str, skill_id: Option<&str>) -> String {
    const MAX_LEN: usize = 64;

    let mut agent = slug(agent_name);
    let Some(skill_id) = skill_id else {
        agent.truncate(MAX_LEN);
        return agent;
    };

    let mut skill = slug(skill_id);
    let hash = short_hash(skill_id);
    // Layout: `{agent}__{skill}_{hash}`. Split the byte budget left after the
    // separators and hash between the two slugs, keeping the skill slug whole
    // when possible and always at least one character of each. Slugs are
    // ASCII, so byte truncation is safe.
    let budget = MAX_LEN - (2 + 1 + hash.len());
    skill.truncate(budget - 1);
    agent.truncate(budget - skill.len());
    format!("{agent}__{skill}_{hash}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parts_to_text_separates_with_newlines() {
        let parts = vec![
            Part::text("hi"),
            Part {
                content: PartContent::Data(json!({"foo": 1})),
                filename: None,
                media_type: None,
                metadata: None,
            },
            Part {
                content: PartContent::Url("https://example.com".into()),
                filename: None,
                media_type: None,
                metadata: None,
            },
        ];
        assert_eq!(
            parts_to_text(&parts),
            "hi\n{\"foo\":1}\nhttps://example.com"
        );
    }

    #[test]
    fn parts_to_text_skips_raw_and_concatenates_adjacent_text() {
        // Adjacent Text parts (with a Raw part skipped between them) are
        // a single Text run — no separator inserted.
        let parts = vec![
            Part::text("hi"),
            Part {
                content: PartContent::Raw(b"binary".to_vec()),
                filename: None,
                media_type: None,
                metadata: None,
            },
            Part::text("bye"),
        ];
        assert_eq!(parts_to_text(&parts), "hibye");
    }

    #[test]
    fn parts_to_text_text_runs_concatenate_then_kind_transitions_separate() {
        // Text + Text + Data: the two Texts concatenate, the Text->Data
        // transition inserts a newline.
        let parts = vec![
            Part::text("Translate to French: "),
            Part::text("hello"),
            Part {
                content: PartContent::Data(json!({"meta": "x"})),
                filename: None,
                media_type: None,
                metadata: None,
            },
        ];
        assert_eq!(
            parts_to_text(&parts),
            "Translate to French: hello\n{\"meta\":\"x\"}"
        );
    }

    #[test]
    fn parts_to_text_empty_input() {
        assert_eq!(parts_to_text(&[]), "");
    }

    #[test]
    fn parts_to_text_limited_fails_past_limit() {
        let parts = vec![Part::text("0123456789")];
        let err = parts_to_text_limited(&parts, 4, "test payload")
            .expect_err("projection over the limit must fail");
        assert!(matches!(err, A2AError::PayloadTooLarge { limit: 4, .. }));
    }

    #[test]
    fn slug_lowercases_and_separates() {
        assert_eq!(slug("Hello World"), "hello-world");
        assert_eq!(slug("agent-1"), "agent-1");
        assert_eq!(slug("Skill_42"), "skill_42");
        assert_eq!(slug("!!!"), "agent");
        assert_eq!(slug(""), "agent");
        assert_eq!(slug("héllo"), "hllo");
    }

    #[test]
    fn parts_to_text_separates_adjacent_data_and_urls() {
        let parts = vec![
            Part {
                content: PartContent::Data(json!({"a": 1})),
                filename: None,
                media_type: None,
                metadata: None,
            },
            Part {
                content: PartContent::Data(json!({"b": 2})),
                filename: None,
                media_type: None,
                metadata: None,
            },
            Part {
                content: PartContent::Url("https://a.example".into()),
                filename: None,
                media_type: None,
                metadata: None,
            },
            Part {
                content: PartContent::Url("https://b.example".into()),
                filename: None,
                media_type: None,
                metadata: None,
            },
        ];
        assert_eq!(
            parts_to_text(&parts),
            "{\"a\":1}\n{\"b\":2}\nhttps://a.example\nhttps://b.example"
        );
    }

    #[test]
    fn short_hash_is_fnv1a32() {
        // Known FNV-1a 32-bit test vectors; pins the hash so tool names stay
        // stable across releases.
        assert_eq!(short_hash(""), "811c9dc5");
        assert_eq!(short_hash("hello"), "4f9f2cab");
    }

    #[test]
    fn projected_tool_names_are_provider_safe_and_disambiguated() {
        let a = tool_name("Remote Agent", Some("do thing"));
        let b = tool_name("Remote Agent", Some("do-thing"));
        assert_ne!(a, b);
        for name in [a, b] {
            assert!(name.len() <= 64, "{name}");
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "{name}"
            );
        }
    }

    #[test]
    fn long_names_are_truncated_but_stay_distinct() {
        let long_skill = "x".repeat(200);
        let other_skill = format!("{}y", "x".repeat(199));
        let a = tool_name("Remote Agent", Some(&long_skill));
        let b = tool_name("Remote Agent", Some(&other_skill));
        assert_ne!(a, b);
        assert!(a.len() <= 64, "{a}");
        assert!(b.len() <= 64, "{b}");
    }
}
