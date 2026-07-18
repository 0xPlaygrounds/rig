//! Shared helpers for projecting A2A payloads into Rig text.
//!
//! Both consumers of a remote agent — the [`tool`](crate::tool) bridge and the
//! [`A2AModel`](crate::model::A2AModel) completion model — collapse agent
//! responses into a single string, so non-text parts (Data, Url) are
//! stringified into the joined buffer with a separator. Binary `Raw` parts are
//! skipped with a debug log because they are not routable to text-only Rig
//! agents.
//!
//! Everything here is a free function over borrowed A2A types: the projection
//! is identical whichever surface consumes it, and only the *classification* of
//! a task state (tool error kind vs completion error) differs per surface.

use a2a::{Artifact, Message, Part, PartContent, Task, TaskState};

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
        // `prev_kind` is `None` only while `out` is still empty, so the
        // emptiness check alone covers the first chunk.
        if !out.is_empty() && !matches!((prev_kind, kind), (Some(Kind::Text), Kind::Text)) {
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

/// Provider-safe Rig tool name derived from a remote agent's card name.
///
/// One remote agent projects to exactly one Rig tool, so the name is the
/// slugged card name capped at a provider-safe length. Slugs are ASCII, so byte
/// truncation is safe.
///
/// The projection is not injective: two remote agents that slug to the same
/// name collide, and Rig's tool registry resolves a collision by replacing the
/// earlier tool. Callers that connect to several same-named agents should set
/// an explicit name with
/// [`A2AClientBuilder::tool_name`](crate::A2AClientBuilder::tool_name).
pub(crate) fn tool_name(agent_name: &str) -> String {
    const MAX_LEN: usize = 64;

    let mut agent = slug(agent_name);
    agent.truncate(MAX_LEN);
    agent
}

/// Wire-format label for a task state, used in the state prefixes that mark
/// non-completed tool output and in error messages.
pub(crate) fn state_label(state: &TaskState) -> &'static str {
    match state {
        TaskState::Unspecified => "unspecified",
        TaskState::Submitted => "submitted",
        TaskState::Working => "working",
        TaskState::Completed => "completed",
        TaskState::Failed => "failed",
        TaskState::Canceled => "canceled",
        TaskState::InputRequired => "input-required",
        TaskState::Rejected => "rejected",
        TaskState::AuthRequired => "auth-required",
    }
}

/// Text carried by a task's status message, if any.
pub(crate) fn status_text_limited(task: &Task, limit: usize) -> Result<String, A2AError> {
    task.status
        .message
        .as_ref()
        .map(|message| parts_to_text_limited(&message.parts, limit, "tool response"))
        .transpose()
        .map(Option::unwrap_or_default)
}

/// Every artifact's text followed by the status message's text.
///
/// This is the payload of a task, with no state markers: callers decide how to
/// label or classify it. Artifacts come first because the status message of a
/// completed task is usually a closing remark rather than the result.
pub(crate) fn task_body_limited(task: &Task, limit: usize) -> Result<String, A2AError> {
    let mut out = String::new();
    if let Some(artifacts) = &task.artifacts {
        for artifact in artifacts {
            push_artifact(&mut out, artifact, limit)?;
        }
    }
    let status = status_text_limited(task, limit)?;
    if !status.is_empty() {
        if !out.is_empty() {
            push_limited(&mut out, "\n", Some((limit, "tool response")))?;
        }
        push_limited(&mut out, &status, Some((limit, "tool response")))?;
    }
    Ok(out)
}

/// Text carried by a bare `Message` response.
pub(crate) fn message_body_limited(message: &Message, limit: usize) -> Result<String, A2AError> {
    parts_to_text_limited(&message.parts, limit, "tool response")
}

fn push_artifact(out: &mut String, artifact: &Artifact, limit: usize) -> Result<(), A2AError> {
    let text = parts_to_text_limited(&artifact.parts, limit, "tool response")?;
    if text.is_empty() {
        return Ok(());
    }
    if !out.is_empty() {
        push_limited(out, "\n", Some((limit, "tool response")))?;
    }
    push_limited(out, &text, Some((limit, "tool response")))
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
    fn tool_name_is_slugged_card_name_capped_at_64() {
        assert_eq!(tool_name("Remote Agent"), "remote-agent");
        assert_eq!(tool_name("!!!"), "agent");

        let long = tool_name(&"x".repeat(200));
        assert_eq!(long.len(), 64);
        assert!(
            long.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "{long}"
        );
    }

    #[test]
    fn state_labels_are_wire_shaped() {
        assert_eq!(state_label(&TaskState::InputRequired), "input-required");
        assert_eq!(state_label(&TaskState::AuthRequired), "auth-required");
        assert_eq!(state_label(&TaskState::Completed), "completed");
    }

    fn task_with(artifacts: Vec<&str>, status: Option<&str>) -> Task {
        use a2a::{Role, TaskStatus, new_artifact_id, new_context_id, new_task_id};
        Task {
            id: new_task_id(),
            context_id: new_context_id(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: status.map(|t| Message::new(Role::Agent, vec![Part::text(t.to_string())])),
                timestamp: None,
            },
            artifacts: (!artifacts.is_empty()).then(|| {
                artifacts
                    .into_iter()
                    .map(|text| Artifact {
                        artifact_id: new_artifact_id(),
                        name: None,
                        description: None,
                        parts: vec![Part::text(text.to_string())],
                        metadata: None,
                        extensions: None,
                    })
                    .collect()
            }),
            history: None,
            metadata: None,
        }
    }

    #[test]
    fn task_body_joins_artifacts_then_status() {
        let task = task_with(vec!["first", "second"], Some("all done"));
        assert_eq!(
            task_body_limited(&task, usize::MAX).unwrap(),
            "first\nsecond\nall done"
        );
    }

    #[test]
    fn task_body_is_empty_without_content() {
        let task = task_with(vec![], None);
        assert_eq!(task_body_limited(&task, usize::MAX).unwrap(), "");
    }

    #[test]
    fn task_body_respects_the_limit() {
        let task = task_with(vec!["0123456789"], None);
        assert!(matches!(
            task_body_limited(&task, 4),
            Err(A2AError::PayloadTooLarge { limit: 4, .. })
        ));
    }
}
