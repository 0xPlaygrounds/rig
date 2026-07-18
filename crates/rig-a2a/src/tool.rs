//! Projection of a remote A2A agent into a single Rig
//! [`DynamicTool`](rig_agent::tool::DynamicTool).
//!
//! One remote agent becomes one tool. The A2A protocol routes every request
//! through the same `message/send` endpoint and does not carry a skill
//! selector, so the skills a card declares are not separate callable
//! endpoints — they are documentation. They are rendered into the tool's
//! description, exactly as [`Agent::into_tool`] renders a sub-agent's identity
//! and system prompt.
//!
//! [`Agent::into_tool`]: rig_agent::agent::Agent::into_tool

use a2a::{AgentCard, AgentSkill, SendMessageResponse, TaskState};
use rig_agent::tool::ToolExecutionError;
use serde_json::Value;

use crate::error::A2AError;
use crate::parts::{
    DEFAULT_TEXT_LIMIT, message_body_limited, state_label, status_text_limited, task_body_limited,
};

/// Budget for the rendered tool description.
///
/// An `AgentCard` is remote-controlled, and its description is cloned into
/// every completion request, so an agent advertising hundreds of verbose skills
/// would otherwise inflate every turn. Rendering truncates rather than fails:
/// a long card should still yield a usable tool.
pub(crate) const DESCRIPTION_LIMIT: usize = 8 * 1024;

/// Longest single skill rendering before it is elided.
const SKILL_LIMIT: usize = 512;

/// Room reserved inside [`DESCRIPTION_LIMIT`] for the "N further skill(s)
/// omitted" note, so a truncated skill list never renders as a complete one.
const NOTE_BUDGET: usize = 64;

/// Output for a completed task that carried neither artifacts nor a closing
/// message. Providers handle empty tool-result blocks inconsistently, so the
/// tool always returns something.
const EMPTY_COMPLETION: &str = "[a2a completed] the remote agent returned no content.";

/// Errors validating the JSON arguments passed to the A2A tool.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ToolArgsError {
    #[error("tool arguments must be a JSON object")]
    NotAnObject,
    #[error("tool arguments must include a string `prompt` field")]
    MissingPrompt,
    #[error("the `prompt` field must not be blank")]
    BlankPrompt,
}

/// The A2A tool's parameter schema is a constant, so build it once and clone it
/// rather than re-allocating the JSON tree on every `definition()` call.
pub(crate) fn parameters_schema() -> Value {
    static SCHEMA: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    SCHEMA
        .get_or_init(|| {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Natural-language instruction for the remote A2A agent. State the whole task; the remote agent decides which of its skills to apply."
                    }
                },
                "required": ["prompt"],
                "additionalProperties": false
            })
        })
        .clone()
}

/// Extract the `prompt` from the tool's JSON arguments.
///
/// Unknown extra fields are ignored: the schema declares
/// `additionalProperties: false`, but a model may still emit them, and
/// rejecting the call over a stray key wastes a turn. A blank prompt *is*
/// rejected: there is nothing for the remote to act on, so sending it would
/// spend a request to learn what the argument check already knows.
pub(crate) fn parse_args(value: Value) -> Result<String, ToolArgsError> {
    let Value::Object(mut map) = value else {
        return Err(ToolArgsError::NotAnObject);
    };
    let prompt = map
        .remove("prompt")
        .and_then(|value| value.as_str().map(str::to_string))
        .ok_or(ToolArgsError::MissingPrompt)?;
    if prompt.trim().is_empty() {
        return Err(ToolArgsError::BlankPrompt);
    }
    Ok(prompt)
}

/// Render a remote agent's card as tool documentation.
///
/// Mirrors the shape [`Agent::into_tool`] uses for sub-agents: what the tool is
/// for, then the agent's identity, then what it can do. Skills appear as a list
/// because they describe the remote's capabilities without being separately
/// callable.
///
/// [`Agent::into_tool`]: rig_agent::agent::Agent::into_tool
pub(crate) fn describe_card(card: &AgentCard) -> String {
    let mut out = String::new();
    out.push_str("Prompt a remote A2A agent to do a task for you.\n\n");
    out.push_str(&format!("Agent name: {}\n", card.name));
    out.push_str(&format!("Agent description: {}\n", card.description));

    if card.skills.is_empty() {
        return truncated(out, DESCRIPTION_LIMIT);
    }

    out.push_str("Skills:\n");
    // Stop at the first skill that does not fit rather than cherry-picking
    // later short ones, so the rendered list stays a contiguous prefix of the
    // card and the omission note is accurate. Every skill but the last reserves
    // room for that note, so a truncated list always says it was truncated.
    let mut rendered_count = 0usize;
    for (index, skill) in card.skills.iter().enumerate() {
        let rendered = truncated(describe_skill(skill), SKILL_LIMIT);
        let reserved = if index + 1 == card.skills.len() {
            0
        } else {
            NOTE_BUDGET
        };
        if out.len() + rendered.len() + reserved > DESCRIPTION_LIMIT {
            break;
        }
        out.push_str(&rendered);
        rendered_count += 1;
    }
    let elided = card.skills.len() - rendered_count;
    if elided > 0 {
        tracing::warn!(
            target: "rig_a2a",
            agent = %card.name,
            elided,
            limit = DESCRIPTION_LIMIT,
            "remote agent card exceeds the tool description budget; some skills were omitted"
        );
        out.push_str(&truncated(
            format!("  … and {elided} further skill(s) omitted.\n"),
            NOTE_BUDGET,
        ));
    }
    truncated(out, DESCRIPTION_LIMIT)
}

fn describe_skill(skill: &AgentSkill) -> String {
    let mut line = format!("  - {} ({}): {}", skill.name, skill.id, skill.description);
    if !skill.tags.is_empty() {
        line.push_str(&format!(" [tags: {}]", skill.tags.join(", ")));
    }
    if let Some(examples) = skill.examples.as_ref().filter(|e| !e.is_empty()) {
        line.push_str(&format!(" [e.g. {}]", examples.join("; ")));
    }
    line.push('\n');
    line
}

/// Truncate on a character boundary, marking that content was dropped.
fn truncated(mut text: String, limit: usize) -> String {
    const MARKER: &str = "…";
    if text.len() <= limit {
        return text;
    }
    let mut cut = limit.saturating_sub(MARKER.len());
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    text.truncate(cut);
    text.push_str(MARKER);
    text
}

/// Project a response into the tool's model-visible output.
///
/// A completed or still-running task returns text; a task the remote failed,
/// rejected, cancelled, or gated behind authentication returns a typed
/// [`ToolExecutionError`] so the kind reaches hooks and telemetry. The remote's
/// own status text is the error message, and
/// [`ToolExecutionError::new`](rig_core::tool::ToolExecutionError) seeds the
/// model-visible output from it, so the model still learns *why* the call
/// failed.
pub(crate) fn response_to_output(
    response: &SendMessageResponse,
) -> Result<String, ToolExecutionError> {
    match response {
        SendMessageResponse::Task(task) => {
            let state = &task.status.state;
            if let Some(error) = task_failure(state, || status_or_default(task)) {
                return Err(error);
            }
            let body = task_body_limited(task, DEFAULT_TEXT_LIMIT)?;
            Ok(match state {
                TaskState::Completed if body.is_empty() => EMPTY_COMPLETION.to_string(),
                TaskState::Completed => body,
                // A non-terminal task is an interim answer, not the result.
                // The prefix tells the model to expect a continuation; no
                // identifiers appear in it, because the host re-attaches them.
                other if body.is_empty() => format!("[a2a {}]", state_label(other)),
                other => format!("[a2a {}] {body}", state_label(other)),
            })
        }
        SendMessageResponse::Message(message) => {
            Ok(message_body_limited(message, DEFAULT_TEXT_LIMIT)?)
        }
    }
}

/// Classify a task state that the caller cannot act on as a typed tool error.
///
/// `status` is lazy so the (fallible, allocating) status projection only runs
/// for states that actually fail.
fn task_failure(
    state: &TaskState,
    status: impl FnOnce() -> Result<String, A2AError>,
) -> Option<ToolExecutionError> {
    let build: fn(String) -> ToolExecutionError = match state {
        TaskState::Failed => ToolExecutionError::provider,
        // The remote declined the work; a refusal, not an outage.
        TaskState::Rejected => ToolExecutionError::refused,
        TaskState::Canceled => ToolExecutionError::cancelled,
        // Non-terminal, but only an operator can satisfy an auth challenge —
        // returning it as text would just burn a turn.
        TaskState::AuthRequired => ToolExecutionError::permission_denied,
        _ => return None,
    };
    Some(match status() {
        Ok(text) => build(format!("remote A2A agent {}: {text}", state_label(state))),
        Err(error) => ToolExecutionError::from(error),
    })
}

fn status_or_default(task: &a2a::Task) -> Result<String, A2AError> {
    let text = status_text_limited(task, DEFAULT_TEXT_LIMIT)?;
    Ok(if text.is_empty() {
        "no status message".to_string()
    } else {
        text
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use a2a::{
        AgentCapabilities, AgentInterface, Artifact, Message, Part, Role, Task, TaskStatus,
        new_artifact_id, new_context_id, new_task_id,
    };
    use rig_agent::tool::ToolErrorKind;

    fn make_task(state: TaskState, artifact: Option<&str>, status: Option<&str>) -> Task {
        Task {
            id: new_task_id(),
            context_id: new_context_id(),
            status: TaskStatus {
                state,
                message: status.map(|t| Message::new(Role::Agent, vec![Part::text(t.to_string())])),
                timestamp: None,
            },
            artifacts: artifact.map(|t| {
                vec![Artifact {
                    artifact_id: new_artifact_id(),
                    name: None,
                    description: None,
                    parts: vec![Part::text(t.to_string())],
                    metadata: None,
                    extensions: None,
                }]
            }),
            history: None,
            metadata: None,
        }
    }

    fn task_output(task: Task) -> Result<String, ToolExecutionError> {
        response_to_output(&SendMessageResponse::Task(task))
    }

    fn card(name: &str, skills: Vec<AgentSkill>) -> AgentCard {
        AgentCard {
            name: name.to_string(),
            description: "A stub agent.".to_string(),
            version: "1.0".to_string(),
            supported_interfaces: vec![AgentInterface {
                url: "http://127.0.0.1:1".to_string(),
                protocol_binding: a2a::TRANSPORT_PROTOCOL_JSONRPC.to_string(),
                protocol_version: a2a::VERSION.to_string(),
                tenant: None,
            }],
            capabilities: AgentCapabilities::default(),
            default_input_modes: vec!["text/plain".to_string()],
            default_output_modes: vec!["text/plain".to_string()],
            skills,
            provider: None,
            documentation_url: None,
            icon_url: None,
            security_schemes: None,
            security_requirements: None,
            signatures: None,
        }
    }

    fn skill(id: &str, name: &str) -> AgentSkill {
        AgentSkill {
            id: id.to_string(),
            name: name.to_string(),
            description: format!("Does {name}."),
            tags: vec!["demo".to_string()],
            examples: Some(vec![format!("please {name}")]),
            input_modes: None,
            output_modes: None,
            security_requirements: None,
        }
    }

    #[test]
    fn completed_task_returns_body_without_markers() {
        let task = make_task(TaskState::Completed, Some("hello"), None);
        assert_eq!(task_output(task).unwrap(), "hello");
    }

    #[test]
    fn completed_task_joins_artifact_and_status() {
        let task = make_task(TaskState::Completed, Some("hello"), Some("done"));
        assert_eq!(task_output(task).unwrap(), "hello\ndone");
    }

    #[test]
    fn empty_completed_task_returns_a_placeholder() {
        let task = make_task(TaskState::Completed, None, None);
        assert_eq!(task_output(task).unwrap(), EMPTY_COMPLETION);
    }

    #[test]
    fn input_required_is_prefixed_and_carries_no_identifiers() {
        let task = make_task(TaskState::InputRequired, None, Some("which file?"));
        let out = task_output(task).expect("input-required is not an error");
        assert_eq!(out, "[a2a input-required] which file?");
        assert!(!out.contains("contextId"), "{out}");
        assert!(!out.contains("taskId"), "{out}");
    }

    #[test]
    fn input_required_without_a_prompt_is_still_labelled() {
        let task = make_task(TaskState::InputRequired, None, None);
        assert_eq!(task_output(task).unwrap(), "[a2a input-required]");
    }

    #[test]
    fn working_task_is_prefixed_with_its_state() {
        let task = make_task(TaskState::Working, None, Some("still going"));
        assert_eq!(task_output(task).unwrap(), "[a2a working] still going");
    }

    #[test]
    fn failure_states_map_to_typed_tool_errors() {
        let cases = [
            (TaskState::Failed, ToolErrorKind::Provider, false),
            (TaskState::Rejected, ToolErrorKind::PermissionDenied, true),
            (TaskState::Canceled, ToolErrorKind::Cancelled, false),
            (
                TaskState::AuthRequired,
                ToolErrorKind::PermissionDenied,
                false,
            ),
        ];
        for (state, kind, refused) in cases {
            let task = make_task(state.clone(), Some("ignored"), Some("quota exceeded"));
            let error = task_output(task).expect_err("state must fail the call");
            assert_eq!(error.kind(), kind, "{state:?}");
            assert_eq!(error.is_refusal(), refused, "{state:?}");
            // The remote's own status text must reach the model, not a
            // redacted placeholder.
            assert!(
                error.model_output().render().contains("quota exceeded"),
                "{state:?}: {}",
                error.model_output().render()
            );
        }
    }

    #[test]
    fn failure_without_a_status_message_still_names_the_state() {
        let task = make_task(TaskState::Failed, None, None);
        let error = task_output(task).expect_err("failed must error");
        assert!(error.to_string().contains("failed"), "{error}");
        assert!(error.to_string().contains("no status message"), "{error}");
    }

    #[test]
    fn message_response_returns_bare_text() {
        let mut message = Message::new(Role::Agent, vec![Part::text("hello")]);
        message.context_id = Some("ctx-1".to_string());
        message.task_id = Some("task-1".to_string());
        let out = response_to_output(&SendMessageResponse::Message(message)).unwrap();
        assert_eq!(out, "hello");
    }

    #[test]
    fn description_renders_card_skills() {
        let rendered = describe_card(&card("greeter", vec![skill("greet", "greet")]));
        assert!(rendered.contains("Agent name: greeter"), "{rendered}");
        assert!(
            rendered.contains("Agent description: A stub agent."),
            "{rendered}"
        );
        assert!(
            rendered.contains("- greet (greet): Does greet."),
            "{rendered}"
        );
        assert!(rendered.contains("[tags: demo]"), "{rendered}");
        assert!(rendered.contains("[e.g. please greet]"), "{rendered}");
    }

    #[test]
    fn description_without_skills_omits_the_section() {
        let rendered = describe_card(&card("bare", vec![]));
        assert!(!rendered.contains("Skills:"), "{rendered}");
        assert!(rendered.contains("Agent name: bare"), "{rendered}");
    }

    #[test]
    fn description_is_capped() {
        let skills = (0..2000)
            .map(|i| skill(&format!("skill-{i}"), &format!("do thing {i}")))
            .collect();
        let rendered = describe_card(&card("verbose", skills));
        assert!(rendered.len() <= DESCRIPTION_LIMIT, "{}", rendered.len());
        assert!(rendered.contains("further skill(s) omitted"), "{rendered}");
    }

    #[test]
    fn overlong_single_skill_is_elided() {
        let mut long = skill("big", "big");
        long.description = "x".repeat(4096);
        let rendered = describe_card(&card("verbose", vec![long]));
        assert!(rendered.len() <= DESCRIPTION_LIMIT);
        assert!(rendered.contains('…'), "{rendered}");
    }

    #[test]
    fn parse_args_prompt_only() {
        assert_eq!(
            parse_args(serde_json::json!({"prompt": "hi"})).unwrap(),
            "hi"
        );
    }

    #[test]
    fn parse_args_ignores_unknown_extras() {
        assert_eq!(
            parse_args(serde_json::json!({"prompt": "hi", "contextId": "ctx", "foo": 1})).unwrap(),
            "hi"
        );
    }

    #[test]
    fn parse_args_missing_prompt_errors() {
        assert!(matches!(
            parse_args(serde_json::json!({"foo": 1})),
            Err(ToolArgsError::MissingPrompt)
        ));
        assert!(matches!(
            parse_args(serde_json::json!("not an object")),
            Err(ToolArgsError::NotAnObject)
        ));
    }
}
