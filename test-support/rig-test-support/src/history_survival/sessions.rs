//! Reasoning across a session boundary: a reasoning turn with a tool call is
//! recorded, its history is persisted the way an application would
//! (serialized to JSON and loaded back), and the conversation continues live
//! from the loaded history, optionally on a different model of the same
//! provider.
//!
//! The run asserts the loaded history equals the original, provenance
//! included. On the same model the recording asserts every signature,
//! ciphertext and id the first reply delivered reaches the continuation in
//! the slot that must carry it; on another model it asserts the call ids
//! still pair and no reasoning state is sent.

use serde_json::Value;

use rig_core::completion::{CompletionRequest, ToolDefinition};
use rig_core::message::{AssistantContent, Message, ToolResultContent, UserContent};

use super::{Dialect, lost_tokens, response_tokens};

/// One session cell.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    /// The provider's cassette directory.
    pub provider: &'static str,
    /// Provider request parameters, typically the reasoning switch.
    pub params: fn() -> Option<Value>,
    /// Output budget per call.
    pub max_tokens: u64,
    /// The reasoning kinds the first reply must deliver: `signature`,
    /// `thought_signature`, `encrypted_content`, `reasoning_id`.
    pub expect: &'static [&'static str],
}

const CODE: &str = "amber-5521";

fn tool() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("lookup_code").expect("tool name"),
        description: "Return the code stored for a record. Always call it before answering."
            .to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "record": { "type": "string" } },
            "required": ["record"]
        }),
        annotations: Default::default(),
    }
}

fn request(cell: Cell, history: Vec<Message>) -> CompletionRequest {
    let mut request = CompletionRequest::from(history);
    request.tools = vec![tool().into()];
    request.max_tokens = Some(cell.max_tokens);
    request.additional_params = (cell.params)();
    request
}

/// Record the first turn on `first`, persist and reload its history, answer
/// the call, and continue on `second` (the same model, or another model of
/// the same provider).
pub async fn run(
    first: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    second: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    cell: Cell,
) {
    let first: rig_core::DynModel<rig_core::operation::Completion> = first.into();
    let second: rig_core::DynModel<rig_core::operation::Completion> = second.into();
    let loaded = turn_one(&first, cell).await;
    let answer = second
        .call(request(cell, loaded))
        .await
        .unwrap_or_else(|error| {
            panic!(
                "[{}] continuation from loaded history: {error}",
                cell.provider
            )
        });
    assert_answer(cell, &answer.choice);
}

/// Run turn one and return its history (prompt, reply, tool result) after a
/// JSON persistence round trip that must preserve it exactly.
async fn turn_one(
    first: &rig_core::DynModel<rig_core::operation::Completion>,
    cell: Cell,
) -> Vec<Message> {
    let prompt = Message::user(
        "Think it through, then call lookup_code for record alpha. Do not guess the code.",
    );
    let reply = first
        .call(request(cell, vec![prompt.clone()]))
        .await
        .unwrap_or_else(|error| panic!("[{}] turn one: {error}", cell.provider));
    let call = reply
        .choice
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!(
                "[{}] turn one called no tool: {:?}",
                cell.provider, reply.choice
            )
        });
    let reasoning: Vec<_> = reply
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert!(
        !reasoning.is_empty(),
        "[{}] turn one reasoned",
        cell.provider
    );
    // Replayable state rides a reasoning block's provider item (signature,
    // ciphertext, item id) or, on Gemini, the function call's.
    let replayable =
        call.native.is_some() || reasoning.iter().any(|reasoning| reasoning.native.is_some());
    assert!(
        cell.expect.is_empty() || replayable,
        "[{}] turn one delivered replayable reasoning state",
        cell.provider
    );

    let history = vec![
        prompt,
        Message::Assistant(reply.head().with_content(reply.choice.clone())),
        Message::User {
            content: vec![UserContent::tool_result(
                call.id.clone(),
                call.function.name.clone(),
                vec![ToolResultContent::text(format!(
                    "record alpha: code {CODE}"
                ))],
            )],
        },
    ];
    let persisted = serde_json::to_string(&history).expect("history serializes");
    let loaded: Vec<Message> = serde_json::from_str(&persisted).expect("history loads");
    assert_eq!(
        loaded, history,
        "[{}] the history survives persistence",
        cell.provider
    );

    loaded
}

fn assert_answer(cell: Cell, choice: &[AssistantContent]) {
    let text: String = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        text.contains(CODE),
        "[{}] the answer uses the result: {text:?}",
        cell.provider
    );
}

/// The first reply's reasoning state reaches a continuation on the same
/// model, each value in its slot, and the expected kinds were delivered.
pub fn assert_recorded(cell: Cell, scenario: &str) {
    let (dialect, reply, next) = recorded(cell, scenario);
    let lost = lost_tokens(dialect, &reply, &next);
    assert!(
        lost.is_empty(),
        "[{}] lost in the continuation: {lost:?}",
        cell.provider
    );
}

/// The recorded first reply and continuation request, after asserting the
/// first reply delivered every expected kind verbatim.
fn recorded(cell: Cell, scenario: &str) -> (Dialect, String, Value) {
    let paths = crate::cassettes::recorded_request_paths(cell.provider, scenario);
    let mut bodies = crate::cassettes::recorded_interaction_bodies(cell.provider, scenario);
    assert_eq!(
        bodies.len(),
        2,
        "[{}] one turn and one continuation",
        cell.provider
    );
    let dialect = Dialect::from_path(&paths[0]);
    let next: Value = serde_json::from_str(&bodies[1].0).expect("continuation request is JSON");
    let reply = bodies.swap_remove(0).1;
    let delivered = response_tokens(dialect, &reply);
    for kind in cell.expect {
        assert!(
            delivered
                .iter()
                .any(|token| token.kind == *kind && !token.value.contains("REDACTED")),
            "[{}] turn one delivered a verbatim {kind}: {delivered:?}",
            cell.provider
        );
    }
    (dialect, reply, next)
}
