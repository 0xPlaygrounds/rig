//! Reasoning contracts beside the shared matrix's effect-log oracle.

use rig_core::streaming::PartKind;
use std::sync::{Arc, Mutex};

use rig_agent::completion::CompletionResponse;

use rig_agent::completion::FinishReason;

use rig_core::effect::EffectKind;

use rig_core::effect::Outcome;

use rig_cassette::effect_log::EffectLog;

use rig_core::message::AssistantContent;

use rig_core::message::Message;

use rig_core::streaming::StreamEvent;

use super::cells::{Cell, ReasoningCase, Thinking, ThinkingWire};

pub(crate) struct Settlement {
    pub(crate) messages: Vec<Message>,
    pub(crate) error: Option<String>,
}

pub(crate) type SettlementCapture = Arc<Mutex<Option<Settlement>>>;

/// The same named observer the corpus program declares; the producer
/// retains its payload so an error cannot hide a wrongly committed turn.
pub(crate) struct RecordSettled(pub SettlementCapture);

impl rig_agent::agent::AgentHook for RecordSettled {
    async fn on_run_settled(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::RunSettled<'_>,
    ) {
        let messages = event
            .messages
            .expect("the capped run reached the engine")
            .to_vec();
        let error = match event.outcome {
            rig_agent::agent::SettledOutcome::Error(error) => Some(error.to_owned()),
            rig_agent::agent::SettledOutcome::Response(_) => None,
        };
        let previous = self
            .0
            .lock()
            .expect("settlement capture")
            .replace(Settlement { messages, error });
        assert!(previous.is_none(), "the run settles exactly once");
    }
}

pub(crate) fn completions(log: &EffectLog) -> Vec<&CompletionResponse> {
    log.records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(Outcome::Completion(response)) => Some(response),
            _ => None,
        })
        .collect()
}

/// Check the provider's normalized reply, not its transport frames.
/// Complete payload equality is checked separately against committed history.
pub(crate) fn assert_log(cell: &Cell, wire: ThinkingWire, log: &EffectLog) {
    let Some(case) = cell.reasoning else { return };
    let responses = completions(log);
    if case == ReasoningCase::Capped {
        assert_eq!(
            responses.len(),
            1,
            "the capped reply is recorded before settlement"
        );
        if cell.program.streamed {
            assert!(
                log.records
                    .iter()
                    .flat_map(|record| record.events.iter().flat_map(|events| events.events()))
                    .any(|event| matches!(
                        event,
                        StreamEvent::Start {
                            kind: PartKind::Reasoning,
                            ..
                        }
                    )),
                "the capped stream records the reasoning prefix"
            );
        }
        for response in &responses {
            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
            assert!(rig_core::message::turn_delivered_no_answer(
                &response.choice
            ));
        }
        return;
    }
    let expected_turns = if case == ReasoningCase::Tool { 2 } else { 1 };
    assert_eq!(
        responses.len(),
        expected_turns,
        "{}: completion turns",
        cell.name
    );
    for (index, response) in responses.iter().enumerate() {
        let parts = &response.choice;
        let thinking = cell.thinking == Thinking::On
            || (cell.thinking == Thinking::SecondTurnOnly && index == 1);
        let reasoning: Vec<_> = parts
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Reasoning(block) => Some(block),
                _ => None,
            })
            .collect();
        // Row 2 requires reasoning on the tool turn. Its final answer may
        // need no further thinking; preserve any reasoning the wire does send.
        let requires_reasoning = thinking
            && !(case == ReasoningCase::Tool && cell.thinking == Thinking::On && index == 1);
        if thinking
            && !matches!(wire, ThinkingWire::OpenAiChat)
            && (requires_reasoning || !reasoning.is_empty())
        {
            assert!(
                !reasoning.is_empty(),
                "{}: visible reasoning: {parts:?}",
                cell.name
            );
            assert!(matches!(
                parts.first(),
                Some(AssistantContent::Reasoning(_))
            ));
            match wire {
                ThinkingWire::OpenAiResponses => {
                    let item = |block: &&rig_core::message::Reasoning, key: &str| {
                        block
                            .native
                            .as_ref()
                            .and_then(|native| native.item.get(key))
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|value| !value.is_empty())
                    };
                    assert!(reasoning.iter().any(|block| item(block, "id")));
                    assert!(
                        reasoning
                            .iter()
                            .any(|block| item(block, "encrypted_content"))
                    );
                }
                ThinkingWire::Gemini => {
                    // A signature stays on the part that carried it: its
                    // provider item holds it.
                    assert!(
                        parts.iter().any(|part| part
                            .native_item()
                            .and_then(|item| item.get("thoughtSignature"))
                            .is_some()),
                        "{}: the Gemini signature is preserved",
                        cell.name
                    );
                }
                _ => {}
            }
        } else {
            assert!(
                reasoning.is_empty(),
                "{}: no visible reasoning: {parts:?}",
                cell.name
            );
        }
        if requires_reasoning
            && matches!(
                wire,
                ThinkingWire::OpenAiChat
                    | ThinkingWire::OpenAiResponses
                    | ThinkingWire::Gemini
                    | ThinkingWire::DeepSeek
            )
        {
            assert!(
                response.usage.reasoning_tokens.is_some_and(|n| n > 0),
                "{}: reasoning usage: {:?}",
                cell.name,
                response.usage
            );
        }
        if matches!(wire, ThinkingWire::Doubleword) {
            assert_eq!(
                response.usage.reasoning_tokens, None,
                "{}: Doubleword's reasoning count is not part of its completion count, so it \
                 is unreported",
                cell.name
            );
        }
        if !thinking {
            assert_eq!(
                response.usage.reasoning_tokens.unwrap_or(0),
                0,
                "{}: reasoning explicitly off",
                cell.name
            );
        }
        let calls: Vec<_> = parts
            .iter()
            .filter_map(|part| match part {
                AssistantContent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();
        if case == ReasoningCase::Output || (case == ReasoningCase::Tool && index == 0) {
            assert_eq!(calls.len(), 1, "{}: one call", cell.name);
            assert_eq!(
                calls[0].function.name,
                if case == ReasoningCase::Output {
                    "final_result"
                } else {
                    "add"
                }
            );
            assert!(
                !parts
                    .iter()
                    .any(|part| matches!(part, AssistantContent::Text(_))),
                "{}: the tool turn has no prose",
                cell.name
            );
        } else {
            assert!(calls.is_empty());
            assert!(
                parts.iter().any(
                    |part| matches!(part, AssistantContent::Text(text) if !text.text.is_empty())
                )
            );
        }
    }
    let tools = log
        .records
        .iter()
        .filter(|record| matches!(record.kind, EffectKind::ToolCall { .. }))
        .count();
    assert_eq!(
        tools,
        usize::from(case == ReasoningCase::Tool),
        "{}: tool executes once",
        cell.name
    );
}

pub(crate) fn assert_history(cell: &Cell, log: &EffectLog, history: &[Message]) {
    if cell.reasoning.is_none() {
        return;
    }
    if cell.reasoning == Some(ReasoningCase::Capped) {
        assert_eq!(
            history.len(),
            1,
            "{}: nothing committed on the capped turn",
            cell.name
        );
        assert!(matches!(history.first(), Some(Message::User { .. })));
        return;
    }
    let actual: Vec<_> = history
        .iter()
        .filter(|message| matches!(message, Message::Assistant(_)))
        .cloned()
        .collect();
    let expected: Vec<_> = completions(log)
        .iter()
        .map(|response| {
            let content = if cell.reasoning == Some(ReasoningCase::Output) {
                // The record retains the output call; committed history keeps
                // its answer as JSON text, avoiding an unanswered tool call.
                let call = response
                    .choice
                    .iter()
                    .find_map(|part| match part {
                        AssistantContent::ToolCall(call)
                            if call.function.name == "final_result" =>
                        {
                            Some(call)
                        }
                        _ => None,
                    })
                    .expect("the recorded output call");
                let mut parts: Vec<_> = response
                    .choice
                    .iter()
                    .filter(|part| !matches!(part, AssistantContent::ToolCall(_)))
                    .cloned()
                    .collect();
                parts.push(AssistantContent::text(
                    call.function.arguments_value().to_string(),
                ));
                parts
            } else {
                response.choice.clone()
            };
            Message::Assistant(response.head().with_content(content))
        })
        .collect();
    assert_eq!(
        actual, expected,
        "{}: complete reasoning payload and provider ids survive commit",
        cell.name
    );
}
