//! Reasoning contracts beside the shared matrix's effect-log oracle.
//!
//! The host observes collected stream deltas through `rig::observe`'s
//! explicit HostAction extension. AdapterEnding describes HTTP closure
//! (`Decoded`/`Terminal`); the run's Ended observation carries `settled`.

use std::sync::{Arc, Mutex};

use bevy_ecs::prelude::*;
use rig::completion::{CompletionResponse, FinishReason};
use rig::effect::{EffectKind, Outcome};
use rig::effect_log::EffectLog;
use rig::message::{AssistantContent, Message, ReasoningContent, canonical_streamed_choice};
use rig::observe::{AdapterEnding, AdapterEvent, Emitter, HostAction, ObservationLog, Stage};
use rig::streaming::{Delta, StreamEvent};
use rig_ecs::agent::{Order, Utterance};
use rig_ecs::bus::{StreamItemsDelivered, Subjects, Witnessing};
use serde::{Deserialize, Serialize};

use super::cells::{Cell, ReasoningCase, Thinking, ThinkingWire};

pub(crate) struct Settlement {
    pub messages: Vec<Message>,
    pub error: Option<String>,
}

pub(crate) type SettlementCapture = Arc<Mutex<Option<Settlement>>>;

/// The same named observer the corpus program declares; the producer
/// retains its payload so an error cannot hide a wrongly committed turn.
pub(crate) struct RecordSettled(pub SettlementCapture);

impl rig::agent::AgentHook for RecordSettled {
    async fn on_run_settled(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::RunSettled<'_>,
    ) {
        let messages = event
            .messages
            .expect("the capped run reached the engine")
            .to_vec();
        let error = match event.outcome {
            rig::agent::SettledOutcome::Error(error) => Some(error.to_owned()),
            rig::agent::SettledOutcome::Response(_) => None,
        };
        let previous = self
            .0
            .lock()
            .expect("settlement capture")
            .replace(Settlement { messages, error });
        assert!(previous.is_none(), "the run settles exactly once");
    }
}

/// A host's observation of a real collected reasoning delta. It is not
/// synthesized from the final answer or from a cassette's frame bytes.
#[derive(Serialize, Deserialize)]
struct ReasoningDelta(Delta);

impl HostAction for ReasoningDelta {
    const KIND: &'static str = "ecs-reasoning/delta";
}

#[derive(Resource, Default)]
pub(crate) struct PendingReasoning(Vec<(Entity, Delta)>);

pub(crate) fn install_delivery_observer(app: &mut bevy_app::App) {
    app.init_resource::<PendingReasoning>().add_observer(
        |delivered: On<StreamItemsDelivered>, mut pending: ResMut<PendingReasoning>| {
            for item in &delivered.items {
                if let Ok(StreamEvent::BlockDelta {
                    delta: delta @ Delta::Reasoning { .. },
                    ..
                }) = item
                {
                    pending.0.push((delivered.effect, delta.clone()));
                }
            }
        },
    );
}

/// Preserve the existing policy-visible emission point after Collect. The
/// synchronous callback captures only new deltas; it does not advance policy.
pub(crate) fn witness_deltas(
    entities: Query<Entity>,
    subjects: Subjects,
    witness: Res<Witnessing>,
    mut pending: ResMut<PendingReasoning>,
) {
    for (entity, delta) in pending.0.drain(..) {
        if entities.contains(entity) {
            witness.emit(
                subjects.of(entity),
                Stage::Collect,
                Emitter::named("ecs-reasoning"),
                ReasoningDelta(delta).action().expect("a delta serializes"),
            );
        }
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

fn rank(part: &AssistantContent) -> u8 {
    match part {
        AssistantContent::Reasoning(_) => 0,
        AssistantContent::Text(_) => 1,
        AssistantContent::ToolCall(_) => 2,
        AssistantContent::Image(_) => 3,
    }
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
                    .flat_map(|record| record.events.iter().flatten())
                    .any(|event| matches!(
                        event,
                        StreamEvent::BlockStart {
                            kind: rig::streaming::BlockKind::Reasoning { .. },
                            ..
                        }
                    )),
                "the capped stream records the reasoning prefix"
            );
        }
        for response in &responses {
            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
            assert!(rig::message::turn_delivered_no_answer(&response.choice));
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
        assert!(
            parts
                .windows(2)
                .all(|pair| rank(&pair[0]) <= rank(&pair[1])),
            "{}: canonical reasoning/text/call/image order: {parts:?}",
            cell.name
        );
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
                    assert!(reasoning.iter().any(|block| block.id.is_some()));
                    assert!(reasoning.iter().any(|block| {
                        block.content.iter().any(|content|
                        matches!(content, ReasoningContent::Encrypted(value) if !value.is_empty()))
                    }));
                }
                ThinkingWire::Gemini => {
                    // Function-call signatures belong to ToolCall.signature;
                    // text/thought signatures belong to the reasoning block.
                    assert!(
                        reasoning
                            .iter()
                            .any(|block| block.first_signature().is_some())
                            || parts.iter().any(|part| matches!(part,
                            AssistantContent::ToolCall(call) if call.signature.is_some())),
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
                    | ThinkingWire::Doubleword
            )
        {
            assert!(
                response.usage.reasoning_tokens > 0,
                "{}: reasoning usage: {:?}",
                cell.name,
                response.usage
            );
        }
        if !thinking {
            assert_eq!(
                response.usage.reasoning_tokens, 0,
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

pub(crate) fn assistant_history(world: &mut World, run: Entity) -> Vec<Message> {
    let mut rows: Vec<_> = world
        .query_filtered::<(Entity, &ChildOf, &Order), With<Utterance>>()
        .iter(world)
        .filter(|(_, parent, _)| parent.parent() == run)
        .map(|(entity, _, order)| {
            (
                order.0,
                rig_ecs::agent::content::parts::read_message(world, entity)
                    .expect("valid content graph")
                    .to_message(),
            )
        })
        .collect();
    rows.sort_by_key(|(order, _)| *order);
    rows.into_iter().map(|(_, message)| message).collect()
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
        .filter(|message| matches!(message, Message::Assistant { .. }))
        .cloned()
        .collect();
    let expected: Vec<_> = completions(log)
        .iter()
        .map(|response| Message::Assistant {
            id: response.message_id.clone(),
            content: if cell.reasoning == Some(ReasoningCase::Output) {
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
                parts.push(AssistantContent::text(call.function.arguments.to_string()));
                parts
            } else if cell.program.streamed {
                canonical_streamed_choice(response.choice.clone())
            } else {
                response.choice.clone()
            },
        })
        .collect();
    assert_eq!(
        actual, expected,
        "{}: complete reasoning payload and provider ids survive commit",
        cell.name
    );
}

pub(crate) fn assert_witness(cell: &Cell, log: &EffectLog, trace: &ObservationLog) {
    if cell.reasoning.is_none() {
        return;
    }
    let expected = if cell.reasoning == Some(ReasoningCase::Capped) {
        "provider"
    } else {
        "settled"
    };
    assert_eq!(crate::stream_faults::endings(trace), [expected]);
    let adapters = crate::stream_faults::adapter_events(trace);
    let endings: Vec<_> = adapters
        .iter()
        .filter_map(|event| match event {
            AdapterEvent::Finished { ending } => Some(ending),
            _ => None,
        })
        .collect();
    assert_eq!(
        endings.len(),
        log.records
            .iter()
            .filter(|record| matches!(record.kind, EffectKind::Completion { .. }))
            .count()
    );
    assert!(
        endings
            .iter()
            .all(|ending| matches!(ending, AdapterEnding::Decoded | AdapterEnding::Terminal)),
        "provider boundaries: {endings:?}"
    );
    let mut current_usage = None;
    let mut reported = Vec::new();
    for event in &adapters {
        match event {
            AdapterEvent::Usage { usage } => current_usage = usage.reasoning_tokens,
            AdapterEvent::Finished { .. } => reported.push(current_usage.take()),
            _ => {}
        }
    }
    for (response, reported) in completions(log).iter().zip(reported) {
        assert_eq!(
            response.usage.reasoning_tokens,
            reported.unwrap_or(0),
            "{}: record usage equals the provider's witnessed counter",
            cell.name
        );
    }
    let recorded: Vec<_> = log
        .records
        .iter()
        .flat_map(|record| record.events.iter().flatten())
        .filter_map(|event| match event {
            StreamEvent::BlockDelta {
                delta: delta @ Delta::Reasoning { .. },
                ..
            } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    let witnessed: Vec<_> = trace
        .trace()
        .observations
        .iter()
        .filter_map(|observation| {
            ReasoningDelta::from_action(&observation.action)
                .map(|fact| fact.expect("the host fact decodes").0)
        })
        .collect();
    if cell.program.streamed
        && completions(log).iter().any(|response| {
            response.choice.iter().any(|part| {
                matches!(part,
            AssistantContent::Reasoning(block) if !block.display_text().is_empty())
            })
        })
    {
        assert!(
            !recorded.is_empty(),
            "{}: streamed reasoning text has deltas",
            cell.name
        );
    }
    assert_eq!(
        witnessed, recorded,
        "{}: witness sees every reasoning delta",
        cell.name
    );
}
