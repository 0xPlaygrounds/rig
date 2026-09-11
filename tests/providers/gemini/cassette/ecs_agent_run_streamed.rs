//! Native streamed-run diagnostics from actual failure and retained graph state.

use super::super::{
    agent_run_support::{
        Add, FORCE_TOOLS_PREAMBLE, history_has_assistant_tool_call, is_tool_result_user_message,
    },
    support::with_gemini_cassette,
};
use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig::{
    effect::EffectKind,
    message::{Message, ToolChoice},
    prelude::*,
    providers::gemini,
};
use rig_ecs::{
    agent::{
        Cancelled, DefaultMaxTurns, Failure, Order, Parts, RunResult, Settled, ToolChoiceSpec,
        Turn, Utterance,
    },
    bus::{BusSet, EffectOutcome, Issued, PendingEffect, RigSchedule},
    systems::spawn_run,
};

#[derive(Resource)]
struct InvalidPolicy {
    resolution: rig_ecs::agent::Resolution,
    seen: Vec<rig_ecs::agent::InvalidCall>,
}

fn judge_invalid(
    mut commands: Commands,
    calls: Query<
        (Entity, &ChildOf, &rig_ecs::agent::InvalidCall),
        Added<rig_ecs::agent::InvalidCall>,
    >,
    turns: Query<&ChildOf, With<Turn>>,
    runs: Query<&rig_ecs::agent::RunStreaming>,
    mut policy: ResMut<InvalidPolicy>,
) {
    for (entity, parent, call) in &calls {
        let run = turns
            .get(parent.parent())
            .expect("invalid call belongs to turn")
            .parent();
        assert!(runs.get(run).expect("run streaming setting").0);
        assert_eq!(call.name, "add");
        assert!(
            call.stream_offset.is_some(),
            "policy receives the delivered name event"
        );
        if matches!(policy.resolution, rig_ecs::agent::Resolution::Skip { .. }) {
            assert!(
                policy.seen.is_empty(),
                "only the first turn restricts tools"
            );
            commands.entity(run).insert(rig_ecs::agent::ToolAccess {
                executable: None,
                allowed: Some(std::collections::BTreeSet::from(["add".into()])),
            });
        }
        policy.seen.push(call.clone());
        commands.entity(entity).insert(policy.resolution.clone());
    }
}

fn install_invalid_policy(ecs: &mut EcsAgent, resolution: rig_ecs::agent::Resolution) {
    ecs.app.insert_resource(InvalidPolicy {
        resolution,
        seen: vec![],
    });
    ecs.app.world_mut().resource_mut::<Schedules>().add_systems(
        RigSchedule,
        judge_invalid.in_set(rig_ecs::systems::RigSet::Judge),
    );
}

async fn drain_completion(ecs: &mut EcsAgent) {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            ecs.app.update();
            let pending = ecs
                .app
                .world_mut()
                .query::<(&PendingEffect, Option<&EffectOutcome>)>()
                .iter(ecs.app.world())
                .any(|(effect, outcome)| {
                    matches!(effect.kind, EffectKind::Completion { .. }) && outcome.is_none()
                });
            if !pending {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("completion drains after policy failure");
}

#[tokio::test]
async fn streamed_invalid_tool_call_fails_fast_mid_stream() {
    with_gemini_cassette(
        "agent_run_streamed/streamed_invalid_tool_call_fails_fast_mid_stream",
        |client| async move {
            let mut ecs = setup(&client);
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(rig_ecs::agent::ToolAccess {
                    allowed: Some(Default::default()),
                    ..Default::default()
                });
            install_invalid_policy(&mut ecs, rig_ecs::agent::Resolution::Fail);
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &[],
                "What is 21 + 21? Use the add tool.",
                true,
                Some(2),
            );
            let failure = ecs
                .wait_for_outcome(run)
                .await
                .expect_err("disallowed call fails");
            let Failure::UnknownToolCall { name } = failure else {
                panic!("expected invalid tool failure, got {failure:?}")
            };
            assert_eq!(name, "add");
            let access = ecs
                .app
                .world_mut()
                .query_filtered::<&rig_ecs::agent::ToolAccess, With<Turn>>()
                .single(ecs.app.world())
                .expect("single turn snapshot");
            assert_eq!(
                access
                    .executable
                    .as_ref()
                    .expect("executable snapshot")
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>(),
                vec!["add".to_string()]
            );
            assert!(
                access
                    .allowed
                    .as_ref()
                    .expect("allowed snapshot")
                    .is_empty()
            );
            assert!(history_has_assistant_tool_call(
                &history(&mut ecs, run),
                "add"
            ));
            assert_eq!(ecs.app.world().resource::<InvalidPolicy>().seen.len(), 1);
            drain_completion(&mut ecs).await;
        },
    )
    .await;
}

#[tokio::test]
async fn streamed_repair_continues_the_same_stream() {
    use super::super::agent_run_support::{Sum, assistant_tool_call_names};
    use crate::support::assert_mentions_expected_number;
    with_gemini_cassette("agent_run_streamed/streamed_repair_continues_the_same_stream", |client| async move {
        let mut ecs = setup(&client);
        ecs.app.world_mut().entity_mut(ecs.agent).insert(ToolChoiceSpec(None));
        ecs.tool(Sum);
        let sum_key = ecs.app.world_mut().query::<&rig_ecs::bus::Bound>().iter(ecs.app.world())
            .find(|bound| matches!(&bound.descriptor.family, rig::effect::FamilyDescriptor::Tool { name, .. } if name == "sum"))
            .expect("sum bound").key.clone();
        ecs.app.world_mut().entity_mut(ecs.agent).insert(rig_ecs::agent::ToolAccess {
            executable: Some(std::collections::BTreeMap::from([("sum".into(), sum_key)])),
            allowed: Some(std::collections::BTreeSet::from(["sum".into()])),
        });
        install_invalid_policy(&mut ecs, rig_ecs::agent::Resolution::Repair { to: "sum".into() });
        let run = spawn_run(ecs.app.world_mut(), ecs.agent, &[], "Use the add tool to compute 2 + 3, then state the result.", true, Some(3));
        let output = ecs.wait_for_success(run).await;
        assert_mentions_expected_number(&output, 5);
        let calls: Vec<_> = ecs.app.world_mut().query::<&PendingEffect>().iter(ecs.app.world())
            .filter_map(|effect| match &effect.kind { EffectKind::ToolCall { name, .. } => Some(name.clone()), _ => None }).collect();
        assert!(!calls.is_empty(), "a repaired tool reaches native dispatch");
        assert!(calls.iter().all(|name| name == "sum"), "{calls:?}");
        let messages = history(&mut ecs, run);
        let recorded: Vec<_> = messages.iter().flat_map(assistant_tool_call_names).collect();
        assert!(!recorded.iter().any(|name| name == "add"), "{recorded:?}");
        assert!(!ecs.app.world().resource::<InvalidPolicy>().seen.is_empty());
    }).await;
}

struct CompletedTurn {
    order: u64,
    request: rig::completion::CompletionRequest,
    usage: rig::completion::Usage,
    text: String,
    events: Vec<rig::streaming::StreamEvent>,
}

fn completed_turns(ecs: &mut EcsAgent, run: Entity) -> Vec<CompletedTurn> {
    let world = ecs.app.world_mut();
    let mut turns: Vec<_> = world
        .query::<(
            &ChildOf,
            &PendingEffect,
            &EffectOutcome,
            &rig_ecs::bus::Streamed,
        )>()
        .iter(world)
        .filter_map(|(parent, effect, outcome, stream)| {
            let turn = parent.parent();
            if world.get::<Turn>(turn).is_none() || world.get::<ChildOf>(turn)?.parent() != run {
                return None;
            }
            let EffectKind::Completion {
                request,
                stream: streaming,
            } = &effect.kind
            else {
                return None;
            };
            assert!(streaming);
            let Ok(rig::effect::Outcome::Completion(response)) = &outcome.0 else {
                panic!("completion must succeed: {outcome:?}")
            };
            assert!(
                stream.errors.is_empty(),
                "successful turn propagates every stream item error"
            );
            Some(CompletedTurn {
                order: world.get::<Order>(turn).expect("turn order").0,
                request: request.clone(),
                usage: response.usage,
                text: stream.text.clone(),
                events: stream.events.clone(),
            })
        })
        .collect();
    turns.sort_by_key(|turn| turn.order);
    turns
}

#[tokio::test]
async fn streamed_skip_abandons_the_turn_and_recovers() {
    use crate::support::assert_nonempty_response;
    use rig::message::UserContent;
    const REASON: &str = "The add tool is disabled for this request.";
    with_gemini_cassette(
        "agent_run_streamed/streamed_skip_abandons_the_turn_and_recovers",
        |client| async move {
            let mut ecs = setup(&client);
            ecs.app.world_mut().entity_mut(ecs.agent).insert((
                ToolChoiceSpec(None),
                rig_ecs::agent::ToolAccess {
                    allowed: Some(Default::default()),
                    ..Default::default()
                },
            ));
            install_invalid_policy(
                &mut ecs,
                rig_ecs::agent::Resolution::Skip {
                    reason: REASON.into(),
                },
            );
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &[],
                "What is 21 + 21? Use the add tool.",
                true,
                Some(3),
            );
            let output = ecs.wait_for_success(run).await;
            assert_nonempty_response(&output);
            assert_eq!(
                ecs.app.world().resource::<InvalidPolicy>().seen.len(),
                1,
                "first turn abandoned exactly once"
            );
            let turns = completed_turns(&mut ecs, run);
            assert!(turns.len() >= 2, "abandoned turn retains its completion");
            for turn in turns.iter().skip(1) {
                let (prompt, history) = turn
                    .request
                    .chat_history
                    .split_last()
                    .expect("retry prompt");
                assert!(history_has_assistant_tool_call(history, "add"));
                assert!(is_tool_result_user_message(prompt));
            }
            let retry = &turns[1].request.chat_history;
            let Message::User { content } = retry.last().expect("retry prompt") else {
                panic!("tool results are a user message")
            };
            let result = content
                .iter()
                .find_map(|part| match part {
                    UserContent::ToolResult(result) => Some(result),
                    _ => None,
                })
                .expect("synthetic skip result");
            assert!(result.call.is_generated());
            assert!(result.provider.is_none());
            let texts = super::super::agent_run_support::user_content_tool_result_texts(
                &UserContent::ToolResult(result.clone()),
            );
            assert_eq!(texts, [REASON]);
            let total = turns
                .iter()
                .fold(rig::completion::Usage::new(), |total, turn| {
                    total + turn.usage
                });
            assert_eq!(
                ecs.app
                    .world()
                    .get::<rig_ecs::agent::Usage>(run)
                    .expect("run usage")
                    .0,
                total
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streamed_hand_driven_multi_turn_run_completes() {
    use super::super::agent_run_support::{Subtract, assert_canonical_assistant_order};
    use crate::support::assert_mentions_expected_number;
    use rig::{
        message::AssistantContent,
        streaming::{BlockId, StreamEvent},
    };
    with_gemini_cassette("agent_run_streamed/streamed_hand_driven_multi_turn_run_completes", |client| async move {
        let mut ecs = setup(&client);
        ecs.app.world_mut().entity_mut(ecs.agent).insert(ToolChoiceSpec(None));
        ecs.tool(Subtract);
        let run = spawn_run(ecs.app.world_mut(), ecs.agent, &[], "Use the tools to compute (7 + 4) - 2: first compute 7 + 4 with the add tool, then subtract 2 from that result with the subtract tool, then state the final result.", true, Some(5));
        // There is no native record-completion API to call on a fresh run.
        // Its graph has no completion or usage before the first scheduled turn.
        assert_eq!(ecs.app.world_mut().query::<&PendingEffect>().iter(ecs.app.world()).count(), 0);
        assert_eq!(ecs.app.world().get::<rig_ecs::agent::Usage>(run).expect("fresh usage").0, rig::completion::Usage::new());
        let output = ecs.wait_for_success(run).await;
        let turns = completed_turns(&mut ecs, run);
        let streamed_text: String = turns.iter().map(|turn| turn.text.as_str()).collect();
        assert_mentions_expected_number(&streamed_text, 9);
        assert_mentions_expected_number(&output, 9);
        let begun = ecs.app.world().get::<rig_ecs::agent::Cursor>(run).expect("cursor").turn;
        assert!(begun >= 2);
        assert_eq!(turns.len(), begun, "exactly one actual completion per turn");
        let total = turns.iter().fold(rig::completion::Usage::new(), |total, turn| total + turn.usage);
        assert_eq!(ecs.app.world().get::<rig_ecs::agent::Usage>(run).expect("usage").0, total);
        assert!(total.total_tokens > 0);
        for event in turns.iter().flat_map(|turn| &turn.events) {
            if let StreamEvent::BlockEnd { id: BlockId::Wire(id), block: Some(AssistantContent::ToolCall(call)), .. } = event {
                assert_eq!(call.provider.as_ref().map(|provider| provider.call_id.as_str()), Some(id.as_str()), "{call:?}");
            }
        }
        let messages = history(&mut ecs, run);
        assert!(history_has_assistant_tool_call(&messages, "add"));
        assert!(history_has_assistant_tool_call(&messages, "subtract"));
        assert_canonical_assistant_order(&messages);
    }).await;
}

fn setup(client: &gemini::Client) -> EcsAgent {
    let mut ecs = EcsAgent::new(
        client.completion_model(gemini::completion::GEMINI_2_5_FLASH),
        FORCE_TOOLS_PREAMBLE,
        1,
    );
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        DefaultMaxTurns(None),
        ToolChoiceSpec(Some(ToolChoice::Required)),
    ));
    ecs.tool(Add);
    ecs
}

fn history(ecs: &mut EcsAgent, run: Entity) -> Vec<Message> {
    let mut messages: Vec<_> = ecs
        .app
        .world_mut()
        .query_filtered::<(&ChildOf, &Order, &Parts), With<Utterance>>()
        .iter(ecs.app.world())
        .filter(|(parent, _, _)| parent.parent() == run)
        .map(|(_, order, parts)| (order.0, parts.0.to_message()))
        .collect();
    messages.sort_by_key(|(order, _)| *order);
    messages.into_iter().map(|(_, message)| message).collect()
}

#[tokio::test]
async fn builtin_streaming_max_turns_error_carries_pending_message() {
    with_gemini_cassette(
        "agent_run_streamed/builtin_streaming_max_turns_error_carries_pending_message",
        |client| async move {
            let mut ecs = setup(&client);
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &[],
                "What is 21 + 21? Use the add tool.",
                true,
                Some(2),
            );
            let error = ecs
                .wait_for_outcome(run)
                .await
                .expect_err("the stream should surface MaxTurnsError");
            let Failure::MaxTurns { limit: max_turns } = error else {
                panic!("expected MaxTurnsError, got {error:?}")
            };
            let mut chat_history = history(&mut ecs, run);
            let prompt = chat_history
                .pop()
                .expect("the actual pending message is retained in the graph");
            // Same portable predicates, on native failure/graph instead of constructing
            // a legacy PromptError that could hide missing native observations.
            assert_eq!(max_turns, 2);
            assert!(
                !chat_history.is_empty(),
                "portable max-turn history is nonempty"
            );
            assert!(
                matches!(&prompt, Message::User { content } if content.iter().next().is_some()),
                "portable pending prompt is retained"
            );
            assert!(
                is_tool_result_user_message(&prompt),
                "MaxTurnsError must carry the pending tool-results message: {prompt:?}"
            );
            assert!(
                history_has_assistant_tool_call(&chat_history, "add"),
                "the error history must include the assistant tool-call turn: {chat_history:?}"
            );
        },
    )
    .await;
}

type Unissued<'w, 's> = Query<
    'w,
    's,
    (&'static ChildOf, &'static PendingEffect),
    (Without<Issued>, Without<EffectOutcome>),
>;
fn cancel_tool_dispatch(
    effects: Unissued,
    turns: Query<&ChildOf, With<Turn>>,
    mut commands: Commands,
) {
    for (parent, pending) in &effects {
        if matches!(&pending.kind, EffectKind::ToolCall { .. }) {
            let run = turns
                .get(parent.parent())
                .expect("tool belongs to a turn")
                .parent();
            commands
                .entity(run)
                .insert(Cancelled("cancelled by test hook".into()));
        }
    }
}

#[derive(Resource, Default)]
struct SawFinal(bool);
fn observe_final(results: Query<&RunResult, Added<Settled>>, mut seen: ResMut<SawFinal>) {
    seen.0 |= !results.is_empty();
}

#[tokio::test]
async fn builtin_streaming_cancellation_history_includes_assistant_turn() {
    with_gemini_cassette(
        "agent_run_streamed/builtin_streaming_cancellation_history_includes_assistant_turn",
        |client| async move {
            let mut ecs = setup(&client);
            ecs.app.init_resource::<SawFinal>().add_systems(
                RigSchedule,
                (
                    cancel_tool_dispatch.in_set(BusSet::Gate),
                    observe_final.after(rig_ecs::systems::RigSet::Settle),
                ),
            );
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &[],
                "What is 21 + 21? Use the add tool.",
                true,
                Some(2),
            );
            let error = ecs
                .wait_for_outcome(run)
                .await
                .expect_err("the hook should cancel the run");
            assert!(
                !ecs.app.world().resource::<SawFinal>().0,
                "a cancelled run must not produce a final response"
            );
            let Failure::Cancelled(report) = error else {
                panic!("expected PromptCancelled, got {error:?}")
            };
            let reason = report.message;
            let chat_history = history(&mut ecs, run);
            assert_eq!(
                reason, "cancelled by test hook",
                "portable cancellation reason"
            );
            assert!(
                reason.contains("cancelled by test hook"),
                "the hook reason must surface: {reason}"
            );
            assert!(
                history_has_assistant_tool_call(&chat_history, "add"),
                "cancellation history must include the recorded assistant turn: {chat_history:?}"
            );
        },
    )
    .await;
}
