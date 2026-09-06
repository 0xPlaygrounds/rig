//! Raw-response hook semantics observed at the native ECS publication and
//! completed-turn boundaries. The two observations execute independently;
//! neither is copied from the other or from an effect replay.

use bevy_ecs::prelude::*;
use rig::{effect::Outcome, prelude::*, providers::anthropic, streaming::StreamEvent};
use rig_ecs::{
    agent::{MaxTokens, MaxTurns, Outputs, Preamble, Retry, Turn},
    bus::{BusSet, EffectOutcome, RigSchedule, Seq, Streamed},
    systems::RigSet,
};
use serde_json::Value;

use super::{
    super::support::{assert_ids_match_recording, recorded_response_body, with_anthropic_cassette},
    raw_capture_agent_matrix::{
        assert_all_populated, assert_distinct_msg_ids, ids_of, recorded_message_ids,
        recorded_stop_reasons,
    },
};
use crate::ecs_agent::EcsAgent;

#[derive(Resource, Default, Clone)]
struct Seen {
    responses: Vec<Value>,
    streamed: Vec<bool>,
    turns: Vec<Value>,
    tool_calls: Vec<bool>,
    retry_first: bool,
}

fn observe_response(
    effects: Query<(&EffectOutcome, Option<&Streamed>), Added<EffectOutcome>>,
    mut seen: ResMut<Seen>,
) {
    for (outcome, stream) in &effects {
        if let Ok(Outcome::Completion(response)) = &outcome.0 {
            seen.responses.push(response.raw.clone());
            seen.streamed.push(stream.is_some());
            seen.tool_calls.push(
                response
                    .choice
                    .iter()
                    .any(|part| matches!(part, rig::message::AssistantContent::ToolCall(_))),
            );
        }
    }
}

#[derive(Component)]
struct CompletionObserved;

type CompletedTurns<'w, 's> =
    Query<'w, 's, (Entity, &'static Outputs), (With<Turn>, Without<CompletionObserved>)>;

fn observe_turn(
    turns: CompletedTurns,
    effects: Query<(&ChildOf, &EffectOutcome)>,
    mut seen: ResMut<Seen>,
    mut commands: Commands,
) {
    for (turn, outputs) in &turns {
        if !outputs.done {
            continue;
        }
        let raws: Vec<_> = effects
            .iter()
            .filter_map(|(parent, outcome)| {
                if parent.parent() != turn {
                    return None;
                }
                match &outcome.0 {
                    Ok(Outcome::Completion(response)) => Some(response.raw.clone()),
                    _ => None,
                }
            })
            .collect();
        assert_eq!(raws.len(), 1, "one model outcome for the completed turn");
        seen.turns.extend(raws);
        // Outputs can change after completion (for example, usage accounting).
        // Observe the lifecycle boundary once per turn, never deduplicate raws.
        commands.entity(turn).insert(CompletionObserved);
        if seen.retry_first && seen.turns.len() == 1 {
            commands.entity(turn).insert(Retry { feedback: None });
        }
    }
}

fn setup(client: &anthropic::Client) -> EcsAgent {
    let mut ecs = EcsAgent::new(
        client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5),
        "",
        1,
    );
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert((Preamble(None), MaxTokens(Some(32))));
    ecs.app.init_resource::<Seen>().add_systems(
        RigSchedule,
        (
            observe_response.after(BusSet::Collect).before(RigSet::Fold),
            observe_turn.after(RigSet::Fold).before(RigSet::Judge),
        ),
    );
    ecs
}

#[tokio::test]
async fn hooks_observe_raw_blocking() {
    let mut observed = None;
    let output = &mut observed;
    with_anthropic_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_blocking",
        move |client| async move {
            let mut ecs = setup(&client);
            ecs.prompt("Reply with exactly: agent raw probe", false)
                .await;
            *output = Some(ecs.app.world().resource::<Seen>().clone());
        },
    )
    .await;
    let seen = observed.expect("execution completed");
    assert_eq!(seen.responses.len(), 1, "one response publication");
    assert_eq!(seen.streamed, [false]);
    assert_eq!(seen.turns.len(), 1, "one completed-turn observation");
    let raw = &seen.responses[0];
    assert!(!raw.is_null());
    assert_eq!(&seen.turns[0], raw);
    let scenario = "raw_capture_agent_matrix/hooks_observe_raw_blocking";
    let recorded = recorded_message_ids(scenario, false);
    assert_eq!(recorded.len(), 1);
    assert_ids_match_recording(&ids_of(&seen.responses, "id"), &recorded, scenario);
    let body = recorded_response_body(scenario);
    assert_eq!(raw["stop_reason"], body["stop_reason"]);
    assert_eq!(raw["model"], body["model"]);
    assert_eq!(
        raw["usage"]["output_tokens"],
        body["usage"]["output_tokens"]
    );
}

#[tokio::test]
async fn hooks_observe_raw_streamed() {
    let mut observed = None;
    let output = &mut observed;
    with_anthropic_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_streamed",
        move |client| async move {
            let mut ecs = setup(&client);
            ecs.prompt("Reply with exactly: agent raw probe", true)
                .await;
            let mut query = ecs.app.world_mut().query::<&Streamed>();
            let finals: Vec<_> = query
                .iter(ecs.app.world())
                .flat_map(|s| s.events.iter())
                .filter_map(|e| match e {
                    StreamEvent::Final(final_) => Some(final_),
                    _ => None,
                })
                .collect();
            assert_eq!(finals.len(), 1);
            assert!(!finals[0].raw.is_null());
            *output = Some(ecs.app.world().resource::<Seen>().clone());
        },
    )
    .await;
    let seen = observed.expect("execution completed");
    assert_eq!(seen.responses.len(), 1);
    assert_eq!(seen.streamed, [true]);
    assert_eq!(seen.turns.len(), 1);
    let raw = &seen.responses[0];
    assert!(!raw.is_null());
    assert_eq!(&seen.turns[0], raw);
    assert!(raw.get("message_id").is_some() && raw.get("usage").is_some());
    let scenario = "raw_capture_agent_matrix/hooks_observe_raw_streamed";
    let recorded = recorded_message_ids(scenario, true);
    assert_eq!(recorded.len(), 1);
    assert_ids_match_recording(&ids_of(&seen.responses, "message_id"), &recorded, scenario);
    assert_eq!(
        ids_of(&seen.responses, "stop_reason"),
        recorded_stop_reasons(scenario, true)
    );
}

// Collect attempt records independently of the two lifecycle observers.
fn attempt_raws(world: &mut World) -> Vec<Value> {
    let mut query = world.query::<(&Seq, &EffectOutcome)>();
    let mut attempts: Vec<_> = query
        .iter(world)
        .filter_map(|(seq, outcome)| match &outcome.0 {
            Ok(Outcome::Completion(response)) => Some((*seq, response.raw.clone())),
            _ => None,
        })
        .collect();
    attempts.sort_by_key(|(seq, _)| *seq);
    attempts.into_iter().map(|(_, raw)| raw).collect()
}

type Attempts = (Vec<Value>, Seen, Vec<Value>);

async fn run_two_attempts(client: anthropic::Client, streamed: bool, retry: bool) -> Attempts {
    let mut ecs = setup(&client);
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(MaxTurns(3));
    let prompt = if retry {
        ecs.app.world_mut().resource_mut::<Seen>().retry_first = true;
        "Reply with exactly: agent raw probe"
    } else {
        ecs.app.world_mut().entity_mut(ecs.agent).insert((
            Preamble(Some(crate::support::TOOLS_PREAMBLE.into())),
            MaxTokens(Some(1024)),
        ));
        ecs.tool(crate::support::Adder);
        "What is 2 + 3? Use the tool, then state the result."
    };
    ecs.prompt(prompt, streamed).await;
    let raws = attempt_raws(ecs.app.world_mut());
    assert_eq!(raws.len(), 2, "both attempts retained");
    let mut terminals = Vec::new();
    if streamed {
        let mut query = ecs.app.world_mut().query::<(&Seq, &Streamed)>();
        let mut streams: Vec<_> = query.iter(ecs.app.world()).collect();
        streams.sort_by_key(|(seq, _)| **seq);
        terminals = streams
            .into_iter()
            .flat_map(|(_, stream)| stream.events.iter())
            .filter_map(|event| match event {
                StreamEvent::Final(final_) => Some(final_.raw.clone()),
                _ => None,
            })
            .collect();
        assert!(!terminals.is_empty());
    }
    (raws, ecs.app.world().resource::<Seen>().clone(), terminals)
}

fn assert_two_attempts(
    scenario: &str,
    streamed: bool,
    retry: bool,
    final_only: bool,
    observed: Attempts,
) {
    let (raws, seen, terminals) = observed;
    let recorded = recorded_message_ids(scenario, streamed);
    assert_distinct_msg_ids(&recorded, scenario);
    assert_eq!(recorded.len(), 2);
    let id_key = if streamed { "message_id" } else { "id" };
    if final_only {
        let last = terminals.last().expect("terminal");
        assert_eq!(last, &raws[1]);
        assert_ids_match_recording(
            &ids_of(std::slice::from_ref(last), id_key),
            &recorded[1..],
            scenario,
        );
        assert_eq!(
            recorded_stop_reasons(scenario, true),
            [Some("tool_use".into()), Some("end_turn".into())]
        );
        assert_eq!(
            ids_of(std::slice::from_ref(last), "stop_reason"),
            [Some("end_turn".into())]
        );
        return;
    }
    assert_all_populated(&raws, scenario);
    assert_ne!(raws[0], raws[1]);
    assert_ids_match_recording(&ids_of(&raws, id_key), &recorded, scenario);
    assert_eq!(seen.responses, raws);
    assert_eq!(seen.turns, raws);
    if !retry {
        let stops = recorded_stop_reasons(scenario, streamed);
        assert_eq!(stops[0].as_deref(), Some("tool_use"));
        assert_eq!(ids_of(&raws, "stop_reason"), stops);
        assert_eq!(seen.tool_calls, [true, false]);
        if streamed {
            assert_eq!(seen.streamed, [true, true]);
        } else {
            assert_eq!(raws[0]["content"][0]["type"], "tool_use");
        }
    }
}

#[tokio::test]
async fn multi_turn_tool_run_records_distinct_raw_blocking() {
    let mut observed = None;
    let output = &mut observed;
    with_anthropic_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking",
        move |client| async move {
            *output = Some(run_two_attempts(client, false, false).await);
        },
    )
    .await;
    assert_two_attempts(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking",
        false,
        false,
        false,
        observed.expect("execution completed"),
    );
}

#[tokio::test]
async fn multi_turn_tool_run_records_distinct_raw_streamed() {
    let mut observed = None;
    let output = &mut observed;
    with_anthropic_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_streamed",
        move |client| async move {
            *output = Some(run_two_attempts(client, true, false).await);
        },
    )
    .await;
    assert_two_attempts(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_streamed",
        true,
        false,
        false,
        observed.expect("execution completed"),
    );
}

#[tokio::test]
async fn streamed_final_carries_final_turn_raw() {
    let mut observed = None;
    let output = &mut observed;
    with_anthropic_cassette(
        "raw_capture_agent_matrix/streamed_final_carries_final_turn_raw",
        move |client| async move {
            *output = Some(run_two_attempts(client, true, false).await);
        },
    )
    .await;
    assert_two_attempts(
        "raw_capture_agent_matrix/streamed_final_carries_final_turn_raw",
        true,
        false,
        true,
        observed.expect("execution completed"),
    );
}

#[tokio::test]
async fn retried_turn_records_retried_attempt_raw_blocking() {
    let mut observed = None;
    let output = &mut observed;
    with_anthropic_cassette(
        "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_blocking",
        move |client| async move {
            *output = Some(run_two_attempts(client, false, true).await);
        },
    )
    .await;
    assert_two_attempts(
        "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_blocking",
        false,
        true,
        false,
        observed.expect("execution completed"),
    );
}

#[tokio::test]
async fn retried_turn_records_retried_attempt_raw_streamed() {
    let mut observed = None;
    let output = &mut observed;
    with_anthropic_cassette(
        "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_streamed",
        move |client| async move {
            *output = Some(run_two_attempts(client, true, true).await);
        },
    )
    .await;
    assert_two_attempts(
        "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_streamed",
        true,
        true,
        false,
        observed.expect("execution completed"),
    );
}
