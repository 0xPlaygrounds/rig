//! The world-only rows of the grid, the ones with no rig-agent twin:
//! §8.1/§10.1's batch holds (the #2499 semantics), error facts through
//! the world (#2495/#2499), the id-less wire's minted ids (owed from
//! #2499), and `despawn_run` on a run cancelled with a stream in flight.

use std::time::Instant;

use bevy_app::App;
use bevy_ecs::prelude::*;
use rig::completion::CompletionModel;
use rig::error::ErrorKind;
use rig::message::{AssistantContent, UserContent};
use rig::observe::{AdapterEnding, AdapterErrorBoundary, AdapterEvent};
use rig::streaming::{Delta, StreamEvent};
use rig_ecs::{
    agent::{
        AdditionalParams, Cancelled, Failed, Failure, MaxTokens, MessageParts, Order, Parts,
        Preamble, Settled, ToolCallSlot, ToolPolicy, Turn, Utterance,
        scene::{load_world, save_world},
    },
    bus::{BusSet, EffectOutcome, Held, PendingEffect, RigSchedule, Streamed, release_hold},
    systems::{BatchHeld, RigSet, RunBusy, despawn_run, spawn_run},
};
use rig_effect_log::RequestCheck;

use super::cells::{self, Cell};
use super::world::{one_pass, open, tool_outputs};
use super::{Wire, corpus};
use crate::ecs_agent::EcsAgent;
use crate::goldens::families;
use crate::stream_faults::{adapter_events, endings, witnessed};

const GUARD: std::time::Duration = std::time::Duration::from_secs(300);

/// A 4xx the wire records, as the request the recording holds.
pub(crate) struct ErrorProbe {
    pub prompt: &'static str,
    pub max_tokens: Option<u64>,
    pub additional_params: Option<serde_json::Value>,
    pub streamed: bool,
    /// The recorded status.
    pub status: u16,
    /// The report's `code`: the provider's own machine-readable code when
    /// the transport reported one apart from the body (a gRPC code, an AWS
    /// exception type — `ProviderResponseError::code`); `None` on every
    /// HTTP wire here, whose reply is a status and a body kept verbatim on
    /// the report's `provider_response.body` (the body's own `error.code`
    /// or Gemini's `status` is read off it; the witness projects Gemini's
    /// into its envelope fact). A contract gap the ledger records.
    pub code: Option<&'static str>,
}

/// Row 13: a 4xx the wire records, driven through `spawn_run`: the run
/// fails as the provider's response, the record's report carries the
/// status table's verdict and the body's code, and the witness's ending
/// names the same.
pub(crate) async fn error_facts<M: CompletionModel + 'static>(model: M, probe: ErrorProbe) {
    let mut ecs = EcsAgent::new(model, "", 2);
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        Preamble(None),
        MaxTokens(probe.max_tokens),
        AdditionalParams(probe.additional_params.clone()),
    ));
    let trace = witnessed(&mut ecs.app);
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        probe.prompt,
        probe.streamed,
        None,
    );
    let outcome = ecs.wait_for_outcome(run).await;
    let report = match &outcome {
        Err(Failure::Provider(report)) => report.clone(),
        other => panic!("the run fails as the provider's response, not {other:?}"),
    };
    for _ in 0..64 {
        ecs.app.update();
        tokio::task::yield_now().await;
    }
    assert_eq!(report.kind, ErrorKind::ProviderResponse, "{report:?}");
    assert_eq!(report.http_status, Some(probe.status), "{report:?}");
    assert_eq!(
        report.retryable,
        rig::error::retryable_status(Some(probe.status)),
        "the status table's verdict on a {}: {report:?}",
        probe.status
    );
    assert_eq!(report.code.as_deref(), probe.code, "{report:?}");
    let response = report
        .provider_response
        .as_ref()
        .expect("the reply is kept on the report");
    assert_eq!(
        response.status.map(|status| status.as_u16()),
        Some(probe.status)
    );
    assert!(!response.body.is_empty(), "the body is kept");
    // The record holds the same report.
    let log = ecs.effect_log();
    assert_eq!(families(&log), [rig::effect::EffectFamily::Completion]);
    let recorded = log.records[0]
        .outcome
        .as_ref()
        .expect_err("the record's outcome is the provider's error");
    assert_eq!(recorded.kind, report.kind);
    assert_eq!(recorded.http_status, report.http_status);
    assert_eq!(recorded.retryable, report.retryable);
    assert_eq!(recorded.code, report.code);
    // The witness names the same ending.
    assert_eq!(endings(&trace), ["provider"]);
    let events = adapter_events(&trace);
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Error {
                boundary: AdapterErrorBoundary::ProviderResponse,
                kind: "provider_response".into(),
                status: Some(probe.status),
                retryable: rig::error::retryable_status(Some(probe.status)),
            }
        }),
        "the adapter's ending: {events:?}"
    );
    assert!(
        events.contains(&AdapterEvent::Response {
            status: probe.status
        }),
        "the adapter's status: {events:?}"
    );
    // The failed run despawns: its one effect has landed.
    despawn_run(ecs.app.world_mut(), run).expect("a failed run despawns");
}

/// How a host approves a batch-held call (CONTRACT §8.1).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Approval {
    /// Remove `Held`.
    RemoveHeld,
    /// `release_hold(entity, "rig-ecs/batch")`.
    ReleaseBatchOwner,
}

/// Row 11: the two-signal program under a concurrency of one holds its
/// second call as `rig-ecs/batch`; a host that approves it by either route
/// dispatches it, the batch counts it as active, the run lands with both
/// results in call order, and a scene saved afterwards loads in a fresh
/// world (`approving_a_batch_held_call_by_any_route_keeps_the_batch_and_the_scene_consistent`,
/// on real provider bytes).
pub(crate) async fn batch_hold<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    approval: Approval,
) {
    let cell = &cells::SERVING_CONCURRENT_CONCURRENCY_ONE;
    let mut program = wire.program(cell);
    program.fixture = cell.name;
    let (mut app, agent, recorder) = open(wire, cell, &program);
    let run = spawn_run(
        app.world_mut(),
        agent,
        &[],
        program.prompt,
        program.streamed,
        program.max_turns,
    );
    app.world_mut()
        .entity_mut(run)
        .insert(ToolPolicy { concurrency: 1 });
    // One schedule pass at a time: an `update` runs to quiescence and
    // would land the batch before the hold is observable.
    let start = Instant::now();
    let held = loop {
        one_pass(app.world_mut());
        let held = app
            .world_mut()
            .query_filtered::<Entity, (With<BatchHeld>, With<Held>)>()
            .iter(app.world())
            .next();
        if let Some(held) = held {
            break held;
        }
        assert!(
            app.world().get::<Failed>(run).is_none() && app.world().get::<Settled>(run).is_none(),
            "{approval:?}: the run ended before the batch held a call"
        );
        assert!(
            start.elapsed() < GUARD,
            "{approval:?}: the batch never held a call"
        );
        tokio::task::yield_now().await;
    };
    match approval {
        Approval::RemoveHeld => {
            app.world_mut().entity_mut(held).remove::<Held>();
        }
        Approval::ReleaseBatchOwner => {
            release_hold(app.world_mut(), held, "rig-ecs/batch").expect("the batch owner releases");
        }
    }
    app.update();
    assert!(
        app.world().get::<BatchHeld>(held).is_none(),
        "{approval:?}: the batch marker follows the hold"
    );
    loop {
        app.update();
        let world = app.world_mut();
        let ended = world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some();
        let open = world
            .query_filtered::<(), (With<PendingEffect>, Without<EffectOutcome>)>()
            .iter(world)
            .count();
        if ended && open == 0 {
            break;
        }
        assert!(
            start.elapsed() < GUARD,
            "{approval:?}: the run did not land after the approval"
        );
        tokio::task::yield_now().await;
    }
    assert!(
        app.world().get::<Settled>(run).is_some(),
        "{approval:?}: the run lands after the approval: {:?}",
        app.world().get::<Failed>(run)
    );
    let log = recorder.log();
    assert_eq!(families(&log), cell.families, "{approval:?}");
    assert_eq!(
        tool_outputs(&log),
        [
            crate::support::ALPHA_SIGNAL_OUTPUT,
            crate::support::BETA_SIGNAL_OUTPUT
        ],
        "{approval:?}: both calls served, in call order"
    );
    // The scene saved afterwards loads in a fresh world whose handlers are
    // the log's replayers.
    let saved = save_world(app.world_mut()).expect("saves");
    let corpus::world::Opened { mut app, .. } =
        corpus::world::open(&program, &log, RequestCheck::Payload);
    load_world(&saved, app.world_mut())
        .unwrap_or_else(|error| panic!("{approval:?}: the scene loads: {error}"));
}

/// Row 14: the id-less wire's two calls in one turn carry distinct minted
/// ids, both answered in the adjacent utterance, the history canonical
/// (one assistant utterance with both calls, one user utterance with both
/// results, in call order). Returns the ids, for a re-run to compare.
/// Gemini's cell alone.
#[allow(dead_code)]
pub(crate) async fn minted_ids<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
) -> Vec<String> {
    let cell = &cells::SERVING_CONCURRENT_CONCURRENCY_TWO;
    let program = wire.program(cell);
    let (mut app, agent, recorder) = open(wire, cell, &program);
    let run = spawn_run(
        app.world_mut(),
        agent,
        &[],
        program.prompt,
        program.streamed,
        program.max_turns,
    );
    app.world_mut()
        .entity_mut(run)
        .insert(ToolPolicy { concurrency: 2 });
    settle(&mut app, run).await;
    let log = recorder.log();
    assert_eq!(families(&log), cell.families);
    let world = app.world_mut();
    let turns: Vec<Entity> = world
        .query_filtered::<(Entity, &ChildOf), With<Turn>>()
        .iter(world)
        .filter(|(_, parent)| parent.parent() == run)
        .map(|(turn, _)| turn)
        .collect();
    let mut slots: Vec<ToolCallSlot> = world
        .query::<(&ChildOf, &ToolCallSlot)>()
        .iter(world)
        .filter(|(parent, _)| turns.contains(&parent.parent()))
        .map(|(_, slot)| slot.clone())
        .collect();
    slots.sort_by_key(|slot| slot.index);
    assert_eq!(slots.len(), 2, "two calls in one turn: {slots:?}");
    assert_ne!(slots[0].id, slots[1].id, "distinct minted ids: {slots:?}");
    let mut utterances: Vec<(u64, MessageParts)> = world
        .query_filtered::<(&ChildOf, &Order, &Parts), With<Utterance>>()
        .iter(world)
        .filter(|(parent, _, _)| parent.parent() == run)
        .map(|(_, order, parts)| (order.0, parts.0.clone()))
        .collect();
    utterances.sort_by_key(|(order, _)| *order);
    // prompt, the call turn, the results, the answer
    assert_eq!(utterances.len(), 4, "{utterances:?}");
    let calls: Vec<_> = match &utterances[1].1 {
        MessageParts::Assistant { content, .. } => content
            .iter()
            .filter_map(|part| match part {
                AssistantContent::ToolCall(call) => Some(call.id.clone()),
                _ => None,
            })
            .collect(),
        other => panic!("the call turn: {other:?}"),
    };
    let results: Vec<_> = match &utterances[2].1 {
        MessageParts::User { content } => content
            .iter()
            .filter_map(|part| match part {
                UserContent::ToolResult(result) => Some(result.call.clone()),
                _ => None,
            })
            .collect(),
        other => panic!("the results: {other:?}"),
    };
    let ids: Vec<_> = slots.iter().map(|slot| slot.id.clone()).collect();
    assert_eq!(calls, ids, "the assistant utterance names the minted ids");
    assert_eq!(
        results, ids,
        "the adjacent utterance answers both, in call order"
    );
    despawn_run(world, run).expect("a settled run despawns");
    ids.iter().map(|id| id.to_string()).collect()
}

#[allow(dead_code)]
async fn settle(app: &mut App, run: Entity) {
    let start = Instant::now();
    loop {
        app.update();
        let world = app.world_mut();
        let ended = world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some();
        let open = world
            .query_filtered::<(), (With<PendingEffect>, Without<EffectOutcome>)>()
            .iter(world)
            .count();
        if ended && open == 0 {
            break;
        }
        assert!(start.elapsed() < GUARD, "the run did not end");
        tokio::task::yield_now().await;
    }
    for _ in 0..8 {
        app.update();
        tokio::task::yield_now().await;
    }
}

#[derive(Resource, Default)]
struct CancelledOnce(bool);

/// `Cancelled` on the first text delta, and nothing else: the stream is
/// left to its handler (CONTRACT §9.1).
fn cancel_on_text_delta(
    streams: Query<(&ChildOf, &Streamed), Without<EffectOutcome>>,
    turns: Query<&ChildOf, With<Turn>>,
    mut once: ResMut<CancelledOnce>,
    mut commands: Commands,
) {
    if once.0 {
        return;
    }
    for (parent, stream) in &streams {
        let text = stream.events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::BlockDelta {
                    delta: Delta::Text { text },
                    ..
                } if !text.is_empty()
            )
        });
        if text {
            let run = turns
                .get(parent.parent())
                .expect("the stream's turn")
                .parent();
            commands
                .entity(run)
                .insert(Cancelled(corpus::STOP_ON_TEXT_DELTA.to_owned()));
            once.0 = true;
        }
    }
}

/// Row 15: a run cancelled while its completion still streams refuses
/// `despawn_run` with `RunBusy::InFlight` until the stream has drained,
/// then despawns.
pub(crate) async fn despawn_waits_for_the_stream<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
) {
    // The delta hook's cell without the hook: no delta gate on the model,
    // no despawn of the stream — a bare `Cancelled`.
    let mut cell = *cell;
    cell.program.hooks = &[];
    let program = wire.program(&cell);
    let (mut app, agent, _recorder) = open(wire, &cell, &program);
    app.init_resource::<CancelledOnce>().add_systems(
        RigSchedule,
        cancel_on_text_delta
            .after(BusSet::Collect)
            .before(RigSet::Fold),
    );
    let run = spawn_run(
        app.world_mut(),
        agent,
        &[],
        program.prompt,
        true,
        program.max_turns,
    );
    let start = Instant::now();
    loop {
        app.update();
        if app.world().get::<Failed>(run).is_some() {
            break;
        }
        assert!(
            app.world().get::<Settled>(run).is_none(),
            "the run is cancelled before it answers"
        );
        assert!(start.elapsed() < GUARD, "the run was not cancelled");
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(
            &app.world().get::<Failed>(run).expect("failed").0,
            Failure::Cancelled(report) if report.message == corpus::STOP_ON_TEXT_DELTA
        ),
        "{:?}",
        app.world().get::<Failed>(run)
    );
    let in_flight = app
        .world_mut()
        .query_filtered::<(), (With<PendingEffect>, Without<EffectOutcome>)>()
        .iter(app.world())
        .count();
    assert_eq!(in_flight, 1, "the stream is left to its handler");
    assert_eq!(despawn_run(app.world_mut(), run), Err(RunBusy::InFlight));
    loop {
        app.update();
        let open = app
            .world_mut()
            .query_filtered::<(), (With<PendingEffect>, Without<EffectOutcome>)>()
            .iter(app.world())
            .count();
        if open == 0 {
            break;
        }
        assert!(start.elapsed() < GUARD, "the stream did not drain");
        tokio::task::yield_now().await;
    }
    for _ in 0..8 {
        app.update();
        tokio::task::yield_now().await;
    }
    despawn_run(app.world_mut(), run).expect("a drained run despawns");
    assert!(app.world().get_entity(run).is_err(), "the run is gone");
}
