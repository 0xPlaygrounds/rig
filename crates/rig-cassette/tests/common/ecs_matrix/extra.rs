//! The world-only rows of the grid, the ones with no rig-agent twin:
//! §8.1/§10.1's batch holds (the #2499 semantics), error facts through
//! the world (#2495/#2499), the id-less wire's minted ids (owed from
//! #2499), and `despawn_run` on a run cancelled with a stream in flight.

use std::time::Instant;

use bevy_app::App;
use bevy_ecs::prelude::*;

use rig_agent::completion::CompletionModel;

use rig_core::error::ErrorKind;

use rig_core::message::AssistantContent;

use rig_core::message::UserContent;

use rig_core::observe::AdapterEnding;

use rig_core::observe::AdapterErrorBoundary;

use rig_core::observe::AdapterEvent;

use rig_core::streaming::Delta;

use rig_core::streaming::StreamEvent;

use rig_cassette::ecs::identity::stamp_run;
use rig_cassette::effect_log::{EffectLog, RequestCheck};
use rig_ecs::{
    agent::{
        AdditionalParams, Cancelled, Failed, Failure, MaxTokens, MessageParts, Preamble, Settled,
        ToolCallSlot, ToolPolicy, Turn, Utterance,
    },
    bus::{BusSet, EffectOutcome, Held, PendingEffect, RigSchedule, Streamed, release_hold},
    checkpoint::{RestoreMode, load_world, save_world},
    systems::{BatchHeld, RigSet, RunBusy, RunCommands},
};

use super::cells::{self, Cell};
use super::world::{one_pass, open, open_gated, tool_outputs};
use super::{Wire, corpus};
use crate::ecs_agent::EcsAgent;
use crate::goldens::families;
use crate::stream_faults::{adapter_events, endings, witnessed};

const GUARD: std::time::Duration = std::time::Duration::from_secs(300);

#[cfg(test)]
#[path = "extra/tests.rs"]
mod tests;

/// A 4xx the wire records, as the request the recording holds.
pub(crate) struct ErrorProbe {
    pub(crate) prompt: &'static str,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) additional_params: Option<serde_json::Value>,
    pub(crate) streamed: bool,
    /// The recorded status.
    pub(crate) status: u16,
    /// The report's `code`: the transport's own machine code when it gave
    /// one apart from the body (a gRPC code, an AWS exception type), else
    /// the string the body names under `error.code`, `error.status` or
    /// `error.type` (`ProviderResponseError::machine_code`, CONTRACT §5);
    /// `None` on a wire whose envelope is prose (Venice's `{"error":"…"}`).
    pub(crate) code: Option<&'static str>,
}

/// Row 13: a 4xx the wire records, driven through `spawn_run`: the run
/// fails as the provider's response, the record's report carries the
/// status table's verdict and the body's code, and the witness's ending
/// names the same.
pub(crate) async fn error_facts<M: CompletionModel + 'static>(
    model: M,
    probe: ErrorProbe,
    golden: impl FnOnce(&EffectLog),
) {
    let mut ecs = EcsAgent::new(model, "", 2);
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        Preamble(None),
        MaxTokens(probe.max_tokens),
        AdditionalParams(probe.additional_params.clone()),
    ));
    let trace = witnessed(&mut ecs.app);
    let run = ecs
        .app
        .world_mut()
        .spawn_run(ecs.agent, &[], probe.prompt, probe.streamed, None);
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
        rig_core::error::retryable_status(Some(probe.status)),
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
    golden(&log);
    assert_eq!(families(&log), [rig_core::effect::EffectFamily::Completion]);
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
                retryable: rig_core::error::retryable_status(Some(probe.status)),
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
    ecs.app
        .world_mut()
        .despawn_run(run)
        .expect("a failed run despawns");
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
    golden: impl FnOnce(&EffectLog),
) {
    let cell = &cells::SERVING_CONCURRENT_CONCURRENCY_ONE;
    let mut program = wire.program(cell);
    program.fixture = cell.name;
    let (mut app, agent, recorder, _gates) = open(wire, cell, &program);
    let run = app.world_mut().spawn_run(
        agent,
        &[],
        program.prompt,
        program.streamed,
        program.max_turns,
    );
    app.world_mut()
        .entity_mut(run)
        .insert(ToolPolicy { concurrency: 1 });
    stamp_run(app.world_mut(), run, &recorder).expect("stamp batch run");
    crate::goldens::capture_world_program(app.world_mut(), run, &recorder.log());
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
    golden(&log);
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
    load_world(&saved, app.world_mut(), RestoreMode::Strict, [])
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
    golden: impl FnOnce(&EffectLog),
) -> Vec<String> {
    let cell = &cells::SERVING_CONCURRENT_CONCURRENCY_TWO;
    let program = wire.program(cell);
    let (mut app, agent, recorder, _gates) = open(wire, cell, &program);
    let run = app.world_mut().spawn_run(
        agent,
        &[],
        program.prompt,
        program.streamed,
        program.max_turns,
    );
    app.world_mut()
        .entity_mut(run)
        .insert(ToolPolicy { concurrency: 2 });
    stamp_run(app.world_mut(), run, &recorder).expect("stamp minted-id run");
    crate::goldens::capture_world_program(app.world_mut(), run, &recorder.log());
    settle(&mut app, run).await;
    let log = recorder.log();
    golden(&log);
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
    let mut utterances: Vec<(usize, MessageParts)> = world
        .query_filtered::<(Entity, &ChildOf), With<Utterance>>()
        .iter(world)
        .filter(|(_, parent)| parent.parent() == run)
        .map(|(entity, _)| {
            (
                crate::ecs_agent::sibling_index(world, entity).expect("a child of the run"),
                rig_ecs::agent::content::parts::read_message(world, entity)
                    .expect("valid content graph"),
            )
        })
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
    world.despawn_run(run).expect("a settled run despawns");
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

/// Where a bare `Cancelled` lands on a streaming run (row 11: every cut
/// of a stream).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cut {
    /// The first non-empty text delta.
    FirstTextDelta,
    /// The first tool-call delta (a name or arguments).
    FirstToolCallDelta,
    /// The terminal record has landed in `Collect`; `Fold` has not run.
    AfterTerminal,
}

#[derive(Resource)]
struct CancelAt {
    cut: Cut,
    done: bool,
}

/// `Cancelled` at the cut, and nothing else: the stream is left to its
/// handler (CONTRACT §9.1).
fn cancel_at_cut(
    streams: Query<(&ChildOf, &Streamed, Option<&EffectOutcome>)>,
    turns: Query<&ChildOf, With<Turn>>,
    mut at: ResMut<CancelAt>,
    mut commands: Commands,
) {
    if at.done {
        return;
    }
    for (parent, stream, outcome) in &streams {
        let reached = match at.cut {
            Cut::FirstTextDelta => stream.events.iter().any(|event| {
                matches!(
                    event,
                    StreamEvent::BlockDelta {
                        delta: Delta::Text { text },
                        ..
                    } if !text.is_empty()
                )
            }),
            Cut::FirstToolCallDelta => stream
                .events
                .iter()
                .any(super::world::is_tool_call_progress),
            // A folded terminal can precede EOF. This cut promises no live
            // stream, so wait for the collector's completed effect outcome.
            Cut::AfterTerminal => outcome.is_some(),
        };
        if reached {
            let run = turns
                .get(parent.parent())
                .expect("the stream's turn")
                .parent();
            commands
                .entity(run)
                .insert(Cancelled(corpus::STOP_ON_TEXT_DELTA.to_owned()));
            at.done = true;
        }
    }
}

/// Row 15: a run cancelled while its completion still streams refuses
/// `despawn_run` with `RunBusy::InFlight` until the stream has drained,
/// then despawns.
pub(crate) async fn despawn_waits_for_the_stream<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    golden: impl FnOnce(&EffectLog),
) {
    cancel_at(wire, cell, Cut::FirstTextDelta, golden).await;
}

/// Row 11 of the failure rows: a bare `Cancelled` at `cut` of a streaming
/// run. The stream is left to its handler: where the handler had not
/// finished, `despawn_run` is `InFlight` until it drains and the record is
/// the handler's; where the terminal had landed, the record is a whole
/// completion and the run despawns at once. Nothing is committed either
/// way, and no tool the turn called is dispatched.
pub(crate) async fn cancel_at<M: CompletionModel + Clone + 'static>(
    wire: &Wire<M>,
    cell: &Cell,
    cut: Cut,
    golden: impl FnOnce(&EffectLog),
) {
    // The delta hook's cell without the hook: no delta gate on the model,
    // no despawn of the stream — a bare `Cancelled`.
    let mut cell = *cell;
    cell.program.hooks = &[];
    let program = wire.program(&cell);
    // The tool-call cut parks the stream at its first tool delta: a
    // scripted stream would otherwise finish within the pass that
    // published it, leaving nothing in flight to refuse the despawn.
    let gate = (cut == Cut::FirstToolCallDelta).then_some(true);
    let (mut app, agent, recorder, gates) = open_gated(wire, &cell, &program, gate);
    app.insert_resource(CancelAt { cut, done: false })
        .add_systems(
            RigSchedule,
            cancel_at_cut.after(BusSet::Collect).before(RigSet::Fold),
        );
    let run = app
        .world_mut()
        .spawn_run(agent, &[], program.prompt, true, program.max_turns);
    stamp_run(app.world_mut(), run, &recorder).expect("stamp cancellation run");
    crate::goldens::capture_world_program(app.world_mut(), run, &recorder.log());
    let start = Instant::now();
    loop {
        app.update();
        if app.world().get::<Failed>(run).is_some() {
            break;
        }
        assert!(
            app.world().get::<Settled>(run).is_none(),
            "{cut:?}: the run is cancelled before it answers"
        );
        assert!(
            start.elapsed() < GUARD,
            "{cut:?}: the run was not cancelled"
        );
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
    match cut {
        Cut::FirstTextDelta | Cut::FirstToolCallDelta => {
            assert_eq!(in_flight, 1, "{cut:?}: the stream is left to its handler");
            assert_eq!(app.world_mut().despawn_run(run), Err(RunBusy::InFlight));
            gates.stream.add_permits(1);
        }
        Cut::AfterTerminal => {
            assert_eq!(in_flight, 0, "{cut:?}: the terminal had landed");
        }
    }
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
        assert!(start.elapsed() < GUARD, "{cut:?}: the stream did not drain");
        tokio::task::yield_now().await;
    }
    for _ in 0..8 {
        app.update();
        tokio::task::yield_now().await;
    }
    // The record is the handler's: the one completion, whole where the
    // handler finished it; no tool was dispatched, nothing committed.
    let log = recorder.log();
    golden(&log);
    assert_eq!(
        families(&log),
        [rig_core::effect::EffectFamily::Completion],
        "{cut:?}: the one completion, no tool"
    );
    if cut == Cut::AfterTerminal {
        assert!(
            log.records[0].outcome.is_ok(),
            "{cut:?}: a whole completion: {:?}",
            log.records[0].outcome
        );
    }
    assert_eq!(
        crate::stream_faults::utterance_roles(app.world_mut(), run),
        [rig_ecs::agent::Role::User],
        "{cut:?}: nothing is committed"
    );
    app.world_mut()
        .despawn_run(run)
        .expect("a drained run despawns");
    assert!(app.world().get_entity(run).is_err(), "the run is gone");
}
