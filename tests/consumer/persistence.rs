//! Supported pre-delivery and post-write cuts: library scene plus declared host state.
//! The mutation ledger and workspace image are resources, not implicit ECS data.

use super::{
    Approval, Case, Error, Evidence, Host, Observation, build, drive, workspace::Workspace,
};
use bevy_ecs::prelude::*;
use rig_ecs::{
    agent::{
        Run,
        scene::{SceneExtensions, WorldScene, load_world, save_world},
    },
    bus::{EffectLogResource, IdCounter},
};
use rig_effect_log::EffectLog;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Component, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct MaintenanceRun {
    pub operation: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct HostSnapshot {
    contents: String,
    writes: usize,
    proposal: Option<String>,
    approval: Option<Approval>,
    validated: bool,
    observations: Vec<Observation>,
    seen: BTreeMap<u64, (usize, bool)>,
    primary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repair: Option<super::repair::Snapshot>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Checkpoint {
    pub cut: String,
    pub scene: WorldScene,
    host: HostSnapshot,
    pub next_effect: u64,
    pub next_id: u64,
}

#[derive(Default, Resource)]
pub(super) struct Checkpoints(pub Vec<Checkpoint>);

pub(super) fn install(world: &mut World) -> Result<(), Error> {
    let mut extensions = SceneExtensions::default();
    extensions.register_component::<MaintenanceRun>("maintenance/run/v1")?;
    world.insert_resource(extensions);
    world.init_resource::<Checkpoints>();
    Ok(())
}

pub(super) fn capture_early(world: &mut World) {
    if world.resource::<Host>().case.capture_zero
        && world.resource::<Host>().observations.is_empty()
    {
        capture(world);
    }
}

pub(super) fn capture(world: &mut World) {
    let early = world.resource::<Host>().case.capture_zero
        && world.resource::<Host>().observations.is_empty();
    let restart = if early {
        world.query_filtered::<(&rig_ecs::bus::Issued, Option<&rig_ecs::bus::Streamed>), With<rig_ecs::bus::InFlight>>().iter(world)
        .find(|(_, stream)| stream.is_none_or(|stream| stream.events.is_empty() && stream.outcome.is_none()))
        .map(|(id, _)| id.0.as_u64())
    } else {
        None
    };
    let cut = if restart.is_some() {
        "before-first-delivery"
    } else {
        "after-write"
    };
    if world
        .resource::<Checkpoints>()
        .0
        .iter()
        .any(|checkpoint| checkpoint.cut == cut)
    {
        return;
    }
    let host = world.resource::<Host>();
    // The cut is after the observed write, before the next model turn. It
    // contains no unfinished delivered stream prefix and no transient inbox.
    let wrote = host.observations.iter().any(|item| {
        if let Some(state) = &host.repair {
            item.boundary == "collect.publication"
                && item
                    .data
                    .get("applied")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                && item
                    .data
                    .get("operation")
                    .and_then(serde_json::Value::as_str)
                    == state
                        .ledger
                        .last()
                        .map(|receipt| receipt.patch.operation.as_str())
        } else {
            item.boundary == "collect.outcome"
                && item.data.get("key").and_then(serde_json::Value::as_str)
                    == Some("maintenance/tool:apply_edit")
        }
    });
    let completed = host
        .repair
        .as_ref()
        .map_or(host.workspace.writes == 1, |state| {
            state.ledger.len() == 2 && state.final_report.is_none()
        });
    if restart.is_none() && (!completed || !wrote) {
        return;
    }
    let repair = match host
        .repair
        .as_ref()
        .map(|state| state.snapshot())
        .transpose()
    {
        Ok(snapshot) => snapshot,
        Err(error) => {
            world.resource_mut::<Host>().failure = Some(error.to_string());
            return;
        }
    };
    let snapshot = match host.workspace.read() {
        Ok(contents) => HostSnapshot {
            contents,
            writes: host.workspace.writes,
            proposal: host.proposal.clone(),
            approval: host.approval,
            validated: host.validated,
            observations: host.observations.clone(),
            seen: host.seen.clone(),
            primary: host.primary.clone(),
            repair,
        },
        Err(error) => {
            world.resource_mut::<Host>().failure = Some(error.to_string());
            return;
        }
    };
    match save_world(world) {
        Ok(scene) => {
            let next_id = world.resource::<IdCounter>().0;
            let next_effect = restart.unwrap_or(next_id);
            world.resource_mut::<Checkpoints>().0.push(Checkpoint {
                cut: cut.into(),
                scene,
                host: snapshot,
                next_effect,
                next_id,
            });
        }
        Err(error) => world.resource_mut::<Host>().failure = Some(error.to_string()),
    }
}

pub(crate) fn tail(log: &EffectLog, next_effect: u64) -> EffectLog {
    let mut tail = log.clone();
    tail.records
        .retain(|record| record.id.as_u64() >= next_effect);
    tail.header.signature = rig_core::effect::EffectRow::new();
    for record in &tail.records {
        tail.header
            .signature
            .insert(record.key.clone(), record.kind.family());
    }
    if let Some(deliveries) = &mut tail.header.deliveries {
        deliveries.retain(|delivery| delivery.id.as_u64() >= next_effect);
    }
    tail.header
        .stream_errors
        .retain(|id, _| id.as_u64() >= next_effect);
    tail
}

/// Negative scene checks use the consumer's actual pre-delivery checkpoint.
/// They must refuse before adding even part of the saved graph to the world.
pub(crate) fn check_refusals(case: &Case, evidence: &Evidence) -> Result<Vec<&'static str>, Error> {
    if !case.capture_zero {
        return Ok(Vec::new());
    }
    let cut = evidence
        .checkpoints
        .iter()
        .find(|cut| cut.cut == "before-first-delivery")
        .ok_or_else(|| {
            Error::Invariant("zero-progress case did not capture its required cut".into())
        })?;
    let refuse = |scene: &WorldScene, expected: &str| -> Result<(), Error> {
        let mut app = build(case, Some(&evidence.effects))?;
        let before = app.world().entities().len();
        match load_world(scene, app.world_mut()) {
            Err(error)
                if error.to_string().contains(expected)
                    && app.world().entities().len() == before =>
            {
                Ok(())
            }
            _ => Err(Error::Invariant(format!(
                "scene must refuse {expected} before spawning"
            ))),
        }
    };
    let mut unknown = cut.scene.clone();
    unknown
        .extensions
        .values_mut()
        .next()
        .ok_or_else(|| {
            Error::Invariant("consumer checkpoint lost its registered run extension".into())
        })?
        .insert("maintenance/unregistered/v1".into(), serde_json::json!({}));
    refuse(&unknown, "unregistered scene extension")?;
    let mut passed = vec!["unknown-extension-refused-before-spawn"];
    if case.stream {
        let mut partial = cut.scene.clone();
        let effect = partial
            .effects
            .effects
            .iter_mut()
            .find(|effect| effect.outcome.is_none() && effect.kind.streams())
            .ok_or_else(|| Error::Invariant("stream checkpoint has no unfinished effect".into()))?;
        let recorded = evidence
            .effects
            .records
            .iter()
            .find(|record| Some(record.id) == effect.id)
            .and_then(|record| record.events.as_ref())
            .ok_or_else(|| Error::Invariant("stream case has no recorded prefix".into()))?;
        let stream = effect.streamed.as_mut().ok_or_else(|| {
            Error::Invariant("stream checkpoint has no explicit empty state".into())
        })?;
        for event in recorded {
            stream.events.push(event.clone());
            if let rig_core::streaming::StreamEvent::BlockDelta {
                delta: rig_core::streaming::Delta::Text { text },
                ..
            } = event
            {
                stream.text.push_str(text);
                break;
            }
        }
        if stream.text.is_empty() {
            return Err(Error::Invariant(
                "prefix probe did not observe actual partial text".into(),
            ));
        }
        refuse(&partial, "already delivered progress")?;
        passed.push("unfinished-delivered-prefix-refused-before-spawn");
    }
    Ok(passed)
}

pub(crate) async fn resume(
    case: &Case,
    log: &EffectLog,
    checkpoint: &Checkpoint,
) -> Result<Evidence, Error> {
    let checkpoint: Checkpoint = serde_json::from_str(&serde_json::to_string(checkpoint)?)?;
    if case.repair != checkpoint.host.repair.is_some() {
        return Err(Error::Invariant(
            "checkpoint application state does not match selected workflow".into(),
        ));
    }
    if !matches!(
        checkpoint.cut.as_str(),
        "after-write" | "before-first-delivery"
    ) {
        return Err(Error::Invocation(format!(
            "unsupported cut {}",
            checkpoint.cut
        )));
    }
    let log_tail = tail(log, checkpoint.next_effect);
    if log_tail.records.is_empty() {
        return Err(Error::Invariant(
            "resume must create new effects after its cut".into(),
        ));
    }
    let mut app = build(case, Some(&log_tail))?;
    let workspace = Workspace::restore(&checkpoint.host.contents, checkpoint.host.writes)?;
    {
        let mut host = app.world_mut().resource_mut::<Host>();
        host.workspace = workspace;
        host.proposal = checkpoint.host.proposal.clone();
        host.approval = checkpoint.host.approval;
        host.validated = checkpoint.host.validated;
        host.observations = checkpoint.host.observations.clone();
        host.seen = checkpoint.host.seen.clone();
        host.primary = checkpoint.host.primary.clone();
        host.repair = checkpoint
            .host
            .repair
            .clone()
            .map(super::repair::State::restore)
            .transpose()?;
    }
    let loaded = load_world(&checkpoint.scene, app.world_mut())?;
    for (saved, restored) in checkpoint.scene.effects.effects.iter().zip(&loaded.effects) {
        if saved.outcome.is_some() {
            let world = app.world();
            let outcome = world
                .get::<rig_ecs::bus::EffectOutcome>(*restored)
                .map(|outcome| &outcome.0);
            let stream = world.get::<rig_ecs::bus::Streamed>(*restored);
            let published = world
                .get::<rig_ecs::bus::ToolOutputs>(*restored)
                .map(|output| &output.0);
            if serde_json::to_value(outcome)? != serde_json::to_value(&saved.outcome)?
                || serde_json::to_value(stream)? != serde_json::to_value(&saved.streamed)?
                || serde_json::to_value(published)? != serde_json::to_value(&saved.tool_outputs)?
            {
                return Err(Error::Invariant(format!(
                    "completed effect {:?} lost outcome, stream or published state during load",
                    saved.id
                )));
            }
        }
    }
    if app.world().resource::<IdCounter>().0 != checkpoint.next_id {
        return Err(Error::Invariant(
            "restored effect allocator differs from checkpoint".into(),
        ));
    }
    let run = loaded
        .graph
        .into_iter()
        .find(|entity| app.world().get::<Run>(*entity).is_some())
        .ok_or_else(|| Error::Invariant("checkpoint has no consumer run".into()))?;
    if app.world().get::<MaintenanceRun>(run).is_none() {
        return Err(Error::Invariant(
            "registered consumer extension was not restored".into(),
        ));
    }
    rig_ecs::replay::check_replayable(app.world_mut(), run, log)?;
    let recorder = app.world().resource::<EffectLogResource>().0.clone();
    rig_ecs::replay::stamp_run(app.world_mut(), run, &recorder);
    app.world_mut()
        .resource_mut::<Checkpoints>()
        .0
        .push(checkpoint);
    let evidence = drive(app, run).await?;
    if evidence.effects.records.iter().any(|record| {
        record.id.as_u64()
            < evidence
                .checkpoints
                .first()
                .map_or(0, |cut| cut.next_effect)
    }) {
        return Err(Error::Invariant(
            "completed effect re-executed after restore".into(),
        ));
    }
    Ok(evidence)
}

/// Simulate losing the persisted answer after the consumer really wrote its
/// file. Its external image/operation ledger survives; the effect must reissue
/// with its original id and require reconciliation instead of writing twice.
pub(crate) async fn check_external_recovery(
    case: &Case,
    evidence: &Evidence,
) -> Result<Option<serde_json::Value>, Error> {
    if case.fault != super::Fault::LostWriteOutcome {
        return Ok(None);
    }
    let mut cut = evidence
        .checkpoints
        .iter()
        .find(|cut| cut.cut == "after-write")
        .cloned()
        .ok_or_else(|| {
            Error::Invariant("write recovery requires an actual completed write".into())
        })?;
    let operation = if let Some(repair) = &cut.host.repair {
        if repair.writes != 2 || repair.ledger.len() != 2 {
            return Err(Error::Invariant(
                "repair recovery probe has no completed production edit".into(),
            ));
        }
        repair
            .ledger
            .get(1)
            .ok_or_else(|| Error::Invariant("repair recovery lacks production receipt".into()))?
            .patch
            .operation
            .clone()
    } else {
        "greeting-fix-v1".into()
    };
    if !case.repair && (cut.host.writes != 1 || cut.host.contents != super::TARGET) {
        return Err(Error::Invariant(
            "recovery probe has no external write to preserve".into(),
        ));
    }
    let effect = cut
        .scene
        .effects
        .effects
        .iter_mut()
        .find(|effect| {
            if case.repair {
                effect.key.as_str()==super::tool_key("repo_apply_patch") && matches!(&effect.kind,rig_core::effect::EffectKind::ToolCall{args,..} if serde_json::from_str::<serde_json::Value>(args).ok().is_some_and(|args|args.get("operation").and_then(serde_json::Value::as_str)==Some(operation.as_str())))
            } else {effect.key.as_str()==super::tool_key("apply_edit")}
        })
        .ok_or_else(|| Error::Invariant("write effect absent from checkpoint".into()))?;
    let id = effect
        .id
        .ok_or_else(|| Error::Invariant("write effect has no durable operation identity".into()))?
        .as_u64();
    effect.outcome = None;
    effect.tool_outputs = None;
    cut.next_effect = id;
    cut.host.seen.remove(&id);
    cut.host
        .observations
        .retain(|observation| observation.effect != Some(id));
    let bundle = super::artifacts::candidate(case)?;
    super::artifacts::write(&bundle.join("unanswered-write.checkpoint.json"), &cut)?;
    match resume(case, &evidence.effects, &cut).await {
        Err(error)
            if error.to_string().contains(if case.repair {
                "external edit already happened; reconcile"
            } else {
                "duplicate write operation greeting-fix-v1"
            }) =>
        {
            Ok(Some(
                serde_json::json!({"check":"lost-write-outcome-requires-reconciliation","status":"passed","operation":operation,"effect":id,"fault_input":bundle.join("unanswered-write.checkpoint.json")}),
            ))
        }
        _ => Err(Error::Invariant(
            "unanswered external write was not protected by the persisted operation ledger".into(),
        )),
    }
}
