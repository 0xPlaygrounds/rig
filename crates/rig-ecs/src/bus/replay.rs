//! Record a world's effects into an [`EffectLog`]; replay a log through a
//! world, by id.

use bevy_ecs::prelude::*;
use rig_core::error::ErrorReport;
use rig_effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer, RequestCheck};

use super::{
    effect::{PendingEffect, Reserved},
    handlers::Handlers,
    record::Recording,
};

/// The log recorder as a resource: what [`Recording`] writes into under
/// this feature. Take the log with [`EffectLogRecorder::log`].
#[derive(Resource, Clone, Default)]
pub struct EffectLogResource(pub EffectLogRecorder);

impl EffectLogResource {
    /// Install a recorder on `world` as both this resource and the world's
    /// [`Recording`].
    pub fn install(world: &mut World, recorder: EffectLogRecorder) {
        Recording::install(world, recorder.clone());
        world.insert_resource(Self(recorder));
    }

    /// The log so far.
    pub fn log(&self) -> EffectLog {
        self.0.log()
    }
}

/// Replay a log through a world: register a replayer per key that answers
/// each dispatch **by its id** (the record with that id, or a divergence),
/// and load the log's records as effect entities under their recorded ids.
/// Saved ids let a scene re-issue a subset without positional lookup. A
/// program that mints new ids must reproduce its causal dispatch order.
/// Recorded delivery batches pace collection across schedule passes; equal
/// batch numbers deliver together, in trace order. This records observable
/// collection boundaries, not elapsed time or arbitrary world state.
#[derive(Debug, Clone, Copy, Default)]
pub struct Replay {
    /// How each replayer compares a request with its record.
    pub check: RequestCheck,
    /// Require recorded policy-visible delivery boundaries and kept events
    /// for observed streams. Otherwise available boundaries are honored, but
    /// a folded recording supplies only a final stream answer.
    pub require_delivery: bool,
}

impl Replay {
    /// A replay comparing requests as `check` says.
    pub fn checking(check: RequestCheck) -> Self {
        Self {
            check,
            ..Self::default()
        }
    }

    /// Replay policies that observe answer order or partial stream state.
    /// Refuses recordings without the delivery boundaries or kept stream
    /// events that guarantee needs. The same policy must reproduce recorded
    /// cancellations: a cancelled effect is not given a synthetic answer.
    /// A missing cancellation or inconsistent trace produces [`super::ReplayFailure`].
    ///
    /// Observe `On<Add, EffectOutcome>` or run policy after the entire
    /// `BusSet::Collect`, with the same relevant ordering live and on replay.
    /// Intermediate collector state and handler inboxes are not covered.
    /// World handlers submit [`super::WorldOutcome`] or typed [`super::Answer`];
    /// direct in-flight outcome insertions make this mode refuse the recording.
    ///
    /// Default replay honors available batches but supplies recorded
    /// cancellation errors to exchange consumers and folds streams whose
    /// events were omitted. It does not promise identical policy decisions.
    pub fn policy_visible() -> Self {
        Self {
            require_delivery: true,
            ..Self::default()
        }
    }

    /// Register a by-id replayer for every recorded or required key in `log`,
    /// including all scoped required rows. Refuses conflicting families,
    /// descriptors and inconsistent delivery metadata. Recorded semantic
    /// descriptors are preserved; reapply executable middleware separately.
    pub fn register(
        &self,
        handlers: &mut Handlers<'_, '_>,
        log: &EffectLog,
    ) -> Result<(), ErrorReport> {
        EffectLogReplayer::check_header(log)?;
        let delivery = super::delivery::ReplayDelivery::new(log, self.require_delivery)?;
        for replayer in EffectLogReplayer::for_log_by_id(log)? {
            let key = replayer.key().clone();
            let replayer = if let Some(delivery) = &delivery {
                replayer.reporting_refusals(delivery.refusals())
            } else {
                replayer
            };
            handlers.register_erased(
                key,
                rig_core::serve::ErasedHandler::new(replayer.checking(self.check)),
            )?;
        }
        handlers.replay_delivery(delivery);
        Ok(())
    }

    /// Spawn one [`PendingEffect`] per record of `log`, in id order, each
    /// with its recorded id [`Reserved`] and `ChildOf` the entity of its
    /// recorded parent. A parent outside `log` (a log's tail loaded over a
    /// checkpoint) is not in the world, so that child carries no `ChildOf`
    /// and its new record names no parent. Returns the entities, in record
    /// order.
    pub fn load(world: &mut World, log: &EffectLog) -> Vec<Entity> {
        let mut records: Vec<&rig_core::effect::EffectRecord> = log.iter().collect();
        records.sort_by_key(|record| record.id);
        let mut by_id: Vec<(rig_core::effect::EffectId, Entity)> =
            Vec::with_capacity(records.len());
        let mut spawned = Vec::with_capacity(records.len());
        for record in records {
            let mut entity = world.spawn((
                PendingEffect {
                    key: record.key.clone(),
                    kind: record.kind.clone(),
                },
                Reserved(record.id),
            ));
            if let Some(scope) = &record.scope {
                entity.insert(super::effect::Scope(scope.to_string()));
            }
            if let Some(parent) = record
                .parent
                .and_then(|parent| by_id.iter().find(|(id, _)| *id == parent))
                .map(|(_, entity)| *entity)
            {
                entity.insert(ChildOf(parent));
            }
            let id = entity.id();
            by_id.push((record.id, id));
            spawned.push(id);
        }
        spawned
    }
}
