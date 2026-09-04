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
/// The order the world dispatches them in is then free — `Seq` still
/// orders the intake, but a record is found by id, never by position —
/// which is what lets a scene re-issue a subset of a log.
#[derive(Debug, Clone, Copy, Default)]
pub struct Replay {
    /// How each replayer compares a request with its record.
    pub check: RequestCheck,
}

impl Replay {
    /// A replay comparing requests as `check` says.
    pub fn checking(check: RequestCheck) -> Self {
        Self { check }
    }

    /// Register a by-id replayer for every key in `log`. Refuses a log of
    /// another format, and one whose signature names a family its records
    /// do not answer.
    pub fn register(
        &self,
        handlers: &mut Handlers<'_, '_>,
        log: &EffectLog,
    ) -> Result<(), ErrorReport> {
        EffectLogReplayer::check_header(log)?;
        for replayer in EffectLogReplayer::for_log_by_id(log)? {
            let key = replayer.key().clone();
            handlers.register_erased(
                key,
                rig_core::serve::ErasedHandler::new(replayer.checking(self.check)),
            )?;
        }
        Ok(())
    }

    /// Spawn one [`PendingEffect`] per record of `log`, in id order, each
    /// with its recorded id [`Reserved`] and `ChildOf` the entity of its
    /// recorded parent. Returns the entities, in record order.
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
