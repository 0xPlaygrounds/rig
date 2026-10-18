//! The worker: today's agent hosting exactly one session for `rigd`. It
//! applies the commands `rigd` forwards exactly once and exports the
//! session's events.

use bevy_app::{App, Plugin};
use bevy_ecs::error::{BevyError, ErrorContext};
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_reflect::Reflect;
use rig_ecs::session_log::ReflectSaved;
use rig_protocol::methods::WorkerEvent;

use crate::channel::ChannelEvent;

pub use rig_protocol::SourceKey;

/// The worker's plugin, added by [`worker_app`](crate::worker_app).
pub struct WorkerPlugin;

impl Plugin for WorkerPlugin {
    fn build(&self, _app: &mut App) {}
}

/// The worker's connection to `rigd`, as its session's principal.
#[derive(Resource)]
pub struct WorkerChannel {
    pub client: rig_client::Client,
}

/// On the session's main agent: the last command applied, saved in the same
/// agent-log append as what it produced.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
pub struct Applied(pub u64);

/// Events published this frame, before `rigd` journals them.
#[derive(Resource, Default)]
pub struct WorkerOutbox {
    pub events: Vec<WorkerEvent>,
}

/// Publishes an event to this session's journal.
#[derive(SystemParam)]
pub struct Publish<'w> {
    outbox: ResMut<'w, WorkerOutbox>,
}

impl Publish<'_> {
    pub fn event<E: ChannelEvent>(&mut self, e: &E) {
        let data = serde_json::to_value(e).unwrap_or_default();
        self.raw(E::KIND, data, None);
    }

    /// An event of any kind; `durable` is its saved record's key.
    pub fn raw(&mut self, kind: &str, data: serde_json::Value, durable: Option<SourceKey>) {
        self.outbox.events.push(WorkerEvent {
            kind: kind.to_owned(),
            data,
            source: durable,
        });
    }
}

/// The worker's error handler: a failing system leaves partial mutations,
/// so the worker logs it and exits with 70 for `rigd` to restart it.
pub fn fail(error: BevyError, ctx: ErrorContext) {
    eprintln!("rig: worker failed in {}: {error}", ctx.name());
    std::process::exit(70);
}
