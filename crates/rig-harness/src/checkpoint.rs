//! Every tool call that may change anything reports what it changed. The
//! file tools name the files they write ([`Footprint`] `Writes(Paths)`) and
//! publish [`FilesChanged`] themselves; a call that may write anywhere
//! (`Writes(Anything)`, such as `shell`, or a tool that does not say) is
//! wrapped in rig-tools' [`Checkpointed`], which takes a snapshot of its
//! agent's [`WorkDir`] (else the [`ProjectRoot`]) before and after it runs
//! and publishes the difference as its `FilesChanged`, each change with
//! the hashes of the bytes before and after. A reviewer reads them as any
//! tool's facts:
//!
//! ```
//! use rig_harness::prelude::*;
//! use rig_harness::rig_core::tool::FilesChanged;
//!
//! fn review(calls: Query<&Published, (With<ToolRequest>, Added<Settled>)>) {
//!     for published in &calls {
//!         let changes = published.0.get::<FilesChanged>().ok().flatten();
//!         for change in changes.map(|it| it.0).unwrap_or_default() {
//!             // change.path, change.kind, change.before, change.after, change.diff
//!         }
//!     }
//! }
//! ```
//!
//! The calls of one turn share their snapshots ([`TurnCheckpoint`]): the
//! state after one call is the next one's "before", and the state before
//! the turn's first is kept, to tell what the whole turn changed or read a
//! file as it was. In a git work tree on this machine a snapshot is a tree
//! written with a private index and object directory in the session's
//! [`Scratch`], so the user's repository is never written to; elsewhere it
//! is a scan of the workspace's files. `[checkpoints] enabled = false` in
//! `settings.toml` turns it off ([`Checkpoints`]).
//!
//! What it does not see: files that git or the workspace's listing leave
//! out (ignored, hidden, `target`), and changes made between two calls of
//! a turn by anything but a tool call, such as the user's editor, which
//! the next call reports as its own. Two agents writing the same files at
//! once can see each other's changes. A call of an open tool, which has no
//! [`CallHandler`], is not wrapped, nor is a call whose footprint a plugin
//! narrowed to named paths: a plugin that does so and wants the call's
//! changes wraps its handler in a `Checkpointed` itself.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use rig_core::serve::ErasedHandler;
use rig_core::tool::{Scope, Touches};
use rig_ecs::effects::Handler;
use rig_ecs::prelude::{
    AgentPass, CallHandler, CallOf, Footprint, ForAgent, GateSystems, Settled, Stage, ToolCalls,
    ToolRequest, Undecided,
};
use rig_tools::checkpoint::{Checkpointed, Snapshot};
use serde::Deserialize;

use crate::host::session::Scratch;
use crate::per_call::base;
use crate::preview::PreviewSystems;
use crate::services::{ProjectRoot, WorkDir};
use crate::settings::AppSettingsExt;
use crate::workspace::ProjectWorkspace;

#[cfg(doc)]
use rig_core::tool::FilesChanged;

/// `[checkpoints]` in `settings.toml`: whether a call that may write
/// anywhere reports what it changed. On unless set; read each pass.
#[derive(Resource, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Checkpoints {
    /// Whether such calls are checkpointed.
    pub enabled: bool,
}

impl Default for Checkpoints {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// On a turn: the snapshots its checkpointed calls share.
#[derive(Component, Clone, Debug, Default)]
pub struct TurnCheckpoint {
    first: Arc<Mutex<Option<Snapshot>>>,
    last: Arc<Mutex<Option<Snapshot>>>,
}

impl TurnCheckpoint {
    /// The files as they were before the turn's first checkpointed call
    /// ran, once one did: compare a later snapshot with it
    /// ([`Snapshot::changes_since`]), or read a file as it was
    /// ([`Snapshot::content`], a git snapshot only).
    pub fn first(&self) -> Option<Snapshot> {
        self.first
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The files as the last checkpointed call left them, which the next
    /// one starts from; `None` before the first, and once another call that
    /// may change files settled since.
    pub fn last(&self) -> Option<Snapshot> {
        self.last
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Forgets the last snapshot, so the next call takes a fresh "before".
    fn forget_last(&self) {
        self.last
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
    }
}

/// On a call wrapped in a [`Checkpointed`], so it is wrapped once.
#[derive(Component)]
struct Checkpointing;

/// Wraps every tool call that may write anywhere in a [`Checkpointed`].
pub(crate) struct CheckpointPlugin;

impl Plugin for CheckpointPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_setting::<Checkpoints>("checkpoints")
            .add_systems(
                AgentPass,
                checkpoint
                    .after(GateSystems::After)
                    .after(PreviewSystems)
                    .before(Stage::Start)
                    .run_if(|checkpoints: Res<Checkpoints>| checkpoints.enabled),
            )
            .add_observer(forget_last);
    }
}

/// Wraps each call about to start that may write anywhere, after every
/// gate, so a footprint a plugin declared counts. A call a gate decided is
/// wrapped too, harmlessly: its handler never runs. It reads no
/// [`Decision`](rig_ecs::prelude::Decision), so it needs no order against
/// the systems that decide there.
fn checkpoint(
    mut calls: Query<
        (
            Entity,
            &Footprint,
            &mut CallHandler,
            Option<&ForAgent>,
            Option<&CallOf>,
        ),
        (ToolCalls, Undecided, Without<Checkpointing>),
    >,
    turns: Query<&TurnCheckpoint>,
    dirs: Query<&WorkDir>,
    (workspace, root, scratch): (
        Res<ProjectWorkspace>,
        Res<ProjectRoot>,
        Option<Res<Scratch>>,
    ),
    mut commands: Commands,
) {
    let mut made: HashMap<Entity, TurnCheckpoint> = HashMap::new();
    for (call, footprint, mut handler, agent, turn) in &mut calls {
        if footprint.0 != Touches::Writes(Scope::Anything) {
            continue;
        }
        let shared = match turn {
            Some(&CallOf(turn)) => match turns.get(turn) {
                Ok(shared) => shared.clone(),
                Err(_) => made
                    .entry(turn)
                    .or_insert_with(|| {
                        let shared = TurnCheckpoint::default();
                        commands.entity(turn).try_insert(shared.clone());
                        shared
                    })
                    .clone(),
            },
            None => TurnCheckpoint::default(),
        };
        let (root, _) = base(agent.map(|agent| agent.0), &dirs, &root);
        let wrapped = Checkpointed {
            inner: handler.0.0.clone(),
            workspace: workspace.0.clone(),
            root,
            keep: scratch.as_ref().and_then(|scratch| scratch.0.clone()),
            last: shared.last,
            first: shared.first,
        };
        *handler = CallHandler(Handler(ErasedHandler::new(wrapped)));
        commands.entity(call).insert(Checkpointing);
    }
}

/// When a call that may change files but is not checkpointed settles, such
/// as a file tool's, its turn forgets its last snapshot: the next
/// checkpointed call must not report that call's changes as its own.
fn forget_last(
    settled: On<Add<Settled>>,
    calls: Query<(&Footprint, &CallOf), (With<ToolRequest>, Without<Checkpointing>)>,
    turns: Query<&TurnCheckpoint>,
) {
    let Ok((footprint, &CallOf(turn))) = calls.get(settled.entity) else {
        return;
    };
    if footprint.0.reads_only() {
        return;
    }
    if let Ok(shared) = turns.get(turn) {
        shared.forget_last();
    }
}

#[cfg(test)]
mod tests;
