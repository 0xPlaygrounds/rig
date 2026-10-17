//! Every tool call that may change anything reports what it changed. The
//! file tools name the files they write ([`Footprint`] `Writes(Paths)`) and
//! publish [`FilesChanged`] themselves; a call that may write anywhere
//! (`Writes(Anything)`, such as `shell`, or a tool that does not say) is
//! wrapped in rig-tools' [`Checkpointed`], which takes a snapshot of its
//! agent's [`WorkDir`] (else the [`ProjectRoot`]) just before and just
//! after it runs and publishes the difference as its `FilesChanged`, each
//! change with the [`FileState`] of each side and, where the content was
//! kept, its hunks, the text before and so the text after. A reviewer
//! reads them as any tool's facts:
//!
//! ```
//! use rig_harness::prelude::*;
//! use rig_harness::rig_core::tool::FilesChanged;
//!
//! fn review(calls: Query<&Published, (With<ToolRequest>, Added<Settled>)>) {
//!     for published in &calls {
//!         let changes = published.0.get::<FilesChanged>().ok().flatten();
//!         for change in changes.map(|it| it.0).unwrap_or_default() {
//!             // change.path, change.kind, change.hunks, change.before_text,
//!             // change.after_text(), change.unified()
//!         }
//!     }
//! }
//! ```
//!
//! The state before a turn's first checkpointed call
//! ([`TurnCheckpoint::first`]) and before each call
//! ([`CallCheckpoint::before`]) are kept; [`Snapshot::content`] reads a file
//! from either. In a git work tree on this machine a snapshot is a tree
//! written with a private index in the session's [`Scratch`], never in the
//! user's repository; elsewhere it is a scan. `[checkpoints] enabled = false`
//! turns it off ([`Checkpoints`]).
//!
//! It reports what changed while the call ran, not who changed it: another
//! process's write in that time is in it. Files git or the listing leave out
//! are unseen unless named in [`CheckpointAlso`]. An open tool's call, and a
//! call whose footprint a plugin narrowed, are not wrapped; such a plugin
//! wraps the handler in a `Checkpointed` itself.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use bevy_app::{App, Plugin, Startup};
use bevy_ecs::prelude::*;
use rig_core::serve::ErasedHandler;
use rig_core::tool::{Scope, Touches};
use rig_ecs::effects::Handler;
use rig_ecs::prelude::{
    AgentPass, CallHandler, CallOf, Footprint, ForAgent, GateSystems, ReadyToStart, ToolCalls,
    TurnOf,
};
use rig_tools::checkpoint::{Checkpointed, Snapshot, Taking};
use serde::Deserialize;

use crate::approval::ApprovalSystems;
use crate::preview::PreviewSystems;
use rig_harness::host::session::Scratch;
use rig_harness::per_call::base;
use rig_harness::services::{Processes, ProjectRoot, WorkDir};
use rig_harness::settings::AppSettingsExt;
use rig_harness::workspace::ProjectWorkspace;

#[cfg(doc)]
use rig_core::tool::{FileState, FilesChanged};

/// `[checkpoints]` in `settings.toml`: whether a call that may write
/// anywhere reports what it changed, read each pass, and the ignored paths
/// it counts too ([`CheckpointAlso`]).
#[derive(Resource, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Checkpoints {
    /// Whether such calls are checkpointed. On unless set.
    pub enabled: bool,
    /// Paths every checkpoint counts whole although ignored, merged into
    /// [`CheckpointAlso`] when the plugin builds.
    pub also: Vec<PathBuf>,
}

impl Default for Checkpoints {
    fn default() -> Self {
        Self {
            enabled: true,
            also: Vec::new(),
        }
    }
}

/// Paths every checkpoint counts whole although ignore rules, hidden names
/// or `target` would leave them out, such as a build output a plugin
/// watches: relative to the checkpointed directory, or absolute under it.
/// `[checkpoints] also = ["dist"]` fills it; a plugin adds its own when it
/// builds, read whenever a call is wrapped:
///
/// ```
/// use rig_basics::checkpoint::CheckpointAlso;
/// use rig_harness::prelude::*;
///
/// fn build(app: &mut App) {
///     app.world_mut().get_resource_or_init::<CheckpointAlso>().0.push("dist".into());
/// }
/// # build(&mut App::new());
/// ```
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointAlso(pub Vec<PathBuf>);

/// On a turn: the snapshot before its first checkpointed call.
#[derive(Component, Clone, Debug, Default)]
pub struct TurnCheckpoint {
    first: Arc<Mutex<Option<Snapshot>>>,
}

/// The newest snapshot of each checkpointed directory in the run, which
/// the next call's snapshots start from ([`Checkpointed::newest`]).
#[derive(Resource, Default)]
struct Newest(HashMap<PathBuf, Arc<Mutex<Option<Snapshot>>>>);

impl TurnCheckpoint {
    /// The files as they were before the turn's first checkpointed call
    /// ran, once one did: compare a later snapshot with it
    /// ([`Snapshot::changes_since`]), or read a file as it was
    /// ([`Snapshot::content`], when the session has a scratch to keep it).
    pub fn first(&self) -> Option<Snapshot> {
        self.first
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// On a call wrapped in a [`Checkpointed`] (so it is wrapped once): the
/// snapshot taken just before it ran.
#[derive(Component, Clone, Debug, Default)]
pub struct CallCheckpoint {
    before: Arc<Mutex<Option<Snapshot>>>,
}

impl CallCheckpoint {
    /// The files as they were just before the call ran, once it started:
    /// compare a later snapshot with it ([`Snapshot::changes_since`]), or
    /// read a file as the call found it ([`Snapshot::content`], when the
    /// session has a scratch to keep it).
    pub fn before(&self) -> Option<Snapshot> {
        self.before
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// Wraps every tool call that may write anywhere in a [`Checkpointed`]. A
/// plugin that reads what such calls changed adds it, unless
/// `app.is_plugin_added::<CheckpointPlugin>()`.
#[derive(Default)]
pub struct CheckpointPlugin;

impl Plugin for CheckpointPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<Newest>()
            .init_resource::<Processes>()
            .add_setting::<Checkpoints>("checkpoints");
        let also = app.world().get_resource::<Checkpoints>();
        let also = also.map(|it| it.also.clone()).unwrap_or_default();
        app.world_mut()
            .get_resource_or_init::<CheckpointAlso>()
            .0
            .extend(also);
        if let Err(failure) = app.try_register_required_components::<TurnOf, TurnCheckpoint>() {
            bevy_log::warn!("turns will not share a checkpoint: {failure}");
        }
        app.add_systems(
            AgentPass,
            checkpoint
                .after(GateSystems::After)
                .after(PreviewSystems)
                .after(ApprovalSystems)
                .before(GateSystems::Fallback)
                .run_if(|checkpoints: Res<Checkpoints>| checkpoints.enabled),
        )
        .add_systems(Startup, forget_kept);
    }
}

/// Removes the content earlier runs' scans kept in the session scratch,
/// before any call is checkpointed: no snapshot outlives its run.
fn forget_kept(scratch: Option<Res<Scratch>>) {
    if let Some(dir) = scratch.as_ref().and_then(|scratch| scratch.0.as_deref())
        && let Err(error) = rig_tools::checkpoint::forget_kept(dir)
    {
        bevy_log::warn!("could not remove the content earlier checkpoints kept: {error}");
    }
}

/// Wraps each call about to start that may write anywhere, after every
/// gate, so a footprint a plugin declared counts, and before the kernel's
/// fallback answers, so before the conversations saved ahead of Start (a
/// save reads the whole world). A call a gate decided is
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
        (ToolCalls, ReadyToStart, Without<CallCheckpoint>),
    >,
    turns: Query<&TurnCheckpoint>,
    dirs: Query<&WorkDir>,
    mut newest: ResMut<Newest>,
    (workspace, root, scratch, also, processes): (
        Res<ProjectWorkspace>,
        Res<ProjectRoot>,
        Option<Res<Scratch>>,
        Option<Res<CheckpointAlso>>,
        Res<Processes>,
    ),
    mut commands: Commands,
) {
    for (call, footprint, mut handler, agent, turn) in &mut calls {
        if footprint.0 != Touches::Writes(Scope::Anything) {
            continue;
        }
        let shared = turn
            .and_then(|&CallOf(turn)| turns.get(turn).ok())
            .cloned()
            .unwrap_or_default();
        let (root, _) = base(agent.map(|agent| agent.0), &dirs, &root);
        let checkpoint = CallCheckpoint::default();
        let wrapped = Checkpointed {
            inner: handler.0.0.clone(),
            workspace: workspace.0.clone(),
            newest: newest.0.entry(root.clone()).or_default().clone(),
            root,
            taking: Taking {
                keep: scratch.as_ref().and_then(|scratch| scratch.0.clone()),
                also: also.as_ref().map(|also| also.0.clone()).unwrap_or_default(),
                launcher: processes.launcher(),
            },
            first: shared.first,
            before: checkpoint.before.clone(),
        };
        *handler = CallHandler(Handler(ErasedHandler::new(wrapped)));
        commands.entity(call).insert(checkpoint);
    }
}

#[cfg(test)]
mod tests;
