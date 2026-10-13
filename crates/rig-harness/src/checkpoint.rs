//! Every tool call that may change anything reports what it changed. The
//! file tools name the files they write ([`Footprint`] `Writes(Paths)`) and
//! publish [`FilesChanged`] themselves; a call that may write anywhere
//! (`Writes(Anything)`, such as `shell`, or a tool that does not say) is
//! wrapped in rig-tools' [`Checkpointed`], which takes a snapshot of its
//! agent's [`WorkDir`] (else the [`ProjectRoot`]) just before and just
//! after it runs and publishes the difference as its `FilesChanged`, each
//! change with the [`FileState`] of each side and, where the content was
//! kept, its diff. A reviewer reads them as any tool's facts:
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
//! The state before a turn's first checkpointed call is kept
//! ([`TurnCheckpoint::first`]), to tell what the whole turn changed or read
//! a file as it was ([`Snapshot::content`]). Each call's snapshots start
//! from the newest one taken of its directory in the run, in any turn, so
//! a scan hashes again only what changed. In a git work tree on this
//! machine a snapshot is a tree written with a private index and object
//! directory in the session's [`Scratch`], so the user's repository is
//! never written to; elsewhere it is a scan of the workspace's files, whose
//! content is kept in the scratch for the run. `[checkpoints] enabled =
//! false` in `settings.toml` turns it off ([`Checkpoints`]).
//!
//! What it reports is what changed while the call ran, not who changed it:
//! a write by another process in that time, such as the user's editor, is
//! in it, as telling processes apart would need the operating system to
//! trace them. A call whose footprint names its paths reports exactly what
//! its tool wrote. A change made between two calls is in neither. Files git
//! or the workspace's listing leave out (ignored, hidden, `target`) are not
//! seen unless named in [`CheckpointAlso`]. A call of an open tool, which
//! has no [`CallHandler`], is not wrapped, nor is a call whose footprint a
//! plugin narrowed to named paths: a plugin that does so and wants the
//! call's changes wraps its handler in a `Checkpointed` itself.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use bevy_app::{App, Plugin, Startup};
use bevy_ecs::prelude::*;
use rig_core::serve::ErasedHandler;
use rig_core::tool::{Scope, Touches};
use rig_ecs::effects::Handler;
use rig_ecs::prelude::{
    AgentPass, CallHandler, CallOf, Footprint, ForAgent, GateSystems, Stage, ToolCalls, Undecided,
};
use rig_tools::checkpoint::{Checkpointed, Snapshot, Taking};
use serde::Deserialize;

use crate::host::session::Scratch;
use crate::per_call::base;
use crate::preview::PreviewSystems;
use crate::services::{ProjectRoot, WorkDir};
use crate::settings::AppSettingsExt;
use crate::workspace::ProjectWorkspace;

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

/// On a call wrapped in a [`Checkpointed`], so it is wrapped once.
#[derive(Component)]
struct Checkpointing;

/// Wraps every tool call that may write anywhere in a [`Checkpointed`].
/// [`RigHarnessPlugins`](crate::RigHarnessPlugins) adds it; a small test
/// app without that group adds it alone. It is not idempotent: a plugin
/// never adds it itself.
pub struct CheckpointPlugin;

impl Plugin for CheckpointPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<Newest>()
            .add_setting::<Checkpoints>("checkpoints");
        let also = app.world().get_resource::<Checkpoints>();
        let also = also.map(|it| it.also.clone()).unwrap_or_default();
        app.world_mut()
            .get_resource_or_init::<CheckpointAlso>()
            .0
            .extend(also);
        app.add_systems(
            AgentPass,
            checkpoint
                .after(GateSystems::After)
                .after(PreviewSystems)
                .before(Stage::Start)
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
    mut newest: ResMut<Newest>,
    (workspace, root, scratch, also): (
        Res<ProjectWorkspace>,
        Res<ProjectRoot>,
        Option<Res<Scratch>>,
        Option<Res<CheckpointAlso>>,
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
            newest: newest.0.entry(root.clone()).or_default().clone(),
            root,
            taking: Taking {
                keep: scratch.as_ref().and_then(|scratch| scratch.0.clone()),
                also: also.as_ref().map(|also| also.0.clone()).unwrap_or_default(),
            },
            first: shared.first,
        };
        *handler = CallHandler(Handler(ErasedHandler::new(wrapped)));
        commands.entity(call).insert(Checkpointing);
    }
}

#[cfg(test)]
mod tests;
