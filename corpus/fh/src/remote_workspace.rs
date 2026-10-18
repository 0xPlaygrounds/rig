//! FH-09: 09. Remote workspace.
//!
//! [`RemoteWorkspacePlugin`] points the app's whole [`ProjectWorkspace`]
//! at a directory on one of the hosts `[remote.hosts]` names
//! (rig-basics' [`RemoteSettings`]): `rig --remote bench:~/app`, or
//! `/remote bench:~/app` in a session, and `/remote off` to come back.
//! While remote, the workspace is that host's rig-tools
//! [`Remote`](rig_tools::remote::Remote) and the [`ProjectRoot`] the
//! remote directory, so every file tool, the shell and `@path`
//! attachments act there, as does any plugin that
//! goes through the workspace; the others learn it is remote from its
//! `host()`. Paths written `HOST:PATH` still reach the other hosts. A call
//! whose transport failed fails with "the transport to host failed" (a
//! command's call in Review, from its exit status). The target is saved
//! with the session, so `/resume` points at it again. The status line
//! shows `⇄ host:/path`.

use std::path::PathBuf;
use std::sync::Arc;

use rig_basics::coding_tools::remote::RemoteSettings;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ExitStatus, ToolExecutionError, ToolResult};
use rig_tools::remote::{Hosts, TRANSPORT_FAILED};
use rig_tools::split_status;
use rig_tools::workspace::{Local, Workspace};

/// The remote workspace, `host:/path`, or empty when local. Saved with the
/// session, so `/resume` points at it again.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved, Default)]
pub struct RemoteTarget {
    /// `host:/path`; empty is local.
    pub target: String,
}

/// A remote workspace: the host, by its name in `[remote.hosts]`, and the
/// directory there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    /// The host's name, such as `bench`.
    pub host: String,
    /// The project directory there, absolute or under `~`.
    pub dir: String,
}

impl Target {
    /// `host:/path`, or `None` when it is not one.
    pub fn parse(spec: &str) -> Option<Self> {
        let (host, dir) = spec.trim().split_once(':')?;
        let dir = dir.trim_end_matches('/');
        let dir = if dir.is_empty() { "/" } else { dir };
        let rooted = dir.starts_with('/') || dir == "~" || dir.starts_with("~/");
        (!host.is_empty() && rooted).then(|| Self {
            host: host.to_owned(),
            dir: dir.to_owned(),
        })
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.host, self.dir)
    }
}

/// The plugin's stage system: failing a call whose command's transport
/// failed, in [`ReviewSystems::Edit`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RemoteSystems;

/// The remote workspace, `/remote` and `--remote`.
#[derive(Default)]
pub struct RemoteWorkspacePlugin;

impl Plugin for RemoteWorkspacePlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<RemoteSettings>("remote")
            .add_flag(
                Flag::new("remote", "Work in a directory on a host of [remote.hosts]")
                    .value("HOST:PATH"),
            )
            .init_resource::<Processes>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<Pointed>()
            .register_type::<RemoteTarget>();
        let target = app
            .world()
            .get_resource::<Invoked>()
            .and_then(|invoked| invoked.flag_value("remote"))
            .map(|target| RemoteTarget {
                target: target.to_owned(),
            })
            .unwrap_or_default();
        app.insert_resource(target)
            .add_command(
                "remote",
                "Work on another machine: /remote host:/path, /remote off",
                remote_command,
            )
            .configure_sets(AgentPass, RemoteSystems.in_set(ReviewSystems::Edit))
            .add_systems(AgentPass, fail_lost.in_set(RemoteSystems))
            .add_systems(
                Update,
                point
                    .run_if(resource_changed::<RemoteTarget>)
                    .before(AgentLoop),
            );
        // With `--remote`, every plugin finds the remote workspace when it
        // finishes.
        point(app.world_mut());
        #[cfg(feature = "tui")]
        app.init_resource::<rig_tui::AppStatus>().add_systems(
            PostUpdate,
            show_status
                .run_if(resource_changed::<RemoteTarget>)
                .in_set(rig_tui::StatusSystems::Show),
        );
    }
}

/// The target pointed at last, and the local project directory to come
/// back to.
#[derive(Resource, Default)]
struct Pointed {
    at: Option<Target>,
    local_root: Option<PathBuf>,
}

/// Puts in the workspace [`RemoteTarget`] names, at startup, after a
/// restore and after `/remote`, with the [`ProjectRoot`] there. Either way
/// the workspace reaches every host's `HOST:PATH`.
fn point(world: &mut World) {
    let spec = world
        .get_resource::<RemoteTarget>()
        .map(|target| target.target.clone())
        .unwrap_or_default();
    let launcher = world
        .get_resource::<Processes>()
        .map(Processes::launcher)
        .unwrap_or_default();
    let (remotes, _) = world
        .get_resource::<RemoteSettings>()
        .map(|settings| settings.remotes(&launcher))
        .unwrap_or_default();
    let target = Target::parse(&spec);
    let remote = target.as_ref().and_then(|target| {
        remotes
            .iter()
            .find(|remote| remote.name() == target.host)
            .cloned()
    });
    let telling = world.contains_resource::<Messages<Notice>>();
    if telling && remote.is_none() && !spec.is_empty() {
        let refused = format!("--remote: `{spec}` is no host:/path of a host in [remote.hosts]");
        world.write_message(Notice::error(None, refused));
    }
    let target = target.filter(|_| remote.is_some());
    let root = world
        .get_resource::<ProjectRoot>()
        .map(|root| root.0.clone());
    let Some(mut pointed) = world.get_resource_mut::<Pointed>() else {
        return;
    };
    if pointed.at == target {
        return;
    }
    if pointed.at.is_none() {
        pointed.local_root = root;
    }
    pointed.at.clone_from(&target);
    let local_root = pointed.local_root.clone().unwrap_or_default();
    let (inner, root, said): (Arc<dyn Workspace>, _, _) = match (remote, target) {
        (Some(remote), Some(target)) => (
            Arc::new(remote),
            PathBuf::from(&target.dir),
            format!("Working in {target}."),
        ),
        _ => (
            Arc::new(Local),
            local_root,
            "Working locally again.".to_owned(),
        ),
    };
    world.insert_resource(ProjectWorkspace(Arc::new(Hosts::new(inner, remotes))));
    world.insert_resource(ProjectRoot(root));
    if telling {
        world.write_message(Notice::info(None, said));
    }
}

/// On a call failed for the transport that failed.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Lost;

/// Fails a call whose command ended with the transport's own exit status
/// while remote, since the transport, not the command, failed, with what
/// the command printed before. Once: Review sees the outcome each pass
/// until it settles.
fn fail_lost(
    mut calls: Query<(Entity, &mut Outcome, &Published), (ReadyToSettle, ToolCalls, Without<Lost>)>,
    pointed: Res<Pointed>,
    mut commands: Commands,
) {
    let Some(target) = &pointed.at else {
        return;
    };
    for (call, mut outcome, published) in &mut calls {
        let status = published.0.get::<ExitStatus>().ok().flatten();
        if status != Some(ExitStatus::Exited(TRANSPORT_FAILED)) {
            continue;
        }
        let result = outcome.tool_result();
        let (printed, _) = split_status(result.output().as_text().unwrap_or_default());
        let lost = format!("the transport to {} failed", target.host);
        let why = match printed.trim_end() {
            "" => lost,
            printed => format!("{lost}; it printed before:\n{printed}"),
        };
        *outcome = Outcome::from(ToolResult::failed(ToolExecutionError::other(why)));
        commands.entity(call).insert(Lost);
    }
}

/// `/remote host:/path`, `/remote off`, or `/remote` to say where the
/// workspace is.
fn remote_command(
    In(args): In<CommandArgs>,
    mut target: ResMut<RemoteTarget>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
) {
    match args.args.trim() {
        "" => {
            let said = match Target::parse(&target.target) {
                Some(remote) => format!("The workspace is {remote}."),
                None => format!("The workspace is local: {}.", root.0.display()),
            };
            notices.write(Notice::info(args.agent, said));
        }
        "off" => target.target.clear(),
        spec if Target::parse(spec).is_some() => spec.clone_into(&mut target.target),
        spec => {
            let usage = format!("/remote: `{spec}` is no host:/path; or /remote off");
            notices.write(Notice::error(args.agent, usage));
        }
    }
}

/// `⇄ host:/path` on the status line while remote.
#[cfg(feature = "tui")]
fn show_status(target: Res<RemoteTarget>, mut status: ResMut<rig_tui::AppStatus>) {
    use rig_tui::{Side, StatusItem, Tone};
    /// After the session's name; kept long.
    const REMOTE: StatusItem = StatusItem::new("fh-remote-workspace", Side::Left, 15, 17);
    let text = Target::parse(&target.target)
        .map(|remote| format!("⇄ {remote}"))
        .unwrap_or_default();
    status.0.show(REMOTE.says(text, Tone::Plain));
}

#[cfg(test)]
mod tests;
