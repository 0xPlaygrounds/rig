//! FH-09: 09. Remote workspace over SSH.
//!
//! [`RemoteWorkspacePlugin`] points the app's [`ProjectWorkspace`] at a
//! directory on another machine: `rig --remote user@host:/path`, or
//! `/remote user@host:/path` in a session, and `/remote off` to come back.
//! While remote, the workspace is an [`Ssh`] and the [`ProjectRoot`] the
//! remote directory, so every file tool, the shell, `@path` attachments
//! and the project's instruction files act there, as does any plugin that
//! goes through the workspace; the others learn it is remote from its
//! `host()`. Every call goes over one multiplexed connection; a dropped one
//! fails what was running with "connection to host lost", is reconnected
//! with backoff, and the next calls wait for it (30 s by default). The
//! target is saved with the session, so `/resume` points at it again. The
//! status line shows `⇄ host:/path`.

use std::path::PathBuf;
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};
use rig_harness::prelude::*;
use serde::Deserialize;

mod ssh;

pub use ssh::Ssh;

/// The `[remote]` table of `settings.toml`.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RemoteSettings {
    /// The `ssh` command and options of its own, such as a stand-in.
    pub ssh: Vec<String>,
    /// How long a call waits for a dropped connection to come back.
    pub wait_secs: u64,
}

impl Default for RemoteSettings {
    fn default() -> Self {
        Self {
            ssh: vec!["ssh".to_owned()],
            wait_secs: 30,
        }
    }
}

/// The remote workspace, `user@host:/path`, or empty when local. Saved
/// with the session, so `/resume` reconnects to it.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved, Default)]
pub struct RemoteTarget {
    /// `user@host:/path`; empty is local.
    pub target: String,
}

/// A remote workspace: the host `ssh` reaches and the directory there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    /// What `ssh` connects to, such as `dev@buildbox`.
    pub host: String,
    /// The project directory there, absolute or under `~`.
    pub dir: String,
}

impl Target {
    /// `user@host:/path`, or `None` when it is not one.
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
        let host = self.host.rsplit('@').next().unwrap_or(&self.host);
        write!(f, "{host}:{}", self.dir)
    }
}

/// `text` quoted for `sh`.
pub fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// A remote path quoted for `sh`, a leading `~` kept as the home
/// directory.
pub fn quote_path(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => format!("\"$HOME\"/{}", quote(rest)),
        None if path == "~" => "\"$HOME\"".to_owned(),
        None => quote(path),
    }
}

/// The remote workspace, `/remote` and `--remote`.
#[derive(Default)]
pub struct RemoteWorkspacePlugin;

impl Plugin for RemoteWorkspacePlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<RemoteSettings>("remote")
            .add_flag(
                Flag::new("remote", "Work in a directory on another machine, over SSH")
                    .value("USER@HOST:PATH"),
            )
            .init_resource::<Processes>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .register_type::<RemoteTarget>();
        let (sender, news) = unbounded();
        let target = app
            .world()
            .get_resource::<Invoked>()
            .and_then(|invoked| invoked.flag_value("remote"))
            .map(|target| RemoteTarget {
                target: target.to_owned(),
            })
            .unwrap_or_default();
        app.insert_resource(Pointed {
            news: sender,
            at: None,
            local_root: None,
        })
        .insert_resource(News(Some(news)))
        .insert_resource(target)
        .add_command(
            "remote",
            "Work on another machine: /remote user@host:/path, /remote off",
            remote_command,
        )
        .add_systems(Startup, listen)
        .add_systems(
            Update,
            point
                .run_if(resource_changed::<RemoteTarget>)
                .before(AgentLoop),
        )
        .add_observer(tell_news);
        // With `--remote`, every plugin finds the remote workspace when it
        // finishes.
        point(app.world_mut());
        #[cfg(feature = "tui")]
        app.init_resource::<rig_tui::AppStatus>().add_systems(
            PostUpdate,
            show_status
                .run_if(resource_changed::<RemoteTarget>)
                .in_set(rig_tui::StatusSystems),
        );
    }
}

/// Where the workspace was last pointed, the local project directory to
/// come back to, and where an [`Ssh`]'s reconnects are told.
#[derive(Resource)]
struct Pointed {
    news: UnboundedSender<String>,
    at: Option<Target>,
    local_root: Option<PathBuf>,
}

/// Puts in the workspace [`RemoteTarget`] names, at startup, after a
/// restore and after `/remote`, with the [`ProjectRoot`] there.
fn point(world: &mut World) {
    let spec = world
        .get_resource::<RemoteTarget>()
        .map(|target| target.target.clone())
        .unwrap_or_default();
    let remote = Target::parse(&spec);
    let settings = world
        .get_resource::<RemoteSettings>()
        .cloned()
        .unwrap_or_default();
    let launcher = world
        .get_resource::<Processes>()
        .map(Processes::launcher)
        .unwrap_or_default();
    let root = world
        .get_resource::<ProjectRoot>()
        .map(|root| root.0.clone());
    let telling = world.contains_resource::<Messages<Notice>>();
    if telling && remote.is_none() && !spec.is_empty() {
        let refused = format!("--remote: `{spec}` is no user@host:/path");
        world.write_message(Notice::error(None, refused));
    }
    let Some(mut pointed) = world.get_resource_mut::<Pointed>() else {
        return;
    };
    if pointed.at == remote {
        return;
    }
    let was_remote = pointed.at.is_some();
    if !was_remote {
        pointed.local_root = root;
    }
    pointed.at.clone_from(&remote);
    let news = pointed.news.clone();
    let local_root = pointed.local_root.clone().unwrap_or_default();
    let said = match remote {
        Some(target) => {
            let said = Notice::info(None, format!("Working in {target}, over SSH."));
            let root = ProjectRoot(PathBuf::from(&target.dir));
            let ssh = Ssh::new(target, settings, launcher, news);
            world.insert_resource(ProjectWorkspace(Arc::new(ssh)));
            world.insert_resource(root);
            said
        }
        None => {
            world.insert_resource(ProjectWorkspace::default());
            world.insert_resource(ProjectRoot(local_root));
            Notice::info(None, "Working locally again.")
        }
    };
    if telling {
        world.write_message(said);
    }
}

/// The reconnect news, until [`listen`] streams it into notices.
#[derive(Resource)]
struct News(Option<UnboundedReceiver<String>>);

/// A piece of reconnect news, as [`Running::stream`] puts it on its
/// entity.
struct Told(String);

/// Streams the reconnect news onto an entity, which [`tell_news`] shows.
fn listen(mut news: ResMut<News>, wake: Res<Wake>, mut commands: Commands) {
    if let Some(news) = news.0.take() {
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let told = futures::StreamExt::map(news, Told);
        commands.spawn((
            Name::new("remote link news"),
            Running::stream(pool, &wake, told),
        ));
    }
}

/// Shows each piece of reconnect news as a notice.
fn tell_news(
    told: On<Insert<Done<Told>>>,
    news: Query<&Done<Told>>,
    mut notices: MessageWriter<Notice>,
) {
    if let Ok(Done(Told(text))) = news.get(told.entity) {
        notices.write(Notice::info(None, text.clone()));
    }
}

/// `/remote user@host:/path`, `/remote off`, or `/remote` to say where the
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
                Some(remote) => format!("The workspace is {remote}, over SSH."),
                None => format!("The workspace is local: {}.", root.0.display()),
            };
            notices.write(Notice::info(args.agent, said));
        }
        "off" => target.target.clear(),
        spec if Target::parse(spec).is_some() => spec.clone_into(&mut target.target),
        spec => {
            let usage = format!("/remote: `{spec}` is no user@host:/path; or /remote off");
            notices.write(Notice::error(args.agent, usage));
        }
    }
}

/// `⇄ host:/path` on the status line while remote.
#[cfg(feature = "tui")]
fn show_status(target: Res<RemoteTarget>, mut status: ResMut<rig_tui::AppStatus>) {
    use rig_tui::{Side, StatusItem, Tone};
    /// After the session's name; kept long.
    const REMOTE: StatusItem = StatusItem::at(Side::Left, 15, 17);
    let text = Target::parse(&target.target)
        .map(|remote| format!("⇄ {remote}"))
        .unwrap_or_default();
    status.0.show(REMOTE.says(text, Tone::Plain));
}

#[cfg(test)]
mod tests;
