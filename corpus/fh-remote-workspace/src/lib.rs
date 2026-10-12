//! FH-09: 09. Remote workspace over SSH.
//!
//! [`RemoteWorkspacePlugin`] points the workspace at a directory on another
//! machine: `rig --remote=user@host:/path`, or `/remote user@host:/path`
//! in a session, and `/remote off` to come back. While remote, each call of
//! `read`, `write`, `edit`, `search` and `shell` admitted from then on gets a
//! handler of this crate in [`PrepareSystems::Edit`] that does the same over
//! SSH, with the same names, arguments and result formats. Every call goes
//! over one multiplexed connection (OpenSSH's `ControlMaster`); a call that
//! finds it dropped fails with "connection to host lost", the link
//! reconnects with backoff, and the next calls wait for it (30 s by
//! default). The project's instruction files are read from the remote
//! directory before each message reaches its agent, and the local project
//! and environment prompt sections are turned off meanwhile. The target is
//! saved with the session, so `/resume` points at it again. The status line
//! shows `⇄ host:/path`.

use std::path::Path;

use bevy_tasks::{IoTaskPool, TaskPool};
use futures::channel::mpsc::{UnboundedReceiver, unbounded};
use rig_harness::prelude::*;
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_core::serve::adapters::ToolAdapter;
use rig_harness::rig_ecs::effects::Handler;
use rig_tools::context::{InstructionFile, Instructions, MAX_FILE_BYTES, instruction_file};
use rig_tools::fs::{Prefix, read_prefix};
use serde::Deserialize;

mod link;
mod tools;

pub use link::Link;
use tools::{Remote, RemoteEdit, RemoteRead, RemoteSearch, RemoteShell, RemoteWrite};

/// The prompt sections of rig-basics' `ProjectContextPlugin` that name the
/// local project, turned off while the workspace is remote, by tag.
const LOCAL_SECTIONS: [&str; 2] = ["project_instructions", "environment"];

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
        let dir = normalize(dir);
        let rooted = dir.starts_with('/') || dir == "~" || dir.starts_with("~/");
        (!host.is_empty() && rooted).then(|| Self {
            host: host.to_owned(),
            dir,
        })
    }

    /// `path` resolved against the directory, `..` folded, as the remote
    /// tools name it.
    pub fn join(&self, path: &str) -> String {
        if path.starts_with('/') || path == "~" || path.starts_with("~/") {
            normalize(path)
        } else {
            normalize(&format!("{}/{path}", self.dir))
        }
    }

    /// Whether `path`, as [`join`](Self::join) gives it, lies in the
    /// directory.
    pub fn contains(&self, path: &str) -> bool {
        self.dir == "/"
            || path == self.dir
            || path.starts_with(&format!("{}/", self.dir.trim_end_matches('/')))
    }

    /// The directory and those above it, from the top down.
    fn ancestors(&self) -> Vec<String> {
        let mut dirs: Vec<String> = Path::new(&self.dir)
            .ancestors()
            .map(|dir| dir.to_string_lossy().into_owned())
            .filter(|dir| !dir.is_empty())
            .collect();
        dirs.reverse();
        dirs
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let host = self.host.rsplit('@').next().unwrap_or(&self.host);
        write!(f, "{host}:{}", self.dir)
    }
}

/// `path` with `.`, `..` and repeated slashes folded, lexically: the remote
/// file system is not at hand to follow links.
fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|last| *last != "~") {
                    parts.pop();
                }
            }
            other => parts.push(other),
        }
    }
    let body = parts.join("/");
    if path.starts_with('/') {
        format!("/{body}")
    } else {
        body
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

/// The plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteSystems {
    /// In [`PrepareSystems::Edit`]: gives the coding tools' calls their
    /// remote handlers.
    Handlers,
    /// In [`Stage::Gate`], after the commands and questions: holds each
    /// message until the remote instruction files are read.
    Instructions,
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
            .register_type::<RemoteTarget>();
        let settings = app
            .world()
            .get_resource::<RemoteSettings>()
            .cloned()
            .unwrap_or_default();
        let (sender, news) = unbounded();
        let target = app
            .world()
            .get_resource::<Invoked>()
            .and_then(|invoked| invoked.flag_value("remote"))
            .map(|target| RemoteTarget {
                target: target.to_owned(),
            })
            .unwrap_or_default();
        app.insert_resource(Link::new(settings, sender))
            .insert_resource(News(Some(news)))
            .insert_resource(target);
        app.world_mut().spawn((
            Name::new("prompt:remote"),
            RemoteSection,
            PromptSection::new(PromptSection::ORDER_PROJECT, "remote_workspace", ""),
        ));
        app.add_command(
            "remote",
            "Work on another machine: /remote user@host:/path, /remote off",
            remote_command,
        )
        .add_systems(Startup, listen)
        .add_systems(
            Update,
            point_link
                .run_if(resource_changed::<RemoteTarget>)
                .before(AgentLoop),
        )
        .add_observer(hide_new_section)
        .add_observer(tell_news)
        .add_observer(instructions_read)
        .configure_sets(
            AgentPass,
            (
                RemoteSystems::Handlers.in_set(PrepareSystems::Edit),
                RemoteSystems::Instructions
                    .in_set(Stage::Gate)
                    .after(CommandSystems)
                    .after(AskSystems),
            ),
        )
        .add_systems(
            AgentPass,
            (
                remote_handlers.in_set(RemoteSystems::Handlers),
                read_instructions.in_set(RemoteSystems::Instructions),
            ),
        );
        #[cfg(feature = "tui")]
        app.init_resource::<rig_tui::AppStatus>().add_systems(
            PostUpdate,
            show_status
                .run_if(resource_changed::<RemoteTarget>)
                .in_set(rig_tui::StatusSystems),
        );
    }
}

/// The link's news, until [`listen`] streams it into notices.
#[derive(Resource)]
struct News(Option<UnboundedReceiver<String>>);

/// A piece of the link's news, as [`Running::stream`] puts it on its
/// entity.
struct Told(String);

/// Streams the link's news onto an entity, which [`tell_news`] shows.
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

/// Shows each piece of the link's news as a notice.
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

/// On a prompt section this plugin turned off.
#[derive(Component)]
struct HiddenByRemote;

/// On this plugin's prompt section.
#[derive(Component)]
struct RemoteSection;

/// Points the link where [`RemoteTarget`] says, at startup, after a
/// restore and after `/remote`, and turns the local sections off or on.
fn point_link(
    target: Res<RemoteTarget>,
    link: Res<Link>,
    mut sections: Query<
        (Entity, &PromptSection, Has<HiddenByRemote>),
        (Allow<Disabled>, Without<RemoteSection>),
    >,
    mut ours: Query<&mut PromptSection, (With<RemoteSection>, Without<HiddenByRemote>)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let remote = Target::parse(&target.target);
    if remote.is_none() && !target.target.is_empty() {
        notices.write(Notice::error(
            None,
            format!("--remote: `{}` is no user@host:/path", target.target),
        ));
    }
    link.set_target(remote.clone());
    for (section, prompt, hidden) in &mut sections {
        if !LOCAL_SECTIONS.contains(&prompt.tag.as_str()) {
            continue;
        }
        match (&remote, hidden) {
            (Some(_), false) => {
                commands.entity(section).insert((Disabled, HiddenByRemote));
            }
            (None, true) => {
                commands
                    .entity(section)
                    .remove::<(Disabled, HiddenByRemote)>();
            }
            _ => {}
        }
    }
    for mut section in &mut ours {
        section.text = remote.as_ref().map(environment).unwrap_or_default();
    }
    if let Some(remote) = &remote {
        notices.write(Notice::info(
            None,
            format!("Working in {remote}, over SSH."),
        ));
    } else if target.is_changed() && !target.is_added() {
        notices.write(Notice::info(None, "Working locally again."));
    }
}

/// Turns off a local section spawned while remote, such as a new agent's
/// environment.
fn hide_new_section(
    added: On<Add<PromptSection>>,
    sections: Query<&PromptSection>,
    target: Res<RemoteTarget>,
    mut commands: Commands,
) {
    let local = sections
        .get(added.entity)
        .is_ok_and(|section| LOCAL_SECTIONS.contains(&section.tag.as_str()));
    if local && Target::parse(&target.target).is_some() {
        commands
            .entity(added.entity)
            .insert((Disabled, HiddenByRemote));
    }
}

/// What the model is told of the remote workspace.
fn environment(remote: &Target) -> String {
    format!(
        "Working directory: {}, on {}, a remote machine reached over SSH. The file tools and \
         `shell` act there: paths resolve against {}, and commands run there.",
        remote.dir, remote.host, remote.dir
    )
}

/// Gives each call of the coding tools admitted while remote its remote
/// handler, after the coding tools gave it theirs.
fn remote_handlers(
    mut calls: Query<(&ToolRequest, &mut CallHandler), (Added<Admitted>, ToolCalls)>,
    link: Res<Link>,
    target: Res<RemoteTarget>,
    processes: Res<Processes>,
    paths: Option<Res<SessionPaths>>,
) {
    let Some(target) = Target::parse(&target.target) else {
        return;
    };
    let remote = Remote {
        link: link.clone(),
        target,
        launcher: processes.launcher(),
    };
    for (request, mut handler) in &mut calls {
        let remote = remote.clone();
        let erased = match request.call.function.name.as_str() {
            "read" => ErasedHandler::new(ToolAdapter::new(RemoteRead(remote))),
            "write" => ErasedHandler::new(ToolAdapter::new(RemoteWrite(remote))),
            "edit" => ErasedHandler::new(ToolAdapter::new(RemoteEdit(remote))),
            "search" => ErasedHandler::new(ToolAdapter::new(RemoteSearch(remote))),
            "shell" => ErasedHandler::new(ToolAdapter::new(RemoteShell {
                remote,
                spill: paths.as_deref().map(SessionPaths::spill),
            })),
            _ => continue,
        };
        handler.0 = Handler(erased);
    }
}

/// On a message whose instruction files were read, or are being read.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Instructed;

/// On the entity that holds a message while the remote instruction files
/// are read.
#[derive(Component)]
struct ReadingInstructions;

/// What reading the remote instruction files came to.
type Fetched = Result<InstructionsRead, rig_harness::rig_core::tool::ToolExecutionError>;

/// The remote instruction files as a prompt section's text, for the target
/// they were read from.
struct InstructionsRead {
    target: Target,
    text: String,
}

/// Holds each message to an agent, other than a command, while the remote
/// instruction files are read, so its turn's prompt has them.
fn read_instructions(
    messages: Query<
        (Entity, &Incoming, &Decision),
        (Undecided, IncomingMessages, Without<Instructed>),
    >,
    target: Res<RemoteTarget>,
    link: Res<Link>,
    processes: Res<Processes>,
    home: Option<Res<RigHome>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Some(remote) = Target::parse(&target.target) else {
        return;
    };
    for (message, incoming, decision) in &messages {
        if decision.is_decided() || incoming.command().is_some() {
            continue;
        }
        commands.entity(message).insert(Instructed);
        let (link, launcher, remote) = (link.clone(), processes.launcher(), remote.clone());
        let home = home.as_ref().map(|home| home.0.root().to_path_buf());
        let read = futures::stream::once(rig_tools::blocking(move || {
            Ok(InstructionsRead {
                text: fetch(&link, &launcher, &remote, home.as_deref()),
                target: remote,
            })
        }));
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        commands.spawn((
            Name::new("reading the remote instruction files"),
            ReadingInstructions,
            HoldOf(message),
            Running::stream(pool, &wake, read),
        ));
    }
}

/// Puts the instruction files read in the plugin's section and releases
/// the message.
fn instructions_read(
    read: On<Insert<Done<Fetched>>>,
    reads: Query<&Done<Fetched>>,
    target: Res<RemoteTarget>,
    mut ours: Query<&mut PromptSection, With<RemoteSection>>,
    mut commands: Commands,
) {
    if let Ok(Done(Ok(read))) = reads.get(read.entity)
        && Target::parse(&target.target).as_ref() == Some(&read.target)
    {
        let text = format!("{}\n\n{}", environment(&read.target), read.text);
        for mut section in &mut ours {
            section.text = text.trim_end().to_owned();
        }
    }
    commands.entity(read.entity).despawn();
}

/// The instruction files of `RIG_HOME` and of the remote directory and
/// those above it, as a prompt section's text; what could not be read is
/// left out.
fn fetch(
    link: &Link,
    launcher: &rig_tools::process::Launcher,
    remote: &Target,
    home: Option<&Path>,
) -> String {
    let mut files: Vec<InstructionFile> = home
        .and_then(instruction_file)
        .and_then(|path| {
            let contents = read_prefix(&path, MAX_FILE_BYTES).ok()?;
            Some(InstructionFile { path, contents })
        })
        .into_iter()
        .collect();
    let dirs: Vec<String> = remote
        .ancestors()
        .iter()
        .map(|dir| quote_path(dir))
        .collect();
    let script = format!(
        "for d in {}; do for n in AGENTS.md CLAUDE.md; do if [ -f \"$d/$n\" ]; then \
         printf '\\036%s\\037' \"$d/$n\"; head -c {MAX_FILE_BYTES} -- \"$d/$n\"; break; fi; \
         done; done",
        dirs.join(" ")
    );
    if let Ok(ran) = link.exec(launcher, &script, None) {
        for file in ran.out.split('\u{1e}').skip(1) {
            let Some((path, text)) = file.split_once('\u{1f}') else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }
            files.push(InstructionFile {
                path: format!("{}:{path}", remote.host).into(),
                contents: Prefix {
                    size: text.len() as u64,
                    text: text.to_owned(),
                },
            });
        }
    }
    Instructions {
        files,
        unreadable: Vec::new(),
    }
    .prompt_text()
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
