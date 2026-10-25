//! Plugin corpus: 04. Sandboxed shell (spec `04-shell-sandbox.md`).
//!
//! [`ShellSandboxPlugin`] starts every child process, and so every `shell`
//! command, in a sandbox, by setting the app's [`Processes`] launcher:
//! `bwrap` on Linux, `sandbox-exec` on macOS. The project directory and
//! `$TMPDIR` are writable, the rest of the file system read-only, the home
//! directory hidden but for an allowlist, and the network off unless
//! `/sandbox net on` allowed it for the session. Without a sandbox program,
//! `shell` calls are refused.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ExitStatus, ToolOutput};
use rig_harness::rig_tools::Shell;
use rig_tools::process::Launcher;

/// What a refused call says when no sandbox program was found.
pub const UNAVAILABLE: &str = "sandbox unavailable";

/// What the model is told about the sandbox.
const SANDBOX_RULE: &str = "`shell` runs in a sandbox: only the project directory and $TMPDIR \
     are writable, the home directory is hidden, and the network is off unless the user allowed \
     it with `/sandbox net on`. A command the sandbox blocked says so; do not work around it.";

/// What the command output says when a write outside the writable places
/// failed (Linux, then macOS).
const WRITE_DENIED: &[&str] = &["Read-only file system", "Operation not permitted"];

/// What the command output says when the network is unreachable.
const NETWORK_DENIED: &[&str] = &[
    "Could not resolve host",
    "Temporary failure in name resolution",
    "Name or service not known",
    "Network is unreachable",
    "Could not connect to server",
    "Couldn't connect to server",
    "Connection refused",
];

/// The plugin's stage systems, to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShellSandboxSystems {
    /// In [`Stage::Gate`]: refuses `shell` calls when there is no sandbox.
    Gate,
    /// In [`Stage::Review`]: says when the sandbox blocked a failed command.
    Review,
}

/// Every child process in a sandbox, and `/sandbox`.
#[derive(Default)]
pub struct ShellSandboxPlugin;

impl Plugin for ShellSandboxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .init_resource::<SandboxConfig>()
            .init_resource::<SandboxNetwork>()
            .add_command(
                "sandbox",
                "The shell's sandbox: `/sandbox status`, `/sandbox net on|off`",
                sandbox_command,
            )
            .add_systems(Startup, announce)
            .add_systems(
                Update,
                confine
                    .run_if(resource_changed::<SandboxNetwork>)
                    .before(AgentLoop),
            )
            .configure_sets(
                AgentPass,
                (
                    ShellSandboxSystems::Gate.in_set(Stage::Gate),
                    ShellSandboxSystems::Review.in_set(Stage::Review),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    refuse_unsandboxed.in_set(ShellSandboxSystems::Gate),
                    hint.in_set(ShellSandboxSystems::Review),
                ),
            );
        app.world_mut().spawn(PromptSection::new(
            PromptSection::ORDER_ENVIRONMENT,
            "sandbox",
            SANDBOX_RULE,
        ));
    }

    /// Finds the sandbox program and keeps the launcher the other plugins
    /// built, which the sandbox wraps.
    fn finish(&self, app: &mut App) {
        let world = app.world();
        let backend = world
            .get_resource::<SandboxConfig>()
            .and_then(Backend::find);
        let unconfined = world
            .get_resource::<Processes>()
            .map(Processes::launcher)
            .unwrap_or_default();
        app.insert_resource(Sandbox {
            backend,
            unconfined,
        });
    }
}

/// What the sandbox hides and where it looks for its program. Insert one
/// before adding the plugin to change the defaults, taken from the
/// environment. The project, [`ProjectRoot`], is always writable.
#[derive(Resource, Clone, Debug)]
pub struct SandboxConfig {
    /// The home directory, hidden from commands; `None` hides nothing.
    pub home: Option<PathBuf>,
    /// Paths under `home` commands may still read.
    pub home_allowlist: Vec<PathBuf>,
    /// The temporary directory, writable, and `$TMPDIR` in commands.
    pub tmpdir: PathBuf,
    /// The `PATH` the sandbox program is looked up in.
    pub search_path: OsString,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            home: std::env::var_os("HOME").map(PathBuf::from),
            home_allowlist: vec![PathBuf::from(".cargo/registry"), PathBuf::from(".rustup")],
            tmpdir: std::env::temp_dir(),
            search_path: std::env::var_os("PATH").unwrap_or_default(),
        }
    }
}

/// Whether sandboxed commands may use the network in this session: off
/// unless `/sandbox net on`. Saved with the session.
#[derive(Resource, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct SandboxNetwork {
    /// Whether the network is on.
    pub allowed: bool,
}

/// The sandbox program found once every plugin was built, or `None`, and
/// then every `shell` call is refused; and the launcher it wraps.
#[derive(Resource, Clone, Debug)]
pub struct Sandbox {
    /// The program.
    pub backend: Option<Backend>,
    /// How processes started before the sandbox wrapped them.
    pub unconfined: Launcher,
}

/// A sandbox program.
#[derive(Clone, Debug)]
pub enum Backend {
    /// Bubblewrap, on Linux.
    Bwrap(PathBuf),
    /// `sandbox-exec`, on macOS.
    SandboxExec(PathBuf),
}

impl Backend {
    /// The program this system sandboxes with, looked up in the config's
    /// search path.
    fn find(config: &SandboxConfig) -> Option<Self> {
        let path = &config.search_path;
        if cfg!(target_os = "linux") {
            lookup("bwrap", path).map(Self::Bwrap)
        } else if cfg!(target_os = "macos") {
            lookup("sandbox-exec", path).map(Self::SandboxExec)
        } else {
            None
        }
    }

    /// The program's path.
    fn program(&self) -> &Path {
        match self {
            Self::Bwrap(program) | Self::SandboxExec(program) => program,
        }
    }

    /// The program and options every command runs under.
    fn prefix(&self, project: &Path, config: &SandboxConfig, network: bool) -> Vec<OsString> {
        let mut prefix: Vec<OsString> = vec![self.program().into()];
        match self {
            Self::Bwrap(_) => prefix.extend(bwrap_args(project, config, network)),
            Self::SandboxExec(_) => {
                let profile = seatbelt_profile(project, config, network);
                prefix.extend(["-p".into(), profile.into()]);
            }
        }
        prefix
    }
}

/// The first `program` in the directories of `search_path`.
fn lookup(program: &str, search_path: &OsStr) -> Option<PathBuf> {
    std::env::split_paths(search_path)
        .filter(|dir| !dir.as_os_str().is_empty())
        .map(|dir| dir.join(program))
        .find(|path| path.is_file())
}

/// Bubblewrap's options: everything read-only, a fresh `/dev` and `/proc`,
/// the home directory an empty read-only directory but for the allowlist,
/// the project and `$TMPDIR` writable, and no network unless `network`.
/// The sandbox dies with its parent and takes every process it started with
/// it, since they live in its own pid namespace. Commands start in the
/// directory they were started in, as bwrap keeps it.
fn bwrap_args(project: &Path, config: &SandboxConfig, network: bool) -> Vec<OsString> {
    let mut args: Vec<OsString> = ["--die-with-parent", "--unshare-pid", "--unshare-ipc"]
        .map(OsString::from)
        .to_vec();
    if !network {
        args.push("--unshare-net".into());
    }
    let mut option = |name: &str, paths: &[&Path]| {
        args.push(name.into());
        args.extend(paths.iter().map(|path| path.as_os_str().to_owned()));
    };
    option("--ro-bind", &[Path::new("/"), Path::new("/")]);
    option("--dev", &[Path::new("/dev")]);
    option("--proc", &[Path::new("/proc")]);
    if let Some(home) = &config.home {
        option("--tmpfs", &[home.as_path()]);
        for allowed in &config.home_allowlist {
            let path = home.join(allowed);
            option("--ro-bind-try", &[path.as_path(), path.as_path()]);
        }
    }
    let tmpdir = config.tmpdir.as_path();
    option("--bind", &[project, project]);
    option("--bind-try", &[tmpdir, tmpdir]);
    if let Some(home) = &config.home {
        // Last, once the mount points under it exist; not recursive, so
        // the project and `$TMPDIR` under it stay writable.
        option("--remount-ro", &[home.as_path()]);
        option("--setenv", &[Path::new("HOME"), home.as_path()]);
    }
    option("--setenv", &[Path::new("TMPDIR"), tmpdir]);
    args.push("--".into());
    args
}

/// The macOS sandbox profile: as [`bwrap_args`], in Seatbelt's language;
/// later rules win.
fn seatbelt_profile(project: &Path, config: &SandboxConfig, network: bool) -> String {
    let path = |path: &Path| {
        let text = path.to_string_lossy();
        format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
    };
    let (project, tmpdir) = (path(project), path(&config.tmpdir));
    let mut profile = format!(
        "(version 1)\n(allow default)\n(deny file-write*)\n\
         (allow file-write* (subpath {project}) (subpath {tmpdir}) (literal \"/dev/null\"))\n"
    );
    if let Some(home) = &config.home {
        profile.push_str(&format!("(deny file-read* (subpath {}))\n", path(home)));
        let allowed: String = config
            .home_allowlist
            .iter()
            .map(|allowed| format!(" (subpath {})", path(&home.join(allowed))))
            .collect();
        profile.push_str(&format!(
            "(allow file-read-metadata)\n(allow file-read* (subpath {project}) (subpath {tmpdir}){allowed})\n"
        ));
    }
    if !network {
        profile.push_str("(deny network*)\n");
    }
    profile
}

/// Starts every process from now on in the sandbox, with the network as
/// the session says: at the first frame, after a restore and after each
/// `/sandbox net`. Without a sandbox program nothing changes, and `shell`
/// calls are refused.
fn confine(
    sandbox: Res<Sandbox>,
    network: Res<SandboxNetwork>,
    config: Res<SandboxConfig>,
    root: Res<ProjectRoot>,
    processes: Res<Processes>,
) {
    let Some(backend) = &sandbox.backend else {
        return;
    };
    let unconfined = &sandbox.unconfined;
    let mut prefix = backend.prefix(&root.0, &config, network.allowed);
    prefix.extend(unconfined.prefix.iter().cloned());
    let cwd = unconfined.cwd.clone().unwrap_or_else(|| root.0.clone());
    processes.set(unconfined.clone().wrapped(prefix).in_dir(cwd));
}

/// Refuses each `shell` call when there is no sandbox to run it in.
fn refuse_unsandboxed(
    mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>,
    sandbox: Res<Sandbox>,
) {
    if sandbox.backend.is_some() {
        return;
    }
    for (request, mut decision) in &mut calls {
        if request.call.function.name.as_str() == Shell::NAME {
            decision.decide(
                Decider::plugin("shell-sandbox"),
                Outcome::refused(format!(
                    "{UNAVAILABLE}: no `bwrap` (Linux) or `sandbox-exec` (macOS) on PATH, and \
                 commands never run unsandboxed. Ask the user to install one."
                )),
            );
        }
    }
}

/// On a `shell` call whose result got its hint.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Hinted;

/// Adds a hint to each failed `shell` command the sandbox blocked, before
/// it is committed.
fn hint(
    mut calls: Query<
        (Entity, &ToolRequest, &mut Outcome, &Published),
        (ReadyToSettle, ToolCalls, Without<Hinted>),
    >,
    network: Res<SandboxNetwork>,
    mut commands: Commands,
) {
    for (call, request, mut outcome, published) in &mut calls {
        let failed = published
            .0
            .get::<ExitStatus>()
            .ok()
            .flatten()
            .is_some_and(|status| status != ExitStatus::Exited(0));
        if request.call.function.name.as_str() != Shell::NAME || !failed {
            continue;
        }
        let result = outcome.tool_result();
        let Some(why) = result
            .output()
            .as_text()
            .and_then(|output| blocked(output, network.allowed))
        else {
            continue;
        };
        let text = format!(
            "{}\nblocked by the sandbox: {why}",
            result.output().render()
        );
        *outcome = Outcome::from(result.with_output(ToolOutput::text(text)));
        // Once: Review sees the outcome each pass until it settles.
        commands.entity(call).insert(Hinted);
    }
}

/// Why the sandbox blocked a command that printed `output`, if it did.
fn blocked(output: &str, network: bool) -> Option<&'static str> {
    let says = |markers: &[&str]| markers.iter().any(|marker| output.contains(marker));
    if !network && says(NETWORK_DENIED) {
        Some("the network is off; ask the user to run `/sandbox net on`")
    } else if says(WRITE_DENIED) {
        Some(
            "only the project directory and $TMPDIR are writable; ask the user to run the \
             command outside the agent",
        )
    } else {
        None
    }
}

/// Says once, at startup, that `shell` calls are refused without a
/// sandbox.
fn announce(sandbox: Res<Sandbox>, mut notices: MessageWriter<Notice>) {
    if sandbox.backend.is_none() {
        notices.write(Notice::error(
            None,
            format!("{UNAVAILABLE}: no `bwrap` or `sandbox-exec` on PATH; shell calls are refused"),
        ));
    }
}

/// `/sandbox status` and `/sandbox net on|off`.
fn sandbox_command(
    In(args): In<CommandArgs>,
    sandbox: Res<Sandbox>,
    config: Res<SandboxConfig>,
    root: Res<ProjectRoot>,
    mut network: ResMut<SandboxNetwork>,
    mut notices: MessageWriter<Notice>,
) {
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let said = match words.as_slice() {
        [] | ["status"] => status(&sandbox, &config, &root.0, *network),
        ["net", "on"] => {
            network.allowed = true;
            "Sandbox network on for this session.".to_owned()
        }
        ["net", "off"] => {
            network.allowed = false;
            "Sandbox network off.".to_owned()
        }
        _ => {
            let usage = "usage: /sandbox status | /sandbox net on | /sandbox net off";
            notices.write(Notice::error(args.agent, usage));
            return;
        }
    };
    notices.write(Notice::info(args.agent, said));
}

/// The sandbox's profile, as `/sandbox status` shows it.
fn status(
    sandbox: &Sandbox,
    config: &SandboxConfig,
    project: &Path,
    network: SandboxNetwork,
) -> String {
    let Some(backend) = &sandbox.backend else {
        return format!("{UNAVAILABLE}: shell calls are refused.");
    };
    let hidden = match &config.home {
        Some(home) => {
            let allowed: Vec<String> = config
                .home_allowlist
                .iter()
                .map(|path| format!("~/{}", path.display()))
                .collect();
            format!("{} (except {})", home.display(), allowed.join(", "))
        }
        None => "nothing".to_owned(),
    };
    format!(
        "Sandbox: {}\nwritable: {}, {}\nhidden: {hidden}\nnetwork: {}",
        backend.program().display(),
        project.display(),
        config.tmpdir.display(),
        if network.allowed { "on" } else { "off" },
    )
}

#[cfg(test)]
mod tests;
