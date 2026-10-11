//! Plugin corpus: 04. Sandboxed shell (spec `04-shell-sandbox.md`).
//!
//! [`ShellSandboxPlugin`] turns the default `shell` tool off and adds one of
//! the same name and arguments whose commands run in a sandbox: `bwrap` on
//! Linux, `sandbox-exec` on macOS. The project directory and `$TMPDIR` are
//! writable, the rest of the file system read-only, the home directory
//! hidden but for an allowlist, and the network off unless `/sandbox net on`
//! allowed it for the session. Without a sandbox program, shell calls are
//! refused.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_harness::harness_protocol::env::AGENT_ONLY;
use rig_harness::prelude::*;
use rig_tools::{Shell, ShellArgs};
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool's name, the default shell's.
pub const SHELL: &str = "shell";

/// What a refused call says when no sandbox program was found.
pub const UNAVAILABLE: &str = "sandbox unavailable";

/// The rule the tool adds to the default shell's.
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

/// The sandboxed `shell` and `/sandbox`.
#[derive(Default)]
pub struct ShellSandboxPlugin;

impl Plugin for ShellSandboxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SandboxConfig>()
            .init_resource::<SandboxNetwork>()
            .add_command(
                "sandbox",
                "The shell's sandbox: `/sandbox status`, `/sandbox net on|off`",
                sandbox_command,
            )
            .add_systems(Startup, announce);
    }

    fn finish(&self, app: &mut App) {
        let backend = app
            .world()
            .get_resource::<SandboxConfig>()
            .and_then(Backend::find);
        app.insert_resource(Sandbox(backend));
        let world = app.world_mut();
        let mut tools = world.query::<(Entity, &ToolDef)>();
        let default = tools
            .iter(world)
            .find(|(_, def)| def.0.name.as_str() == SHELL)
            .map(|(tool, _)| tool);
        if let Some(default) = default {
            world.entity_mut(default).insert(Disabled);
        }
        let description = format!(
            "{} It runs in a sandbox: writes outside the project directory and $TMPDIR fail, \
             the home directory is hidden, and the network is off unless the user allowed it.",
            Tool::description(&Shell::default())
        );
        let rules: Vec<&str> = Shell::RULES.iter().copied().chain([SANDBOX_RULE]).collect();
        let options = ToolOptions {
            rules: &rules,
            footprint: Footprint::Exclusive,
        };
        app.add_open_tool(SHELL, &description, options, run_shell);
    }
}

/// Where sandboxed commands may write and what they see, read when the app
/// is finished and at each call. Insert one before adding the plugin to
/// change the defaults, taken from the environment.
#[derive(Resource, Clone, Debug)]
pub struct SandboxConfig {
    /// The project directory: writable, and where commands start.
    pub project: PathBuf,
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
            project: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
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

/// The sandbox program found when the app was finished, or `None`, and
/// then every shell call is refused.
#[derive(Resource, Clone, Debug)]
pub struct Sandbox(pub Option<Backend>);

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

    /// The argument list that runs `command` with `sh -c` in the sandbox.
    fn argv(&self, config: &SandboxConfig, network: bool, command: &str) -> Vec<OsString> {
        let mut argv: Vec<OsString> = vec![self.program().into()];
        match self {
            Self::Bwrap(_) => argv.extend(bwrap_args(config, network)),
            Self::SandboxExec(_) => {
                argv.extend(["-p".into(), seatbelt_profile(config, network).into()]);
            }
        }
        argv.extend(["sh".into(), "-c".into(), command.into()]);
        argv
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
/// it, since they live in its own pid namespace.
fn bwrap_args(config: &SandboxConfig, network: bool) -> Vec<OsString> {
    let mut args: Vec<OsString> = ["--die-with-parent", "--unshare-pid", "--unshare-ipc"]
        .map(OsString::from)
        .to_vec();
    if !network {
        args.push("--unshare-net".into());
    }
    let mut mount = |option: &str, paths: &[&Path]| {
        args.push(option.into());
        args.extend(paths.iter().map(|path| path.as_os_str().to_owned()));
    };
    mount("--ro-bind", &[Path::new("/"), Path::new("/")]);
    mount("--dev", &[Path::new("/dev")]);
    mount("--proc", &[Path::new("/proc")]);
    if let Some(home) = &config.home {
        mount("--tmpfs", &[home.as_path()]);
        for allowed in &config.home_allowlist {
            let path = home.join(allowed);
            mount("--ro-bind-try", &[path.as_path(), path.as_path()]);
        }
    }
    let (project, tmpdir) = (config.project.as_path(), config.tmpdir.as_path());
    mount("--bind", &[project, project]);
    mount("--bind-try", &[tmpdir, tmpdir]);
    if let Some(home) = &config.home {
        // Last, once the mount points under it exist; not recursive, so
        // the project and `$TMPDIR` under it stay writable.
        mount("--remount-ro", &[home.as_path()]);
        mount("--setenv", &[Path::new("HOME"), home.as_path()]);
    }
    mount("--setenv", &[Path::new("TMPDIR"), tmpdir]);
    mount("--chdir", &[project]);
    args.push("--".into());
    args
}

/// The macOS sandbox profile: as [`bwrap_args`], in Seatbelt's language;
/// later rules win.
fn seatbelt_profile(config: &SandboxConfig, network: bool) -> String {
    let path = |path: &Path| {
        let text = path.to_string_lossy();
        format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
    };
    let (project, tmpdir) = (path(&config.project), path(&config.tmpdir));
    let mut profile = format!(
        "(version 1)\n(allow default)\n(deny file-write*)\n\
         (allow file-write* (subpath {project}) (subpath {tmpdir}) (literal \"/dev/null\"))\n"
    );
    if let Some(home) = &config.home {
        profile.push_str(&format!("(deny file-read* (subpath {}))\n", path(home)));
        let allowed = config
            .home_allowlist
            .iter()
            .map(|allowed| format!(" (subpath {})", path(&home.join(allowed))))
            .collect::<String>();
        profile.push_str(&format!(
            "(allow file-read-metadata)\n(allow file-read* (subpath {project}) (subpath {tmpdir}){allowed})\n"
        ));
    }
    if !network {
        profile.push_str("(deny network*)\n");
    }
    profile
}

/// `argv` as one `sh` command line, each word quoted, run from `project`.
fn command_line(project: &Path, argv: &[OsString]) -> String {
    let quote = |word: &OsStr| format!("'{}'", word.to_string_lossy().replace('\'', r"'\''"));
    let words: Vec<String> = argv.iter().map(|word| quote(word)).collect();
    format!(
        "cd {} || exit 126\nexec {}",
        quote(project.as_os_str()),
        words.join(" ")
    )
}

/// The arguments of the sandboxed `shell`, the default shell's.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SandboxedArgs {
    /// The command line.
    command: String,
    /// Seconds before the command is killed.
    timeout_secs: Option<u64>,
}

/// Starts a `shell` call: the command, wrapped in the sandbox, runs through
/// the default shell's runner as the call's [`Running`] task, so an
/// interrupt, which despawns the call, drops it and kills the sandbox. With
/// no sandbox program the call is refused.
fn run_shell(
    called: On<ToolCalled<SandboxedArgs>>,
    sandbox: Res<Sandbox>,
    config: Res<SandboxConfig>,
    network: Res<SandboxNetwork>,
    paths: Option<Res<SessionPaths>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Some(backend) = &sandbox.0 else {
        let refusal = ToolExecutionError::refused(format!(
            "{UNAVAILABLE}: no `bwrap` (Linux) or `sandbox-exec` (macOS) on PATH, and commands \
             never run unsandboxed. Ask the user to install one."
        ));
        let output = ToolOutput(ToolResult::failed(refusal));
        commands.entity(called.call).insert_if_new(output);
        return;
    };
    let argv = backend.argv(&config, network.allowed, &called.args.command);
    let line = command_line(&config.project, &argv);
    let shell = Shell {
        launcher: rig_tools::process::Launcher::default().without_env(AGENT_ONLY),
        spill: paths.map(|paths| paths.spill()),
        ..Shell::default()
    };
    let work = run(shell, line, called.args.timeout_secs, network.allowed);
    let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
    commands
        .entity(called.call)
        .insert(Running::spawn_into::<ToolOutput, _>(pool, &wake, work));
}

/// Runs the wrapped command `line` through `shell`, with a hint when the
/// sandbox blocked it.
async fn run(shell: Shell, line: String, timeout_secs: Option<u64>, network: bool) -> ToolResult {
    let args = ShellArgs {
        timeout_secs,
        ..ShellArgs::new(line)
    };
    match shell.run(args).await.map(|ran| ran.output) {
        Ok(mut output) => {
            if let Some(hint) = hint(&output, network) {
                output.push_str(&format!("\n{hint}"));
            }
            ToolResult::success(output.into())
        }
        Err(error) => ToolResult::failed(error),
    }
}

/// What to tell the model when `output` shows the sandbox blocked the
/// command.
fn hint(output: &str, network: bool) -> Option<String> {
    let says = |markers: &[&str]| markers.iter().any(|marker| output.contains(marker));
    if !network && says(NETWORK_DENIED) {
        Some(
            "blocked by the sandbox: the network is off; ask the user to run `/sandbox net on`"
                .to_owned(),
        )
    } else if says(WRITE_DENIED) {
        Some(
            "blocked by the sandbox: only the project directory and $TMPDIR are writable; \
             ask the user to run the command outside the agent"
                .to_owned(),
        )
    } else {
        None
    }
}

/// Says once, at startup, that shell calls are refused without a sandbox.
fn announce(sandbox: Option<Res<Sandbox>>, mut notices: MessageWriter<Notice>) {
    if sandbox.is_some_and(|sandbox| sandbox.0.is_none()) {
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
    mut network: ResMut<SandboxNetwork>,
    mut notices: MessageWriter<Notice>,
) {
    let words: Vec<&str> = args.args.split_whitespace().collect();
    match words.as_slice() {
        [] | ["status"] => {
            notices.write(Notice::info(
                args.agent,
                status(&sandbox, &config, *network),
            ));
        }
        ["net", "on"] => {
            network.allowed = true;
            notices.write(Notice::info(
                args.agent,
                "Sandbox network on for this session.",
            ));
        }
        ["net", "off"] => {
            network.allowed = false;
            notices.write(Notice::info(args.agent, "Sandbox network off."));
        }
        _ => {
            notices.write(Notice::error(
                args.agent,
                "usage: /sandbox status | /sandbox net on | /sandbox net off",
            ));
        }
    }
}

/// The sandbox's profile, as `/sandbox status` shows it.
fn status(sandbox: &Sandbox, config: &SandboxConfig, network: SandboxNetwork) -> String {
    let Some(backend) = &sandbox.0 else {
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
        config.project.display(),
        config.tmpdir.display(),
        if network.allowed { "on" } else { "off" },
    )
}

#[cfg(test)]
mod tests;
