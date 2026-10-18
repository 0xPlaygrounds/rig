//! The sandbox of a restricted worker: what makes a child's narrower grant
//! real, for its file tools and every command its shell runs alike. A grant
//! this platform can't enforce is refused, never widened.
//!
//! - **Linux:** [`apply`] confines the worker process with Landlock before
//!   any plugin runs; its descendants inherit it. Readable: the system and
//!   toolchain folders ([`READ_ONLY`], plus `~/.cargo` and `~/.rustup`) and
//!   the grant's folders; writable: the grant's writable folders, the
//!   session's own folder (with its own temporary folder, [`tmp_dir`],
//!   which its commands get as `TMPDIR`: [`command_env`]), its sign-in
//!   (`auth/`, which model calls refresh) and `/dev`; `settings.toml` and
//!   `defaults.json` (its model) are read-only; the one Unix socket it
//!   may connect to is `rigd`'s. Nothing else,
//!   so not rig's `run/token`, another session's transcript or the rest of
//!   the home folder, through a symlink or not. Signals and abstract
//!   sockets are scoped to the sandbox where the kernel can (Landlock ABI 6).
//!   A kernel without Landlock refuses the grant.
//! - **macOS:** [`wrap`] starts the worker under `sandbox-exec` with a
//!   profile saying the same.
//! - **Elsewhere** (Windows): a directory grant is refused.
//!
//! **Commands** (Linux): each command a confined worker runs starts through
//! its confine step ([`shell_launch`], [`run_confined`]: the agent binary in
//! the `confine` role), which adds a second Landlock layer before it runs
//! the command: the worker's folders without its sign-in (`auth/`) and
//! settings, and no Unix socket at all, so a command can neither read the
//! model credentials nor reach `rigd`, worker token or not.
//!
//! `net: false` keeps the *shell* off the network: on Linux each command
//! runs in a new network namespace ([`shell_prefix`], `unshare --user --net`);
//! elsewhere it is refused. The worker's own model calls are unaffected.
//!
//! What the sandbox does not cover: the worker's own connection to `rigd`
//! (its principal limits it to its own tree), and a grant over a folder that
//! holds rig's `run/` directory, which [`apply`] refuses.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use rig_protocol::{DirGrant, Grant};

#[derive(Debug)]
pub enum SandboxError {
    /// This platform can't enforce what the grant restricts.
    Unavailable(String),
    Io(std::io::Error),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::Unavailable(what) => write!(f, "the sandbox can't enforce {what}"),
            SandboxError::Io(e) => write!(f, "the sandbox failed: {e}"),
        }
    }
}

impl std::error::Error for SandboxError {}

impl From<SandboxError> for rig_protocol::RpcError {
    fn from(error: SandboxError) -> rig_protocol::RpcError {
        rig_protocol::RpcError::new(rig_protocol::codes::SANDBOX_UNAVAILABLE, error.to_string())
            .with_next("start the helper without `dirs`, or on Linux or macOS")
    }
}

/// System folders a confined worker reads (and runs programs from).
pub const READ_ONLY: &[&str] = &[
    "/usr",
    "/etc",
    "/opt",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/nix",
    "/proc",
    "/sys",
    "/run/current-system",
];

/// Whether a variable of `name` and `value` would hand a confined
/// worker's commands a way out of the sandbox or a credential, so they never
/// inherit it ([`scrubbed_env`]). By rule, not by tool: a name that ends
/// like a socket, token, key, secret or password, or that has an authority
/// as a word (`AUTH`, `*AUTHORITY`, as an ssh agent's or X's), and a value
/// that is (or starts with) the path of a Unix socket, or a `unix:` address
/// (such as a session bus's).
pub fn reaches_out(name: &std::ffi::OsStr, value: &std::ffi::OsStr) -> bool {
    const ENDINGS: &[&str] = &[
        "_SOCK",
        "_SOCKET",
        "_TOKEN",
        "_KEY",
        "_SECRET",
        "_PASSWORD",
        "_PASSWD",
        "_PASS",
        "_CREDENTIALS",
        "_COOKIE",
    ];
    let name = name.to_string_lossy().to_ascii_uppercase();
    let value = value.to_string_lossy();
    // An authority, as a word of the name: `*_AUTH`, `AUTH_*`, `*AUTHORITY`
    // (never `AUTHOR`).
    let authority = name
        .split('_')
        .any(|word| word == "AUTH" || word.ends_with("AUTHORITY"));
    if ENDINGS.iter().any(|end| name.ends_with(end)) || authority {
        return true;
    }
    if value.starts_with("unix:") || value.contains("unix:path=") || value.contains("unix:abstract=") {
        return true;
    }
    // A socket's path, alone or before `:`-separated fields.
    let path = value.split(':').next().unwrap_or_default();
    is_socket(Path::new(path))
}

#[cfg(unix)]
fn is_socket(path: &Path) -> bool {
    use std::os::unix::fs::FileTypeExt;
    path.is_absolute() && std::fs::metadata(path).is_ok_and(|m| m.file_type().is_socket())
}

#[cfg(not(unix))]
fn is_socket(_path: &Path) -> bool {
    false
}

/// The names of this process's variables a confined worker's commands
/// never inherit: each one [`reaches_out`].
pub fn scrubbed_env() -> Vec<OsString> {
    std::env::vars_os()
        .filter(|(name, value)| reaches_out(name, value))
        .map(|(name, _)| name)
        .collect()
}

/// Whether `grant` restricts anything a sandbox must enforce.
pub fn restricts(grant: &Grant) -> bool {
    grant.dirs.is_some() || !grant.net
}

/// What a confined worker may read and write.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Plan {
    pub read: Vec<PathBuf>,
    pub write: Vec<PathBuf>,
    /// Unix sockets it may connect to: `rigd`'s, and no other.
    pub connect: Vec<PathBuf>,
}

/// The folders a worker under `grant` gets, with `session_dir` writable and
/// rig's home at `rig_home`; refused when a granted folder holds rig's
/// `run/` directory (the owner token and socket). `None` when the grant
/// leaves files unrestricted.
pub fn plan(
    grant: &Grant,
    session_dir: &Path,
    rig_home: &Path,
) -> Result<Option<Plan>, SandboxError> {
    let Some(dirs) = &grant.dirs else {
        return Ok(None);
    };
    let run = rig_home.join("run");
    if let Some(over) = dirs.iter().find(|d| run.starts_with(&d.path)) {
        return Err(SandboxError::Unavailable(format!(
            "a grant of {}, which holds rig's run directory",
            over.path.display()
        )));
    }
    let mut read: Vec<PathBuf> = READ_ONLY.iter().map(PathBuf::from).collect();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        read.extend([home.join(".cargo"), home.join(".rustup")]);
    }
    // Its own temporary folder, not the shared one: that holds other
    // sessions' scratch, and may hold rig's home.
    let mut write = vec![
        session_dir.to_path_buf(),
        tmp_dir(session_dir),
        PathBuf::from("/dev"),
        // Its sign-in, which its model calls read and refresh.
        rig_home.join("auth"),
    ];
    read.push(rig_home.join("settings.toml"));
    // The model and reasoning it starts with; a confined worker never
    // changes the user's.
    read.push(rig_home.join("defaults.json"));
    // Its own build, which its commands' confine step runs.
    if let Some(build) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        read.push(build);
    }
    let connect = vec![run.join("rig.sock")];
    for DirGrant { path, write: w } in dirs {
        if *w {
            write.push(path.clone());
        } else {
            read.push(path.clone());
        }
    }
    Ok(Some(Plan {
        read,
        write,
        connect,
    }))
}

/// The temporary folder of a confined worker of `session_dir`, which its
/// commands get as `TMPDIR`.
pub fn tmp_dir(session_dir: &Path) -> PathBuf {
    session_dir.join("tmp")
}

/// The variables a confined worker's commands run with: `TMPDIR` its own
/// temporary folder. Empty when files are unrestricted.
pub fn command_env(grant: &Grant, session_dir: &Path) -> Vec<(OsString, OsString)> {
    match grant.dirs {
        Some(_) => vec![("TMPDIR".into(), tmp_dir(session_dir).into_os_string())],
        None => Vec::new(),
    }
}

/// Confines this process to `grant`, with `session_dir` writable. Called at
/// worker start, before any plugin runs; irreversible.
pub fn apply(grant: &Grant, session_dir: &Path) -> Result<(), SandboxError> {
    let rig_home = std::env::var_os(crate::harness_protocol::env::HOME)
        .map(PathBuf::from)
        .unwrap_or_default();
    if !grant.net {
        shell_prefix(grant)?;
    }
    match plan(grant, session_dir, &rig_home)? {
        None => Ok(()),
        Some(mut plan) => {
            std::fs::create_dir_all(tmp_dir(session_dir)).map_err(SandboxError::Io)?;
            plan.read.retain(|p| p.exists());
            plan.write.retain(|p| p.exists());
            plan.connect.retain(|p| p.exists());
            confine(&plan)
        }
    }
}

#[cfg(target_os = "linux")]
fn confine(plan: &Plan) -> Result<(), SandboxError> {
    use landlock::{
        ABI, Access, AccessFs, CompatLevel, Compatible, RestrictSelfAttr, Ruleset, RulesetAttr,
        RulesetCreatedAttr, RulesetStatus, Scope, path_beneath_rules,
    };
    let abi = ABI::V9;
    let failed = |e: landlock::RulesetError| SandboxError::Unavailable(format!("files: {e}"));
    let status = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .map_err(failed)?
        .set_compatibility(CompatLevel::BestEffort)
        .scope(Scope::AbstractUnixSocket | Scope::Signal)
        .map_err(failed)?
        .create()
        .map_err(failed)?
        .add_rules(path_beneath_rules(&plan.read, AccessFs::from_read(abi)))
        .map_err(failed)?
        .add_rules(path_beneath_rules(&plan.write, AccessFs::from_all(abi)))
        .map_err(failed)?
        .add_rules(path_beneath_rules(&plan.connect, AccessFs::ResolveUnix))
        .map_err(failed)?
        // Every thread, those Bevy's pools started already included, not
        // only this one (Landlock ABI 8).
        .all_threads(true)
        .map_err(failed)?
        .restrict_self()
        .map_err(failed)?;
    if status.ruleset == RulesetStatus::NotEnforced {
        return Err(SandboxError::Unavailable(
            "a directory grant: this kernel has no Landlock".into(),
        ));
    }
    if !status.all_threads && threads() != Some(1) {
        return Err(SandboxError::Unavailable(
            "a directory grant: this kernel can't confine the threads already running (Landlock ABI 8)"
                .into(),
        ));
    }
    Ok(())
}

/// How many threads this process runs.
#[cfg(target_os = "linux")]
fn threads() -> Option<usize> {
    std::fs::read_dir("/proc/self/task")
        .ok()
        .map(Iterator::count)
}

#[cfg(target_os = "macos")]
fn confine(_plan: &Plan) -> Result<(), SandboxError> {
    // macOS confines at spawn: `wrap` started this worker under sandbox-exec.
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn confine(_plan: &Plan) -> Result<(), SandboxError> {
    Err(SandboxError::Unavailable(
        "a directory grant on this platform".into(),
    ))
}

/// Makes `cmd` (a worker's spawn) start confined to `grant`, where the
/// platform confines at spawn: macOS. Elsewhere it only refuses what
/// [`apply`] would.
pub fn wrap(cmd: &mut Command, grant: &Grant) -> Result<(), SandboxError> {
    let rig_home = std::env::var_os(crate::harness_protocol::env::HOME)
        .map(PathBuf::from)
        .unwrap_or_default();
    let session_dir = cmd
        .get_current_dir()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let plan = plan(grant, &session_dir, &rig_home)?;
    wrap_for(cmd, grant, plan.as_ref(), &rig_home)
}

#[cfg(target_os = "macos")]
fn wrap_for(
    cmd: &mut Command,
    grant: &Grant,
    plan: Option<&Plan>,
    rig_home: &Path,
) -> Result<(), SandboxError> {
    if plan.is_none() && grant.net {
        return Ok(());
    }
    let profile = macos_profile(plan, rig_home);
    let mut wrapped = Command::new("/usr/bin/sandbox-exec");
    wrapped
        .arg("-p")
        .arg(profile)
        .arg(cmd.get_program())
        .args(cmd.get_args());
    if let Some(dir) = cmd.get_current_dir() {
        wrapped.current_dir(dir);
    }
    for (name, value) in cmd.get_envs() {
        match value {
            Some(value) => wrapped.env(name, value),
            None => wrapped.env_remove(name),
        };
    }
    *cmd = wrapped;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn wrap_for(
    _cmd: &mut Command,
    grant: &Grant,
    plan: Option<&Plan>,
    _rig_home: &Path,
) -> Result<(), SandboxError> {
    if plan.is_some() && !cfg!(target_os = "linux") {
        return Err(SandboxError::Unavailable(
            "a directory grant on this platform".into(),
        ));
    }
    if !grant.net {
        shell_prefix(grant)?;
    }
    Ok(())
}

/// The `sandbox-exec` profile of `plan`: everything but files as usual,
/// writes only where the plan writes, reads of the home folder only where
/// it reads, and never rig's run directory.
pub fn macos_profile(plan: Option<&Plan>, rig_home: &Path) -> String {
    let quote = |p: &Path| format!("(subpath {:?})", p.to_string_lossy());
    let mut profile = String::from("(version 1)\n(allow default)\n");
    if let Some(plan) = plan {
        profile.push_str("(deny file-write*)\n(allow file-write*");
        for p in &plan.write {
            profile.push(' ');
            profile.push_str(&quote(p));
        }
        profile.push_str(")\n");
        if let Some(home) = std::env::var_os("HOME") {
            profile.push_str(&format!("(deny file-read* {})\n", quote(Path::new(&home))));
        }
        profile.push_str("(allow file-read*");
        for p in plan.read.iter().chain(&plan.write) {
            profile.push(' ');
            profile.push_str(&quote(p));
        }
        profile.push_str(")\n");
    }
    profile.push_str(&format!(
        "(deny file-read* file-write* {})\n",
        quote(&rig_home.join("run"))
    ));
    profile
}

/// The variable that makes this binary the confine step of a confined
/// worker's commands ([`run_confined`]), with [`CONFINE_ENV`] its plan.
pub const CONFINE_ROLE: &str = "confine";
/// The confine step's plan, as JSON.
pub const CONFINE_ENV: &str = "RIG_CONFINE";

/// How a confined worker under `grant` starts each command: the prefix
/// (`exe`, this agent binary, as the confine step; under the network
/// namespace of [`shell_prefix`] for `net: false`) and the variables that
/// make it one. Empty when nothing is restricted.
pub fn shell_launch(
    grant: &Grant,
    session_dir: &Path,
    exe: &Path,
) -> Result<(Vec<OsString>, Vec<(OsString, OsString)>), SandboxError> {
    let mut prefix = shell_prefix(grant)?;
    let mut env = command_env(grant, session_dir);
    let rig_home = std::env::var_os(crate::harness_protocol::env::HOME)
        .map(PathBuf::from)
        .unwrap_or_default();
    // Linux only: macOS confines at spawn, with no second layer for commands.
    let planned = plan(grant, session_dir, &rig_home)?.filter(|_| cfg!(target_os = "linux"));
    if let Some(plan) = planned {
        let commands = commands_plan(plan, &rig_home);
        let json = serde_json::to_string(&commands)
            .map_err(|e| SandboxError::Unavailable(format!("its commands' plan: {e}")))?;
        prefix.push(exe.as_os_str().to_owned());
        env.push((
            crate::harness_protocol::env::ROLE.into(),
            CONFINE_ROLE.into(),
        ));
        env.push((CONFINE_ENV.into(), json.into()));
    }
    Ok((prefix, env))
}

/// What a confined worker's commands get of its `plan`: everything but its
/// sign-in and settings, and no Unix socket at all, not even `rigd`'s.
pub fn commands_plan(mut plan: Plan, rig_home: &Path) -> Plan {
    let private = [rig_home.join("auth"), rig_home.join("settings.toml")];
    plan.read
        .retain(|p| !private.iter().any(|q| p.starts_with(q)));
    plan.write
        .retain(|p| !private.iter().any(|q| p.starts_with(q)));
    plan.connect.clear();
    plan
}

/// The confine step (the agent binary run with `RIG_ROLE=confine`, as
/// [`shell_launch`] prefixes each command): confines this process further,
/// a Landlock layer on the worker's own, to its commands' plan,
/// then runs `argv` in its place, without the step's own variables. Its
/// exit code when `argv` could not start.
#[cfg(unix)]
pub fn run_confined(argv: Vec<OsString>) -> u8 {
    use std::os::unix::process::CommandExt;
    let Some((program, args)) = argv.split_first() else {
        eprintln!("rig: the confine step has no command");
        return 127;
    };
    // Never run unconfined: no plan, or one this process can't take, ends it.
    let plan = std::env::var(CONFINE_ENV)
        .map_err(|e| e.to_string())
        .and_then(|json| serde_json::from_str::<Plan>(&json).map_err(|e| e.to_string()));
    let confined = match plan {
        Ok(mut plan) => {
            plan.read.retain(|p| p.exists());
            plan.write.retain(|p| p.exists());
            confine(&plan).map_err(|e| e.to_string())
        }
        Err(why) => Err(format!("no plan in {CONFINE_ENV}: {why}")),
    };
    if let Err(why) = confined {
        eprintln!("rig: the command was not run, as it could not be confined: {why}");
        return 126;
    }
    let error = Command::new(program)
        .args(args)
        .env_remove(crate::harness_protocol::env::ROLE)
        .env_remove(CONFINE_ENV)
        .exec();
    eprintln!("rig: {}: {error}", program.to_string_lossy());
    127
}

/// The confine step where processes can't be replaced: it never runs a
/// command.
#[cfg(not(unix))]
pub fn run_confined(_argv: Vec<OsString>) -> u8 {
    eprintln!("rig: the confine step runs on Linux only");
    126
}

/// The prefix every shell command of a worker under `grant` runs under:
/// for `net: false` on Linux, a new user and network namespace with only a
/// loopback device; refused elsewhere, or where user namespaces are off.
pub fn shell_prefix(grant: &Grant) -> Result<Vec<OsString>, SandboxError> {
    if grant.net {
        return Ok(Vec::new());
    }
    if !cfg!(target_os = "linux") {
        return Err(SandboxError::Unavailable(
            "no network on this platform".into(),
        ));
    }
    let prefix: Vec<OsString> = ["unshare", "--user", "--map-current-user", "--net", "--"]
        .into_iter()
        .map(OsString::from)
        .collect();
    let works = Command::new("unshare")
        .args(["--user", "--map-current-user", "--net", "--", "true"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if works {
        Ok(prefix)
    } else {
        Err(SandboxError::Unavailable(
            "no network: user namespaces are off here (or `unshare` is missing)".into(),
        ))
    }
}

#[cfg(test)]
mod tests;
