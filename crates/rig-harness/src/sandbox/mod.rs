//! The sandbox of a restricted worker: what makes a child's narrower grant
//! real, for its file tools and every command its shell runs alike. A grant
//! this platform can't enforce is refused, never widened.
//!
//! - **Linux:** [`apply`] confines the worker process with Landlock before
//!   any plugin runs; its descendants inherit it. Readable: the system and
//!   toolchain folders ([`READ_ONLY`], plus `~/.cargo` and `~/.rustup`) and
//!   the grant's folders; writable: the grant's writable folders, the
//!   session's own folder (with its own temporary folder, [`tmp_dir`],
//!   which its commands get as `TMPDIR`: [`command_env`]) and `/dev`. Nothing else,
//!   so not rig's `run/token`, another session's transcript or the rest of
//!   the home folder, through a symlink or not. Signals and abstract
//!   sockets are scoped to the sandbox where the kernel can (Landlock ABI 6).
//!   A kernel without Landlock refuses the grant.
//! - **macOS:** [`wrap`] starts the worker under `sandbox-exec` with a
//!   profile saying the same.
//! - **Elsewhere** (Windows): a directory grant is refused.
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

/// The variables a restricted worker's commands never inherit: the routes
/// out of the sandbox they would hand a shell (the user's fux server, ssh
/// agent, desktop session bus).
pub const SCRUBBED_ENV: &[&str] = &[
    "FUX_SOCKET",
    "FUX_PANE",
    "FUX_SERVER",
    "SSH_AUTH_SOCK",
    "DBUS_SESSION_BUS_ADDRESS",
];

/// Whether `grant` restricts anything a sandbox must enforce.
pub fn restricts(grant: &Grant) -> bool {
    grant.dirs.is_some() || !grant.net
}

/// What a confined worker may read and write.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plan {
    pub read: Vec<PathBuf>,
    pub write: Vec<PathBuf>,
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
    ];
    for DirGrant { path, write: w } in dirs {
        if *w {
            write.push(path.clone());
        } else {
            read.push(path.clone());
        }
    }
    Ok(Some(Plan { read, write }))
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
    std::fs::read_dir("/proc/self/task").ok().map(Iterator::count)
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
