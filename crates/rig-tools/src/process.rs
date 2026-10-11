//! Child processes started in a session of their own, such as `shell`
//! commands, so they never read the terminal and are killed as a group, and
//! the [`Launcher`] that says how every tool starts them.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::{Child, Command};

/// How tools start processes: under a wrapper program (a sandbox such as
/// `bwrap … --`, `nice -n 10`, a container's `exec`), in a directory, with
/// variables set and removed. Every command it builds is [`detach`]ed. A
/// host that hands one launcher to every tool that starts processes
/// ([`Shell`](crate::Shell), a background job, a formatter) puts them all
/// under the same policy.
///
/// ```
/// use rig_tools::process::Launcher;
///
/// let launcher = Launcher::default()
///     .wrapped(["nice", "-n", "10"])
///     .in_dir("/srv/project")
///     .with_env("CI", "1")
///     .without_env(["AGENT_TOKEN"]);
/// let mut command = launcher.command("sh");
/// command.args(["-c", "make test"]);
/// assert_eq!(command.get_program(), "nice");
/// let args: Vec<_> = command.get_args().collect();
/// assert_eq!(args, ["-n", "10", "sh", "-c", "make test"]);
/// assert_eq!(command.get_current_dir(), Some(std::path::Path::new("/srv/project")));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Launcher {
    /// The program and arguments every command runs under; empty runs the
    /// command itself.
    pub prefix: Vec<OsString>,
    /// The directory commands start in; `None` is this process's.
    pub cwd: Option<PathBuf>,
    /// Variables set for every command.
    pub env: Vec<(OsString, OsString)>,
    /// Variables no command inherits, such as those that make a program
    /// act as the agent that runs it.
    pub unset_env: Vec<OsString>,
}

impl Launcher {
    /// The same launcher, running every command under `prefix`.
    pub fn wrapped(mut self, prefix: impl IntoIterator<Item = impl Into<OsString>>) -> Self {
        self.prefix = prefix.into_iter().map(Into::into).collect();
        self
    }

    /// The same launcher, starting commands in `dir`.
    pub fn in_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cwd = Some(dir.into());
        self
    }

    /// The same launcher, setting `name` to `value` for every command.
    pub fn with_env(mut self, name: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.env.push((name.into(), value.into()));
        self
    }

    /// The same launcher, keeping `names` from every command.
    pub fn without_env(mut self, names: impl IntoIterator<Item = impl Into<OsString>>) -> Self {
        self.unset_env.extend(names.into_iter().map(Into::into));
        self
    }

    /// A detached command that runs `program` under this launcher's policy;
    /// add the program's arguments to it.
    pub fn command(&self, program: impl AsRef<OsStr>) -> Command {
        let mut command = match self.prefix.split_first() {
            Some((wrapper, arguments)) => {
                let mut command = Command::new(wrapper);
                command.args(arguments).arg(program);
                command
            }
            None => Command::new(program),
        };
        if let Some(dir) = &self.cwd {
            command.current_dir(dir);
        }
        for name in &self.unset_env {
            command.env_remove(name);
        }
        command.envs(self.env.iter().map(|(name, value)| (name, value)));
        detach(&mut command);
        command
    }
}

/// Starts `command` in a new session, which also makes it the leader of a
/// new process group. With no controlling terminal, a program that opens
/// `/dev/tty` for a prompt (ssh, git, sudo) fails at once instead of
/// drawing over the view and stopping on a terminal read. Git is also told
/// not to prompt.
pub fn detach(command: &mut Command) {
    command.env("GIT_TERMINAL_PROMPT", "0");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: the closure runs in the forked child before `exec` and
        // only calls `setsid`, which is async-signal-safe.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() == -1 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
    }
}

/// Kills the process group `child` leads, created by [`detach`];
/// elsewhere, the child alone.
#[cfg(unix)]
pub fn kill_group(child: &mut Child) {
    kill_group_of(child.id());
}

/// Kills the process group led by the process `leader`. Its id names that
/// group while the leader is not reaped or any member lives: the kernel
/// does not reuse the id of a process group that exists. Once the leader
/// was reaped and the group emptied, the id is free again, so callers kill
/// within moments of the reap.
#[cfg(unix)]
pub fn kill_group_of(leader: u32) {
    if let Ok(group) = i32::try_from(leader) {
        // SAFETY: `kill` takes plain integers; a negative pid names the
        // process group the child leads, created by `detach`.
        unsafe {
            libc::kill(-group, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
pub fn kill_group(child: &mut Child) {
    child.kill().ok();
}
