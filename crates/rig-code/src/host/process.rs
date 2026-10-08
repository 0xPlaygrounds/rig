//! Child processes the agent starts in a session of their own: shell
//! commands and the `/reload` build.

use std::process::{Child, Command};

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

/// Kills the process group led by the process `leader`, which must not have
/// been reaped yet, so its id still names that group.
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
