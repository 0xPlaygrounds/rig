//! Child processes the agent starts and must be able to stop with
//! everything they started: shell commands and the `/reload` build.

use std::process::{Child, Command};

/// Makes `command` start a new process group, so [`kill`] reaches its
/// descendants too.
pub(crate) fn new_group(command: &mut Command) {
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(command, 0);
    #[cfg(not(unix))]
    let _ = command;
}

/// Kills a child started with [`new_group`] and everything it started,
/// then reaps it. The child must not have been reaped yet.
pub(crate) fn kill(child: &mut Child) {
    #[cfg(unix)]
    if let Ok(group) = i32::try_from(child.id()) {
        // SAFETY: `kill` takes no pointers. The group id is the child's pid
        // (it was started as a group leader) and the child is not reaped yet,
        // so the id cannot have been reused.
        unsafe {
            libc::kill(-group, libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}
