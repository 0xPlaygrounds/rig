//! Child processes the agent starts in a process group of their own: shell
//! commands and the `/reload` build.

use std::process::Child;

/// Kills the process group `child` leads, created with `process_group(0)`;
/// elsewhere, the child alone.
#[cfg(unix)]
pub fn kill_group(child: &mut Child) {
    if let Ok(group) = i32::try_from(child.id()) {
        // SAFETY: `kill` takes plain integers; a negative pid names the
        // process group the child leads, created by `process_group(0)`.
        unsafe {
            libc::kill(-group, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
pub fn kill_group(child: &mut Child) {
    child.kill().ok();
}
