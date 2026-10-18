//! Plugin corpus: 51. Git status-line item (spec `51-git-status-item.md`).
//!
//! The terminal view's status line shows the current git branch and how
//! many files are changed, such as `main ±3`: refreshed at startup, after
//! every turn, and after every call of a tool that may change files, at
//! most once a second, with git run off the main thread. A detached HEAD
//! shows its short commit, and a rebase under way `REBASE`; outside a
//! repository the item is hidden. Its place is the `[git_status]` table of
//! the settings. It runs in the session's worker, where the project is;
//! the item reaches every view as the session's status.

mod item;

pub use item::{GitRunner, GitStatusSettings};

use rig_harness::prelude::*;

/// Shows the git branch and the changed files on the status line.
#[derive(Default)]
pub struct GitStatusItemPlugin;

impl Plugin for GitStatusItemPlugin {
    fn build(&self, app: &mut App) {
        item::add(app);
    }
}

#[cfg(test)]
mod tests;
