//! Plugin corpus: 51. Git status-line item (spec `51-git-status-item.md`).
//!
//! The terminal view's status line shows the current git branch and how
//! many files are changed, such as `main ±3`: refreshed at startup, after
//! every turn, and after every call of a tool that may change files, at
//! most once a second, with git run off the main thread. A detached HEAD
//! shows its short commit, and a rebase under way `REBASE`; outside a
//! repository the item is hidden. Its place is the `[git_status]` table of
//! the settings. Without the terminal view (the `tui` feature off, or a run
//! another front takes) the plugin does nothing: no item and no git.

#[cfg(feature = "tui")]
mod item;

#[cfg(feature = "tui")]
pub use item::{GitRunner, GitStatusSettings};

use rig_harness::prelude::*;

/// Shows the git branch and the changed files on the status line.
#[derive(Default)]
pub struct GitStatusItemPlugin;

impl Plugin for GitStatusItemPlugin {
    fn build(&self, _app: &mut App) {}

    /// Once every plugin offered its front, so the run's is known.
    fn finish(&self, app: &mut App) {
        #[cfg(feature = "tui")]
        item::add(app);
        #[cfg(not(feature = "tui"))]
        let _ = app;
    }
}

#[cfg(all(test, feature = "tui"))]
mod tests;
