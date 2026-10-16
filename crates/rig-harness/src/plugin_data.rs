//! Where a plugin keeps files of its own, by how long they live:
//! [`PluginData`], in every run. A plugin names its files under its own
//! name, such as `vim/registers.json`.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! fn save(data: Res<PluginData>) -> std::io::Result<()> {
//!     // Kept with the session.
//!     data.session.write_file("vim/registers.json", b"{}")?;
//!     // This project's, whichever session works in it.
//!     data.project.write_file("vim/marks.json", b"{}")?;
//!     // Every session's.
//!     data.user.write_file("vim/mode.json", b"\"normal\"")
//! }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use rig_core::tool::content_hash;
use rig_harness_protocol::{Home, SessionDir};
use rig_memory::{DirFiles, Files, MemoryStore, PluginFiles, SessionError};

use crate::host::session::StoreOpener;

/// Where plugins keep their files: per session, per project and per
/// user. Not in the project's own directory: a plugin's data is the
/// user's, never committed and never read from a checkout someone else
/// made. Inserted once every plugin is built, in every run, unless the
/// app has one: a run without a saved session keeps `session` in memory.
#[derive(Resource, Clone)]
pub struct PluginData {
    /// This session's: `plugins/` in the session's store, which a delete
    /// removes.
    pub session: Arc<dyn Files>,
    /// This project's, for this user, whichever session works in it:
    /// `$RIG_HOME/projects/<name>-<key>/`, the key from the project's
    /// canonical path, so every spelling of it shares one.
    pub project: Arc<dyn Files>,
    /// Across sessions and projects: `$RIG_HOME/data/`.
    pub user: Arc<dyn Files>,
}

impl PluginData {
    /// The places of a session kept in `session` (in memory with `None`),
    /// working on the project at `root` in the workspace on `host` (`None`
    /// for this machine), under `home`.
    pub(crate) fn new(
        home: &Home,
        session: Option<Arc<dyn Files>>,
        root: &Path,
        host: Option<&str>,
    ) -> Self {
        let session = session.unwrap_or_else(|| Arc::new(MemoryStore::default()));
        Self {
            session: Arc::new(PluginFiles(session)),
            project: Arc::new(DirFiles::new(project_dir(home, root, host))),
            user: Arc::new(DirFiles::new(home.data())),
        }
    }
}

impl PluginData {
    /// The plugin files of the saved session in `dir`, kept in the store
    /// `stores` opens for it (`plugins/` of that store): what
    /// [`PluginData::session`] is in a run of that session.
    ///
    /// # Errors
    ///
    /// When its store does not open.
    pub fn of(stores: &StoreOpener, dir: &SessionDir) -> Result<Arc<dyn Files>, SessionError> {
        let store: Arc<dyn Files> = stores.open(dir)?;
        Ok(Arc::new(PluginFiles(store)))
    }
}

/// The directory of `home` keeping plugins' files for the project at
/// `root` on `host`: its name and the first 8 hex digits of the SHA-256
/// of where it is, canonical on this machine.
fn project_dir(home: &Home, root: &Path, host: Option<&str>) -> PathBuf {
    let root = match host {
        None => root.canonicalize().unwrap_or_else(|_| root.to_owned()),
        Some(_) => root.to_owned(),
    };
    let mut place = host.map(|host| format!("{host}:")).unwrap_or_default();
    place.push_str(&root.to_string_lossy());
    let hash = content_hash(place.as_bytes()).0;
    let name = root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let key = hash.get(..8).unwrap_or(&hash);
    home.projects().join(format!("{name}-{key}"))
}

#[cfg(test)]
mod tests;
