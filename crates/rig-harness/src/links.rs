//! Links about the running session that a plugin publishes for others to
//! show or forward, such as the pull request a command opened or the page
//! where the run is watched. A plugin publishes one by spawning a
//! [`SessionLink`]; a reader observes `Add<SessionLink>` or queries them.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! let mut world = World::new();
//! world.spawn(SessionLink::new("Pull request #7", "https://example.com/pull/7"));
//! ```

use bevy_ecs::prelude::*;

/// A link about this session, on an entity of its own: `label` is what a
/// reader shows, `url` where it leads. Despawning it withdraws nothing a
/// reader already forwarded.
#[derive(Component, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SessionLink {
    /// What the link is, such as `Pull request #7`.
    pub label: String,
    /// The absolute URL.
    pub url: String,
}

impl SessionLink {
    /// The link to `url`, shown as `label`.
    pub fn new(label: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            url: url.into(),
        }
    }
}
