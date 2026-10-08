//! The approval policy new agents start with, from `RIG_HOME/policy.json`:
//!
//! ```json
//! {
//!   "mode": "ask",
//!   "rules": [
//!     { "tool": "shell", "subject": "git status*", "permission": "allow" },
//!     { "tool": "mcp__github__*", "permission": "ask" }
//!   ]
//! }
//! ```
//!
//! Without the file every call runs (`auto`). An agent keeps its policy
//! in the session, so an edit counts for agents started after it;
//! `/approvals` changes a running agent's.

use std::fs;
use std::io::ErrorKind;

use bevy_app::prelude::*;
use rig::code_protocol::Home;

use crate::core::agent::Notice;
use crate::core::approval::{DefaultPolicy, Policy};

/// Reads `policy.json` into the [`DefaultPolicy`].
pub struct PolicyPlugin;

impl Plugin for PolicyPlugin {
    fn build(&self, app: &mut App) {
        let path = Home::from_env().policy();
        let policy = match fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str::<Policy>(&text).map_err(|failure| {
                format!(
                    "{} does not load ({failure}); every tool call runs without asking.",
                    path.display()
                )
            }),
            Err(failure) if failure.kind() == ErrorKind::NotFound => Ok(Policy::default()),
            Err(failure) => Err(format!(
                "Could not read {} ({failure}); every tool call runs without asking.",
                path.display()
            )),
        };
        match policy {
            Ok(policy) => {
                app.insert_resource(DefaultPolicy(policy));
            }
            Err(why) => {
                app.world_mut().write_message(Notice::error(None, why));
            }
        }
    }
}
