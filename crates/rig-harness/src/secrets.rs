//! Secrets a plugin needs before any front runs, such as the passphrase of
//! an encrypted session store, which opens before the session is restored.
//!
//! A plugin asks for one in `build` with `app.add_secret(..)`
//! ([`AppCliExt`](crate::host::cli::AppCliExt)) and reads it from
//! [`Secrets`], there from `build` on. The launcher learns what the build
//! asks for when it describes it ([`describe`](crate::describe)) and,
//! when stdin is a terminal, whatever the front, asks the user for each
//! one the environment does not hold, once per launcher, whatever the
//! restarts. A secret reaches the agent's process alone, as the variable
//! [`rig_harness_protocol::secret_variable`] of its name
//! (`RIG_SECRET_CRYPT`); a piped run gets only what the environment
//! has. A secret not given is `None`, and the plugin decides what then.
//!
//! No process rig starts gets a secret: not cargo or any other launcher
//! child, not `/reload`'s build, not a command started through
//! [`Processes`](crate::services::Processes) (the shell and every tool), not
//! the terminal view's editor. A plugin that starts a process with a bare
//! `std::process::Command` passes its secrets on: start processes with
//! `Processes`. What this protects from: tools, the model's commands,
//! build scripts and proc macros, and helper programs. What it does not:
//! the same user reading the agent's environment (`/proc/<pid>/environ`),
//! and plugin code, which is trusted as all Bevy plugin code is.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! #[derive(Default)]
//! struct CryptPlugin;
//!
//! impl Plugin for CryptPlugin {
//!     fn build(&self, app: &mut App) {
//!         app.add_secret(Secret::new("crypt", "Passphrase for this machine's sessions"));
//!         let passphrase = app
//!             .world()
//!             .get_resource::<Secrets>()
//!             .and_then(|secrets| secrets.get("crypt"))
//!             .map(str::to_owned);
//!         // Open the session store with it, or refuse to start without.
//!         # drop(passphrase);
//!     }
//! }
//! ```

use std::collections::BTreeMap;
use std::fmt;

use bevy_ecs::prelude::*;
use rig_harness_protocol::{env, secret_variable};

/// A secret a plugin asks for: its name, which names its variable, and
/// what the launcher asks.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct Secret {
    /// Its name: lowercase letters, digits and dashes.
    pub name: &'static str,
    /// What to ask for, such as `Passphrase for this machine's sessions`.
    pub help: &'static str,
}

impl Secret {
    /// The secret `name`, asked for as `help`.
    pub fn new(name: &'static str, help: &'static str) -> Self {
        Self { name, help }
    }
}

/// The secrets this agent was given. The harness takes them from the
/// process's environment while it is built, unless the app inserted its
/// own, as a test does: `Secrets::from_iter([("crypt", "passphrase")])`.
/// Its `Debug` shows names only.
#[derive(Resource, Clone, Default)]
pub struct Secrets(BTreeMap<String, String>);

impl Secrets {
    /// Every secret in this process's environment.
    pub fn from_env() -> Self {
        Self(
            std::env::vars_os()
                .filter_map(|(name, value)| {
                    Some((name.into_string().ok()?, value.into_string().ok()?))
                })
                .filter(|(name, _)| name.starts_with(env::SECRET_PREFIX))
                .collect(),
        )
    }

    /// The secret `name`, if given.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0.get(&secret_variable(name)).map(String::as_str)
    }
}

/// Secrets by name, such as a test's.
impl<N: AsRef<str>, V: Into<String>> FromIterator<(N, V)> for Secrets {
    fn from_iter<I: IntoIterator<Item = (N, V)>>(secrets: I) -> Self {
        Self(
            secrets
                .into_iter()
                .map(|(name, value)| (secret_variable(name.as_ref()), value.into()))
                .collect(),
        )
    }
}

impl fmt::Debug for Secrets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.0.keys()).finish()
    }
}
