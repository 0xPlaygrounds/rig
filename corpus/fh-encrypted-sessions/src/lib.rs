//! FH-18: encrypted sessions at rest.
//!
//! With `RIG_HOME/crypt.toml` saying so (`enabled = true`, or `projects =
//! ["/path"]` for some projects), every session's store, this one's,
//! forks' and those `/resume` lists, is a [`CryptStore`]: every file the
//! harness and its plugins keep of a session (agent logs, saved plugin
//! state, blobs, the effect log, the listing, a fork's parent, the
//! terminal view's notices and draft, a plugin's own files) is sealed with
//! ChaCha20-Poly1305 under a key Argon2id derives from the passphrase,
//! the secret `crypt`: the launcher asks for it once on an interactive
//! start, and a headless run takes it from `RIG_SECRET_CRYPT`. A run
//! without it, or with a wrong one, is refused and keeps nothing. A file
//! that does not authenticate fails to load with [`TAMPERED`] and is left
//! as it is. What rig keeps plain by path (cut tool output, pasted images,
//! the text log) is kept nowhere, and typed prompts stay out of the prompt
//! history.
//!
//! `rig crypt migrate` encrypts the plain sessions in place; `rig crypt
//! rekey` seals every session again under the passphrase in
//! `RIG_SECRET_CRYPT_NEW`.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_cassette::journal::JsonlDirStore;
use rig_harness::harness_protocol::{Home, secret_variable};
use rig_harness::prelude::*;
use serde::Deserialize;

mod reseal;
mod store;

pub use reseal::{migrate, rekey};
pub use store::{Crypt, CryptStore, Kdf, TAMPERED, new_salt};

/// The secret holding the passphrase.
pub const SECRET: &str = "crypt";

/// The secret `rig crypt rekey` takes the new passphrase from.
pub const NEW_SECRET: &str = "crypt-new";

/// The command, `rig crypt migrate|rekey`.
const COMMAND: &str = "crypt";

/// `RIG_HOME/crypt.toml`.
#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct CryptConfig {
    /// Encrypt every session.
    pub enabled: bool,
    /// Encrypt the sessions of these projects (and their subdirectories).
    pub projects: Vec<PathBuf>,
    /// Argon2id's memory in KiB.
    pub kdf_memory_kib: Option<u32>,
    /// Argon2id's passes.
    pub kdf_iterations: Option<u32>,
}

impl CryptConfig {
    /// The file in `home`, or no encryption without one.
    ///
    /// # Errors
    ///
    /// When it exists and does not read as a config.
    pub fn read(home: &Home) -> Result<Self, String> {
        match fs::read_to_string(home.root().join("crypt.toml")) {
            Ok(text) => toml::from_str(&text).map_err(|failure| format!("crypt.toml: {failure}")),
            Err(_) => Ok(Self::default()),
        }
    }

    /// Whether sessions of `project` are encrypted.
    pub fn applies_to(&self, project: &Path) -> bool {
        self.enabled || self.projects.iter().any(|root| project.starts_with(root))
    }

    /// The key derivation's cost.
    pub fn kdf(&self) -> Kdf {
        let default = Kdf::default();
        Kdf {
            memory_kib: self.kdf_memory_kib.unwrap_or(default.memory_kib),
            iterations: self.kdf_iterations.unwrap_or(default.iterations),
        }
    }
}

/// Where the salt and the key check live: `RIG_HOME/crypt/`.
fn key_dir(home: &Home) -> PathBuf {
    home.root().join("crypt")
}

/// The salt of `home`'s key, made on first use.
fn salt(home: &Home) -> Result<Vec<u8>, String> {
    let path = key_dir(home).join("salt");
    if let Ok(salt) = fs::read(&path) {
        return Ok(salt);
    }
    let salt = new_salt().map_err(|failure| failure.to_string())?;
    fs::create_dir_all(key_dir(home))
        .and_then(|()| fs::write(&path, salt))
        .map_err(|failure| format!("could not write the salt: {failure}"))?;
    Ok(salt.to_vec())
}

/// The key of `home` from `passphrase`, checked against the key check
/// (made with the first key).
///
/// # Errors
///
/// Without a passphrase, with a wrong one, or when the key files cannot
/// be read or written.
pub fn unlock(
    home: &Home,
    config: &CryptConfig,
    passphrase: Option<&str>,
) -> Result<Crypt, String> {
    let Some(passphrase) = passphrase.filter(|passphrase| !passphrase.is_empty()) else {
        return Err(format!(
            "sessions are encrypted and there is no key: give the passphrase when rig asks, \
             or set {}",
            secret_variable(SECRET)
        ));
    };
    let crypt = Crypt::derive(passphrase.as_bytes(), &salt(home)?, config.kdf())
        .map_err(|failure| failure.to_string())?;
    crypt
        .check(&key_dir(home).join("check"))
        .map_err(|failure| format!("cannot unlock the encrypted sessions: {failure}"))?;
    Ok(crypt)
}

/// The key this run unlocked, or why it has none.
#[derive(Resource, Clone)]
struct Unlocked(Result<Crypt, String>);

/// Keeps the sessions encrypted at rest.
#[derive(Default)]
pub struct EncryptedSessionsPlugin;

impl Plugin for EncryptedSessionsPlugin {
    fn build(&self, app: &mut App) {
        app.add_secret(Secret::new(SECRET, "Passphrase of the encrypted sessions"))
            .add_front(Front::new(COMMAND).headless().sessionless().on_command(
                COMMAND,
                "migrate|rekey",
                "Encrypt the plain sessions, or seal them all under RIG_SECRET_CRYPT_NEW",
            ));
        let world = app.world();
        let home = RigHome::of(world);
        let project = world
            .get_resource::<ProjectRoot>()
            .cloned()
            .unwrap_or_default();
        let command = world
            .get_resource::<Invoked>()
            .and_then(Invoked::command)
            .is_some_and(|(name, _)| name == COMMAND);
        let passphrase = world
            .get_resource::<Secrets>()
            .and_then(|secrets| secrets.get(SECRET));
        let (applies, unlocked) = match CryptConfig::read(&home) {
            Ok(config) => {
                let applies = config.applies_to(&project.0);
                let unlocked = (applies || command).then(|| unlock(&home, &config, passphrase));
                (applies, unlocked)
            }
            Err(why) => (true, Some(Err(why))),
        };
        let Some(unlocked) = unlocked else {
            return;
        };
        if applies {
            let sealing = unlocked.clone();
            app.insert_resource(SessionStores::new(move |dir| {
                let crypt = sealing.clone().map_err(io::Error::other)?;
                let plain = Arc::new(JsonlDirStore::new(dir.path()));
                Ok(Arc::new(CryptStore::new(plain, crypt)) as Arc<dyn JournalStore>)
            }));
            // A refused run keeps its text log, where the launcher points
            // and which says why; it starts no session to leak.
            if unlocked.is_ok() {
                app.insert_resource(Scratch(None));
                #[cfg(feature = "tui")]
                app.add_systems(Startup, keep_no_history);
            }
        }
        app.insert_resource(Unlocked(unlocked));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(COMMAND) {
            app.add_systems(Startup, crypt_command);
        }
    }
}

/// The terminal view keeps typed prompts in a plain file across sessions:
/// not those of an encrypted one.
#[cfg(feature = "tui")]
fn keep_no_history(history: Option<ResMut<rig_tui::PromptHistory>>) {
    if let Some(mut history) = history {
        history.file = None;
    }
}

/// `rig crypt migrate|rekey`: prints what it did, and exits.
fn crypt_command(
    invoked: Res<Invoked>,
    home: Res<RigHome>,
    paths: Res<SessionPaths>,
    unlocked: Option<Res<Unlocked>>,
    secrets: Option<Res<Secrets>>,
    mut commands: Commands,
) {
    let arguments = invoked.command().map(|(_, arguments)| arguments.join(" "));
    let done = unlocked
        .map_or_else(|| Err("no key".to_owned()), |unlocked| unlocked.0.clone())
        .and_then(|crypt| match arguments.as_deref() {
            Some("migrate") => migrate(&home.0, paths.path(), &crypt),
            Some("rekey") => {
                let new = secrets
                    .as_deref()
                    .and_then(|secrets| secrets.get(NEW_SECRET));
                rekey(&home.0, paths.path(), &crypt, new)
            }
            _ => Err("usage: rig crypt migrate|rekey".to_owned()),
        });
    let reason = match done {
        Ok(text) => {
            println!("{text}");
            ExitReason::Quit
        }
        Err(why) => {
            eprintln!("rig crypt: {why}");
            ExitReason::Failed(1)
        }
    };
    commands.trigger(Exit(reason));
}

#[cfg(test)]
mod tests;
