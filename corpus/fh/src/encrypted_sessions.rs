//! FH-18: encrypted sessions at rest.
//!
//! With `RIG_HOME/crypt.toml` saying so (`enabled = true`, or `projects =
//! ["/path"]` for some projects), every session's store, this one's and
//! those `/resume` lists, is a [`CryptStore`]: every file the harness and
//! its plugins keep of a session (agent logs, saved plugin state, blobs,
//! the effect log, the listing, the terminal view's notices and draft, a
//! plugin's own files) is sealed with ChaCha20-Poly1305 under a key
//! Argon2id derives from the passphrase,
//! the secret `crypt`: the launcher asks for it once when started at a
//! terminal, and a scripted run takes it from `RIG_SECRET_CRYPT`. A run
//! without it, or with a wrong one, is refused and keeps nothing. A file
//! that does not authenticate fails to load with [`TAMPERED`] and is left
//! as it is. What rig keeps plain by path (cut tool output, pasted images,
//! the text log) is kept nowhere, and typed prompts stay out of the prompt
//! history.
//!
//! A session is told sealed or plain by its logs' header: a plain session
//! of a project that is not encrypted opens as it is, and one of a project
//! that is is refused until `rig crypt migrate` encrypts the plain
//! sessions in place. `rig crypt rekey` seals every session again under
//! the passphrase in `RIG_SECRET_CRYPT_NEW`, and `rig crypt list` shows
//! each session's date and size without the key.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use rig_harness::harness_protocol::{Home, SessionDir, secret_variable};
use rig_harness::prelude::*;
use rig_harness::rig_memory::{DirStore, SessionError};
use serde::Deserialize;

mod reseal;
mod store;

pub use reseal::{list, migrate, rekey};
pub use store::{Crypt, CryptStore, Kdf, Sealing, TAMPERED, new_salt};

/// The secret holding the passphrase.
pub const SECRET: &str = "crypt";

/// The secret `rig crypt rekey` takes the new passphrase from.
pub const NEW_SECRET: &str = "crypt-new";

/// The command, `rig crypt list|migrate|rekey`.
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

/// How this run opens a session's store: a sealed one through a
/// [`CryptStore`] under the key, unlocked on first need; a plain one as
/// it is, unless its project is encrypted; a new one sealed when this run
/// is encrypted.
#[derive(Resource, Clone)]
struct Vault(Arc<VaultState>);

struct VaultState {
    home: Home,
    config: Result<CryptConfig, String>,
    /// Whether this run's project is encrypted, or its session sealed.
    encrypted: bool,
    passphrase: Option<String>,
    key: OnceLock<Result<Crypt, String>>,
}

impl Vault {
    /// The key, or why there is none.
    fn key(&self) -> Result<Crypt, String> {
        let state = &*self.0;
        state
            .key
            .get_or_init(|| {
                let config = state.config.clone()?;
                unlock(&state.home, &config, state.passphrase.as_deref())
            })
            .clone()
    }

    /// The store of the session in `dir`.
    fn open(&self, dir: &SessionDir) -> Result<Arc<dyn Store>, SessionError> {
        let config = self.0.config.clone().map_err(io::Error::other)?;
        let plain = Arc::new(DirStore::open(dir.path())?);
        let sealed = match Sealing::of(&*plain)? {
            Sealing::Sealed => true,
            Sealing::Empty => self.0.encrypted,
            Sealing::Plain => {
                let project = dir.working_directory();
                if project.map_or(self.0.encrypted, |project| config.applies_to(&project)) {
                    let why = "this session is plain; encrypt it with `rig crypt migrate` first";
                    return Err(io::Error::other(why).into());
                }
                false
            }
        };
        if !sealed {
            return Ok(plain);
        }
        let crypt = self.key().map_err(io::Error::other)?;
        Ok(Arc::new(CryptStore::new(plain, crypt)))
    }
}

/// Keeps the sessions encrypted at rest.
#[derive(Default)]
pub struct EncryptedSessionsPlugin;

impl Plugin for EncryptedSessionsPlugin {
    fn build(&self, app: &mut App) {
        app.add_secret(Secret::new(SECRET, "Passphrase of the encrypted sessions"))
            .add_front(Front::new(COMMAND).sessionless().on_command(
                COMMAND,
                "list|migrate|rekey",
                "List the sessions without the key, encrypt the plain ones, \
                 or seal them all under RIG_SECRET_CRYPT_NEW",
            ));
        let world = app.world();
        let home = RigHome::of(world);
        let project = world
            .get_resource::<ProjectRoot>()
            .cloned()
            .unwrap_or_default()
            .0;
        let passphrase = world
            .get_resource::<Secrets>()
            .and_then(|secrets| secrets.get(SECRET))
            .map(str::to_owned);
        let config = CryptConfig::read(&home);
        let sealing = world
            .get_resource::<SessionPaths>()
            .and_then(|paths| Sealing::of(&DirStore::open(paths.path()).ok()?).ok());
        let encrypted = sealing == Some(Sealing::Sealed)
            || config
                .as_ref()
                .map_or(true, |config| config.applies_to(&project));
        // Only a session kept sealed keeps no plain files: a plain one
        // keeps them, whether it runs or is refused.
        let seals = encrypted && sealing != Some(Sealing::Plain);
        let vault = Vault(Arc::new(VaultState {
            home,
            config,
            encrypted,
            passphrase,
            key: OnceLock::new(),
        }));
        let opening = vault.clone();
        app.insert_resource(StoreOpener::new(move |dir| opening.open(dir)))
            .insert_resource(vault);
        if seals {
            app.insert_resource(Scratch(None));
            #[cfg(feature = "tui")]
            app.add_systems(Startup, keep_no_history);
        }
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

/// `rig crypt list|migrate|rekey`: prints what it did, and exits.
fn crypt_command(
    invoked: Res<Invoked>,
    vault: Res<Vault>,
    paths: Res<SessionPaths>,
    secrets: Option<Res<Secrets>>,
    mut commands: Commands,
) {
    let arguments = invoked.command().map(|(_, arguments)| arguments.join(" "));
    let (home, current) = (&vault.0.home, paths.path());
    let done = match arguments.as_deref() {
        Some("list") => Ok(list(home, current)),
        Some("migrate") => vault.key().and_then(|crypt| migrate(home, current, &crypt)),
        Some("rekey") => vault.key().and_then(|crypt| {
            let new = secrets
                .as_deref()
                .and_then(|secrets| secrets.get(NEW_SECRET));
            rekey(home, current, &crypt, new)
        }),
        _ => Err("usage: rig crypt list|migrate|rekey".to_owned()),
    };
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
