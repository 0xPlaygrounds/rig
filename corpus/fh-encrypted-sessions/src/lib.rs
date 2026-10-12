//! FH-18: encrypted sessions at rest.
//!
//! With `RIG_HOME/crypt.toml` saying so (`enabled = true`, or `projects =
//! ["/path"]` for some projects), the session's journal (every agent's
//! conversation, the plugins' saved state, subagents' logs and stored
//! images) is kept by a [`CryptStore`] in place of the plain one: sealed
//! with ChaCha20-Poly1305 under a key Argon2id derives from a passphrase.
//! The passphrase comes from an environment variable (`passphrase_env`,
//! [`PASSPHRASE_ENV`] by default), read once at startup; a run without it,
//! or with a wrong one, refuses to start and keeps nothing. A file that
//! does not authenticate fails to load with [`TAMPERED`] and is left as it
//! is.
//!
//! What the stack would otherwise write in plain text, the plugin keeps
//! out where it can: the effect log gets no records, `/resume` lists the
//! session as "encrypted session" with its date and cost, and a fork is
//! sealed once written. `/crypt migrate` encrypts the other plain sessions
//! in place, `/crypt rekey` seals every session again under the passphrase
//! in [`NEW_PASSPHRASE_ENV`]. Other plugins seal their own files in the
//! session's directory with the [`Crypt`] resource.

use std::fs::{self, File, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_harness::harness_protocol::{Home, SessionDir};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::{StopTurns, WriteJournal};
use rig_harness::rig_tools::fs::{read_json, write_json};
use rig_sessions::{Meta, SessionTitle, write_meta};
use serde::Deserialize;

mod store;

pub use store::{
    Crypt, CryptStore, Kdf, TAMPERED, encrypt_session, new_salt, rekey_session, shred,
};

/// The environment variable the passphrase is read from by default.
pub const PASSPHRASE_ENV: &str = "RIG_CRYPT_PASSPHRASE";

/// The environment variable `/crypt rekey` reads the new passphrase from.
pub const NEW_PASSPHRASE_ENV: &str = "RIG_CRYPT_NEW_PASSPHRASE";

/// What `/resume` lists an encrypted session as, without its title.
pub const LISTED_AS: &str = "encrypted session";

/// A forked session's parent file, which the harness writes through the
/// session's store.
const PARENT: &str = "parent.json";

/// A session's listing, which rig-sessions writes through its store.
const META: &str = "meta.json";

/// `RIG_HOME/crypt.toml`.
#[derive(Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct CryptConfig {
    /// Encrypt every session.
    pub enabled: bool,
    /// Encrypt the sessions of these projects (and their subdirectories).
    pub projects: Vec<PathBuf>,
    /// The environment variable holding the passphrase.
    pub passphrase_env: Option<String>,
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

    /// The environment variable holding the passphrase.
    pub fn passphrase_env(&self) -> &str {
        self.passphrase_env.as_deref().unwrap_or(PASSPHRASE_ENV)
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
            "sessions are encrypted and there is no key: set {} (no OS keyring or \
             prompt is supported). Nothing was saved.",
            config.passphrase_env()
        ));
    };
    let crypt = Crypt::derive(passphrase.as_bytes(), &salt(home)?, config.kdf())
        .map_err(|failure| failure.to_string())?;
    crypt
        .check(&key_dir(home).join("check"))
        .map_err(|failure| format!("cannot unlock the encrypted sessions: {failure}"))?;
    Ok(crypt)
}

/// Why this run refused to start a session.
#[derive(Resource, Clone, Debug)]
pub struct CryptRefusal(pub String);

/// Keeps the session encrypted at rest.
#[derive(Default)]
pub struct EncryptedSessionsPlugin;

impl Plugin for EncryptedSessionsPlugin {
    fn build(&self, app: &mut App) {
        let home = RigHome::of(app.world());
        let project = app.world().get_resource::<ProjectRoot>().cloned();
        let config = match CryptConfig::read(&home) {
            Ok(config) => config,
            Err(why) => return refuse(app, why),
        };
        if !config.applies_to(&project.unwrap_or_default().0) {
            return;
        }
        let passphrase = std::env::var(config.passphrase_env()).ok();
        let crypt = match unlock(&home, &config, passphrase.as_deref()) {
            Ok(crypt) => crypt,
            Err(why) => return refuse(app, why),
        };
        // Every session's store the app opens, this one's, forks' and those
        // `/resume` lists, seals what it keeps.
        let sealing = crypt.clone();
        app.insert_resource(SessionStores::new(move |dir| {
            let store = CryptStore::new(dir.path(), sealing.clone());
            Ok(Arc::new(store) as Arc<dyn JournalStore>)
        }));
        app.insert_resource(crypt)
            .add_command(
                "crypt",
                "Encrypted sessions: /crypt migrate, /crypt rekey",
                crypt_command,
            )
            .add_systems(Startup, hide_title)
            .add_systems(Update, seal_forks)
            .add_systems(
                Last,
                drop_effect_records.after(StopTurns).before(WriteJournal),
            );
    }
}

/// Keeps nothing of the session, and exits at startup saying why: no
/// session's store opens.
fn refuse(app: &mut App, why: String) {
    let refused = why.clone();
    app.insert_resource(SessionStores::new(move |_| {
        Err(io::Error::other(refused.clone()))
    }));
    app.insert_resource(CryptRefusal(why))
        .add_systems(Startup, refuse_to_start)
        .add_systems(
            Last,
            drop_effect_records.after(StopTurns).before(WriteJournal),
        );
}

fn refuse_to_start(
    refusal: Res<CryptRefusal>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    notices.write(Notice::error(None, refusal.0.clone()));
    commands.trigger(Exit(ExitReason::Failed(2)));
}

/// The effect log is plain JSON lines its plugin writes itself: it gets no
/// record of an encrypted session.
fn drop_effect_records(mut records: ResMut<Messages<Recorded>>) {
    records.clear();
}

/// `/resume` lists sessions from their plain `meta.json`, whose title is
/// the start of the first message typed: this session's says only that it
/// is encrypted.
fn hide_title(title: Option<ResMut<SessionTitle>>) {
    if let Some(mut title) = title
        && title.typed.is_empty()
    {
        title.typed = LISTED_AS.to_owned();
    }
}

/// The harness writes a fork as a plain session: seals it as soon as it
/// is written, and takes the forked message's text out of its parent file.
fn seal_forks(
    mut forked: MessageReader<SessionForked>,
    home: Res<RigHome>,
    crypt: Res<Crypt>,
    mut notices: MessageWriter<Notice>,
) {
    for fork in forked.read() {
        let dir = home.0.session(&fork.to);
        let sealed = encrypt_session(dir.path(), &crypt).map_err(|failure| failure.to_string());
        if let Err(why) = sealed.and_then(|_| forget_fork_text(&dir)) {
            let why = format!("Could not encrypt the fork {}: {why}", fork.to);
            notices.write(Notice::error(None, why));
        }
    }
}

/// Removes the text of the message a fork stopped before from its parent
/// file, when it is plain.
fn forget_fork_text(dir: &SessionDir) -> Result<(), String> {
    let file = dir.path().join(PARENT);
    let Some(mut parent) = read_json::<serde_json::Map<String, serde_json::Value>>(&file) else {
        return Ok(());
    };
    if parent.shift_remove("text").is_some() {
        write_json(&file, &parent).map_err(|failure| failure.to_string())?;
    }
    Ok(())
}

/// Whether a launcher runs `session`.
fn runs_elsewhere(session: &SessionDir) -> bool {
    File::options()
        .write(true)
        .open(session.launcher_lock())
        .is_ok_and(|lock| matches!(lock.try_lock(), Err(TryLockError::WouldBlock)))
}

/// The session directories of `home` other than `current`.
fn other_sessions(home: &Home, current: Option<&Path>) -> Vec<SessionDir> {
    let Ok(entries) = fs::read_dir(home.sessions()) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
        .map(|id| home.session(&id))
        .filter(|dir| Some(dir.path()) != current)
        .collect()
}

/// `/crypt migrate`: encrypts every other plain session no launcher runs.
/// `/crypt rekey`: seals every session again under the passphrase in
/// [`NEW_PASSPHRASE_ENV`], once no other session runs.
fn crypt_command(
    In(args): In<CommandArgs>,
    home: Res<RigHome>,
    paths: Option<Res<SessionPaths>>,
    crypt: Res<Crypt>,
    mut notices: MessageWriter<Notice>,
) {
    let current = paths.as_deref().map(|paths| paths.path());
    let text = match args.args.as_str() {
        "migrate" => migrate(&home.0, current, &crypt),
        "rekey" => rekey(&home.0, current, &crypt),
        _ => Err("Use /crypt migrate or /crypt rekey.".to_owned()),
    };
    notices.write(match text {
        Ok(text) => Notice::info(args.agent, text),
        Err(why) => Notice::error(args.agent, why),
    });
}

fn migrate(home: &Home, current: Option<&Path>, crypt: &Crypt) -> Result<String, String> {
    let (mut sessions, mut skipped) = (0, 0);
    for dir in other_sessions(home, current) {
        if runs_elsewhere(&dir) {
            skipped += 1;
            continue;
        }
        let files = encrypt_session(dir.path(), crypt)
            .map_err(|failure| format!("{}: {failure}", dir.path().display()))?;
        forget_fork_text(&dir)?;
        if let Some(meta) = read_json::<Meta>(&dir.path().join(META)) {
            let title = SessionTitle {
                typed: LISTED_AS.to_owned(),
                ..meta.title
            };
            let store = CryptStore::new(dir.path(), crypt.clone());
            write_meta(&store, &Meta { title, ..meta }).map_err(|failure| failure.to_string())?;
        }
        sessions += usize::from(files > 0);
    }
    Ok(format!(
        "Encrypted {sessions} sessions; {skipped} running elsewhere were left."
    ))
}

fn rekey(home: &Home, current: Option<&Path>, crypt: &Crypt) -> Result<String, String> {
    let config = CryptConfig::read(home)?;
    let passphrase = std::env::var(NEW_PASSPHRASE_ENV)
        .ok()
        .filter(|passphrase| !passphrase.is_empty())
        .ok_or_else(|| format!("Set {NEW_PASSPHRASE_ENV} to the new passphrase first."))?;
    let others = other_sessions(home, current);
    if others.iter().any(runs_elsewhere) {
        return Err("Another rig runs an encrypted session; quit it first.".to_owned());
    }
    let salt = new_salt().map_err(|failure| failure.to_string())?;
    let new = Crypt::derive(passphrase.as_bytes(), &salt, config.kdf())
        .map_err(|failure| failure.to_string())?;
    let dirs = others.iter().map(SessionDir::path).chain(current);
    let mut count = 0;
    for dir in dirs {
        // A session still plain is sealed first.
        encrypt_session(dir, crypt).map_err(|failure| failure.to_string())?;
        rekey_session(dir, crypt, &new)
            .map_err(|failure| format!("{}: {failure}", dir.display()))?;
        count += 1;
    }
    let check = key_dir(home).join("check");
    fs::remove_file(&check).ok();
    fs::write(key_dir(home).join("salt"), salt)
        .map_err(|failure| failure.to_string())
        .and_then(|()| new.check(&check).map_err(|failure| failure.to_string()))?;
    crypt.take_key_of(&new);
    Ok(format!("Sealed {count} sessions under the new passphrase."))
}

#[cfg(test)]
mod tests;
