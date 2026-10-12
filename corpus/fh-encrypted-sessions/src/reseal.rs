//! `rig crypt migrate` and `rig crypt rekey`: each session is copied into
//! a new directory through a store that seals under the key it should
//! have, which then takes the session's place, and the old files are
//! overwritten before they are removed.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_cassette::journal::JsonlDirStore;
use rig_harness::harness_protocol::{Home, SessionDir};
use rig_harness::prelude::JournalStore;

use crate::store::{Crypt, CryptStore, new_salt};
use crate::{CryptConfig, key_dir};

/// The plain files of a session directory the launcher reads, which hold
/// no conversation: copied as they are.
const MARKERS: [&str; 5] = ["launcher.lock", "directory", "switch", "started", "saved"];

/// Encrypts every plain session of `home` but `current` that no launcher
/// runs.
///
/// # Errors
///
/// When a session cannot be read or written; the sessions done stay done.
pub fn migrate(home: &Home, current: &Path, crypt: &Crypt) -> Result<String, String> {
    let (mut sealed, mut skipped) = (0, 0);
    for dir in sessions(home, current) {
        let plain = Arc::new(JsonlDirStore::new(dir.path()));
        if !is_plain(&*plain) {
            continue;
        }
        if runs_elsewhere(&dir) {
            skipped += 1;
            continue;
        }
        reseal(dir.path(), &*plain, crypt).map_err(|failure| failure.to_string())?;
        sealed += 1;
    }
    Ok(format!(
        "Encrypted {sealed} sessions; {skipped} running elsewhere were left plain."
    ))
}

/// Seals every encrypted session of `home` but `current` again under
/// `passphrase`, once no other session runs, then makes it the key.
///
/// # Errors
///
/// Without a new passphrase, while another session runs, or when a
/// session cannot be read or written.
pub fn rekey(
    home: &Home,
    current: &Path,
    crypt: &Crypt,
    passphrase: Option<&str>,
) -> Result<String, String> {
    let passphrase = passphrase
        .filter(|passphrase| !passphrase.is_empty())
        .ok_or("set RIG_SECRET_CRYPT_NEW to the new passphrase first")?;
    let others = sessions(home, current);
    if others.iter().any(runs_elsewhere) {
        return Err("another rig runs a session; quit it first".to_owned());
    }
    let config = CryptConfig::read(home)?;
    let salt = new_salt().map_err(|failure| failure.to_string())?;
    let new = Crypt::derive(passphrase.as_bytes(), &salt, config.kdf())
        .map_err(|failure| failure.to_string())?;
    let mut count = 0;
    for dir in others {
        let plain: Arc<dyn JournalStore> = Arc::new(JsonlDirStore::new(dir.path()));
        if is_plain(&*plain) {
            continue;
        }
        let old = CryptStore::new(plain, crypt.clone());
        reseal(dir.path(), &old, &new)
            .map_err(|failure| format!("{}: {failure}", dir.path().display()))?;
        count += 1;
    }
    let check = key_dir(home).join("check");
    fs::remove_file(&check).ok();
    fs::write(key_dir(home).join("salt"), salt)
        .and_then(|()| new.check(&check))
        .map_err(|failure| failure.to_string())?;
    Ok(format!("Sealed {count} sessions under the new passphrase."))
}

/// The saved sessions of `home` but `current`.
fn sessions(home: &Home, current: &Path) -> Vec<SessionDir> {
    let Ok(entries) = fs::read_dir(home.sessions()) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
        .map(|id| home.session(&id))
        .filter(|dir| dir.path() != current && dir.is_saved())
        .collect()
}

/// Whether the session in `store` is plain: its logs are JSON lines,
/// where a sealed log starts with a frame's length.
fn is_plain(store: &dyn JournalStore) -> bool {
    let agents = store.agents().unwrap_or_default();
    agents
        .iter()
        .filter_map(|agent| store.read(agent).ok())
        .any(|log| log.first() == Some(&b'{'))
}

/// Whether a launcher holds `session`'s lock.
fn runs_elsewhere(session: &SessionDir) -> bool {
    File::options()
        .write(true)
        .open(session.launcher_lock())
        .is_ok_and(|lock| matches!(lock.try_lock(), Err(TryLockError::WouldBlock)))
}

/// `dir` with `suffix` after its name.
fn beside(dir: &Path, suffix: &str) -> PathBuf {
    let mut name = OsString::from(dir.as_os_str());
    name.push(suffix);
    PathBuf::from(name)
}

/// Copies the session `from` keeps in `dir` into a new directory, sealed
/// with `to`, puts it in the session's place, and shreds the old files.
fn reseal(dir: &Path, from: &dyn JournalStore, to: &Crypt) -> io::Result<()> {
    let staging = beside(dir, ".sealing");
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    let target = CryptStore::new(Arc::new(JsonlDirStore::new(&staging)), to.clone());
    let agents = from.agents()?;
    for agent in &agents {
        target.append(agent, &from.read(agent)?)?;
    }
    for blob in from.blobs()? {
        target.put_blob(&blob, &from.blob(&blob)?)?;
    }
    for name in files(dir, Path::new(""))? {
        let log = name
            .strip_suffix(".jsonl")
            .is_some_and(|agent| agents.iter().any(|known| known == agent));
        if log || name.starts_with("blobs/") || name.ends_with(".tmp") {
            continue;
        }
        if MARKERS.contains(&name.as_str()) {
            fs::copy(dir.join(&name), staging.join(&name))?;
        } else {
            target.write_file(&name, &from.read_file(&name)?)?;
        }
    }
    let old = beside(dir, ".old");
    fs::rename(dir, &old)?;
    fs::rename(&staging, dir)?;
    for name in files(&old, Path::new(""))? {
        shred(&old.join(name))?;
    }
    fs::remove_dir_all(&old)
}

/// Every file under `dir`, by its `/`-separated name below `prefix`.
fn files(dir: &Path, prefix: &Path) -> io::Result<Vec<String>> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir.join(prefix))? {
        let entry = entry?;
        let name = prefix.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            found.extend(files(dir, &name)?);
        } else if let Some(name) = name.to_str() {
            found.push(name.replace('\\', "/"));
        }
    }
    Ok(found)
}

/// Overwrites `path` with zeros, then removes it: as far as a file system
/// that writes in place allows, the old bytes are gone.
fn shred(path: &Path) -> io::Result<()> {
    let len = fs::metadata(path)?.len();
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.write_all(&vec![0u8; usize::try_from(len).unwrap_or(0)])?;
    file.sync_all()?;
    fs::remove_file(path)
}
