//! `rig crypt list`, which needs no key, and `rig crypt migrate` and
//! `rig crypt rekey`: each session is copied into
//! a new directory through a store that seals under the key it should
//! have, which then takes the session's place, and the old files are
//! overwritten before they are removed.

use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use rig_harness::harness_protocol::{Home, SessionDir};
use rig_harness::prelude::{Files, Store};
use rig_harness::rig_memory::DirStore;

use crate::encrypted_sessions::store::{Crypt, CryptStore, Sealing, new_salt};
use crate::encrypted_sessions::{CryptConfig, key_dir};

/// Each saved session of `home` but `current`: its id, whether it is
/// sealed, the day it was last saved and its size on disk, read without
/// the key.
pub fn list(home: &Home, current: &Path) -> String {
    let mut lines: Vec<(u64, String)> = sessions(home, current)
        .iter()
        .map(|dir| {
            let paths = walk(dir.path()).unwrap_or_default();
            let metadata: Vec<_> = paths
                .iter()
                .filter_map(|path| fs::metadata(path).ok())
                .collect();
            let size: u64 = metadata.iter().map(fs::Metadata::len).sum();
            let saved = metadata
                .iter()
                .filter_map(|file| file.modified().ok()?.duration_since(UNIX_EPOCH).ok())
                .map(|since| since.as_secs())
                .max()
                .unwrap_or_default();
            let sealing = match plain(dir).and_then(|plain| Sealing::of(&plain)) {
                Ok(Sealing::Sealed) => "sealed",
                Ok(_) => "plain",
                Err(_) => "unreadable",
            };
            let id = dir
                .id()
                .map(|id| id.as_str().to_owned())
                .unwrap_or_default();
            let line = format!(
                "{id}  {}  {:>9} KiB  {sealing}",
                day(saved),
                size.div_ceil(1024)
            );
            (saved, line)
        })
        .collect();
    lines.sort_by_key(|(saved, _)| std::cmp::Reverse(*saved));
    let lines: Vec<String> = lines.into_iter().map(|(_, line)| line).collect();
    lines.join("\n")
}

/// The day `seconds` after the Unix epoch falls on, as `YYYY-MM-DD` in
/// UTC (the civil calendar from a day count).
fn day(seconds: u64) -> String {
    let days = i64::try_from(seconds / 86_400).unwrap_or_default() + 719_468;
    let era = days.div_euclid(146_097);
    let of_era = days.rem_euclid(146_097);
    let year_of_era = (of_era - of_era / 1460 + of_era / 36_524 - of_era / 146_096) / 365;
    let of_year = of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted = (5 * of_year + 2) / 153;
    let day = of_year - (153 * shifted + 2) / 5 + 1;
    let month = if shifted < 10 {
        shifted + 3
    } else {
        shifted - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Encrypts every plain session of `home` but `current` that no launcher
/// runs.
///
/// # Errors
///
/// When a session cannot be read or written; the sessions done stay done.
pub fn migrate(home: &Home, current: &Path, crypt: &Crypt) -> Result<String, String> {
    let (mut sealed, mut skipped) = (0, 0);
    for dir in sessions(home, current) {
        let plain = plain(&dir).map_err(|failure| failure.to_string())?;
        if Sealing::of(&plain).map_err(|failure| failure.to_string())? != Sealing::Plain {
            continue;
        }
        if dir.runs_elsewhere() {
            skipped += 1;
            continue;
        }
        reseal(&dir, &plain, crypt)
            .map_err(|failure| format!("{}: {failure}", dir.path().display()))?;
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
    if others.iter().any(SessionDir::runs_elsewhere) {
        return Err("another rig runs a session; quit it first".to_owned());
    }
    let config = CryptConfig::read(home)?;
    let salt = new_salt().map_err(|failure| failure.to_string())?;
    let new = Crypt::derive(passphrase.as_bytes(), &salt, config.kdf())
        .map_err(|failure| failure.to_string())?;
    let mut count = 0;
    for dir in others {
        let plain = plain(&dir).map_err(|failure| failure.to_string())?;
        if Sealing::of(&plain).map_err(|failure| failure.to_string())? != Sealing::Sealed {
            continue;
        }
        let old = CryptStore::new(Arc::new(plain), crypt.clone());
        reseal(&dir, &old, &new)
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

/// The store `dir` keeps as it is, as the harness's default store does.
fn plain(dir: &SessionDir) -> io::Result<DirStore> {
    Ok(DirStore::open(dir.path())?)
}

/// `dir` with `suffix` after its name.
fn beside(dir: &Path, suffix: &str) -> PathBuf {
    let mut name = OsString::from(dir.as_os_str());
    name.push(suffix);
    PathBuf::from(name)
}

/// Copies the session `from` keeps in `dir` into a new directory, sealed
/// with `to`, puts it in the session's place, and shreds the old files:
/// its logs and files (blobs included) through the stores, and the plain files
/// beside them as they are, but for the text log and the plain output
/// and images (folders) an encrypted session keeps nowhere.
fn reseal(dir: &SessionDir, from: &dyn Store, to: &Crypt) -> io::Result<()> {
    let staging = SessionDir::at(beside(dir.path(), ".sealing"));
    if staging.path().exists() {
        fs::remove_dir_all(staging.path())?;
    }
    fs::create_dir_all(staging.path())?;
    let target = CryptStore::new(Arc::new(plain(&staging)?), to.clone());
    for name in from.logs()? {
        target.append_log(&name, &from.read_log(&name)?)?;
    }
    for name in from.files()? {
        target.write_file(&name, &from.read_file(&name)?)?;
    }
    for entry in fs::read_dir(dir.path())? {
        let file = entry?.path();
        if file.is_file()
            && file != dir.log()
            && let Some(name) = file.file_name()
        {
            fs::copy(&file, staging.path().join(name))?;
        }
    }
    let old = beside(dir.path(), ".old");
    fs::rename(dir.path(), &old)?;
    fs::rename(staging.path(), dir.path())?;
    for file in walk(&old)? {
        shred(&file)?;
    }
    fs::remove_dir_all(&old)
}

/// Every file under `dir`.
fn walk(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            found.extend(walk(&entry.path())?);
        } else {
            found.push(entry.path());
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
