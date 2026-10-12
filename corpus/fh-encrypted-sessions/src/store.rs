//! The keys and the encrypted session store: [`Crypt`] seals and opens
//! bytes with ChaCha20-Poly1305 under a key derived from a passphrase with
//! Argon2id, and [`CryptStore`] keeps a session's journal sealed in its
//! directory, laid out as rig-cassette's `JsonlDirStore` lays it out, so
//! the harness still finds its agent logs by name.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use argon2::{Algorithm, Argon2, Params, Version};
use rig_cassette::journal::JournalStore;
use rig_harness::prelude::Resource;
use ring::aead::{Aad, CHACHA20_POLY1305, LessSafeKey, NONCE_LEN, Nonce, UnboundKey};
use ring::hmac;
use ring::rand::{SecureRandom, SystemRandom};

/// The error a file that does not authenticate gives.
pub const TAMPERED: &str = "session file corrupted or tampered";

/// What the key check file holds, sealed.
const CHECK: &[u8] = b"rig-crypt-check-v1";

/// The extension of an agent log, as the harness names it.
const LOG: &str = "jsonl";

/// The effect log, which is no agent's log.
const EFFECTS: &str = "effects.jsonl";

fn tampered() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, TAMPERED)
}

/// How slow the key derivation is: Argon2id's memory in KiB and passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kdf {
    /// Memory, in KiB.
    pub memory_kib: u32,
    /// Passes over it.
    pub iterations: u32,
}

impl Default for Kdf {
    fn default() -> Self {
        Self {
            memory_kib: 64 * 1024,
            iterations: 3,
        }
    }
}

struct Keys {
    seal: LessSafeKey,
    name: hmac::Key,
}

/// The session key, shared by the store and any plugin writing files into
/// the session's directory ([`Crypt::write_file`]). Cloning shares it;
/// [`Crypt::take_key_of`] swaps it for every clone.
#[derive(Resource, Clone)]
pub struct Crypt(Arc<RwLock<Arc<Keys>>>);

impl Crypt {
    /// The key `passphrase` derives with `salt`.
    ///
    /// # Errors
    ///
    /// When the KDF's parameters are refused.
    pub fn derive(passphrase: &[u8], salt: &[u8], kdf: Kdf) -> io::Result<Self> {
        let params = Params::new(kdf.memory_kib, kdf.iterations, 1, Some(64))
            .map_err(|failure| io::Error::other(failure.to_string()))?;
        let mut bytes = [0u8; 64];
        Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
            .hash_password_into(passphrase, salt, &mut bytes)
            .map_err(|failure| io::Error::other(failure.to_string()))?;
        let (seal, name) = bytes.split_at(32);
        let seal = UnboundKey::new(&CHACHA20_POLY1305, seal)
            .map_err(|_| io::Error::other("bad key length"))?;
        let keys = Keys {
            seal: LessSafeKey::new(seal),
            name: hmac::Key::new(hmac::HMAC_SHA256, name),
        };
        bytes.fill(0);
        Ok(Self(Arc::new(RwLock::new(Arc::new(keys)))))
    }

    fn keys(&self) -> Arc<Keys> {
        self.0
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Takes the key of `other` for this one and every clone of it, such
    /// as the running session's store after [`rekey_session`].
    pub fn take_key_of(&self, other: &Self) {
        let keys = other.keys();
        *self.0.write().unwrap_or_else(PoisonError::into_inner) = keys;
    }

    /// `plain` sealed: a random nonce, then the ciphertext and its tag,
    /// bound to `context`, such as which file and which frame it is.
    ///
    /// # Errors
    ///
    /// When there is no randomness.
    pub fn seal(&self, context: &[u8], plain: &[u8]) -> io::Result<Vec<u8>> {
        let mut nonce = [0u8; NONCE_LEN];
        SystemRandom::new()
            .fill(&mut nonce)
            .map_err(|_| io::Error::other("no randomness"))?;
        let mut sealed = plain.to_vec();
        self.keys()
            .seal
            .seal_in_place_append_tag(
                Nonce::assume_unique_for_key(nonce),
                Aad::from(context),
                &mut sealed,
            )
            .map_err(|_| io::Error::other("could not seal"))?;
        let mut out = nonce.to_vec();
        out.append(&mut sealed);
        Ok(out)
    }

    /// What [`Crypt::seal`] sealed under `context`.
    ///
    /// # Errors
    ///
    /// [`TAMPERED`] when it does not authenticate: another key, another
    /// context, or a changed byte.
    pub fn open(&self, context: &[u8], sealed: &[u8]) -> io::Result<Vec<u8>> {
        let (nonce, body) = sealed.split_at_checked(NONCE_LEN).ok_or_else(tampered)?;
        let nonce = Nonce::try_assume_unique_for_key(nonce).map_err(|_| tampered())?;
        let mut body = body.to_vec();
        let plain = self
            .keys()
            .seal
            .open_in_place(nonce, Aad::from(context), &mut body)
            .map_err(|_| tampered())?;
        Ok(plain.to_vec())
    }

    /// A file name for `name` that does not tell it, such as a blob's
    /// content hash.
    fn hide(&self, name: &str) -> String {
        let tag = hmac::sign(&self.keys().name, name.as_bytes());
        tag.as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    /// Writes `bytes` sealed to `path`, such as a plugin's file in the
    /// session's directory, through a temporary file.
    ///
    /// # Errors
    ///
    /// When it cannot be written.
    pub fn write_file(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        let sealed = self.seal(b"file", bytes)?;
        replace(path, &sealed)
    }

    /// What [`Crypt::write_file`] wrote to `path`.
    ///
    /// # Errors
    ///
    /// When it cannot be read, or [`TAMPERED`].
    pub fn read_file(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.open(b"file", &fs::read(path)?)
    }

    /// Whether this key sealed the check at `path`, or, without one yet,
    /// makes it.
    ///
    /// # Errors
    ///
    /// "wrong passphrase", or when the check cannot be read or written.
    pub fn check(&self, path: &Path) -> io::Result<()> {
        match fs::read(path) {
            Ok(sealed) => match self.open(b"check", &sealed) {
                Ok(plain) if plain == CHECK => Ok(()),
                _ => Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "wrong passphrase",
                )),
            },
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => {
                replace(path, &self.seal(b"check", CHECK)?)
            }
            Err(failure) => Err(failure),
        }
    }
}

/// Random bytes for a new salt.
///
/// # Errors
///
/// When there is no randomness.
pub fn new_salt() -> io::Result<[u8; 16]> {
    let mut salt = [0u8; 16];
    SystemRandom::new()
        .fill(&mut salt)
        .map_err(|_| io::Error::other("no randomness"))?;
    Ok(salt)
}

/// Writes `bytes` to `path` through a temporary file beside it.
fn replace(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    let mut file = File::create(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)
}

/// Overwrites `path` with zeros, then removes it: as far as a file system
/// that writes in place allows, the old bytes are gone.
pub fn shred(path: &Path) -> io::Result<()> {
    let len = fs::metadata(path)?.len();
    let mut file = OpenOptions::new().write(true).open(path)?;
    let zeros = vec![0u8; usize::try_from(len).unwrap_or(0)];
    file.write_all(&zeros)?;
    file.sync_all()?;
    fs::remove_file(path)
}

/// The frames of a sealed log: each a big-endian `u32` length, then that
/// many sealed bytes.
fn frames(bytes: &[u8]) -> io::Result<Vec<&[u8]>> {
    let mut rest = bytes;
    let mut frames = Vec::new();
    while !rest.is_empty() {
        let (len, after) = rest.split_at_checked(4).ok_or_else(tampered)?;
        let len = u32::from_be_bytes(len.try_into().map_err(|_| tampered())?);
        let len = usize::try_from(len).map_err(|_| tampered())?;
        let (frame, after) = after.split_at_checked(len).ok_or_else(tampered)?;
        frames.push(frame);
        rest = after;
    }
    Ok(frames)
}

/// What a frame of `agent`'s log is bound to: its log and its place, so
/// frames cannot be moved between logs or reordered.
fn frame_context(agent: &str, index: usize) -> Vec<u8> {
    format!("log\0{agent}\0{index}").into_bytes()
}

/// A frame of `agent`'s log at `index`, holding `plain`.
fn frame(crypt: &Crypt, agent: &str, index: usize, plain: &[u8]) -> io::Result<Vec<u8>> {
    let sealed = crypt.seal(&frame_context(agent, index), plain)?;
    let len = u32::try_from(sealed.len()).map_err(io::Error::other)?;
    let mut out = len.to_be_bytes().to_vec();
    out.extend(sealed);
    Ok(out)
}

/// Opens every frame of `agent`'s sealed log `bytes`.
fn open_log(crypt: &Crypt, agent: &str, bytes: &[u8]) -> io::Result<Vec<u8>> {
    let mut plain = Vec::new();
    for (index, sealed) in frames(bytes)?.into_iter().enumerate() {
        plain.extend(crypt.open(&frame_context(agent, index), sealed)?);
    }
    Ok(plain)
}

/// A session's journal, sealed, in its directory: `<agent>.jsonl` holds
/// an agent's log as sealed frames, one per append, and `blobs/<hidden>`
/// each blob with its name, under a name that does not tell its content
/// hash.
pub struct CryptStore {
    dir: PathBuf,
    crypt: Crypt,
    /// Each log's length and frames when last seen.
    counts: Mutex<HashMap<String, (u64, usize)>>,
}

impl CryptStore {
    /// The session in `dir`, sealed with `crypt`.
    pub fn new(dir: impl Into<PathBuf>, crypt: Crypt) -> Self {
        Self {
            dir: dir.into(),
            crypt,
            counts: Mutex::new(HashMap::new()),
        }
    }

    fn log(&self, agent: &str) -> PathBuf {
        self.dir.join(format!("{agent}.{LOG}"))
    }

    fn blob_path(&self, name: &str) -> PathBuf {
        self.dir.join("blobs").join(self.crypt.hide(name))
    }

    fn counts(&self) -> MutexGuard<'_, HashMap<String, (u64, usize)>> {
        self.counts.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The frames of `agent`'s log: cached by the log's length, so a log
    /// rewritten elsewhere, such as by [`rekey_session`], is counted again.
    fn count(&self, agent: &str) -> io::Result<usize> {
        let len = match fs::metadata(self.log(agent)) {
            Ok(metadata) => metadata.len(),
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(failure) => return Err(failure),
        };
        if let Some((known, count)) = self.counts().get(agent)
            && *known == len
        {
            return Ok(*count);
        }
        let count = frames(&fs::read(self.log(agent))?)?.len();
        self.counts().insert(agent.to_owned(), (len, count));
        Ok(count)
    }
}

/// The agent logs of `dir`, by agent id, as the harness lists them.
fn agent_logs(dir: &Path) -> io::Result<Vec<String>> {
    let entries = match fs::read_dir(dir) {
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        entries => entries?,
    };
    Ok(entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let stem = path.file_stem()?.to_str()?;
            let log = path.extension()? == LOG && !stem.is_empty();
            (log && path.file_name()? != EFFECTS).then(|| stem.to_owned())
        })
        .collect())
}

impl JournalStore for CryptStore {
    fn agents(&self) -> io::Result<Vec<String>> {
        agent_logs(&self.dir)
    }

    fn read(&self, agent: &str) -> io::Result<Vec<u8>> {
        open_log(&self.crypt, agent, &fs::read(self.log(agent))?)
    }

    fn truncate(&self, agent: &str, len: u64) -> io::Result<()> {
        let mut plain = self.read(agent)?;
        plain.truncate(usize::try_from(len).map_err(io::Error::other)?);
        let sealed = if plain.is_empty() {
            Vec::new()
        } else {
            frame(&self.crypt, agent, 0, &plain)?
        };
        replace(&self.log(agent), &sealed)?;
        let frames = usize::from(!plain.is_empty());
        self.counts()
            .insert(agent.to_owned(), (sealed.len() as u64, frames));
        Ok(())
    }

    fn append(&self, agent: &str, bytes: &[u8]) -> io::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let index = self.count(agent)?;
        let framed = frame(&self.crypt, agent, index, bytes)?;
        fs::create_dir_all(&self.dir)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.log(agent))?;
        file.write_all(&framed)?;
        let len = file.metadata()?.len();
        self.counts().insert(agent.to_owned(), (len, index + 1));
        Ok(())
    }

    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let path = self.blob_path(name);
        if path.exists() {
            return Ok(());
        }
        let len = u16::try_from(name.len()).map_err(io::Error::other)?;
        let mut plain = len.to_be_bytes().to_vec();
        plain.extend_from_slice(name.as_bytes());
        plain.extend_from_slice(bytes);
        replace(&path, &self.crypt.seal(b"blob", &plain)?)
    }

    fn blob(&self, name: &str) -> io::Result<Vec<u8>> {
        let (stored, bytes) = open_blob(&self.crypt, &fs::read(self.blob_path(name))?)?;
        if stored == name {
            Ok(bytes)
        } else {
            Err(tampered())
        }
    }

    fn blobs(&self) -> io::Result<Vec<String>> {
        let entries = match fs::read_dir(self.dir.join("blobs")) {
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            entries => entries?,
        };
        let mut names = Vec::new();
        for entry in entries.filter_map(Result::ok) {
            if entry.path().extension().is_some_and(|ext| ext == "tmp") {
                continue;
            }
            names.push(open_blob(&self.crypt, &fs::read(entry.path())?)?.0);
        }
        Ok(names)
    }

    fn remove(&self, agent: &str) -> io::Result<()> {
        self.counts().remove(agent);
        match fs::remove_file(self.log(agent)) {
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => Ok(()),
            removed => removed,
        }
    }
}

/// A sealed blob's name and content.
fn open_blob(crypt: &Crypt, sealed: &[u8]) -> io::Result<(String, Vec<u8>)> {
    let plain = crypt.open(b"blob", sealed)?;
    let (len, rest) = plain.split_at_checked(2).ok_or_else(tampered)?;
    let len = usize::from(u16::from_be_bytes(len.try_into().map_err(|_| tampered())?));
    let (name, bytes) = rest.split_at_checked(len).ok_or_else(tampered)?;
    let name = String::from_utf8(name.to_vec()).map_err(|_| tampered())?;
    Ok((name, bytes.to_vec()))
}

/// Whether a log file is still plain JSON lines: sealed frames start with
/// a length, whose first byte is zero for any frame under 16 MiB.
fn is_plain(bytes: &[u8]) -> bool {
    bytes.first().is_some_and(|byte| *byte == b'{')
}

/// Encrypts the plain logs and blobs of the session in `dir` in place: each
/// sealed file is written first, then the plain one overwritten and
/// removed. The effect log, which the stack keeps only in plain text, is
/// shredded. Returns how many files it encrypted.
///
/// # Errors
///
/// When a file cannot be read or written; the files done stay done.
pub fn encrypt_session(dir: &Path, crypt: &Crypt) -> io::Result<usize> {
    let store = CryptStore::new(dir, crypt.clone());
    let mut done = 0;
    for agent in agent_logs(dir)? {
        let path = store.log(&agent);
        let bytes = fs::read(&path)?;
        if !is_plain(&bytes) {
            continue;
        }
        let sealed = frame(crypt, &agent, 0, &bytes)?;
        let temporary = path.with_extension("sealed");
        fs::write(&temporary, &sealed)?;
        shred(&path)?;
        fs::rename(&temporary, &path)?;
        done += 1;
    }
    let blobs = dir.join("blobs");
    if let Ok(entries) = fs::read_dir(&blobs) {
        for entry in entries.filter_map(Result::ok) {
            // A plain blob is named by its content hash and extension; a
            // sealed one by a bare HMAC.
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if !name.contains('.') || name.ends_with(".tmp") {
                continue;
            }
            store.put_blob(&name, &fs::read(entry.path())?)?;
            shred(&entry.path())?;
            done += 1;
        }
    }
    let effects = dir.join(EFFECTS);
    if effects.exists() {
        shred(&effects)?;
    }
    Ok(done)
}

/// Seals every log and blob of the session in `dir` again under `to`
/// instead of `from`. Every new file is written before any old one is
/// replaced.
///
/// # Errors
///
/// When a file cannot be read, or [`TAMPERED`] under `from`.
pub fn rekey_session(dir: &Path, from: &Crypt, to: &Crypt) -> io::Result<()> {
    let old = CryptStore::new(dir, from.clone());
    let mut staged: Vec<(PathBuf, Vec<u8>)> = Vec::new();
    for agent in agent_logs(dir)? {
        let plain = old.read(&agent)?;
        let sealed = if plain.is_empty() {
            Vec::new()
        } else {
            frame(to, &agent, 0, &plain)?
        };
        staged.push((old.log(&agent), sealed));
    }
    let new = CryptStore::new(dir, to.clone());
    let mut blobs = Vec::new();
    for name in old.blobs()? {
        blobs.push((old.blob_path(&name), name.clone(), old.blob(&name)?));
    }
    for (path, sealed) in staged {
        replace(&path, &sealed)?;
    }
    for (path, name, bytes) in blobs {
        new.put_blob(&name, &bytes)?;
        fs::remove_file(path)?;
    }
    Ok(())
}
