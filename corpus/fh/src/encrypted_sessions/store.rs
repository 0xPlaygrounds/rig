//! The key and the sealing store: [`Crypt`] seals and opens bytes with
//! ChaCha20-Poly1305 under a key Argon2id derives from a passphrase, and
//! [`CryptStore`] keeps a session's files sealed in another store, which
//! keeps them where and how it likes.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use argon2::{Algorithm, Argon2, Params, Version};
use rig_harness::prelude::{Files, Resource, Store};
use ring::aead::{Aad, CHACHA20_POLY1305, LessSafeKey, NONCE_LEN, Nonce, UnboundKey};
use ring::rand::{SecureRandom, SystemRandom};

/// The error a file that does not authenticate gives.
pub const TAMPERED: &str = "session file corrupted or tampered";

/// What the key check file holds, sealed.
const CHECK: &[u8] = b"rig-crypt-check-v1";

/// What every sealed log and session file starts with, before its first
/// frame: how a sealed session is told from a plain one.
const MAGIC: &[u8] = b"rig-crypt-v1\n";

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

/// The sessions' key. Cloning shares it.
#[derive(Resource, Clone)]
pub struct Crypt(Arc<LessSafeKey>);

impl Crypt {
    /// The key `passphrase` derives with `salt`.
    ///
    /// # Errors
    ///
    /// When the KDF's parameters are refused.
    pub fn derive(passphrase: &[u8], salt: &[u8], kdf: Kdf) -> io::Result<Self> {
        let params = Params::new(kdf.memory_kib, kdf.iterations, 1, Some(32))
            .map_err(|failure| io::Error::other(failure.to_string()))?;
        let mut bytes = [0u8; 32];
        Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
            .hash_password_into(passphrase, salt, &mut bytes)
            .map_err(|failure| io::Error::other(failure.to_string()))?;
        let seal = UnboundKey::new(&CHACHA20_POLY1305, &bytes)
            .map_err(|_| io::Error::other("bad key length"))?;
        bytes.fill(0);
        Ok(Self(Arc::new(LessSafeKey::new(seal))))
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
        self.0
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
            .0
            .open_in_place(nonce, Aad::from(context), &mut body)
            .map_err(|_| tampered())?;
        Ok(plain.to_vec())
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
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, self.seal(b"check", CHECK)?)
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

/// The frames of sealed bytes: after [`MAGIC`], each a big-endian `u32`
/// length, then that many sealed bytes.
fn frames(bytes: &[u8]) -> io::Result<Vec<&[u8]>> {
    let mut frames = Vec::new();
    if bytes.is_empty() {
        return Ok(frames);
    }
    let mut rest = bytes.strip_prefix(MAGIC).ok_or_else(tampered)?;
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

/// A file kept as sealed frames: an agent's log, or a session file by
/// name.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Framed<'a> {
    Log(&'a str),
    File(&'a str),
}

impl Framed<'_> {
    /// What its frame at `index` is bound to: the file and the place, so
    /// frames cannot be moved between files, reordered or left out
    /// before the last.
    fn context(self, index: usize) -> Vec<u8> {
        match self {
            Framed::Log(agent) => format!("log\0{agent}\0{index}").into_bytes(),
            Framed::File(name) => format!("file\0{name}\0{index}").into_bytes(),
        }
    }

    fn key(self) -> (bool, String) {
        match self {
            Framed::Log(agent) => (true, agent.to_owned()),
            Framed::File(name) => (false, name.to_owned()),
        }
    }
}

/// A session's files sealed in another store, `inner`: an agent's log and
/// each other file (blobs included) as sealed frames, one per append,
/// under their own names.
pub struct CryptStore {
    inner: Arc<dyn Store>,
    crypt: Crypt,
    /// The frames of each framed file, once counted.
    counts: Mutex<HashMap<(bool, String), usize>>,
}

impl CryptStore {
    /// The session `inner` keeps, sealed with `crypt`.
    pub fn new(inner: Arc<dyn Store>, crypt: Crypt) -> Self {
        Self {
            inner,
            crypt,
            counts: Mutex::new(HashMap::new()),
        }
    }

    fn counts(&self) -> MutexGuard<'_, HashMap<(bool, String), usize>> {
        self.counts.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `plain` as the frame of `file` at `index`, the first after
    /// [`MAGIC`].
    fn frame(&self, file: Framed<'_>, index: usize, plain: &[u8]) -> io::Result<Vec<u8>> {
        let sealed = self.crypt.seal(&file.context(index), plain)?;
        let len = u32::try_from(sealed.len()).map_err(io::Error::other)?;
        let mut out = if index == 0 {
            MAGIC.to_vec()
        } else {
            Vec::new()
        };
        out.extend(len.to_be_bytes());
        out.extend(sealed);
        Ok(out)
    }

    /// Opens every frame of `file`, kept as `sealed`, and remembers how
    /// many there are.
    fn open(&self, file: Framed<'_>, sealed: &[u8]) -> io::Result<Vec<u8>> {
        let mut plain = Vec::new();
        let frames = frames(sealed)?;
        let count = frames.len();
        for (index, frame) in frames.into_iter().enumerate() {
            plain.extend(self.crypt.open(&file.context(index), frame)?);
        }
        self.counts().insert(file.key(), count);
        Ok(plain)
    }

    /// The frames `file` has, reading it the first time.
    fn count(
        &self,
        file: Framed<'_>,
        sealed: impl FnOnce() -> io::Result<Vec<u8>>,
    ) -> io::Result<usize> {
        if let Some(count) = self.counts().get(&file.key()) {
            return Ok(*count);
        }
        let count = match sealed() {
            Ok(sealed) => frames(&sealed)?.len(),
            Err(failure) if failure.kind() == io::ErrorKind::NotFound => 0,
            Err(failure) => return Err(failure),
        };
        self.counts().insert(file.key(), count);
        Ok(count)
    }

    /// `plain` as all of `file`: one frame, or none.
    fn whole(&self, file: Framed<'_>, plain: &[u8]) -> io::Result<Vec<u8>> {
        let sealed = if plain.is_empty() {
            Vec::new()
        } else {
            self.frame(file, 0, plain)?
        };
        self.counts()
            .insert(file.key(), usize::from(!plain.is_empty()));
        Ok(sealed)
    }
}

impl Store for CryptStore {
    fn logs(&self) -> io::Result<Vec<String>> {
        self.inner.logs()
    }

    fn read_log(&self, name: &str) -> io::Result<Vec<u8>> {
        self.open(Framed::Log(name), &self.inner.read_log(name)?)
    }

    fn replace_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let sealed = self.whole(Framed::Log(name), bytes)?;
        let replaced = self.inner.replace_log(name, &sealed);
        if replaced.is_err() {
            self.counts().remove(&Framed::Log(name).key());
        }
        replaced
    }

    fn append_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let file = Framed::Log(name);
        let index = self.count(file, || self.inner.read_log(name))?;
        self.inner
            .append_log(name, &self.frame(file, index, bytes)?)?;
        self.counts().insert(file.key(), index + 1);
        Ok(())
    }

    fn destroy(&self) -> io::Result<()> {
        self.counts().clear();
        self.inner.destroy()
    }
}

impl Files for CryptStore {
    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        let sealed = self.whole(Framed::File(name), bytes)?;
        self.inner.write_file(name, &sealed)
    }

    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let file = Framed::File(name);
        let index = self.count(file, || self.inner.read_file(name))?;
        self.inner
            .append_file(name, &self.frame(file, index, bytes)?)?;
        self.counts().insert(file.key(), index + 1);
        Ok(())
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        self.open(Framed::File(name), &self.inner.read_file(name)?)
    }

    fn remove_file(&self, name: &str) -> io::Result<()> {
        self.counts().remove(&Framed::File(name).key());
        self.inner.remove_file(name)
    }

    fn files(&self) -> io::Result<Vec<String>> {
        self.inner.files()
    }
}

/// What a session's store holds: no log yet, plain logs, or logs a
/// [`CryptStore`] sealed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sealing {
    /// No agent wrote a log yet.
    Empty,
    /// Its logs are plain.
    Plain,
    /// Its logs are sealed.
    Sealed,
}

impl Sealing {
    /// What the session `store` keeps, as is (not through a
    /// [`CryptStore`]), holds, by its first log that is not empty.
    ///
    /// # Errors
    ///
    /// When a log cannot be read.
    pub fn of(store: &dyn Store) -> io::Result<Self> {
        for name in store.logs()? {
            let log = store.read_log(&name)?;
            if !log.is_empty() {
                let sealed = log.starts_with(MAGIC);
                return Ok(if sealed { Self::Sealed } else { Self::Plain });
            }
        }
        Ok(Self::Empty)
    }
}
