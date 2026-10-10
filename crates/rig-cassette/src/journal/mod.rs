//! Where an agent session's journal is kept: one append-only log per
//! agent and content-addressed blobs such as images, behind a
//! [`JournalStore`]. [`MemoryStore`] keeps a session in memory;
//! with feature `jsonl`, [`JsonlDirStore`] keeps it as JSON-lines files in
//! a directory. A logged message names its images' blobs in place of their
//! data ([`store_images`]), and gets the data back on reading
//! ([`load_images`]).

#[cfg(feature = "jsonl")]
mod jsonl;
#[cfg(feature = "jsonl")]
pub use jsonl::{EFFECT_LOG, JsonlDirStore};

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rig_core::completion::Message;
use rig_core::message::DocumentSourceKind::{Base64, Raw, Unknown, Url};
use rig_core::message::{Image, ImageMediaType};
use sha2::{Digest, Sha256};

/// Storage for a session's journal. Every method takes `&self`, so one
/// store is shared by every agent's log.
pub trait JournalStore: Send + Sync + 'static {
    /// The ids of the agents that have a log, in no particular order.
    fn agents(&self) -> io::Result<Vec<String>>;

    /// The whole log of `agent`.
    fn read(&self, agent: &str) -> io::Result<Vec<u8>>;

    /// Cuts the log of `agent` to its first `len` bytes, such as a torn
    /// last line a crash left.
    fn truncate(&self, agent: &str, len: u64) -> io::Result<()>;

    /// Appends `bytes` to the log of `agent`, which it starts when there is
    /// none.
    fn append(&self, agent: &str, bytes: &[u8]) -> io::Result<()>;

    /// Stores the blob `name`, unless it is stored already: a name is a
    /// hash of the content.
    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// The blob `name`.
    fn blob(&self, name: &str) -> io::Result<Vec<u8>>;
}

/// How a logged message names an image stored as a blob, in place of its
/// data: `blob:<sha256>.<ext>`.
const BLOB: &str = "blob:";

/// `message` as it is logged: the data of each image in it stored in
/// `blobs` once per content, and named there instead. Borrowed when it has
/// no image data to store; data that is not valid base64 stays inline.
pub fn store_images<'m>(
    message: &'m Message,
    blobs: &dyn JournalStore,
) -> io::Result<Cow<'m, Message>> {
    let inline = |image: &Image| matches!(image.data, Base64(_) | Raw(_));
    if !message.images().any(inline) {
        return Ok(Cow::Borrowed(message));
    }
    let mut message = message.clone();
    for image in message.images_mut() {
        let bytes = match &image.data {
            Base64(data) => STANDARD.decode(data).ok(),
            Raw(bytes) => Some(bytes.clone()),
            _ => None,
        };
        if let Some(bytes) = bytes {
            let extension = image.media_type.as_ref();
            let extension = extension.map_or("bin", ImageMediaType::extension);
            let name = format!("{:x}.{extension}", Sha256::digest(&bytes));
            blobs.put_blob(&name, &bytes)?;
            image.data = Url(format!("{BLOB}{name}"));
        }
    }
    Ok(Cow::Owned(message))
}

/// Puts the data of each image `message` names in `blobs` back in place.
/// An image whose blob cannot be read has no data any more ([`Unknown`]),
/// which a request sends as a placeholder; the first such failure is
/// returned.
pub fn load_images(message: &mut Message, blobs: &dyn JournalStore) -> io::Result<()> {
    let mut gone = Ok(());
    for image in message.images_mut() {
        if let Url(url) = &image.data
            && let Some(name) = url.strip_prefix(BLOB)
        {
            image.data = match blobs.blob(name) {
                Ok(bytes) => Base64(STANDARD.encode(bytes)),
                Err(failure) => {
                    gone = gone.and(Err(failure));
                    Unknown
                }
            };
        }
    }
    gone
}

/// A session kept in memory, such as for tests or where there is no file
/// system. Cloning shares the contents.
#[derive(Clone, Default)]
pub struct MemoryStore(Arc<Mutex<Memory>>);

#[derive(Default)]
struct Memory {
    logs: BTreeMap<String, Vec<u8>>,
    blobs: BTreeMap<String, Vec<u8>>,
}

impl MemoryStore {
    fn memory(&self) -> MutexGuard<'_, Memory> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl JournalStore for MemoryStore {
    fn agents(&self) -> io::Result<Vec<String>> {
        Ok(self.memory().logs.keys().cloned().collect())
    }

    fn read(&self, agent: &str) -> io::Result<Vec<u8>> {
        self.memory()
            .logs
            .get(agent)
            .cloned()
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    fn truncate(&self, agent: &str, len: u64) -> io::Result<()> {
        let len = usize::try_from(len).map_err(io::Error::other)?;
        if let Some(log) = self.memory().logs.get_mut(agent) {
            log.truncate(len);
        }
        Ok(())
    }

    fn append(&self, agent: &str, bytes: &[u8]) -> io::Result<()> {
        self.memory()
            .logs
            .entry(agent.to_owned())
            .or_default()
            .extend_from_slice(bytes);
        Ok(())
    }

    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.memory()
            .blobs
            .entry(name.to_owned())
            .or_insert_with(|| bytes.to_vec());
        Ok(())
    }

    fn blob(&self, name: &str) -> io::Result<Vec<u8>> {
        self.memory()
            .blobs
            .get(name)
            .cloned()
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }
}

#[cfg(test)]
mod tests;
