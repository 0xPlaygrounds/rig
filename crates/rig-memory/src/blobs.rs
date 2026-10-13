//! Images kept as blobs: a saved message names each image's blob,
//! `blob:<sha256>.<ext>`, in place of its data.

use std::borrow::Cow;
use std::io;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rig_core::completion::Message;
use rig_core::conversation::{Change, Entry};
use rig_core::message::DocumentSourceKind::{Base64, Raw, Unknown, Url};
use rig_core::message::{Image, ImageMediaType};
use sha2::{Digest, Sha256};

use crate::Store;

/// How a saved message names an image stored as a blob, in place of its
/// data: `blob:<sha256>.<ext>`.
const BLOB: &str = "blob:";

fn inline(image: &Image) -> bool {
    matches!(image.data, Base64(_) | Raw(_))
}

/// `message` as it is saved: the data of each image in it stored in
/// `store` once per content, and named there instead. Borrowed when it has
/// no image data to store; data that is not valid base64 stays inline.
///
/// # Errors
///
/// When the store cannot store a blob.
pub fn store_images<'m>(message: &'m Message, store: &dyn Store) -> io::Result<Cow<'m, Message>> {
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
            store.put_blob(&name, &bytes)?;
            image.data = Url(format!("{BLOB}{name}"));
        }
    }
    Ok(Cow::Owned(message))
}

/// Puts the data of each image `message` names in `store` back in place.
/// An image whose blob cannot be read has no data any more ([`Unknown`]),
/// which a request sends as a placeholder; the first such failure is
/// returned.
///
/// # Errors
///
/// The first blob that could not be read.
pub fn load_images(message: &mut Message, store: &dyn Store) -> io::Result<()> {
    load_images_with(message, &|name| store.blob(name))
}

/// [`load_images`] from the blobs `blob` reads.
pub(crate) fn load_images_with(
    message: &mut Message,
    blob: &dyn Fn(&str) -> io::Result<Vec<u8>>,
) -> io::Result<()> {
    let mut gone = Ok(());
    for image in message.images_mut() {
        if let Url(url) = &image.data
            && let Some(name) = url.strip_prefix(BLOB)
        {
            image.data = match blob(name) {
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

/// Adds the name of every blob `bytes` names (`blob:<name>`) to `named`.
pub(crate) fn named_blobs(bytes: &[u8], named: &mut std::collections::BTreeSet<String>) {
    let name_byte = |byte: &u8| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-');
    let mut rest = bytes;
    while let Some(at) = rest
        .windows(BLOB.len())
        .position(|window| window == BLOB.as_bytes())
    {
        let after = rest.get(at + BLOB.len()..).unwrap_or_default();
        let len = after.iter().take_while(|byte| name_byte(byte)).count();
        if let Some(name) = after
            .get(..len)
            .and_then(|name| std::str::from_utf8(name).ok())
            && !name.is_empty()
        {
            named.insert(name.to_owned());
        }
        rest = after.get(len..).unwrap_or_default();
    }
}

/// The entries a change puts in.
pub(crate) fn entries_mut(change: &mut Change) -> &mut [Entry] {
    match change {
        Change::Push { entries } | Change::Splice { with: entries, .. } => entries,
        _ => &mut [],
    }
}

fn entries(change: &Change) -> &[Entry] {
    match change {
        Change::Push { entries } | Change::Splice { with: entries, .. } => entries,
        _ => &[],
    }
}

/// `change` as it is saved, its entries' images stored as blobs
/// ([`store_images`]); `None` when it has none to store. The whole value
/// of a [`Change::Reset`] keeps its images inline.
pub(crate) fn store_change(change: &Change, store: &dyn Store) -> io::Result<Option<Change>> {
    let has_inline = |entry: &Entry| entry.message.images().any(inline);
    if !entries(change).iter().any(has_inline) {
        return Ok(None);
    }
    let mut change = change.clone();
    for entry in entries_mut(&mut change) {
        if let Cow::Owned(message) = store_images(&entry.message, store)? {
            entry.message = message;
        }
    }
    Ok(Some(change))
}
