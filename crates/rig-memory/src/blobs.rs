//! Images kept as blobs: a saved message names each image's blob,
//! `blob:<sha256>.<ext>`, in place of its data. A message is saved in a
//! push, a splice or a branch aside ([`Branch`]).

use std::borrow::Cow;
use std::io;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rig_core::completion::Message;
use rig_core::conversation::{Aside, Branch, Change};
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

/// Calls `f` on each message `change` saves: the entries of a push or a
/// splice, and those of a branch aside and the branches inside it. A
/// branch is written back only when one of its messages holds an image.
pub(crate) fn messages_mut(
    change: &mut Change,
    f: &mut dyn FnMut(&mut Message) -> io::Result<()>,
) -> io::Result<()> {
    match change {
        Change::Push { entries } | Change::Splice { with: entries, .. } => entries
            .iter_mut()
            .try_for_each(|entry| f(&mut entry.message)),
        Change::Aside {
            aside: Some(aside), ..
        } => branch_messages(aside, f),
        _ => Ok(()),
    }
}

fn branch_messages(
    aside: &mut Aside,
    f: &mut dyn FnMut(&mut Message) -> io::Result<()>,
) -> io::Result<()> {
    let Some(mut branch) = Branch::of(aside).filter(|branch| has_image(branch, &|_| true)) else {
        return Ok(());
    };
    for entry in &mut branch.entries {
        f(&mut entry.message)?;
    }
    for (_, inner) in &mut branch.asides {
        branch_messages(inner, f)?;
    }
    aside.value = serde_json::to_value(branch)?;
    Ok(())
}

/// Whether a message of `branch`, or of a branch inside it, holds an image
/// `which` takes.
fn has_image(branch: &Branch, which: &dyn Fn(&Image) -> bool) -> bool {
    let mut entries = branch.entries.iter();
    entries.any(|entry| entry.message.images().any(which))
        || branch
            .asides
            .iter()
            .filter_map(|(_, aside)| Branch::of(aside))
            .any(|inner| has_image(&inner, which))
}

/// `change` as it is saved, its messages' images stored as blobs
/// ([`store_images`], [`messages_mut`]); `None` when it has none to store.
/// The whole value of a [`Change::Reset`] keeps its images inline.
pub(crate) fn store_change(change: &Change, store: &dyn Store) -> io::Result<Option<Change>> {
    let has_inline = match change {
        Change::Push { entries } | Change::Splice { with: entries, .. } => entries
            .iter()
            .any(|entry| entry.message.images().any(inline)),
        Change::Aside {
            aside: Some(aside), ..
        } => Branch::of(aside).is_some_and(|branch| has_image(&branch, &inline)),
        _ => false,
    };
    if !has_inline {
        return Ok(None);
    }
    let mut change = change.clone();
    messages_mut(&mut change, &mut |message| {
        if let Cow::Owned(stored) = store_images(message, store)? {
            *message = stored;
        }
        Ok(())
    })?;
    Ok(Some(change))
}
