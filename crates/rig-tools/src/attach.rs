//! Files a message names as `@path`, read to go with it: an image as an
//! image, any other file as its text, capped the way `read` shows it, so
//! the model needs no `read` call for it.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use base64::Engine;
use rig_core::completion::Message;
use rig_core::conversation::{Conversation, EntryRef, FILE_CLOSE, FILE_OPEN, attached_file};
use rig_core::message::{ImageMediaType, UserContent};

use crate::workspace::{Stat, Workspace};
use crate::{MAX_LINES, show_lines};

/// The largest image attached: providers refuse one whose base64 is over
/// 5 MB, which this many bytes make.
pub(crate) const MOST_BYTES: u64 = 5 * 1024 * 1024 / 4 * 3;

/// The most bytes of a text file attached; the model reads on with `read`.
const MOST_TEXT_BYTES: usize = 32 * 1024;

/// The file extensions looked at as images.
const EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// Whether `path` names an image by its extension.
pub fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            EXTENSIONS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// What a message's `@path` references came to.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Attached {
    /// The files read, each with its label: its path as the message named
    /// it, `~/` made the home directory.
    pub files: Vec<(String, UserContent)>,
    /// A line for the user about each file that is there but could not be
    /// attached, and why.
    pub notes: Vec<String>,
    /// The references that name a path with no file there, as the message
    /// spelled them.
    pub missing: Vec<String>,
}

/// The files `text` names with `@path`, each once, resolved from `base` in
/// `workspace` and read from it. Every `@` token that names a file there is
/// attached, whatever its shape (`@notes.md`, `@src/lib.rs`, `@Makefile`,
/// `@.env`); one with no file there is [missing](Attached::missing) when it
/// [looks like a path](is_path_reference), and otherwise stays text, such
/// as `@alice` or `@alice.smith`. A token's trailing punctuation is not
/// part of its path. A leading `~/` is the user's home on this machine; in
/// another workspace it is not attached, with a note.
///
/// Reads `workspace`: on a thread that must not wait, call it only for a
/// local one ([`Workspace::host`] is `None`).
pub fn attachments(text: &str, base: &Path, workspace: &dyn Workspace) -> Attached {
    let mut attached = Attached::default();
    let mut seen: Vec<PathBuf> = Vec::new();
    for token in text.split_whitespace() {
        let Some(reference) = token.strip_prefix('@') else {
            continue;
        };
        let trimmed = reference.trim_end_matches([',', '.', ';', ':', '!', '?', ')', '"', '\'']);
        if trimmed.starts_with("~/") && workspace.host().is_some() {
            attached.notes.push(format!(
                "{trimmed} is not attached: home paths are this machine's."
            ));
            continue;
        }
        let found = [reference, trimmed].into_iter().find_map(|named| {
            let label = expand(named);
            let path = base.join(&label);
            matches!(workspace.stat(&path), Ok(Stat::File { .. }))
                .then(|| (label.display().to_string(), path))
        });
        let Some((label, path)) = found else {
            if is_path_reference(trimmed) && !attached.missing.iter().any(|seen| seen == trimmed) {
                attached.missing.push(trimmed.to_owned());
            }
            continue;
        };
        if seen.contains(&path) {
            continue;
        }
        let read = if is_image_path(&path) {
            read_image(workspace, &path)
        } else {
            read_file(workspace, &path, &label)
        };
        match read {
            Ok(content) => attached.files.push((label, content)),
            Err(why) => attached
                .notes
                .push(format!("{label} is not attached: {why}.")),
        }
        seen.push(path);
    }
    attached
}

/// Whether `reference`, an `@` token without its `@`, looks like a path by
/// its shape alone: it holds a `/`, or ends in a dot and one to four
/// letters or digits. [`attachments`] reports such a token as missing when
/// no file is there; any other token (a mention, an address, or a file
/// with no extension) is attached only when its file is there, and is
/// otherwise left as text.
///
/// ```
/// use rig_tools::attach::is_path_reference;
///
/// assert!(is_path_reference("src/main.rs"));
/// assert!(is_path_reference("notes.md"));
/// assert!(!is_path_reference("reviewer"));
/// ```
pub fn is_path_reference(reference: &str) -> bool {
    reference.contains('/')
        || reference.rsplit_once('.').is_some_and(|(name, extension)| {
            !name.is_empty()
                && (1..=4).contains(&extension.len())
                && extension.chars().all(|c| c.is_ascii_alphanumeric())
        })
}

/// `path` with a leading `~/` made the home directory.
fn expand(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
    }
}

/// The image at `path` in `workspace` as message content, its type told
/// by its first bytes; why not, for a file that cannot be read, is not a
/// PNG, JPEG, GIF or WebP image, or is larger than a provider takes.
pub fn read_image(workspace: &dyn Workspace, path: &Path) -> Result<UserContent, String> {
    let image = image(workspace, path).map_err(|failure| failure.to_string())?;
    Ok(UserContent::image_base64(
        image.base64(),
        Some(image.media_type),
        None,
    ))
}

/// An image file, read whole.
pub(crate) struct Image {
    /// Its bytes.
    pub bytes: Vec<u8>,
    /// Its type, told by its first bytes.
    pub media_type: ImageMediaType,
}

impl Image {
    /// Its bytes in base64, as message content carries them.
    pub fn base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.bytes)
    }
}

/// Why [`image`] read no image.
#[derive(Debug)]
pub(crate) enum NoImage {
    /// The file could not be read.
    Read(io::Error),
    /// Its first bytes are not a PNG, JPEG, GIF or WebP image's.
    Other,
    /// It is an image of this many bytes, more than a provider takes.
    TooLarge(u64),
}

impl fmt::Display for NoImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(failure) => write!(formatter, "{failure}"),
            Self::Other => write!(formatter, "it is not a PNG, JPEG, GIF or WebP image"),
            Self::TooLarge(size) => write!(
                formatter,
                "it is an image of {} KB, more than the {} KB a provider takes",
                size.div_ceil(1024),
                MOST_BYTES / 1024
            ),
        }
    }
}

/// The image at `path` in `workspace`, whose first bytes tell its type.
pub(crate) fn image(workspace: &dyn Workspace, path: &Path) -> Result<Image, NoImage> {
    let contents = workspace.read(path, MOST_BYTES).map_err(NoImage::Read)?;
    let media_type = ImageMediaType::sniff(&contents.bytes).ok_or(NoImage::Other)?;
    if contents.size > MOST_BYTES {
        return Err(NoImage::TooLarge(contents.size));
    }
    Ok(Image {
        bytes: contents.bytes,
        media_type,
    })
}

/// The text file at `path` as message content: its lines the way `read`
/// shows them, at most [`MOST_TEXT_BYTES`], with a line saying how
/// many were left out, between a `<file path="label">` line and `</file>`.
fn read_file(workspace: &dyn Workspace, path: &Path, label: &str) -> Result<UserContent, String> {
    let text = workspace
        .read_text(path)
        .map_err(|failure| failure.to_string())?;
    if text.contains('\0') {
        return Err("it is not a text file".to_owned());
    }
    let lines = show_lines(&text, 1, MAX_LINES, MOST_TEXT_BYTES);
    Ok(UserContent::text(format!(
        "{FILE_OPEN}{label}\">\n{lines}{FILE_CLOSE}"
    )))
}

/// What the user typed in message `at` of `conversation`: the text items
/// of a message of the user's own
/// ([`EntryRef::is_users_own`]), never of
/// one an agent or a plugin delivered, such as a note, nor a file
/// attached, each trimmed, joined by a blank line; `None` when it has
/// none.
pub fn typed(conversation: &Conversation, at: usize) -> Option<String> {
    typed_in(conversation.entries().nth(at)?)
}

/// What the user typed in each message of `conversation` that has some
/// ([`typed`]), oldest first.
pub fn typed_prompts(conversation: &Conversation) -> impl Iterator<Item = String> + '_ {
    conversation.entries().filter_map(typed_in)
}

/// What the user typed in `entry` ([`typed`]).
fn typed_in(entry: EntryRef<'_>) -> Option<String> {
    let Message::User { content } = entry.message else {
        return None;
    };
    if !entry.is_users_own() {
        return None;
    }
    let typed: Vec<&str> = content
        .iter()
        .filter_map(|content| match content {
            UserContent::Text(text) if attached_file(&text.text).is_none() => {
                Some(text.text.trim())
            }
            _ => None,
        })
        .filter(|text| !text.is_empty())
        .collect();
    (!typed.is_empty()).then(|| typed.join("\n\n"))
}

/// What the user typed first in `conversation` ([`typed`]).
pub fn first_typed(conversation: &Conversation) -> Option<String> {
    (0..conversation.messages().len()).find_map(|at| typed(conversation, at))
}

#[cfg(test)]
mod tests;
