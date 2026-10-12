//! Files a message names as `@path`, read to go with it: an image as an
//! image, any other file as its text, numbered and capped the way `read`
//! shows it, so the model needs no `read` call for it.

use std::path::{Path, PathBuf};

use base64::Engine;
use rig_core::message::{ImageMediaType, UserContent};

use crate::workspace::{Stat, Workspace};
use crate::{MAX_LINES, numbered};

/// The largest image attached: providers refuse bigger ones (Anthropic
/// takes 5 MB per image).
const MOST_BYTES: u64 = 5 * 1024 * 1024;

/// The most bytes of a text file attached; the model reads on with `read`.
const MOST_TEXT_BYTES: usize = 32 * 1024;

/// How an attached text file starts, followed by its label and `">`.
const FILE_OPEN: &str = "<file path=\"";

/// How an attached text file ends.
const FILE_CLOSE: &str = "</file>";

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
/// `workspace` and read from it. A token is a path reference when it holds a
/// `/` or ends in a dot and one to four letters or digits (`@notes.md`,
/// `@src/lib.rs`); one with no file there is [missing](Attached::missing).
/// Any other `@word`, such as `@alice` or `@alice.smith`, stays text. A
/// token's trailing punctuation is not part of its path. A leading `~/` is
/// the user's home on this machine; in another workspace it is not
/// attached, with a note.
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

/// Whether `reference`, an `@` token without its `@`, names a path: it
/// holds a `/`, or ends in a dot and one to four letters or digits. Any
/// other `@` token (a mention, an address) is left as text.
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

/// The label and the number of lines of an attached text file, when
/// `text` is one, so a view can show it in a line instead of whole.
pub fn attached_file(text: &str) -> Option<(&str, usize)> {
    let rest = text.strip_prefix(FILE_OPEN)?.strip_suffix(FILE_CLOSE)?;
    let (label, body) = rest.split_once("\">\n")?;
    Some((label, body.lines().count()))
}

/// `path` with a leading `~/` made the home directory.
fn expand(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
    }
}

/// The image at `path` in `workspace` as message content, its type told
/// by its first bytes; why not, for a file that cannot be read, is larger
/// than a provider takes, or is not a PNG, JPEG, GIF or WebP image.
pub fn read_image(workspace: &dyn Workspace, path: &Path) -> Result<UserContent, String> {
    let contents = workspace
        .read(path, MOST_BYTES)
        .map_err(|failure| failure.to_string())?;
    if contents.size > MOST_BYTES {
        return Err(format!(
            "it has {} KB, more than the {} KB a provider takes",
            contents.size / 1024,
            MOST_BYTES / 1024
        ));
    }
    let bytes = contents.bytes;
    let media_type = ImageMediaType::sniff(&bytes)
        .ok_or("it is not a PNG, JPEG, GIF or WebP image".to_owned())?;
    let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(UserContent::image_base64(data, Some(media_type), None))
}

/// The text file at `path` as message content: its lines numbered the way
/// `read` shows them, at most [`MOST_TEXT_BYTES`], with a line saying how
/// many were left out, between a `<file path="label">` line and `</file>`.
fn read_file(workspace: &dyn Workspace, path: &Path, label: &str) -> Result<UserContent, String> {
    let text = workspace
        .read_text(path)
        .map_err(|failure| failure.to_string())?;
    if text.contains('\0') {
        return Err("it is not a text file".to_owned());
    }
    let lines = numbered(&text, 1, MAX_LINES, MOST_TEXT_BYTES);
    Ok(UserContent::text(format!(
        "{FILE_OPEN}{label}\">\n{lines}{FILE_CLOSE}"
    )))
}

#[cfg(test)]
mod tests;
