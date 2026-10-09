//! Files in the user's messages. A message names a file as `@path`; the
//! file is read when the message is sent and goes with it as an
//! [`Attachment`], before its text, and the `@path` stays in the text so
//! the model knows which file it is. An image goes as an image, which the
//! core sends only to a model that reads images, telling the user
//! otherwise. Any other file goes as its text, numbered and capped the way
//! the `read` tool shows it, so the model needs no `read` call for it.

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use rig_core::message::{ImageMediaType, UserContent};

use rig_ecs::inbox::Attachment;
use rig_tools::fs::read_text;
use rig_tools::{MAX_LINES, numbered};

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

/// The files `text` names as `@path`, read, and a line for the user about
/// each one that could not be.
pub fn attachments(text: &str) -> (Vec<Attachment>, Vec<String>) {
    let mut attachments = Vec::new();
    let mut notes = Vec::new();
    for path in file_references(text) {
        let label = path.display().to_string();
        let read = if is_image_path(&path) {
            read_image(&path)
        } else {
            read_file(&path, &label)
        };
        match read {
            Ok(content) => attachments.push(Attachment { label, content }),
            Err(why) => notes.push(format!("{label} is not attached: {why}.")),
        }
    }
    (attachments, notes)
}

/// The label and the number of lines of an attached text file, when
/// `text` is one, so a view can show it in a line instead of whole.
pub fn attached_file(text: &str) -> Option<(&str, usize)> {
    let rest = text.strip_prefix(FILE_OPEN)?.strip_suffix(FILE_CLOSE)?;
    let (label, body) = rest.split_once("\">\n")?;
    Some((label, body.lines().count()))
}

/// The files `text` names as `@path`, in order, each once. A token's
/// trailing punctuation is not part of its path.
fn file_references(text: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();
    for token in text.split_whitespace() {
        let Some(reference) = token.strip_prefix('@') else {
            continue;
        };
        let trimmed = reference.trim_end_matches([',', '.', ';', ':', '!', '?', ')', '"', '\'']);
        let found = [reference, trimmed]
            .into_iter()
            .map(expand)
            .find(|path| path.is_file());
        if let Some(path) = found
            && !paths.contains(&path)
        {
            paths.push(path);
        }
    }
    paths
}

/// `path` with a leading `~/` made the home directory.
fn expand(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
    }
}

/// The image at `path` as message content, its type told by its first
/// bytes.
fn read_image(path: &Path) -> Result<UserContent, String> {
    let size = fs::metadata(path)
        .map_err(|failure| failure.to_string())?
        .len();
    if size > MOST_BYTES {
        return Err(format!(
            "it has {} KB, more than the {} KB a provider takes",
            size / 1024,
            MOST_BYTES / 1024
        ));
    }
    let bytes = fs::read(path).map_err(|failure| failure.to_string())?;
    let media_type = ImageMediaType::sniff(&bytes)
        .ok_or("it is not a PNG, JPEG, GIF or WebP image".to_owned())?;
    let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(UserContent::image_base64(data, Some(media_type), None))
}

/// The text file at `path` as message content: its lines numbered the way
/// `read` shows them, at most [`MOST_TEXT_BYTES`], with a line saying how
/// many were left out, between a `<file path="label">` line and `</file>`.
fn read_file(path: &Path, label: &str) -> Result<UserContent, String> {
    let text = read_text(&path.to_string_lossy()).map_err(|failure| failure.to_string())?;
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
