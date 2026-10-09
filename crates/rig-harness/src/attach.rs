//! Images in the user's messages. A message names an image file as
//! `@path`; the file is read when the message is sent and goes with it as
//! an [`Attachment`], before its text, and the `@path` stays in the text so
//! the model knows which file it is. The core sends an image only to a
//! model that reads images, and tells the user otherwise.

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use rig_core::message::{ImageMediaType, UserContent};

use crate::core::inbox::Attachment;

/// The largest image attached: providers refuse bigger ones (Anthropic
/// takes 5 MB per image).
const MOST_BYTES: u64 = 5 * 1024 * 1024;

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

/// The images `text` names as `@path`, read, and a line for the user about
/// each one that could not be.
pub fn attachments(text: &str) -> (Vec<Attachment>, Vec<String>) {
    let mut attachments = Vec::new();
    let mut notes = Vec::new();
    for path in image_references(text) {
        let label = path.display().to_string();
        match read_image(&path) {
            Ok(content) => attachments.push(Attachment { label, content }),
            Err(why) => notes.push(format!("{label} is not attached: {why}.")),
        }
    }
    (attachments, notes)
}

/// The image files `text` names as `@path`, in order, each once. A token's
/// trailing punctuation is not part of its path.
fn image_references(text: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();
    for token in text.split_whitespace() {
        let Some(reference) = token.strip_prefix('@') else {
            continue;
        };
        let trimmed = reference.trim_end_matches([',', '.', ';', ':', '!', '?', ')', '"', '\'']);
        let found = [reference, trimmed]
            .into_iter()
            .map(expand)
            .find(|path| is_image_path(path) && path.is_file());
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
