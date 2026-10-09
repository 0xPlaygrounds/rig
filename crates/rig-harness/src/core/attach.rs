//! Images in user messages. A message names an image file as `@path`; when
//! the agent's model reads images, the file goes with the message as an
//! image, before its text, and the `@path` stays in the text so the model
//! knows which file it is. A model that does not read images gets the text
//! alone, and the user is told.

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::message::{ImageMediaType, UserContent};

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

/// The user message for `text`, with the images it names as `@path`
/// attached when `spec`'s model reads images, and a line for the user
/// about each image that was not attached.
pub fn user_message(text: &str, spec: Option<&ModelSpec>) -> (Message, Vec<String>) {
    let mut content = Vec::new();
    let mut notes = Vec::new();
    for path in image_references(text) {
        let shown = path.display();
        match spec {
            Some(spec) if !spec.input.image => notes.push(format!(
                "{} does not read images, so {shown} is sent as its path only.",
                spec.display_name
            )),
            None => notes.push(format!(
                "No model is connected, so {shown} is sent as its path only."
            )),
            Some(_) => match read_image(&path) {
                Ok(image) => content.push(image),
                Err(why) => notes.push(format!("{shown} is not attached: {why}.")),
            },
        }
    }
    content.push(UserContent::text(text));
    (Message::User { content }, notes)
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
