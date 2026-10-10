//! Images into the input: Ctrl+V saves the clipboard's image in the
//! session's [`SessionDir::images`](rig::harness_protocol::SessionDir::images)
//! and types `@path` for it, and a pasted path to an image file (a file
//! dropped on the terminal) becomes `@path` too. The message attaches what
//! `@path` names (see [`front`](crate::front)). The clipboard is
//! read by `wl-paste`, `xclip` or `pngpaste` on a thread of its own, which
//! wakes the loop when it is done.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};

use super::editor::Editor;
use super::view::TuiView;
use crate::front;
use rig_core::message::ImageMediaType;
use rig_ecs::agent::Notice;
use rig_ecs::calls::Wake;

/// Clipboard reads in flight; each sends the saved image's path, or why
/// there is none.
#[derive(Resource)]
pub(crate) struct Clipboard {
    sender: Sender<Result<PathBuf, String>>,
    results: Receiver<Result<PathBuf, String>>,
}

impl Default for Clipboard {
    fn default() -> Self {
        let (sender, results) = crossbeam_channel::unbounded();
        Self { sender, results }
    }
}

impl Clipboard {
    /// Reads the clipboard's image into `directory` on a thread.
    pub(crate) fn paste_image(&self, directory: PathBuf, wake: Wake) {
        let sender = self.sender.clone();
        let spawned = std::thread::Builder::new()
            .name("rig-harness-clipboard".to_owned())
            .spawn(move || {
                sender.send(save_clipboard_image(&directory)).ok();
                wake.wake();
            });
        if let Err(failure) = spawned {
            self.sender
                .send(Err(format!("could not read the clipboard: {failure}")))
                .ok();
        }
    }
}

/// Types `@path` for each image read from the clipboard.
pub(crate) fn receive_images(
    clipboard: Res<Clipboard>,
    mut view: ResMut<TuiView>,
    mut notices: MessageWriter<Notice>,
) {
    for result in clipboard.results.try_iter() {
        match result {
            Ok(path) => type_path(&mut view.editor, &path),
            Err(why) => {
                notices.write(Notice::info(view.agent, why));
            }
        }
    }
}

/// Types `@path` at the cursor, after a space when the text before it does
/// not end in whitespace, so `@path` is a token of its own.
pub(crate) fn type_path(editor: &mut Editor, path: &Path) {
    let before = editor.text().get(..editor.cursor()).unwrap_or_default();
    let space = if before.is_empty() || before.ends_with(char::is_whitespace) {
        ""
    } else {
        " "
    };
    editor.insert(&format!("{space}@{} ", path.display()));
}

/// The path a paste holds when it is one image file, as a terminal pastes
/// a dropped file: quoted, with escaped spaces, or as a `file://` URL. A
/// path with whitespace in it is not one, since `@path` ends at a space.
pub(crate) fn dropped_image(text: &str) -> Option<PathBuf> {
    let text = text.trim();
    let text = text
        .strip_prefix('\'')
        .and_then(|text| text.strip_suffix('\''))
        .or_else(|| {
            text.strip_prefix('"')
                .and_then(|text| text.strip_suffix('"'))
        })
        .unwrap_or(text);
    let text = text.strip_prefix("file://").unwrap_or(text);
    if text.contains(char::is_whitespace) {
        return None;
    }
    let path = Path::new(text);
    (front::is_image_path(path) && path.is_file()).then(|| path.to_path_buf())
}

/// The commands that print the clipboard's image, in the order tried.
fn readers() -> Vec<(&'static str, Vec<&'static str>)> {
    let mut readers = Vec::new();
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        readers.push(("wl-paste", vec!["--no-newline", "--type", "image/png"]));
    }
    if std::env::var_os("DISPLAY").is_some() {
        readers.push((
            "xclip",
            vec!["-selection", "clipboard", "-target", "image/png", "-out"],
        ));
    }
    if cfg!(target_os = "macos") {
        readers.push(("pngpaste", vec!["-"]));
    }
    readers
}

/// Saves the clipboard's image in `directory`, named by the time.
fn save_clipboard_image(directory: &Path) -> Result<PathBuf, String> {
    let readers = readers();
    if readers.is_empty() {
        return Err(
            "No clipboard to read an image from: Ctrl+V needs wl-paste, xclip or \
                    pngpaste."
                .to_owned(),
        );
    }
    let image = readers.iter().find_map(|(program, args)| {
        let output = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        let extension = ImageMediaType::sniff(&output.stdout)?.extension();
        output
            .status
            .success()
            .then_some((output.stdout, extension))
    });
    let Some((bytes, extension)) = image else {
        let names: Vec<&str> = readers.iter().map(|(program, _)| *program).collect();
        return Err(format!(
            "No image in the clipboard (read with {}).",
            names.join(", ")
        ));
    };
    std::fs::create_dir_all(directory)
        .map_err(|failure| format!("Could not save the pasted image: {failure}."))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or_default();
    let path = directory.join(format!("pasted-{stamp}.{extension}"));
    std::fs::write(&path, bytes)
        .map_err(|failure| format!("Could not save the pasted image: {failure}."))?;
    Ok(path)
}
