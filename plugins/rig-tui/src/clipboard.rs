//! The view's clipboard: Ctrl+V saves the clipboard's image in the
//! session's [`Scratch::images`](rig_harness::prelude::Scratch::images)
//! directory and types `@path` for it, or says there is none, and a pasted
//! path to an image file (a file dropped on the terminal) becomes `@path`
//! too. The message attaches what `@path` names (see
//! [`attach`](rig_tools::attach)). A [`CopyToClipboard`] goes to the
//! terminal (OSC 52) and to this machine's clipboard as well, for a
//! terminal that ignores OSC 52. The clipboard is read and written through
//! [`rig_tools::clipboard`], with programs started as the app's
//! [`Processes`](rig_harness::prelude::Processes) start every program, on a
//! thread of its own; a plugin uses that module the same way.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use bevy_ecs::prelude::*;
use bevy_log::warn;
use crossbeam_channel::{Receiver, Sender};

use super::editor::{Editor, MainEditor};
use super::terminal::CopyToClipboard;
use rig_ecs::agent::Notice;
use rig_ecs::tasks::Wake;
use rig_harness::main_agent::ShownAgent;
use rig_harness::prelude::Processes;
use rig_tools::clipboard;
use rig_tools::process::Launcher;

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
    /// Reads the clipboard's image into `directory` on a thread, starting
    /// the reader as `launcher` says.
    pub(crate) fn paste_image(&self, directory: PathBuf, launcher: Launcher, wake: Wake) {
        let sender = self.sender.clone();
        let spawned = std::thread::Builder::new()
            .name("rig-harness-clipboard".to_owned())
            .spawn(move || {
                sender.send(read_image(&launcher, &directory)).ok();
                wake.wake();
            });
        if let Err(failure) = spawned {
            self.sender
                .send(Err(format!("could not read the clipboard: {failure}")))
                .ok();
        }
    }
}

/// Saves the clipboard's image in `directory`, or says why there is none.
fn read_image(launcher: &Launcher, directory: &Path) -> Result<PathBuf, String> {
    match clipboard::read_image(launcher, directory) {
        Ok(Some(path)) => Ok(path),
        Ok(None) => Err("No image in the clipboard.".to_owned()),
        Err(failure) if failure.kind() == ErrorKind::NotFound => {
            Err("No image in the clipboard.".to_owned())
        }
        Err(failure) if failure.kind() == ErrorKind::Unsupported => Err(
            "No clipboard to read an image from: Ctrl+V needs wl-paste, xclip or pngpaste."
                .to_owned(),
        ),
        Err(failure) => Err(format!("Could not paste the clipboard's image: {failure}.")),
    }
}

/// Puts each text copied to the clipboard in this machine's clipboard too,
/// on a thread, as the terminal may ignore the escape code that copies it.
pub(crate) fn copy_to_system(
    mut copies: MessageReader<CopyToClipboard>,
    processes: Option<Res<Processes>>,
) {
    for CopyToClipboard(text) in copies.read() {
        let launcher = processes.as_deref().map(Processes::launcher);
        let launcher = launcher.unwrap_or_default();
        let text = text.clone();
        let spawned = std::thread::Builder::new()
            .name("rig-harness-copy".to_owned())
            .spawn(move || match clipboard::write_text(&launcher, &text) {
                Err(failure) if failure.kind() != ErrorKind::Unsupported => {
                    warn!("could not copy to the clipboard: {failure}");
                }
                _ => {}
            });
        if let Err(failure) = spawned {
            warn!("could not copy to the clipboard: {failure}");
        }
    }
}

/// Types `@path` for each image read from the clipboard.
pub(crate) fn receive_images(
    clipboard: Res<Clipboard>,
    shown: Res<ShownAgent>,
    mut input: Query<&mut Editor, With<MainEditor>>,
    mut notices: MessageWriter<Notice>,
) {
    for result in clipboard.results.try_iter() {
        match result {
            Ok(path) => {
                if let Ok(mut input) = input.single_mut() {
                    type_path(&mut input, &path);
                }
            }
            Err(why) => {
                notices.write(Notice::info(shown.0, why));
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
    (rig_tools::attach::is_image_path(path) && path.is_file()).then(|| path.to_path_buf())
}
