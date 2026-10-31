//! The view's clipboard.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_tasks::{IoTaskPool, TaskPool};

use super::editor::{Editor, MainEditor};
use super::terminal::CopyToClipboard;
use rig_ecs::agent::Notice;
use rig_ecs::tasks::{Running, Wake};
use rig_harness::main_agent::ShownAgent;
use rig_harness::prelude::Processes;
use rig_tools::attach::Reference;
use rig_tools::process::Launcher;
use rig_tools::{blocking, clipboard};

/// The clipboard's image saved by a paste, or why there is none: what its
/// task ends as.
#[derive(Component)]
pub(crate) struct PastedImage(Result<PathBuf, String>);

impl From<Result<PathBuf, String>> for PastedImage {
    fn from(pasted: Result<PathBuf, String>) -> Self {
        Self(pasted)
    }
}

/// Reads the clipboard's image into `directory` off the main thread,
/// starting the reader as `launcher` says.
pub(crate) fn paste_image(
    directory: PathBuf,
    launcher: Launcher,
    wake: &Wake,
    commands: &mut Commands,
) {
    let read = blocking(move || Ok(read_image(&launcher, &directory)));
    let read = async move {
        let failed = |failure| Err(format!("could not read the clipboard: {failure}"));
        read.await.unwrap_or_else(failed)
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let running = Running::spawn_into::<PastedImage, _>(pool, wake, read);
    commands.spawn((Name::new("rig-tui: paste"), running));
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
/// off the main thread, as the terminal may ignore the escape code that
/// copies it.
pub(crate) fn copy_to_system(
    mut copies: MessageReader<CopyToClipboard>,
    processes: Option<Res<Processes>>,
) {
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    for CopyToClipboard(text) in copies.read() {
        let launcher = processes.as_deref().map(Processes::launcher);
        let launcher = launcher.unwrap_or_default();
        let text = text.clone();
        let copy = blocking(move || Ok(clipboard::write_text(&launcher, &text)));
        let copy = async move {
            match copy.await {
                Ok(Err(failure)) if failure.kind() == ErrorKind::Unsupported => {}
                Ok(Err(failure)) => warn!("could not copy to the clipboard: {failure}"),
                Err(failure) => warn!("could not copy to the clipboard: {failure}"),
                Ok(Ok(())) => {}
            }
        };
        pool.spawn(copy).detach();
    }
}

/// Types `@path` for an image a paste read from the clipboard, or says why
/// there is none.
pub(crate) fn type_pasted(
    pasted: On<Insert<PastedImage>>,
    images: Query<&PastedImage>,
    shown: Res<ShownAgent>,
    mut input: Query<&mut Editor, With<MainEditor>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    match images.get(pasted.entity).map(|image| &image.0) {
        Ok(Ok(path)) => {
            if let Ok(mut input) = input.single_mut() {
                type_path(&mut input, path);
            }
        }
        Ok(Err(why)) => {
            notices.write(Notice::info(shown.0, why.clone()));
        }
        Err(_) => {}
    }
    commands.entity(pasted.entity).despawn();
}

/// Types `path` as a [`Reference`] at the cursor, after a space when the
/// text before it does not end in whitespace, so it is a token of its own.
pub(crate) fn type_path(editor: &mut Editor, path: &Path) {
    let before = editor.text().get(..editor.cursor()).unwrap_or_default();
    let space = if before.is_empty() || before.ends_with(char::is_whitespace) {
        ""
    } else {
        " "
    };
    let reference = Reference::format(&path.to_string_lossy());
    editor.insert(&format!("{space}{reference} "));
}

/// The path a paste holds when it is one image file, as a terminal pastes
/// a dropped file: quoted, with escaped spaces, or as a `file://` URL.
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
    [text.to_owned(), text.replace("\\ ", " ")]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| rig_tools::attach::is_image_path(path) && path.is_file())
}
