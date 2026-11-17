//! The view's clipboard.

use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_tasks::{IoTaskPool, TaskPool};

use super::editor::{Editor, MainEditor};
use super::terminal::CopyToClipboard;
use super::view::TuiView;
use rig_ecs::agent::Notice;
use rig_ecs::requests::RequestOf;
use rig_ecs::tasks::{Running, Wake};
use rig_harness::main_agent::ShownAgent;
use rig_harness::prelude::Processes;
use rig_tools::attach::Reference;
use rig_tools::process::Launcher;
use rig_tools::{blocking, clipboard};

/// What a Ctrl+V found in the clipboard.
pub(crate) enum Paste {
    /// Its image, saved as this file.
    Image(PathBuf),
    /// Its text.
    Text(String),
}

/// What a Ctrl+V found in the clipboard, or why it found nothing: what its
/// task ends as.
#[derive(Component)]
pub(crate) struct Pasted(Result<Paste, String>);

impl From<Result<Paste, String>> for Pasted {
    fn from(pasted: Result<Paste, String>) -> Self {
        Self(pasted)
    }
}

/// Reads the clipboard off the main thread, its image saved into
/// `directory`, else its text, starting each reader as `launcher` says,
/// for the editor `editor`: a [`RequestOf`] it, which goes with it.
pub(crate) fn paste(
    directory: PathBuf,
    launcher: Launcher,
    (editor, wake): (Entity, &Wake),
    commands: &mut Commands,
) {
    let read = blocking(move || Ok(read(&launcher, &directory)));
    let read = async move {
        let failed = |failure| Err(format!("Could not read the clipboard: {failure}."));
        read.await.unwrap_or_else(failed)
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let running = Running::spawn_into::<Pasted, _>(pool, wake, read);
    commands.spawn((Name::new("rig-tui: paste"), RequestOf(editor), running));
}

/// The clipboard's image saved in `directory`, else its text, or why there
/// is neither.
fn read(launcher: &Launcher, directory: &Path) -> Result<Paste, String> {
    read_either(
        || clipboard::read_image(launcher, directory),
        || clipboard::read_text(launcher),
    )
}

/// What `image` found, else what `text` found, or why neither found any.
/// An image reader that is not installed, as `pngpaste` often is on macOS
/// beside `pbpaste`, still lets the text be read; it is named when there is
/// no text, as the clipboard may hold an image it would read.
fn read_either(
    image: impl FnOnce() -> io::Result<Option<PathBuf>>,
    text: impl FnOnce() -> io::Result<Option<String>>,
) -> Result<Paste, String> {
    let failed = |failure: io::Error| match failure.kind() {
        ErrorKind::Unsupported => "No clipboard is reachable from this machine (for example \
                                   over SSH): Ctrl+V reads it with wl-paste, xclip or pngpaste."
            .to_owned(),
        ErrorKind::TimedOut => format!("Gave up reading the clipboard: {failure}."),
        _ => format!("Could not read the clipboard: {failure}."),
    };
    let missing = match image() {
        Ok(Some(path)) => return Ok(Paste::Image(path)),
        Ok(None) => None,
        Err(failure) if failure.kind() == ErrorKind::NotFound => Some(failure),
        Err(failure) => return Err(failed(failure)),
    };
    match (text().map_err(failed)?, missing) {
        (Some(text), _) if !text.is_empty() => Ok(Paste::Text(text)),
        (_, Some(missing)) => Err(format!(
            "The clipboard holds no text, and its image cannot be read: {missing}."
        )),
        _ => Err("The clipboard holds no image or text.".to_owned()),
    }
}

/// A copy that reached no clipboard program, as none is installed: what
/// its task ends as, with why.
#[derive(Component)]
pub(crate) struct CopyMissed(Option<io::Error>);

impl From<Option<io::Error>> for CopyMissed {
    fn from(missed: Option<io::Error>) -> Self {
        Self(missed)
    }
}

/// Puts each text copied to the clipboard in this machine's clipboard too,
/// off the main thread, as the terminal may ignore the escape code that
/// copies it.
pub(crate) fn copy_to_system(
    mut copies: MessageReader<CopyToClipboard>,
    processes: Option<Res<Processes>>,
    wake: Option<Res<Wake>>,
    mut commands: Commands,
) {
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let wake = wake.as_deref().cloned().unwrap_or_default();
    for CopyToClipboard(text) in copies.read() {
        let launcher = processes.as_deref().map(Processes::launcher);
        let launcher = launcher.unwrap_or_default();
        let text = text.clone();
        let copy = blocking(move || Ok(clipboard::write_text(&launcher, &text)));
        let copy = async move {
            match copy.await {
                Ok(Err(failure)) if failure.kind() == ErrorKind::NotFound => Some(failure),
                Ok(Err(failure)) if failure.kind() == ErrorKind::Unsupported => None,
                Ok(Err(failure)) => {
                    warn!("could not copy to the clipboard: {failure}");
                    None
                }
                Err(failure) => {
                    warn!("could not copy to the clipboard: {failure}");
                    None
                }
                Ok(Ok(())) => None,
            }
        };
        let running = Running::spawn_into::<CopyMissed, _>(pool, &wake, copy);
        commands.spawn((Name::new("rig-tui: copy"), running));
    }
}

/// Says once a session that a copy reached only the terminal, naming the
/// clipboard program that is not installed.
pub(crate) fn warn_copy_missed(
    missed: On<Insert<CopyMissed>>,
    copies: Query<&CopyMissed>,
    shown: Res<ShownAgent>,
    mut warned: Local<bool>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if let Ok(CopyMissed(Some(why))) = copies.get(missed.entity)
        && !*warned
    {
        *warned = true;
        let why = format!("Copied with OSC 52 only, which some terminals ignore: {why}.");
        notices.write(Notice::warn(shown.0, why));
    }
    commands.entity(missed.entity).despawn();
}

/// Puts what a Ctrl+V found in the editor that asked for it, or says why
/// there is nothing.
pub(crate) fn type_pasted(
    pasted: On<Insert<Pasted>>,
    found: Query<(&Pasted, &RequestOf)>,
    (shown, mut view): (Res<ShownAgent>, ResMut<TuiView>),
    mut editors: Query<(&mut Editor, Has<MainEditor>)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    match found.get(pasted.entity) {
        Ok((Pasted(Ok(paste)), &RequestOf(editor))) => {
            if let Ok((mut input, main)) = editors.get_mut(editor) {
                match paste {
                    Paste::Image(path) => type_path(&mut input, path),
                    // As a bracketed paste, it opens no completion list.
                    Paste::Text(text) => {
                        paste_text(&mut input, text);
                        view.pasted |= main && !text.is_empty();
                    }
                }
            }
        }
        Ok((Pasted(Err(why)), _)) => {
            notices.write(Notice::info(shown.0, why.clone()));
        }
        Err(_) => {}
    }
    commands.entity(pasted.entity).despawn();
}

/// Inserts pasted `text` at the cursor; a dropped image file becomes
/// `@path`, which attaches it.
pub(crate) fn paste_text(editor: &mut Editor, text: &str) {
    match dropped_image(text) {
        Some(path) => type_path(editor, &path),
        None => editor.insert(text),
    }
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
/// a dropped file: quoted (taken as it is), shell-escaped, such as
/// `shot\ \(1\).png` (unescaped, except on Windows, where `\` separates
/// directories), or as a `file://` URL.
pub(crate) fn dropped_image(text: &str) -> Option<PathBuf> {
    let text = text.trim();
    let quoted = text
        .strip_prefix('\'')
        .and_then(|text| text.strip_suffix('\''))
        .or_else(|| {
            text.strip_prefix('"')
                .and_then(|text| text.strip_suffix('"'))
        });
    let path = match quoted {
        Some(quoted) => quoted.to_owned(),
        None if cfg!(windows) => text.to_owned(),
        None => unescape(text),
    };
    let path = PathBuf::from(path.strip_prefix("file://").unwrap_or(&path));
    (rig_tools::attach::is_image_path(&path) && path.is_file()).then_some(path)
}

/// `text` with each `\` that escapes the next character, as a shell
/// writes a path, taken out.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(character) = chars.next() {
        match character {
            '\\' => out.extend(chars.next()),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests;
