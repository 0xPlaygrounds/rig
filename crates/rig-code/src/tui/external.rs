//! Ctrl+G edits the input in `$VISUAL` or `$EDITOR`, as pi and codex do
//! (`references/pi/packages/coding-agent/src/core/keybindings.ts:127`,
//! `references/codex/codex-rs/tui/src/external_editor.rs:39-56`). The
//! editor runs on a thread of its own with the terminal handed over, and
//! the input thread paused so it does not take the editor's keys. The app
//! keeps running meanwhile (a turn goes on), it just does not draw.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use bevy_ecs::prelude::*;
use crossbeam_channel::Receiver;

use super::input::TerminalInput;
use super::terminal::Tui;
use super::view::TuiView;
use crate::core::agent::Notice;
use crate::core::calls::Wake;
use crate::core::save::SessionPaths;

/// Ctrl+G was pressed: open the editor this frame.
#[derive(Resource)]
pub(crate) struct EditRequested;

/// The editor is open on `file`; `done` gets its exit status.
#[derive(Resource)]
pub(crate) struct ExternalEdit {
    done: Receiver<io::Result<ExitStatus>>,
    file: PathBuf,
}

/// The editor to run: `$VISUAL`, else `$EDITOR`.
fn editor() -> Option<String> {
    ["VISUAL", "EDITOR"]
        .into_iter()
        .filter_map(|name| std::env::var(name).ok())
        .find(|value| !value.trim().is_empty())
}

/// The editor command for `file`. The variable may hold arguments, so the
/// shell splits it, with the path passed as `$1`.
#[cfg(unix)]
fn command(editor: &str, file: &Path) -> Command {
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(format!("{editor} \"$1\""))
        .arg("sh")
        .arg(file);
    command
}

#[cfg(not(unix))]
fn command(editor: &str, file: &Path) -> Command {
    let mut command = Command::new("cmd");
    command.arg("/C").arg(editor).arg(file);
    command
}

/// Writes the input to the draft file, hands the terminal to the editor
/// and waits for it on a thread.
pub(crate) fn start_external_edit(
    mut tui: ResMut<Tui>,
    input: Res<TerminalInput>,
    view: Res<TuiView>,
    paths: Option<Res<SessionPaths>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    commands.remove_resource::<EditRequested>();
    let Some(editor) = editor() else {
        notices.write(Notice::error(
            view.agent,
            "Set VISUAL or EDITOR to edit the input in an editor.",
        ));
        return;
    };
    let file = paths.map_or_else(
        || std::env::temp_dir().join(format!("rig-code-draft-{}.md", std::process::id())),
        |paths| paths.draft(),
    );
    if let Err(failure) = std::fs::write(&file, view.editor.text()) {
        notices.write(Notice::error(
            view.agent,
            format!("Could not write {}: {failure}", file.display()),
        ));
        return;
    }
    input.pause();
    let started = tui
        .suspend()
        .and_then(|screen| spawn(&editor, &file, screen, wake.clone()));
    match started {
        Ok(done) => commands.insert_resource(ExternalEdit { done, file }),
        Err(failure) => {
            input.resume();
            if let Err(again) = tui.resume() {
                bevy_log::error!("could not take the terminal back: {again}");
            }
            notices.write(Notice::error(
                view.agent,
                format!("Could not start {editor}: {failure}"),
            ));
        }
    }
}

/// Runs the editor on a thread, drawing to `screen`, and sends its status.
fn spawn(
    editor: &str,
    file: &Path,
    screen: File,
    wake: Wake,
) -> io::Result<Receiver<io::Result<ExitStatus>>> {
    let mut command = command(editor, file);
    command.stdout(screen.try_clone()?).stderr(screen);
    let (sender, done) = crossbeam_channel::bounded(1);
    std::thread::Builder::new()
        .name("rig-code-editor".to_owned())
        .spawn(move || {
            sender.send(command.status()).ok();
            wake.wake();
        })?;
    Ok(done)
}

/// Takes the terminal back when the editor exits, and the edited text into
/// the input when it exited cleanly.
pub(crate) fn finish_external_edit(
    edit: Res<ExternalEdit>,
    mut tui: ResMut<Tui>,
    input: Res<TerminalInput>,
    mut view: ResMut<TuiView>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(status) = edit.done.try_recv() else {
        return;
    };
    commands.remove_resource::<ExternalEdit>();
    input.resume();
    if let Err(failure) = tui.resume() {
        bevy_log::error!("could not take the terminal back: {failure}");
    }
    match status {
        Ok(status) if status.success() => match std::fs::read_to_string(&edit.file) {
            Ok(text) => view
                .editor
                .set(text.trim_end_matches(['\n', '\r']).to_owned()),
            Err(failure) => {
                notices.write(Notice::error(
                    view.agent,
                    format!("Could not read {}: {failure}", edit.file.display()),
                ));
            }
        },
        Ok(status) => {
            notices.write(Notice::error(
                view.agent,
                format!("The editor exited with {status}; the input is unchanged."),
            ));
        }
        Err(failure) => {
            notices.write(Notice::error(
                view.agent,
                format!("Could not run the editor: {failure}"),
            ));
        }
    }
    std::fs::remove_file(&edit.file).ok();
    // The whole screen is drawn again.
    view.set_changed();
}
