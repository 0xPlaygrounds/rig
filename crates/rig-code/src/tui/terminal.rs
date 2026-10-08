//! Terminal ownership: raw mode, the alternate screen, and keeping stray
//! output off the screen.

use std::fs::File;
use std::io;
use std::path::Path;

use bevy_app::AppExit;
use bevy_ecs::prelude::*;
use bevy_log::error;
use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::execute;
use crossterm::terminal::{
    Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use super::editor::Editor;
use super::input::TerminalInput;
use super::view::TuiView;
use crate::core::calls::Wake;
use rig::code_protocol::{Home, RELOAD_EXIT_CODE};

use crate::core::save::SessionPaths;

/// The terminal, drawn to through a private copy of stdout. Dropping it
/// restores the terminal, keeping the alternate screen for a reload so the
/// next build draws over the same screen.
#[derive(Resource)]
pub(crate) struct Tui {
    pub(super) terminal: Terminal<CrosstermBackend<File>>,
    /// Another copy of the screen, for a program the terminal is handed to.
    screen: File,
    keep_screen: bool,
}

impl Tui {
    /// Take over the terminal. On unix, stdout and stderr are then pointed
    /// at `log`, so output from plugins, dependencies or panics lands there
    /// instead of on the screen.
    fn open(log: Option<&Path>) -> io::Result<Self> {
        let screen = screen()?;
        let spare = screen.try_clone()?;
        if let Some(log) = log {
            redirect_output(log)?;
        }
        enable_raw_mode()?;
        let mut backend = CrosstermBackend::new(screen);
        // After a reload the alternate screen still shows the previous
        // build's frame.
        execute!(
            backend,
            EnterAlternateScreen,
            EnableBracketedPaste,
            Clear(ClearType::All)
        )?;
        let mut terminal = Terminal::new(backend)?;
        terminal.hide_cursor()?;
        Ok(Self {
            terminal,
            screen: spare,
            keep_screen: false,
        })
    }
}

impl Tui {
    /// Gives the terminal back for another program, such as `$EDITOR`, and
    /// returns a copy of the screen for that program's output: stdout and
    /// stderr point at the log.
    pub(crate) fn suspend(&mut self) -> io::Result<File> {
        execute!(
            self.terminal.backend_mut(),
            DisableBracketedPaste,
            LeaveAlternateScreen
        )?;
        self.terminal.show_cursor()?;
        disable_raw_mode()?;
        self.screen.try_clone()
    }

    /// Takes the terminal back after [`Tui::suspend`] and draws the next
    /// frame whole.
    pub(crate) fn resume(&mut self) -> io::Result<()> {
        enable_raw_mode()?;
        execute!(
            self.terminal.backend_mut(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            Clear(ClearType::All)
        )?;
        self.terminal.hide_cursor()?;
        self.terminal.clear()
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        execute!(self.terminal.backend_mut(), DisableBracketedPaste).ok();
        if !self.keep_screen {
            execute!(self.terminal.backend_mut(), LeaveAlternateScreen).ok();
        }
        self.terminal.show_cursor().ok();
        disable_raw_mode().ok();
    }
}

/// Opens the terminal and starts reading it at startup, or exits with code
/// 1 when there is none. The input editor gets the prompt history.
pub(crate) fn open_terminal(
    paths: Option<Res<SessionPaths>>,
    wake: Res<Wake>,
    mut view: ResMut<TuiView>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    let opened = Tui::open(paths.map(|paths| paths.log()).as_deref())
        .and_then(|tui| Ok((tui, TerminalInput::start(wake.clone())?)));
    match opened {
        Ok((tui, input)) => {
            view.editor = Editor::with_history(Home::from_env().history());
            commands.insert_resource(tui);
            commands.insert_resource(input);
        }
        Err(failure) => {
            error!("could not open the terminal: {failure}");
            exit.write(AppExit::from_code(1));
        }
    }
}

/// Keeps the alternate screen when the app exits to reload.
pub(crate) fn keep_screen_on_reload(mut exits: MessageReader<AppExit>, tui: Option<ResMut<Tui>>) {
    if let Some(mut tui) = tui
        && exits
            .read()
            .any(|exit| *exit == AppExit::from_code(RELOAD_EXIT_CODE))
    {
        tui.keep_screen = true;
    }
}

#[cfg(unix)]
fn screen() -> io::Result<File> {
    use std::os::fd::AsFd;
    Ok(File::from(io::stdout().as_fd().try_clone_to_owned()?))
}

#[cfg(windows)]
fn screen() -> io::Result<File> {
    use std::os::windows::io::AsHandle;
    Ok(File::from(io::stdout().as_handle().try_clone_to_owned()?))
}

#[cfg(unix)]
fn redirect_output(log: &Path) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)?;
    for target in [libc::STDOUT_FILENO, libc::STDERR_FILENO] {
        // SAFETY: both descriptors are open for the duration of the call;
        // `dup2` only replaces `target` with a copy of the log's descriptor.
        if unsafe { libc::dup2(file.as_raw_fd(), target) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Other platforms keep their standard streams; only the screen copy is
/// drawn to.
#[cfg(not(unix))]
fn redirect_output(_log: &Path) -> io::Result<()> {
    Ok(())
}
