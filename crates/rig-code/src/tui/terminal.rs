//! Terminal ownership: raw mode, the alternate screen, and keeping stray
//! output off the screen.

use std::fs::File;
use std::io;
use std::path::Path;

use bevy_app::AppExit;
use bevy_ecs::prelude::*;
use bevy_log::error;
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::core::session::SessionPaths;

/// The terminal, drawn to through a private copy of stdout. Dropping it
/// restores the terminal.
#[derive(Resource)]
pub struct Tui {
    pub(super) terminal: Terminal<CrosstermBackend<File>>,
}

impl Tui {
    /// Take over the terminal. On unix, stdout and stderr are then pointed
    /// at `log`, so output from plugins, dependencies or panics lands there
    /// instead of on the screen.
    fn open(log: Option<&Path>) -> io::Result<Self> {
        let screen = screen()?;
        if let Some(log) = log {
            redirect_output(log)?;
        }
        enable_raw_mode()?;
        let mut backend = CrosstermBackend::new(screen);
        execute!(backend, EnterAlternateScreen)?;
        let mut terminal = Terminal::new(backend)?;
        terminal.hide_cursor()?;
        Ok(Self { terminal })
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen).ok();
        self.terminal.show_cursor().ok();
        disable_raw_mode().ok();
    }
}

/// Opens the terminal at startup, or exits with code 1 when there is none.
pub fn open_terminal(
    paths: Option<Res<SessionPaths>>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    match Tui::open(paths.map(|paths| paths.log()).as_deref()) {
        Ok(tui) => commands.insert_resource(tui),
        Err(failure) => {
            error!("could not open the terminal: {failure}");
            exit.write(AppExit::from_code(1));
        }
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
