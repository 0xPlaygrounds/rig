//! The terminal view: one plugin over the agents' components.
//!
//! It never owns the app's loop. Input is polled without blocking in
//! `PreUpdate`, the screen is drawn in `PostUpdate`, and both only read
//! components or write the core's messages. On unix the plugin keeps the
//! terminal for itself and points stdout and stderr at the session log, so
//! no other code can write over the screen.

use std::io::Write;

use bevy::prelude::*;
use crossterm::{
    execute,
    terminal::{
        Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
        enable_raw_mode,
    },
};
use ratatui::{Terminal, backend::CrosstermBackend};

mod draw;
mod input;
pub mod view;

/// Where the terminal UI writes.
type Output = Box<dyn Write + Send + Sync>;

/// The terminal, restored when the app drops it, which also happens when
/// the main thread unwinds from a panic.
#[derive(Resource)]
pub struct Tui {
    terminal: Terminal<CrosstermBackend<Output>>,
    /// Stay on the alternate screen, so the binary started by a reload
    /// draws over the same screen.
    keep_screen: bool,
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        if !self.keep_screen {
            let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        }
        let _ = self.terminal.show_cursor();
    }
}

/// The terminal UI.
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        let tui = match open() {
            Ok(tui) => tui,
            Err(error) => {
                // Without a terminal there is no view to report through.
                eprintln!("rig-code: cannot open the terminal: {error}");
                app.world_mut().write_message(AppExit::error());
                return;
            }
        };
        // Panics are logged rather than printed: a panic in a plugin is
        // caught and must not write over the screen.
        std::panic::set_hook(Box::new(|info| error!("{info}")));
        app.insert_resource(tui)
            .init_resource::<view::Focus>()
            .init_resource::<view::Composer>()
            .init_resource::<view::Scroll>()
            .init_resource::<view::Picker>()
            .init_resource::<view::NoticeLog>()
            .add_systems(PreUpdate, (view::keep_focus, input::read_terminal).chain())
            .add_systems(PostUpdate, (view::read_core_messages, draw::draw).chain())
            .add_systems(
                Last,
                keep_screen_on_reload
                    .in_set(bevy::app::OnAppExitSystems)
                    .run_if(on_message::<AppExit>),
            );
    }
}

/// On a reload exit, leave the alternate screen to the next binary.
fn keep_screen_on_reload(mut exits: MessageReader<AppExit>, mut tui: ResMut<Tui>) {
    let reload = AppExit::from_code(crate::RELOAD_EXIT_CODE);
    if exits.read().any(|exit| *exit == reload) {
        tui.keep_screen = true;
    }
}

/// Take the terminal: keep a handle to it, send stdout and stderr to the
/// log, enter raw mode and the alternate screen.
fn open() -> std::io::Result<Tui> {
    let mut output = take_terminal()?;
    enable_raw_mode()?;
    // `Terminal::clear` would query the cursor through stdout, which now
    // goes to the log, so the screen is cleared directly.
    execute!(output, EnterAlternateScreen, Clear(ClearType::All))?;
    Ok(Tui {
        terminal: Terminal::new(CrosstermBackend::new(output))?,
        keep_screen: false,
    })
}

/// A handle to the terminal, with fds 1 and 2 redirected to the session log
/// so stray prints from plugins, panics and child processes land there.
#[cfg(unix)]
fn take_terminal() -> std::io::Result<Output> {
    use std::os::fd::{AsRawFd, FromRawFd};

    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(crate::ecs::paths::log_file())?;
    // SAFETY: `dup` returns a fresh descriptor, owned from here on by the
    // `File`; `dup2` only takes descriptor numbers.
    let terminal = unsafe { libc::dup(libc::STDOUT_FILENO) };
    if terminal < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let terminal = unsafe { std::fs::File::from_raw_fd(terminal) };
    for stdio in [libc::STDOUT_FILENO, libc::STDERR_FILENO] {
        if unsafe { libc::dup2(log.as_raw_fd(), stdio) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(Box::new(terminal))
}

/// The terminal is stdout; other platforms do not redirect it.
#[cfg(not(unix))]
fn take_terminal() -> std::io::Result<Output> {
    Ok(Box::new(std::io::stdout()))
}
