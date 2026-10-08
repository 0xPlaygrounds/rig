//! The terminal view. It never owns the main loop: a thread reads terminal
//! events into a channel that a `PreUpdate` system drains, and a
//! `PostUpdate` system draws. Input becomes the same entity events any view
//! would trigger.

mod input;
mod render;
pub mod view;

use std::io::{Stdout, stdout};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

/// The terminal view: raw mode on the alternate screen, the input thread,
/// the [`view::View`] state and drawing.
#[derive(Default)]
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        let tui = match Tui::open() {
            Ok(tui) => tui,
            Err(error) => {
                error!("cannot open the terminal: {error}");
                app.add_systems(Startup, |mut exit: MessageWriter<AppExit>| {
                    exit.write(AppExit::error());
                });
                return;
            }
        };
        app.insert_resource(tui)
            .insert_resource(input::Input::spawn())
            .init_resource::<view::View>()
            .add_observer(view::on_notice)
            .add_observer(view::on_open_picker)
            .add_systems(PreUpdate, input::read_input)
            .add_systems(PostUpdate, render::render);
    }
}

/// The terminal, in raw mode on the alternate screen until dropped. The app
/// drops it on exit and while unwinding from a panic, which restores the
/// terminal.
#[derive(Resource)]
pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl Tui {
    fn open() -> std::io::Result<Self> {
        let terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
        enable_raw_mode()?;
        let mut tui = Self { terminal };
        execute!(
            tui.terminal.backend_mut(),
            EnterAlternateScreen,
            EnableBracketedPaste
        )?;
        tui.terminal.clear()?;
        Ok(tui)
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
        let _ = self.terminal.show_cursor();
    }
}
