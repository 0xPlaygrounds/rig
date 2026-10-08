//! The terminal view: one view over the agent components.
//!
//! A reader thread blocks on terminal events and hands them over a channel,
//! so the app's loop never waits on the terminal and any runner can drive
//! it. Key presses become [`Submit`](crate::agent::Submit),
//! [`Interrupt`](crate::agent::Interrupt) and [`Quit`](crate::agent::Quit)
//! messages; view state lives in [`TuiView`], apart from the agents.

mod draw;
mod input;
mod view;

use std::io::{Stdout, stdout};

use bevy_app::{App, AppExit, Last, Plugin, Startup, Update};
use bevy_ecs::prelude::*;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    crossterm::{
        event::{DisableBracketedPaste, EnableBracketedPaste},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
};

pub use view::{Picker, TuiView};

use crate::agent::RigSet;

/// Draws the agents in the terminal and reads the keyboard.
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TuiView>()
            .add_systems(Startup, open_terminal)
            .add_systems(
                Update,
                (
                    input::read_input
                        .run_if(resource_exists::<input::TuiInput>)
                        .before(RigSet::Input),
                    view::collect_messages.after(RigSet::Finish),
                ),
            )
            .add_systems(Last, draw::draw.run_if(resource_exists::<Screen>));
    }
}

/// The terminal, restored to normal when the app drops it.
#[derive(Resource)]
struct Screen(Terminal<CrosstermBackend<Stdout>>);

impl Drop for Screen {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.0.backend_mut(),
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
        let _ = self.0.show_cursor();
    }
}

/// Take over the terminal and start the reader thread. Without a terminal
/// the app exits with an error.
fn open_terminal(mut commands: Commands, mut exit: MessageWriter<AppExit>) {
    let opened = enable_raw_mode()
        .and_then(|()| execute!(stdout(), EnterAlternateScreen, EnableBracketedPaste))
        .and_then(|()| Terminal::new(CrosstermBackend::new(stdout())));
    match opened {
        Ok(terminal) => {
            commands.insert_resource(Screen(terminal));
            commands.insert_resource(input::TuiInput::spawn());
        }
        Err(error) => {
            let _ = disable_raw_mode();
            bevy_log::error!("cannot open the terminal: {error}");
            exit.write(AppExit::error());
        }
    }
}
