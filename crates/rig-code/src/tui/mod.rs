//! The terminal view: one view over the agent components. It draws to the
//! controlling terminal, reads keys without blocking the app's loop, and
//! turns them into the same messages any other view would send.

mod input;
mod render;
mod view;

use std::io::Write;

use bevy::prelude::*;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::{cursor, event, execute, terminal};

use crate::core::{
    Agent, AgentStatus, Conversation, EffortChoice, ModelChoice, ModelEndpoint, StreamingText,
    ToolCallDone, ToolCallRun,
};

pub(crate) use view::TuiView;

/// Adds the terminal view. Without it the agent runs headless.
#[derive(Default)]
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TuiView>()
            .add_systems(Startup, open_terminal)
            .add_systems(PreUpdate, input::read_input.run_if(resource_exists::<Tui>))
            .add_systems(Update, view::collect_messages)
            .add_systems(
                PostUpdate,
                draw.run_if(resource_exists::<Tui>.and_then(needs_redraw)),
            );
    }
}

/// The terminal, in raw mode on the alternate screen. Dropping it, which
/// happens when the app ends, on any exit and while unwinding from a panic,
/// gives the terminal back.
#[derive(Resource)]
struct Tui(Terminal<CrosstermBackend<Box<dyn Write + Send + Sync>>>);

impl Tui {
    /// The controlling terminal, `/dev/tty`, so nothing else the process
    /// prints reaches the screen; stdout where there is none.
    fn open() -> std::io::Result<Self> {
        let output: Box<dyn Write + Send + Sync> =
            match std::fs::OpenOptions::new().write(true).open("/dev/tty") {
                Ok(tty) => Box::new(tty),
                Err(_) => Box::new(std::io::stdout()),
            };
        terminal::enable_raw_mode()?;
        let mut backend = CrosstermBackend::new(output);
        execute!(
            backend,
            terminal::EnterAlternateScreen,
            event::EnableBracketedPaste
        )?;
        Ok(Self(Terminal::new(backend)?))
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        // Nothing is left to report a failure to.
        let _ = execute!(
            self.0.backend_mut(),
            event::DisableBracketedPaste,
            terminal::LeaveAlternateScreen,
            cursor::Show
        );
        let _ = terminal::disable_raw_mode();
    }
}

fn open_terminal(mut commands: Commands) {
    match Tui::open() {
        Ok(tui) => commands.insert_resource(tui),
        Err(error) => error!("no terminal view: {error}"),
    }
}

/// Whether anything the view shows changed this frame.
fn needs_redraw(
    view: Res<TuiView>,
    agents: Query<
        (),
        (
            With<Agent>,
            Or<(
                Changed<Conversation>,
                Changed<AgentStatus>,
                Changed<ModelChoice>,
                Changed<EffortChoice>,
                Changed<ModelEndpoint>,
            )>,
        ),
    >,
    streams: Query<(), Changed<StreamingText>>,
    calls: Query<(), Or<(Added<ToolCallRun>, Added<ToolCallDone>)>>,
) -> bool {
    view.is_changed() || !agents.is_empty() || !streams.is_empty() || !calls.is_empty()
}

/// Draws the focused agent.
fn draw(
    mut tui: ResMut<Tui>,
    view: Res<TuiView>,
    agents: Query<render::AgentView, With<Agent>>,
    streams: Query<&StreamingText>,
    calls: Query<(&ToolCallRun, Option<&ToolCallDone>)>,
) {
    let Some(agent) = agents.iter().next() else {
        return;
    };
    let scene = render::Scene::new(&agent, &streams, &calls);
    if let Err(error) = tui.0.draw(|frame| render::frame(frame, &view, &scene)) {
        error!("cannot draw: {error}");
    }
}
