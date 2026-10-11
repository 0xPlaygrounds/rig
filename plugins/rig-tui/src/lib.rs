//! The terminal view. It reads agent components, keeps its own view state
//! in a resource, and sends the same requests any other view would. It
//! never owns the app's loop: a thread reads the terminal and wakes the
//! loop, `PreUpdate` handles what it read, and drawing is a system in
//! `PostUpdate` that runs when something drawn changed.
//!
//! The input is a multiline editor with the prompt history and `/` and `@`
//! completion. Answers are drawn as markdown and
//! edits as diffs; a plugin draws its own tools' calls with
//! [`AppToolRenderersExt::add_tool_renderer`].
//!
//! It runs when someone sits at the terminal
//! ([`Invoked::interactive`](rig_harness::prelude::Invoked::interactive)):
//! no `--print` and stdin a terminal.
//!
//! A plugin adds to the view without touching it: a [`TuiPanel`] beside
//! the transcript or over the screen, drawn by the plugin's own system in
//! [`TuiSystems::Draw`] (see [`panel`]); a [`RequestRedraw`] for a frame;
//! the agent shown, [`Focused`], and showing another, [`Focus`]; a choice
//! for the user, [`PickRequest`]; the status line's items
//! ([`StatusItems`], [`AppStatus`]); and the [`TuiScreen`]'s size. These
//! are there in every run, also one with another front such as `--print`,
//! which never draws a frame. [`ratatui`] is re-exported so a plugin draws
//! with the same version.

mod clipboard;
mod complete;
pub mod diff;
mod editor;
mod input;
pub mod markdown;
pub mod panel;
mod render;
mod renderers;
mod status;
mod terminal;
mod transcript;
mod view;
mod wrap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_harness::prelude::Invoked;

pub use panel::{Focused, PanelCanvas, Placement, RequestRedraw, TuiPanel, TuiScreen, TuiSystems};
pub use ratatui;
pub use renderers::{
    AppToolRenderersExt, RESULT_LINES, RenderToolCall, ToolCallView, ToolRenderer, excerpt,
};
pub use status::{AppStatus, Side, StatusItem, StatusItems, StatusSystems, Tone};
pub use view::{Focus, PickItem, PickRequest};

/// Owns the terminal and draws the focused agent.
#[derive(Default)]
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        // What other plugins use is there in any run; without the
        // terminal, no frame is ever due.
        status::add(app);
        app.init_resource::<render::FrameLayout>()
            .init_resource::<TuiScreen>()
            .add_message::<RequestRedraw>()
            .add_message::<PickRequest>()
            .configure_sets(
                PostUpdate,
                (
                    TuiSystems::Prepare,
                    TuiSystems::Layout,
                    TuiSystems::Draw.run_if(render::frame_due),
                    TuiSystems::Render.run_if(render::frame_due),
                )
                    .chain()
                    .after(StatusSystems),
            );
        // Anything else, such as a print run, has stdout for its own
        // output.
        if !app
            .world()
            .get_resource::<Invoked>()
            .is_some_and(Invoked::interactive)
        {
            return;
        }
        app.init_resource::<view::TuiView>()
            .init_resource::<complete::FileIndex>()
            .init_resource::<clipboard::Clipboard>()
            .add_systems(Startup, terminal::open_terminal)
            .add_systems(
                PreUpdate,
                (
                    input::read_input.run_if(resource_exists::<input::TerminalInput>),
                    complete::receive_paths,
                    clipboard::receive_images,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    (view::focus_agent, view::mark_focused).chain(),
                    // A picker that cannot open writes a notice instead.
                    (view::open_pickers, view::collect_notices).chain(),
                    view::recall_messages,
                ),
            )
            .add_systems(
                PostUpdate,
                (
                    render::layout
                        .in_set(TuiSystems::Layout)
                        .run_if(resource_exists::<terminal::Tui>.and_then(render::needs_redraw)),
                    render::render
                        .in_set(TuiSystems::Render)
                        .run_if(resource_exists::<terminal::Tui>),
                ),
            )
            .add_systems(Last, terminal::keep_screen_on_reload)
            .add_observer(view::on_focus);
    }
}

#[cfg(test)]
mod tests;
