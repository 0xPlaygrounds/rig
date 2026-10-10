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
//! A plugin adds to the view without touching it: a [`TuiPanel`] beside
//! the transcript or over the screen, drawn by the plugin's own system in
//! [`TuiSystems::Draw`] (see [`panel`]); a [`RequestRedraw`] for a frame;
//! the agent shown, [`Focused`]; and the [`TuiScreen`]'s size. What the
//! agents do is their [`Activity`](crate::plugins::activity::Activity),
//! which the view adds unless it is there, and [`ratatui`] is re-exported
//! so a plugin draws with the same version.

mod clipboard;
mod complete;
pub mod diff;
mod editor;
mod input;
pub mod markdown;
pub mod panel;
mod render;
mod renderers;
mod terminal;
mod transcript;
mod view;
mod wrap;

use crate::front::{Front, RunMode};
use crate::host::reload::ReloadStatus;
use crate::plugins::activity::{ActivityPlugin, ActivitySystems};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

pub use panel::{Focused, PanelCanvas, Placement, RequestRedraw, TuiPanel, TuiScreen, TuiSystems};
pub use ratatui;
pub use renderers::{
    AppToolRenderersExt, RESULT_LINES, RenderToolCall, ToolCallView, ToolRenderer, excerpt,
};

/// Owns the terminal and draws the focused agent.
#[derive(Default)]
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        // A print run has stdout for its own output.
        let world = app.world();
        if world.contains_resource::<Front>()
            || world
                .get_resource::<RunMode>()
                .is_some_and(RunMode::is_headless)
        {
            return;
        }
        app.insert_resource(Front("tui".to_owned()));
        renderers::add_builtin_renderers(app);
        if !app.is_plugin_added::<ActivityPlugin>() {
            app.add_plugins(ActivityPlugin);
        }
        app.init_resource::<view::TuiView>()
            .init_resource::<render::FrameLayout>()
            .init_resource::<TuiScreen>()
            .add_message::<RequestRedraw>()
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
            .configure_sets(
                PostUpdate,
                (
                    TuiSystems::Prepare,
                    TuiSystems::Layout,
                    TuiSystems::Draw.run_if(render::frame_due),
                    TuiSystems::Render.run_if(render::frame_due),
                )
                    .chain()
                    .after(ActivitySystems),
            )
            .add_systems(
                PostUpdate,
                (
                    render::layout.in_set(TuiSystems::Layout).run_if(
                        resource_exists::<terminal::Tui>.and_then(
                            render::needs_redraw
                                .or_eager(resource_changed_or_removed::<ReloadStatus>),
                        ),
                    ),
                    render::render
                        .in_set(TuiSystems::Render)
                        .run_if(resource_exists::<terminal::Tui>),
                ),
            )
            .add_systems(Last, terminal::keep_screen_on_reload)
            .add_observer(view::on_focus);
    }
}
