//! The terminal view. It reads agent components, keeps its own view state
//! in a resource, and sends the same requests any other view would. It
//! never owns the app's loop: a thread reads the terminal and wakes the
//! loop, `PreUpdate` handles what it read, and drawing is a system in
//! `PostUpdate` that runs when something drawn changed.
//!
//! The input is a multiline editor with the prompt history, `/` and `@`
//! completion and Ctrl+G for `$EDITOR`. Answers are drawn as markdown and
//! edits as diffs; a plugin draws its own tools' calls with
//! [`AppToolRenderersExt::add_tool_renderer`].

mod clipboard;
mod complete;
pub mod diff;
mod editor;
mod external;
mod input;
pub mod markdown;
mod render;
mod renderers;
mod terminal;
mod transcript;
mod view;
mod wrap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

pub use renderers::{
    AppToolRenderersExt, RESULT_LINES, RenderToolCall, ToolCallView, ToolRenderer, excerpt,
};

/// Owns the terminal and draws the focused agent.
#[derive(Default)]
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        renderers::add_builtin_renderers(app);
        app.init_resource::<view::TuiView>()
            .init_resource::<complete::FileIndex>()
            .init_resource::<clipboard::Clipboard>()
            .init_resource::<crate::host::sessions::SessionName>()
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
                    view::focus_agent,
                    view::show_reload_failures,
                    // A picker that cannot open writes a notice instead.
                    (view::open_pickers, view::collect_notices).chain(),
                    view::recall_messages,
                    external::start_external_edit.run_if(
                        resource_exists::<external::EditRequested>
                            .and_then(resource_exists::<terminal::Tui>)
                            .and_then(resource_exists::<input::TerminalInput>)
                            .and_then(not(resource_exists::<external::ExternalEdit>)),
                    ),
                    external::finish_external_edit.run_if(
                        resource_exists::<external::ExternalEdit>
                            .and_then(resource_exists::<terminal::Tui>)
                            .and_then(resource_exists::<input::TerminalInput>),
                    ),
                ),
            )
            .add_systems(
                PostUpdate,
                render::render.run_if(
                    resource_exists::<terminal::Tui>
                        .and_then(not(resource_exists::<external::ExternalEdit>))
                        .and_then(render::needs_redraw.or_eager(
                            resource_changed_or_removed::<crate::host::reload::ReloadBuild>,
                        )),
                ),
            )
            .add_systems(Last, terminal::keep_screen_on_reload);
    }
}
