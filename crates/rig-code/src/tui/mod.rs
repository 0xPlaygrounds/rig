//! The terminal view. It reads agent components, keeps its own view state
//! in [`TuiView`], and sends the same requests any other view would. It
//! never owns the app's loop: input is a zero-timeout poll in `PreUpdate`
//! and drawing is a system in `PostUpdate` that runs when something drawn
//! changed.

mod input;
mod render;
mod terminal;
mod view;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

pub use terminal::Tui;
pub use view::{PickValue, Picker, TuiView};

/// Owns the terminal and draws the focused agent.
#[derive(Default)]
pub struct TuiPlugin;

impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TuiView>()
            .add_systems(Startup, terminal::open_terminal)
            .add_systems(PreUpdate, input::read_input.run_if(resource_exists::<Tui>))
            .add_systems(
                Update,
                (
                    view::focus_agent,
                    view::open_pickers,
                    view::collect_notices,
                    view::show_reload_failures,
                ),
            )
            .add_systems(
                PostUpdate,
                render::render.run_if(
                    resource_exists::<Tui>.and_then(
                        render::needs_redraw.or_eager(
                            resource_changed_or_removed::<crate::host::reload::ReloadBuild>,
                        ),
                    ),
                ),
            )
            .add_systems(Last, terminal::keep_screen_on_reload);
    }
}
