//! A minimal coding agent built as a Bevy app on Rig.
//!
//! Every agent is an entity: its conversation, model, effort, system prompt,
//! tool access and status are components, and each model or tool call in
//! flight is an entity tied to its agent. Tools and slash commands are
//! registered by Bevy plugins through
//! [`AppExt`](crate::core::registry::AppExt), the same way for built-ins
//! and third-party plugins. Every model and tool call goes through one
//! dispatch path that records it in the session's effect log.
//!
//! ```no_run
//! fn main() -> rig_code::bevy_app::AppExit {
//!     rig_code::app().run()
//! }
//! ```

pub mod builtin;
pub mod core;
#[cfg(feature = "tui")]
pub mod tui;

use std::time::Duration;

use bevy_app::{App, ScheduleRunnerPlugin, TaskPoolPlugin, plugin_group};
use bevy_log::LogPlugin;

pub use bevy_app;
pub use bevy_ecs;
pub use rig_core;

/// The exit code that asks the launcher to start the freshly built agent.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// The names a plugin author needs: the registration API, the agent
/// components and the events views and commands use.
pub mod prelude {
    pub use crate::core::agent::{
        Agent, AgentId, CallOf, Calls, Conversation, Effort, Model, Status, SystemPrompt,
        ToolAccess,
    };
    pub use crate::core::registry::{
        AppExt, CommandInput, Notice, NoticeLevel, OpenPicker, PickerOption, RunCommand,
        SlashCommand, ToolSpec,
    };
    pub use crate::core::save::ReflectSaved;
    pub use crate::core::turn::{Stop, Submit, TurnEnded};
    pub use crate::{RigCodePlugins, app};
}

plugin_group! {
    /// The agent's built-in plugins, in build order.
    pub struct RigCodePlugins {
        core:::CorePlugin,
        builtin::tools:::BuiltinToolsPlugin,
        builtin::commands:::BuiltinCommandsPlugin,
        core::save:::SavePlugin,
        core::reload:::ReloadPlugin,
        #[cfg(feature = "tui")]
        tui:::TuiPlugin,
    }
}

/// The agent app: Bevy's task pools, a loop ticking every 16 ms, logging to
/// the session's `agent.log`, plugin errors logged as warnings, and
/// [`RigCodePlugins`]. Add plugins to it, then call `run`.
pub fn app() -> App {
    let mut app = App::new();
    core::session::open(&mut app);
    app.add_plugins((
        TaskPoolPlugin::default(),
        ScheduleRunnerPlugin::run_loop(Duration::from_millis(16)),
        LogPlugin {
            fmt_layer: core::session::log_layer,
            ..Default::default()
        },
    ))
    .set_error_handler(bevy_ecs::error::warn)
    .add_plugins(RigCodePlugins);
    app
}
