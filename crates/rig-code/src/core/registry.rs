//! How plugins add tools and slash commands, and the events commands and
//! views exchange. Built-in tools and commands register through the same
//! [`AppExt`] methods as any third-party plugin.
//!
//! ```no_run
//! use rig_code::bevy_app::prelude::*;
//! use rig_code::bevy_ecs::prelude::*;
//! use rig_code::prelude::*;
//!
//! fn hello(input: In<CommandInput>, mut commands: Commands) {
//!     commands.trigger(Notice::info(input.agent, "Hello!"));
//! }
//!
//! fn hello_plugin(app: &mut App) {
//!     app.add_command("hello", "Say hello", hello);
//! }
//! ```

use std::panic::{AssertUnwindSafe, catch_unwind};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::{RegisteredSystemError, SystemId};
use bevy_log::error;
use rig_core::completion::ToolDefinition;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::tool::{Tool, tool_definition};

/// A registered tool: what the model is told about it, and the handler a
/// call is dispatched to. One entity per tool.
#[derive(Component, Clone, Debug)]
pub struct ToolSpec {
    /// The name, description and argument schema sent to the model.
    pub definition: ToolDefinition,
    /// The tool as an effect handler.
    pub handler: ErasedHandler,
}

/// A registered slash command. It sits on the entity of the one-shot
/// system that runs the command.
#[derive(Component, Clone, Debug)]
pub struct SlashCommand {
    /// The name typed after `/`.
    pub name: String,
    /// One line for `/help`.
    pub help: String,
}

/// What a slash command system receives: the agent it was run for and the
/// text after the command name, trimmed.
#[derive(Clone, Debug)]
pub struct CommandInput {
    /// The agent the command was run for.
    pub agent: Entity,
    /// The arguments.
    pub args: String,
}

/// Registration of tools and slash commands on an [`App`].
pub trait AppExt {
    /// Registers `tool`. A second tool with the same name is refused with
    /// an error in the log; the first one stays.
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self;

    /// Registers `system` as the slash command `/name`. A second command
    /// with the same name is refused with an error in the log.
    fn add_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandInput>, (), M> + 'static,
    ) -> &mut Self;
}

impl AppExt for App {
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self {
        let definition = tool_definition(&tool);
        let world = self.world_mut();
        let taken = world
            .query::<&ToolSpec>()
            .iter(world)
            .any(|spec| spec.definition.name.as_str() == definition.name.as_str());
        if taken {
            error!("a tool named `{}` is already registered", definition.name);
            return self;
        }
        world.spawn(ToolSpec {
            definition,
            handler: ErasedHandler::new(ToolAdapter::new(tool)),
        });
        self
    }

    fn add_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandInput>, (), M> + 'static,
    ) -> &mut Self {
        let world = self.world_mut();
        let taken = world
            .query::<&SlashCommand>()
            .iter(world)
            .any(|command| command.name == name);
        if taken {
            error!("a command named `/{name}` is already registered");
            return self;
        }
        let id = self.register_system(system);
        self.world_mut()
            .entity_mut(id.entity())
            .insert(SlashCommand {
                name: name.to_owned(),
                help: help.to_owned(),
            });
        self
    }
}

/// Runs a slash command for an agent. `line` is what was typed, such as
/// `/model openai/gpt-5.5`; the leading `/` is optional.
#[derive(EntityEvent, Clone, Debug)]
pub struct RunCommand {
    /// The agent.
    pub entity: Entity,
    /// The command line.
    pub line: String,
}

/// How a notice is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoticeLevel {
    /// Information.
    Info,
    /// Something failed.
    Error,
}

/// A message for whoever views the agent.
#[derive(EntityEvent, Clone, Debug)]
pub struct Notice {
    /// The agent.
    pub entity: Entity,
    /// How it is shown.
    pub level: NoticeLevel,
    /// The text.
    pub text: String,
}

impl Notice {
    /// An informational notice.
    pub fn info(entity: Entity, text: impl Into<String>) -> Self {
        Self {
            entity,
            level: NoticeLevel::Info,
            text: text.into(),
        }
    }

    /// A notice that something failed.
    pub fn error(entity: Entity, text: impl Into<String>) -> Self {
        Self {
            entity,
            level: NoticeLevel::Error,
            text: text.into(),
        }
    }
}

/// One choice of an [`OpenPicker`].
#[derive(Clone, Debug)]
pub struct PickerOption {
    /// What the list shows and the filter matches.
    pub label: String,
    /// Shown beside the label.
    pub detail: String,
    /// The argument the command runs with.
    pub value: String,
}

/// Asks a view to let the user choose one of `options`. The choice runs
/// `/<command> <value>` for the agent.
#[derive(EntityEvent, Clone, Debug)]
pub struct OpenPicker {
    /// The agent.
    pub entity: Entity,
    /// The picker's title.
    pub title: String,
    /// The command a choice runs.
    pub command: String,
    /// The choices.
    pub options: Vec<PickerOption>,
}

/// Finds the command a [`RunCommand`] names and runs its system. A command
/// that panics is caught and reported. Bevy cannot put a panicked one-shot
/// system back, so that command stays unavailable until the next reload.
pub(crate) fn run_command(
    run: On<RunCommand>,
    registered: Query<(Entity, &SlashCommand)>,
    mut commands: Commands,
) {
    let agent = run.entity;
    let line = run.line.trim().trim_start_matches('/');
    let (name, args) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    let Some((system, _)) = registered.iter().find(|(_, command)| command.name == name) else {
        commands.trigger(Notice::error(
            agent,
            format!("Unknown command /{name}. Type /help for the list."),
        ));
        return;
    };
    let id = SystemId::<In<CommandInput>>::from_entity(system);
    let input = CommandInput {
        agent,
        args: args.trim().to_owned(),
    };
    let name = name.to_owned();
    commands.queue(move |world: &mut World| {
        let ran = catch_unwind(AssertUnwindSafe(|| world.run_system_with(id, input)));
        let text = match ran {
            Ok(Ok(()) | Err(RegisteredSystemError::Skipped(_))) => return,
            Ok(Err(RegisteredSystemError::SystemMissing(_))) => {
                format!("/{name} panicked earlier and is unavailable until /reload.")
            }
            Ok(Err(error)) => format!("/{name} failed: {error}"),
            Err(_) => format!("/{name} panicked; see agent.log. It is unavailable until /reload."),
        };
        world.trigger(Notice::error(agent, text));
    });
}
