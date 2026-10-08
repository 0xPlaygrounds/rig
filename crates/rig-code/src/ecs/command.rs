//! Slash commands are registered one-shot systems. A plugin adds one with
//! [`RigAppExt::add_command`], which registers the system and puts a
//! [`SlashCommand`] on its entity.
//! [`CommandsPlugin`] adds `/model`, `/effort`, `/help` and `/quit` that way.

use bevy::{
    ecs::system::{IntoSystem, SystemId},
    prelude::*,
};

use super::{
    ChoiceKind, ChoiceRequested, Notice, RigAppExt,
    agent::{AgentStatus, EffortChoice, ModelChoice},
    catalog::{self, Providers},
};

/// Names a registered command system: `/name` runs it.
#[derive(Component, Clone, Debug)]
pub struct SlashCommand {
    /// The name, without the slash.
    pub name: String,
    /// One line for `/help`.
    pub help: String,
}

/// What a command system receives: the agent it was typed for, and the text
/// after its name, trimmed.
#[derive(Clone, Debug)]
pub struct CommandInput {
    /// The agent.
    pub agent: Entity,
    /// The arguments.
    pub args: String,
}

/// The id a command system runs under.
pub type CommandId = SystemId<In<CommandInput>>;

/// Register `system` as the command `/name`, unless that name is taken.
pub(super) fn register<M>(
    app: &mut App,
    name: &str,
    help: &str,
    system: impl IntoSystem<In<CommandInput>, (), M> + 'static,
) {
    let world = app.world_mut();
    if world
        .query::<&SlashCommand>()
        .iter(world)
        .any(|command| command.name == name)
    {
        warn!("a command named `/{name}` is already registered; the second one is ignored");
        return;
    }
    let id = app.register_system(system);
    app.world_mut().entity_mut(id.entity()).insert((
        Name::new(format!("/{name}")),
        SlashCommand {
            name: name.to_owned(),
            help: help.to_owned(),
        },
    ));
}

/// The built-in commands.
pub struct CommandsPlugin;

impl Plugin for CommandsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "model",
            "Pick the model; `/model vendor/model` sets it directly",
            model,
        )
        .add_command(
            "effort",
            "Pick how much the model reasons; `/effort <level>` sets it directly",
            effort,
        )
        .add_command("help", "List the commands", help)
        .add_command("quit", "Leave the agent", quit);
    }
}

fn model(
    In(input): In<CommandInput>,
    providers: Res<Providers>,
    mut agents: Query<(&mut ModelChoice, &mut EffortChoice, &AgentStatus)>,
    mut notices: MessageWriter<Notice>,
    mut choices: MessageWriter<ChoiceRequested>,
) {
    let agent = input.agent;
    // The loop reads the model before each step, so a switch mid-turn would
    // change provider halfway through the turn.
    if agents
        .get(agent)
        .is_ok_and(|(_, _, status)| *status != AgentStatus::Idle)
    {
        notices.write(Notice::error(
            agent,
            "A turn is running. Stop it first, then switch models.",
        ));
        return;
    }
    if input.args.is_empty() {
        choices.write(ChoiceRequested {
            agent,
            kind: ChoiceKind::Model,
        });
        return;
    }
    let Some(spec) = catalog::resolve(&input.args) else {
        notices.write(Notice::error(
            agent,
            format!("`{}` is not in the model catalog", input.args),
        ));
        return;
    };
    if !spec.tools {
        notices.write(Notice::error(
            agent,
            format!("{} does not call tools", spec.display_name),
        ));
        return;
    }
    if !providers.available(spec) {
        let variable = spec
            .provider
            .api_key_env()
            .map(|name| format!(": set {name}"))
            .unwrap_or_default();
        notices.write(Notice::error(
            agent,
            format!("no credential for {}{variable}", spec.provider.vendor()),
        ));
        return;
    }
    let Ok((mut model, mut effort, _)) = agents.get_mut(agent) else {
        return;
    };
    let reference = catalog::reference(spec);
    model.0 = Some(reference.clone());
    notices.write(Notice::info(
        agent,
        format!("Model: {} ({reference})", spec.display_name),
    ));
    if !catalog::effort_options(spec).contains(&effort) {
        notices.write(Notice::info(
            agent,
            format!(
                "Effort `{}` is not offered by this model; it is now `default`",
                effort.name()
            ),
        ));
        *effort = EffortChoice::Default;
    }
}

fn effort(
    In(input): In<CommandInput>,
    mut agents: Query<(&ModelChoice, &mut EffortChoice)>,
    mut notices: MessageWriter<Notice>,
    mut choices: MessageWriter<ChoiceRequested>,
) {
    let agent = input.agent;
    let Ok((model, mut effort)) = agents.get_mut(agent) else {
        return;
    };
    let Some(spec) = model.0.as_deref().and_then(catalog::resolve) else {
        notices.write(Notice::error(agent, "Pick a model with /model first"));
        return;
    };
    if input.args.is_empty() {
        choices.write(ChoiceRequested {
            agent,
            kind: ChoiceKind::Effort,
        });
        return;
    }
    let options = catalog::effort_options(spec);
    match options.iter().find(|option| option.name() == input.args) {
        Some(choice) => {
            *effort = *choice;
            notices.write(Notice::info(agent, format!("Effort: {}", choice.name())));
        }
        None => {
            let names = options
                .iter()
                .map(EffortChoice::name)
                .collect::<Vec<_>>()
                .join(", ");
            notices.write(Notice::error(
                agent,
                format!("{} takes one of: {names}", spec.display_name),
            ));
        }
    }
}

fn help(
    In(input): In<CommandInput>,
    commands: Query<&SlashCommand>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines = commands
        .iter()
        .map(|command| format!("/{}  {}", command.name, command.help))
        .collect::<Vec<_>>();
    lines.sort();
    notices.write(Notice::info(input.agent, lines.join("\n")));
}

fn quit(In(_input): In<CommandInput>, mut exit: MessageWriter<AppExit>) {
    exit.write(AppExit::Success);
}
