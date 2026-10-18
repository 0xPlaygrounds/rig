//! Choosing a model and its reasoning setting: `--model`, `/model`,
//! `/effort` ([`ModelsPlugin`]), with the terminal view's pickers and both
//! in each agent's status line under the default `tui` feature, and the
//! ones a new session starts with, the last ones chosen
//! ([`DefaultsPlugin`]).
//!
//! `/model` lists every catalog model, those that can be reached first and
//! each other one with why not (no key, no tool calls, the provider's
//! reason), and completes its argument with them; a provider plugin adds
//! to the catalog through rig-core's [`Connector`] (in [`Models`]). It
//! lists the [`ModelItem`]s other plugins spawn too, for choices
//! that are no catalog model, such as a router that picks the model per
//! request. Such a plugin says what an agent's model is with a
//! [`ModelLabel`], which the status line shows in place of its
//! [`ModelChoice`].
//!
//! [`Connector`]: rig_harness::rig_core::catalog::Connector

use rig_harness::prelude::*;

pub mod defaults;
#[cfg(feature = "tui")]
mod tui;

pub use defaults::DefaultsPlugin;

/// Gives the agent the user talks to the model `--model` names, or no
/// model when the catalog lists none by that name, and adds `/model` and
/// `/effort`; with the `tui` feature, each is a picker without arguments,
/// and each agent's status line shows its model and reasoning setting.
#[derive(Default)]
pub struct ModelsPlugin;

impl Plugin for ModelsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "model",
            "Pick the model, or set it with /model vendor/model",
            model,
        )
        .add_command(
            "effort",
            "Pick the reasoning setting, or set it with /effort <level>",
            effort,
        )
        .add_systems(First, choose_invoked_model);
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// A choice `/model` offers besides the catalog's models, such as a router
/// that picks the model per request: spawn one. `/model <name>` runs its
/// `command`, and the picker lists it, before the catalog's models, by
/// `name`. Bevy's `Disabled` on the entity leaves it out.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default)]
pub struct ModelItem {
    /// What `/model` takes for it, such as `router`.
    pub name: String,
    /// What the picker shows, such as `router  the cheapest model that fits`.
    pub label: String,
    /// The command line it runs, without the `/`, such as `route on`.
    pub command: String,
}

impl ModelItem {
    /// `/model name`, shown as `label`, runs `command`.
    pub fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        command: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            command: command.into(),
        }
    }
}

/// On an agent: what its model is, when no [`ModelChoice`] says it, such
/// as `router` for a plugin that picks the model per request. The status
/// line shows it in place of the choice. It never changes in place.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[component(immutable)]
#[reflect(Component, Clone, Debug)]
pub struct ModelLabel(pub String);

/// Gives the agent the user talks to the model `--model` names, once the
/// session is restored and a remembered model given, so `--model` wins;
/// a turn the restored session continues ends first, as a model is not
/// changed under a running turn, and the agent is told so. A model the
/// catalog does not list leaves the agent with none at once, even under a
/// continued turn, with one error notice naming the closest ones: no
/// request goes to a model nobody named.
fn choose_invoked_model(
    invoked: Option<Res<Invoked>>,
    (agents, busy, models): (MainAgentQuery, Query<Has<ActiveTurn>>, Res<Models>),
    (mut done, mut told): (Local<bool>, Local<bool>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if *done {
        return;
    }
    let model = invoked.and_then(|invoked| invoked.args.model.clone());
    let (Some(model), Some(agent)) = (model, main_agent(&agents)) else {
        *done = true;
        return;
    };
    if let Err(why) = models.0.catalog().resolve(&model) {
        *done = true;
        commands.entity(agent).remove::<(ModelChoice, Connection)>();
        let stop = if why.suggestions.is_empty() { "." } else { "" };
        let text = format!("--model: {why}{stop} No model is picked.");
        notices.write(Notice::error(agent, text));
        return;
    }
    if busy.get(agent).unwrap_or_default() {
        if !*told {
            let text = format!("--model {model} applies once the continued turn ends.");
            notices.write(Notice::info(agent, text));
            *told = true;
        }
        return;
    }
    *done = true;
    commands.trigger(SetModel {
        entity: agent,
        model,
    });
}

/// `/model vendor/model` sets the agent's model, and `/model <name>` runs
/// the [`ModelItem`] of that name; without arguments, the terminal view
/// lets the user pick one.
fn model(
    In(args): In<CommandArgs>,
    items: Query<&ModelItem>,
    #[cfg(feature = "tui")] models: Res<Models>,
    #[cfg(feature = "tui")] mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if args.args.is_empty() {
        #[cfg(feature = "tui")]
        tui::pick_model(args.agent, (&models, &items), &mut notices, &mut commands);
        return;
    }
    if let Some(item) = items.iter().find(|item| item.name == args.args.trim()) {
        commands.trigger(RunCommand::new(args.agent, item.command.clone()));
        return;
    }
    commands.trigger(SetModel {
        entity: args.agent,
        model: args.args,
    });
}

/// `/effort <level>` sets the agent's reasoning setting; without
/// arguments, the terminal view lets the user pick one.
fn effort(
    In(args): In<CommandArgs>,
    agents: Query<&Connection>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(Connection { spec, .. }) = agents.get(args.agent) else {
        notices.write(Notice::info(args.agent, "Pick a model with /model first."));
        return;
    };
    if args.args.is_empty() {
        #[cfg(feature = "tui")]
        tui::pick_effort(args.agent, spec, &mut commands);
        return;
    }
    match spec.reasoning.named(&args.args) {
        Ok(choice) => {
            commands.trigger(SetReasoning {
                entity: args.agent,
                effort: ReasoningSetting(choice.reasoning),
            });
        }
        Err(why) => {
            let why = format!("{}: {why}.", spec.display_name);
            notices.write(Notice::error(args.agent, why));
        }
    }
}

#[cfg(test)]
mod tests;
