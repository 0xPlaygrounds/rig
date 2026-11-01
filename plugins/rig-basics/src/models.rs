//! Choosing a model and its reasoning setting: `--model`, `/model`,
//! `/effort` ([`ModelsPlugin`]), with the terminal view's pickers and both
//! in each agent's status line under the default `tui` feature, and the
//! ones a new session starts with, the last ones the user chose
//! ([`DefaultsPlugin`](defaults::DefaultsPlugin)).
//!
//! `/model` lists the catalog's models that can be used now (a key is set,
//! a sign-in offers them, or a local server is configured), says how to
//! make one usable when none is, opens on the agent's model, and completes
//! its argument with them; a provider plugin adds
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
mod pick;
#[cfg(feature = "tui")]
mod tui;

/// Gives the agent the user talks to the model `--model` names, or no
/// model when the catalog lists none by that name, and adds `/model` and
/// `/effort`, each a picker without arguments; with the `tui` feature,
/// each agent's status line shows its model and reasoning setting.
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
        .init_resource::<TypedChoices>()
        .add_systems(First, choose_invoked_model);
        let relist = pick::relist.run_if(any_with_component::<pick::ModelList>);
        #[cfg(feature = "tui")]
        let relist = relist.before(rig_tui::ChoiceSystems);
        app.add_systems(Update, relist);
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

/// What a model or reasoning setting chosen with a command, `/model`,
/// `/effort` or their pickers, is in this run: [`ChoiceKind::ByUser`]
/// unless a front whose client owns the choice inserts
/// [`ChoiceKind::ForThisRun`], as ACP does, where the editor sets the
/// model of every session.
#[derive(Resource, Clone, Copy, Debug)]
pub struct TypedChoices(pub ChoiceKind);

impl Default for TypedChoices {
    fn default() -> Self {
        Self(ChoiceKind::ByUser)
    }
}

/// The model a run that names none starts on, such as a headless run's
/// or an editor's new session in a home with no default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FirstPick {
    /// A sign-in is still finding out which of its models the account
    /// offers: ask again once the [`Models`] changed.
    Checking,
    /// The first model that can be reached, by its reference.
    Model(String),
    /// None can be reached, with why a sign-in's models cannot be used
    /// when that is why.
    Nothing(Option<String>),
}

impl FirstPick {
    /// The first pick among `models`, in the order [`Connector::reachable`]
    /// gives.
    ///
    /// [`Connector::reachable`]: rig_harness::rig_core::catalog::Connector::reachable
    pub fn of(models: &Models) -> Self {
        use rig_harness::rig_core::catalog::ModelStatus;
        let statuses = models.0.statuses();
        if statuses
            .iter()
            .any(|(_, status)| *status == ModelStatus::Checking)
        {
            return Self::Checking;
        }
        if let Some((spec, _)) = statuses
            .iter()
            .find(|(_, status)| *status == ModelStatus::Usable)
        {
            return Self::Model(spec.reference());
        }
        // Why a sign-in's models cannot be used: it could not ask its
        // account, else the account offers none of them.
        let signed_in = || {
            let statuses = statuses.iter();
            statuses.filter(|(spec, _)| models.0.plan(spec).is_some())
        };
        let unasked = signed_in().find_map(|(_, status)| match status {
            ModelStatus::Unavailable(why) => Some(why.clone()),
            _ => None,
        });
        let refused = unasked.or_else(|| {
            signed_in().find_map(|(_, status)| match status {
                ModelStatus::NotOffered(why) => Some(why.clone()),
                _ => None,
            })
        });
        Self::Nothing(refused)
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
/// catalog does not list, or one that does not connect, such as one with
/// no key, leaves the agent with none at once, even under a continued
/// turn, with one error notice naming the closest ones or why: no request
/// goes to a model nobody named.
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
    let refused = match models.0.catalog().resolve(&model) {
        Ok(found) => {
            let reference = found.spec.reference();
            let connected = models.0.connect(&reference);
            connected
                .err()
                .map(|why| format!("cannot use {reference}: {why}."))
        }
        Err(why) => {
            let stop = if why.suggestions.is_empty() { "." } else { "" };
            Some(format!("{why}{stop}"))
        }
    };
    if let Some(why) = refused {
        *done = true;
        let mut picked = commands.entity(agent);
        picked.remove::<(ModelChoice, Connection, ModelRefused)>();
        let text = format!("--model: {why} No model is picked.");
        notices.write(Notice::error(agent, text));
        return;
    }
    if busy.get(agent).unwrap_or_default() {
        if !*told {
            let text = format!("--model {model} applies once the current turn ends.");
            notices.write(Notice::info(agent, text));
            *told = true;
        }
        return;
    }
    *done = true;
    commands.trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
}

/// `/model vendor/model` sets the agent's model, and `/model <name>` runs
/// the [`ModelItem`] of that name; without arguments, a picker lets the
/// user pick one.
fn model(
    In(args): In<CommandArgs>,
    items: Query<&ModelItem>,
    typed: Res<TypedChoices>,
    (models, chosen): (Res<Models>, Query<&ModelChoice>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if args.args.is_empty() {
        let current = chosen.get(args.agent).ok();
        let listed = (&*models, &items, current);
        pick::pick_model(args.agent, listed, &mut notices, &mut commands);
        return;
    }
    if let Some(item) = items.iter().find(|item| item.name == args.args.trim()) {
        commands.trigger(RunCommand::new(args.agent, item.command.clone()));
        return;
    }
    commands.trigger(SetModel {
        entity: args.agent,
        model: args.args,
        kind: typed.0,
    });
}

/// `/effort <level>` sets the agent's reasoning setting; without
/// arguments, a picker lets the user pick one. Refused without a model.
fn effort(
    In(args): In<CommandArgs>,
    agents: Query<&Connection>,
    typed: Res<TypedChoices>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(Connection { spec, .. }) = agents.get(args.agent) else {
        notices.write(Notice::error(
            args.agent,
            "Pick a model first: /model <vendor/model>.",
        ));
        return;
    };
    if args.args.is_empty() {
        pick::pick_effort(args.agent, spec, &mut commands);
        return;
    }
    match spec.reasoning.named(&args.args) {
        Ok(choice) => {
            commands.trigger(SetReasoning {
                entity: args.agent,
                effort: ReasoningSetting(choice.reasoning),
                kind: typed.0,
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
