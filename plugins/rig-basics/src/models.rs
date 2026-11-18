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
//! A run that names no model with `--model`, restores none with its
//! session and has no saved default starts on the [`FirstPick`], for that
//! run only, in every front alike.
//!
//! [`Connector`]: rig_harness::rig_core::catalog::Connector

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ReasoningSupport;
use rig_harness::run::{RunStart, RunStartSystems};

pub mod defaults;
mod pick;
#[cfg(feature = "tui")]
mod tui;

/// Gives the agent the user talks to the model `--model` names, or no
/// model when the catalog lists none by that name, else the [`FirstPick`]
/// when it has none, and adds `/model` and `/effort`, each a picker
/// without arguments; with the `tui` feature, each agent's status line
/// shows its model and reasoning setting.
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
            "Pick the reasoning setting, or set it with /effort <level> or /effort <n> tokens",
            effort,
        )
        .init_resource::<TypedChoices>()
        .add_systems(First, choose_invoked_model)
        .configure_sets(
            Update,
            FirstPickSystems.after(RunStartSystems).before(AgentLoop),
        )
        .add_systems(
            Update,
            start_on_first_pick
                .in_set(FirstPickSystems)
                .run_if(resource_exists::<RunStart>),
        );
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

/// The model a run that names none starts on, in a home with no default:
/// the terminal view's, a headless run's or an editor's new session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FirstPick {
    /// A sign-in is still finding out which of its models the account
    /// offers: ask again once the [`Models`] changed.
    Checking,
    /// The first model that can be reached, by its reference.
    Model(String),
    /// None can be reached, with why when a saved key file cannot be read
    /// or a sign-in's models cannot be used.
    Nothing(Option<String>),
}

impl FirstPick {
    /// The first pick among `models`: [`Self::Checking`] while a sign-in
    /// is still asking its account; then the ChatGPT plan's default model
    /// when the account offers it; else the first model that can be
    /// reached, in the order [`Connector::reachable`] gives: a provider's
    /// with a key set or a sign-in's, in catalog order, then a local
    /// server's. A sign-in goes before keys because the user signed in
    /// within rig, while a key in the environment may be there for other
    /// tools.
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
        let signed_in = crate::login::default_model();
        if statuses
            .iter()
            .any(|(spec, status)| *status == ModelStatus::Usable && spec.reference() == signed_in)
        {
            return Self::Model(signed_in);
        }
        if let Some((spec, _)) = statuses
            .iter()
            .find(|(_, status)| *status == ModelStatus::Usable)
        {
            return Self::Model(spec.reference());
        }
        let unreadable =
            (statuses.iter()).find_map(|(_, status)| crate::login::keys::unreadable_key(status));
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
        let refused = unreadable.or(unasked).or_else(|| {
            signed_in().find_map(|(_, status)| match status {
                ModelStatus::NotOffered(why) => Some(why.clone()),
                _ => None,
            })
        });
        Self::Nothing(refused)
    }
}

/// The system that gives the agent the user talks to the [`FirstPick`], in
/// `Update` after the run's start ends and before the agent loop: a front
/// that reads the agent's model in that frame orders itself after it.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FirstPickSystems;

/// Gives the agent the user talks to the [`FirstPick`] for this run once
/// the run's start is over, when `--model` names none and neither the
/// restored session nor the saved default gave it one. It waits while a
/// sign-in checks its account or a turn runs; a model chosen meanwhile
/// stands.
fn start_on_first_pick(
    (start, invoked): (Res<RunStart>, Option<Res<Invoked>>),
    (agents, unchosen): (MainAgentQuery, Query<Has<ActiveTurn>, Without<ModelChoice>>),
    models: Res<Models>,
    mut done: Local<bool>,
    mut commands: Commands,
) {
    if *done || !start.is_over() {
        return;
    }
    let Some(agent) = main_agent(&agents) else {
        return;
    };
    let named = invoked.is_some_and(|invoked| invoked.args.model.is_some());
    let busy = match unchosen.get(agent) {
        Ok(busy) if !named => busy,
        _ => {
            *done = true;
            return;
        }
    };
    if busy {
        return;
    }
    match FirstPick::of(&models) {
        // The answer wakes the loop and changes the models.
        FirstPick::Checking => {}
        FirstPick::Model(model) => {
            *done = true;
            commands.trigger(SetModel {
                entity: agent,
                model,
                kind: ChoiceKind::ForThisRun,
            });
        }
        FirstPick::Nothing(_) => *done = true,
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

/// `/effort <setting>` sets the agent's reasoning setting, as its model
/// reads the text ([`ReasoningSupport::parse`]); without arguments, a
/// picker lets the user pick one. Refused without a model.
///
/// [`ReasoningSupport::parse`]: rig_harness::rig_core::catalog::ReasoningSupport::parse
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
    match spec.reasoning.parse(&args.args) {
        Ok(reasoning) => {
            commands.trigger(SetReasoning {
                entity: args.agent,
                effort: ReasoningSetting(reasoning),
                kind: typed.0,
            });
        }
        Err(why) => {
            let why = format!("{}: {why}.", spec.display_name);
            notices.write(Notice::error(args.agent, why));
        }
    }
}

/// The text of the agent's reasoning setting as its connected model reads
/// it back, such as `high` for a budget the model calls that; without a
/// connection, as any model reads it.
pub(crate) fn effort_name(connection: Option<&Connection>, effort: ReasoningSetting) -> String {
    let any = ReasoningSupport::default();
    let reasoning = connection.map_or(&any, |connection| &connection.spec.reasoning);
    reasoning.name_of(effort.0)
}

#[cfg(test)]
mod tests;
