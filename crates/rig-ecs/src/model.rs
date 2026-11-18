//! An agent's model: the catalog model it chose, the reasoning setting it
//! sends, and how the choice is connected. The catalog and how its models
//! are reached are rig-core's [`Connector`]. A model whose catalog entry
//! gives no context window takes the one its local server says it serves.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::{Connector, ModelSpec};
use rig_core::completion::{CompletionRequest, Reasoning, UnsupportedOption};
use rig_core::effect::EffectKind;
use rig_core::error::ProviderError;
use serde::{Deserialize, Serialize};

use super::agent::CallOf;
use super::agent::{ActiveTurn, Agent, Notice};
use super::effects::Handler;
use super::session_log::ReflectSaved;
use super::stage::{Effect, ForAgent, HoldOf, ReadyToStart, SupersededBy};
use super::tasks::{Running, Wake};
use super::turn::{Adapted, ModelRequest, TurnCall};

/// The models agents can use and how each is reached: rig's built-in
/// catalog by default. A plugin adds a sign-in with
/// [`Connector::add_sign_in`], or replaces the catalog. After a change, the
/// next frame connects the agents whose chosen model was not usable before.
/// The [`Connector`] is `Send` and `Sync` except on browser wasm, where its
/// providers may hold JavaScript values, and Bevy asks for both on every
/// target.
#[derive(Resource, Clone, Default)]
pub struct Models(pub Connector);

// SAFETY: browser wasm without the `atomics` target feature runs one
// thread, so `Models` is never sent to or shared with another one.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    not(target_feature = "atomics")
))]
unsafe impl Send for Models {}
// SAFETY: as for `Send`: there is no other thread.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    not(target_feature = "atomics")
))]
unsafe impl Sync for Models {}

impl Models {
    /// A connection to the catalog model `reference` (`vendor/model`), such
    /// as for a turn or a call a plugin sends to another model than its
    /// agent's; or why it cannot be used.
    pub fn connect(&self, reference: &str) -> Result<Connection, String> {
        match self.0.connect(reference) {
            Ok((spec, handler)) => Ok(Connection {
                spec,
                handler: Handler(handler),
            }),
            Err(why) => Err(format!("Cannot use {reference}: {why}.")),
        }
    }
}

/// The chosen catalog model, as `vendor/model`. It never changes in place:
/// choosing another model inserts a new one, which rebuilds the agent's
/// [`Connection`]. Saved with the session as a string under rig-memory's
/// `MODEL` (its type path, `rig::model`), which a library program reads.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[component(immutable)]
#[reflect(Component, Saved, Clone)]
#[type_path = "rig"]
#[type_name = "model"]
#[serde(transparent)]
pub struct ModelChoice(pub String);

/// On an agent whose [`ModelChoice`] could not be connected: why, as the
/// error notice that refused it said. Gone once a model connects.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct ModelRefused(pub String);

impl ModelChoice {
    /// The model and reasoning setting of an agent spawned by an agent with
    /// `parent`'s: `model`, a reference to a catalog model that calls tools,
    /// when given, else the parent's; the reasoning setting `effort` names
    /// for that model ([`ReasoningSupport::parse`]) when given, else the
    /// parent's with the parent's model.
    ///
    /// [`ReasoningSupport::parse`]: rig_core::catalog::ReasoningSupport::parse
    pub fn inherit(
        models: &Connector,
        parent: (Option<&ModelChoice>, ReasoningSetting),
        model: Option<&str>,
        effort: Option<&str>,
    ) -> Result<(Option<ModelChoice>, ReasoningSetting), String> {
        fn asked(text: Option<&str>) -> Option<&str> {
            text.map(str::trim).filter(|text| !text.is_empty())
        }
        let resolve = |text: &str| models.catalog().resolve(text).map(|found| found.spec);
        let model = match asked(model) {
            Some(asked) => {
                let spec = resolve(asked).map_err(|why| format!("{why}; use vendor/model"))?;
                if !spec.tools {
                    return Err(format!("{asked} cannot call tools"));
                }
                Some(ModelChoice(spec.reference()))
            }
            None => parent.0.cloned(),
        };
        let effort = match (asked(effort), &model) {
            (Some(text), Some(model)) => {
                let spec = resolve(&model.0).map_err(|why| why.to_string())?;
                let parsed = spec.reasoning.parse(text);
                ReasoningSetting(parsed.map_err(|why| format!("{}: {why}", spec.display_name))?)
            }
            _ if model.is_some() && model.as_ref() == parent.0 => parent.1,
            _ => ReasoningSetting::default(),
        };
        Ok((model, effort))
    }
}

/// The connected model of an agent with a [`ModelChoice`]: its catalog
/// entry and the effect handler every model call is dispatched to. Built
/// from the environment once per choice, and not saved: restoring the
/// choice rebuilds it.
#[derive(Component, Reflect, Clone)]
#[reflect(opaque, Component, Clone)]
pub struct Connection {
    /// The model's catalog entry.
    pub spec: Arc<ModelSpec>,
    /// The model as an effect handler.
    pub handler: Handler,
}

/// The reasoning setting sent with each request, or `None` for the
/// provider's default. It never changes in place. Saved with the session.
#[derive(
    Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize,
)]
#[component(immutable)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct ReasoningSetting(pub Option<Reasoning>);

/// Who made a model or reasoning choice, which says whether it outlives
/// the run.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Clone, Debug, PartialEq)]
pub enum ChoiceKind {
    /// The user, with a command of their own such as `/model`: a plugin
    /// may keep it for later runs.
    ByUser,
    /// For this run only, such as a command-line flag, an editor that sets
    /// the model of each session, an automatic pick or a plugin's own.
    ForThisRun,
}

/// Choose the agent's model by catalog reference (`vendor/model`).
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SetModel {
    /// The agent.
    pub entity: Entity,
    /// The catalog reference.
    pub model: String,
    /// Who chose it.
    pub kind: ChoiceKind,
}

/// Choose the agent's reasoning setting; `None` is the provider default.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SetReasoning {
    /// The agent.
    pub entity: Entity,
    /// The setting.
    pub effort: ReasoningSetting,
    /// Who chose it.
    pub kind: ChoiceKind,
}

/// A [`SetModel`] or [`SetReasoning`] the agent took: the new value is on
/// it. A refused one triggers nothing.
#[derive(EntityEvent, Clone, Debug)]
pub struct ChoiceTaken {
    /// The agent.
    pub entity: Entity,
    /// Who chose it.
    pub kind: ChoiceKind,
    /// The model chosen, or `None` for a reasoning setting.
    pub model: Option<ModelChoice>,
}

/// Chooses the agent's model: a catalog model that connects replaces the
/// agent's [`ModelChoice`] and connects it, or keeps its [`Connection`] when
/// it is to the same model, and is [`ChoiceTaken`]. A connected agent keeps
/// its reasoning setting by the name its model gives it, read again by the
/// new model ([`carried`]). An unknown one, or another one that does not
/// connect, such as one with no key, keeps it and its connection, with an
/// error notice naming the closest models or why, and the one kept. The
/// agent's own model, chosen again when it no longer connects, is refused
/// as [`connect`] refuses it.
pub(crate) fn on_set_model(
    set: On<SetModel>,
    agents: Query<
        (
            Has<ActiveTurn>,
            Option<&ModelChoice>,
            Option<&Connection>,
            Option<&ReasoningSetting>,
        ),
        With<Agent>,
    >,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((busy, current, connected, effort)) = agents.get(set.entity) else {
        return;
    };
    if busy {
        notices.write(Notice::turn_running(set.entity));
        return;
    }
    let why = match models.0.catalog().resolve(&set.model) {
        Ok(found) => {
            let model = ModelChoice(found.spec.reference());
            match models.connect(&model.0) {
                Ok(connection) => {
                    let mut agent = commands.entity(set.entity);
                    match connected {
                        Some(old) if old.spec.reference() == model.0 => {
                            agent.insert(model.clone());
                        }
                        // One insert, so `check_effort` never sees the old
                        // setting with the new model.
                        Some(old) => {
                            let effort = effort.copied().unwrap_or_default();
                            let effort = carried(&old.spec, effort, &connection.spec);
                            notices.write(Notice::info(set.entity, switched(&connection)));
                            agent
                                .insert((model.clone(), connection, effort))
                                .remove::<ModelRefused>();
                        }
                        None => {
                            agent
                                .insert((model.clone(), connection))
                                .remove::<ModelRefused>();
                        }
                    }
                    commands.trigger(ChoiceTaken {
                        entity: set.entity,
                        kind: set.kind,
                        model: Some(model),
                    });
                    return;
                }
                Err(why) if current == Some(&model) => {
                    let mut agent = commands.entity(set.entity);
                    agent
                        .remove::<Connection>()
                        .insert(ModelRefused(why.clone()));
                    notices.write(Notice::error(set.entity, why));
                    return;
                }
                Err(why) => why,
            }
        }
        Err(why) => {
            let stop = if why.suggestions.is_empty() { "." } else { "" };
            format!("Not picked: {why}{stop}")
        }
    };
    let kept = match current {
        Some(ModelChoice(model)) => format!(" Still on {model}."),
        None => String::new(),
    };
    notices.write(Notice::error(set.entity, format!("{why}{kept}")));
}

/// Connects an agent whose [`ModelChoice`] was inserted, by [`SetModel`],
/// by restoring a session or by a plugin, unless it is connected to that
/// model already: each choice is connected, or refused, once, so requests
/// never re-resolve the provider. Only a change of a connected agent's
/// model is announced: spawning and restoring an agent are silent, since
/// views show its model anyway.
pub(crate) fn connect(
    inserted: On<Insert<ModelChoice>>,
    agents: Query<(&ModelChoice, Option<&Connection>)>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = inserted.entity;
    let Ok((choice, connected)) = agents.get(agent) else {
        return;
    };
    if connected.is_some_and(|connection| connection.spec.reference() == choice.0) {
        return;
    }
    match models.connect(&choice.0) {
        Ok(connection) => {
            if connected.is_some() {
                notices.write(Notice::info(agent, switched(&connection)));
            }
            commands
                .entity(agent)
                .insert(connection)
                .remove::<ModelRefused>();
        }
        Err(why) => {
            let refused = ModelRefused(why.clone());
            commands
                .entity(agent)
                .remove::<Connection>()
                .insert(refused);
            notices.write(Notice::error(agent, why));
        }
    }
}

/// The context windows asked of the servers of models whose catalog entry
/// gives none, by reference: `None` while asking, or when the server did
/// not say. A server's answer, with or without a window or with a client
/// error status, is final for the run, and so is its silence past the ask's
/// deadline, so a server that hangs costs one deadline. A server that
/// could not be reached or answered with a server error has no entry, so
/// the next turn asks again.
#[derive(Resource, Default)]
pub(crate) struct ServedWindows(HashMap<String, Option<u32>>);

/// The ask of a model's server for its context window, by reference, on an
/// entity of its own until [`WindowAnswer`]ed.
#[derive(Component)]
pub(crate) struct AskingWindow(String);

/// How a model's server answered the ask for its window.
#[derive(Component)]
pub(crate) enum WindowAnswer {
    /// The window it serves, or `None` when its answer named none or it
    /// did not answer in time.
    Said(Option<u32>),
    /// It could not be reached, or answered with a server error.
    Unreached,
}

/// Gives a [`Connection`] whose model's catalog entry has no context window
/// the window its server serves it with: the one learned already, or else
/// asks the server ([`Connector::context_window`]), such as a local
/// Ollama or llama.cpp server, so compaction and output limits hold to it.
/// A model whose server cannot say keeps no window.
pub(crate) fn ask_window(
    inserted: On<Insert<Connection>>,
    mut connections: Query<&mut Connection>,
    (models, wake): (Res<Models>, Res<Wake>),
    mut windows: ResMut<ServedWindows>,
    mut commands: Commands,
) {
    let Ok(mut connection) = connections.get_mut(inserted.entity) else {
        return;
    };
    if connection.spec.input_limit().is_some() {
        return;
    }
    if let Some(&Some(window)) = windows.0.get(&connection.spec.reference()) {
        connection.spec = with_window(&connection.spec, window);
        return;
    }
    start_asking(
        &connection.spec,
        &models,
        &wake,
        &mut windows,
        &mut commands,
    );
}

/// Asks the server of `spec` for its window unless it was asked already,
/// and says whether it did.
fn start_asking(
    spec: &ModelSpec,
    models: &Models,
    wake: &Wake,
    windows: &mut ServedWindows,
    commands: &mut Commands,
) -> bool {
    let reference = spec.reference();
    if windows.0.contains_key(&reference) {
        return false;
    }
    let Some(ask) = models.0.context_window(spec) else {
        return false;
    };
    windows.0.insert(reference.clone(), None);
    let asked = reference.clone();
    let work = async move {
        ask.await.map_or_else(
            |error| {
                bevy_log::debug!("{asked} has no context window yet: its server said {error}");
                let unreached = matches!(error, ProviderError::Http(_))
                    || error
                        .provider_response_status()
                        .is_some_and(|status| status.is_server_error());
                if unreached {
                    WindowAnswer::Unreached
                } else {
                    WindowAnswer::Said(None)
                }
            },
            |window| WindowAnswer::Said(Some(window)),
        )
    };
    commands.spawn((
        Name::new(format!("context window of {reference}")),
        AskingWindow(reference),
        Running::spawn_into::<WindowAnswer, _>(super::turn::model_pool(), wake, work),
    ));
    true
}

/// Keeps the window a model's server answered with, and gives it to every
/// [`Connection`] to that model without one. The turns' calls held for the
/// answer ([`wait_for_window`]) are released.
pub(crate) fn learn_window(
    answered: On<Insert<WindowAnswer>>,
    asks: Query<(&AskingWindow, &WindowAnswer)>,
    holds: Query<(Entity, &WaitingForWindow)>,
    mut connections: Query<&mut Connection>,
    mut windows: ResMut<ServedWindows>,
    mut commands: Commands,
) {
    let Ok((AskingWindow(reference), answer)) = asks.get(answered.entity) else {
        return;
    };
    commands.entity(answered.entity).despawn();
    let window = match *answer {
        WindowAnswer::Said(window) => window,
        WindowAnswer::Unreached => {
            windows.0.remove(reference);
            None
        }
    };
    for (hold, WaitingForWindow(waited)) in &holds {
        if waited == reference {
            commands.entity(hold).despawn();
        }
    }
    let Some(window) = window else {
        return;
    };
    windows.0.insert(reference.clone(), Some(window));
    for mut connection in &mut connections {
        if connection.spec.reference() == *reference && connection.spec.input_limit().is_none() {
            connection.spec = with_window(&connection.spec, window);
        }
    }
}

/// A hold on a turn's model call while the server of its model, named by
/// reference, is asked for the model's window.
#[derive(Component)]
pub(crate) struct WaitingForWindow(String);

/// Names the model whose window a turn last waited for, so a failed ask
/// holds no more of the turn's calls while it stays on that model.
#[derive(Component)]
pub(crate) struct WaitedForWindow(String);

/// Holds the calls about to start on their agent's own connection, to a
/// model with no context window, while that model's server is asked for one,
/// in [`Stage::Gate`](super::stage::Stage::Gate), asking it again first when
/// the last ask did not reach it: measured without the window, a request
/// would not be compacted near it. A turn does not wait again for the model
/// it last waited for, though moving to another model and back can, and its
/// calls ready together wait for the same ask. A call measured without
/// a window its model now has is built anew, once for that model, from its
/// agent. A call a plugin routed to a connection of its own, or left behind
/// by its agent's switch to another model, goes out as it is.
pub(crate) fn wait_for_window(
    calls: Query<
        (
            Entity,
            &Adapted,
            &ModelRequest,
            &CallOf,
            &ForAgent,
            &Effect,
            Option<&BuiltWithWindow>,
        ),
        (ReadyToStart, With<TurnCall>),
    >,
    agents: Query<&Connection, With<Agent>>,
    waited: Query<&WaitedForWindow>,
    asks: Query<&AskingWindow>,
    (models, wake): (Res<Models>, Res<Wake>),
    mut windows: ResMut<ServedWindows>,
    mut commands: Commands,
) {
    let mut started = Vec::new();
    for (call, Adapted(adapted), model, &CallOf(turn), &ForAgent(agent), effect, built) in &calls {
        let Ok(own) = agents.get(agent) else {
            continue;
        };
        let reference = own.spec.reference();
        // Connecting makes a new handler, so a connection a plugin made to
        // this same model does not share the agent's.
        let on_own = adapted.spec.reference() == reference
            && model
                .connection
                .as_ref()
                .is_none_or(|to| to.handler.0.ptr_eq(&own.handler.0));
        let built_for_it = built.is_some_and(|BuiltWithWindow(built)| *built == reference);
        if !on_own || built_for_it || adapted.spec.input_limit().is_some() {
            continue;
        }
        if windows.0.get(&reference).is_some_and(Option::is_some) {
            commands
                .spawn((
                    TurnCall,
                    CallOf(turn),
                    ForAgent(agent),
                    Effect::with_parent_id(effect.parent),
                    BuiltWithWindow(reference),
                ))
                .add_related::<SupersededBy>(&[call]);
            continue;
        }
        let turn_waited = waited
            .get(turn)
            .is_ok_and(|WaitedForWindow(waited)| *waited == reference);
        if own.spec.input_limit().is_some() || turn_waited {
            continue;
        }
        if start_asking(&own.spec, &models, &wake, &mut windows, &mut commands) {
            started.push(reference.clone());
        }
        let asking = started.contains(&reference)
            || asks.iter().any(|AskingWindow(asked)| *asked == reference);
        if !asking {
            continue;
        }
        commands.spawn((
            Name::new(format!("waiting for the context window of {reference}")),
            WaitingForWindow(reference.clone()),
            HoldOf(call),
        ));
        commands.entity(turn).try_insert(WaitedForWindow(reference));
    }
}

/// On a turn's call [`wait_for_window`] built anew, naming the model it was
/// built for, so a plugin that keeps a connection with no window on the call
/// cannot make it build the call for that model forever.
#[derive(Component)]
pub(crate) struct BuiltWithWindow(String);

/// `spec` with a context window of `window`.
fn with_window(spec: &ModelSpec, window: u32) -> Arc<ModelSpec> {
    Arc::new(spec.clone().with_context_window(window))
}

/// The notice that an agent now uses the model of `connection`.
fn switched(connection: &Connection) -> String {
    let spec = &connection.spec;
    format!("Model: {} ({}).", spec.display_name, spec.reference())
}

/// `effort`, set on the model `from`, for the model `to`: by the name
/// `from` gives it, so a budget `from` calls `high` is `to`'s `high`. Kept
/// as it is when `to` takes no setting by that name; [`check_effort`] then
/// resets it out loud.
fn carried(from: &ModelSpec, effort: ReasoningSetting, to: &ModelSpec) -> ReasoningSetting {
    let name = from.reasoning.name_of(effort.0);
    to.reasoning.parse(&name).map_or(effort, ReasoningSetting)
}

/// Connects each agent whose [`ModelChoice`] is not connected once the
/// [`Models`] changed, such as a model a plugin found or added to the
/// catalog after the choice was restored, quietly as [`connect`] does for a
/// restored choice. A choice still not usable is left as it is: its
/// refusal was shown when it was made.
pub(crate) fn reconnect(
    agents: Query<(Entity, &ModelChoice), (With<Agent>, Without<Connection>)>,
    models: Res<Models>,
    mut commands: Commands,
) {
    for (agent, choice) in &agents {
        if let Ok(connection) = models.connect(&choice.0) {
            let mut agent = commands.entity(agent);
            agent.try_insert(connection).try_remove::<ModelRefused>();
        }
    }
}

impl Connection {
    /// Why this model cannot take `effort`: its catalog entry refuses it
    /// ([`ModelSpec::validate`]), or its route does, as the handler's check
    /// says with nothing sent. `None` when it takes it.
    fn refuses(&self, effort: Option<Reasoning>) -> Option<UnsupportedOption> {
        let options = self.spec.default_options(effort);
        if let Err(refusal) = self.spec.validate(&options) {
            return Some(refusal);
        }
        let kind = EffectKind::Completion {
            request: CompletionRequest::new("").options(options).into(),
            stream: true,
        };
        let refused = self.handler.0.check(&kind);
        refused
            .into_iter()
            .find(|refusal| refusal.option == "reasoning")
    }
}

/// Resets a reasoning setting the agent's model does not take, whichever
/// of the two was inserted last, with a warning.
pub(crate) fn check_effort(
    inserted: On<Insert<(Connection, ReasoningSetting)>>,
    agents: Query<(&Connection, &ReasoningSetting)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = inserted.entity;
    let Ok((connection, effort)) = agents.get(agent) else {
        return;
    };
    if let Some(refusal) = connection.refuses(effort.0) {
        commands.entity(agent).insert(ReasoningSetting(None));
        notices.write(Notice::warn(
            agent,
            format!("Reasoning reset to default: {refusal}."),
        ));
    }
}

/// Sets the agent's reasoning setting after checking it against the model,
/// and says it was [`ChoiceTaken`].
pub(crate) fn on_set_reasoning(
    set: On<SetReasoning>,
    agents: Query<(Option<&Connection>, Has<ActiveTurn>), With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((connection, busy)) = agents.get(set.entity) else {
        return;
    };
    if busy {
        notices.write(Notice::turn_running(set.entity));
        return;
    }
    let Some(connection) = connection else {
        notices.write(Notice::no_model(set.entity));
        return;
    };
    match connection.refuses(set.effort.0) {
        None => {
            commands.entity(set.entity).insert(set.effort);
            commands.trigger(ChoiceTaken {
                entity: set.entity,
                kind: set.kind,
                model: None,
            });
            notices.write(Notice::info(
                set.entity,
                format!(
                    "Reasoning: {}.",
                    connection.spec.reasoning.name_of(set.effort.0)
                ),
            ));
        }
        Some(refusal) => {
            notices.write(Notice::error(set.entity, format!("{refusal}.")));
        }
    }
}

#[cfg(test)]
mod tests;
