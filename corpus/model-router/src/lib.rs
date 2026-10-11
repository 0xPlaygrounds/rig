//! A virtual model, `router`, that sends each model request of an agent to
//! one of a few configured models: `/model router` (also in the `/model`
//! picker) turns it on for the agent, and choosing any other model turns it
//! off. The rules come from `RIG_HOME/router.json`; the first that matches
//! a request picks its target, so one turn may mix models:
//!
//! ```json
//! {
//!   "targets": { "cheap": "deepseek/deepseek-flash", "strong": "deepseek/deepseek-v4-pro" },
//!   "rules": [
//!     { "when": { "context_over": 60000 }, "use": "strong" },
//!     { "when": { "first_request_mentions": ["plan", "design", "why", "debug"] }, "use": "strong" },
//!     { "when": "after_read_only_tools", "use": "cheap" },
//!     { "when": "always", "use": "default" }
//!   ]
//! }
//! ```
//!
//! A routed request goes to its target as the turn's [`Connection`], so the
//! effect log records it under the real model; the agent's reasoning
//! setting goes along when the target takes it, else the target's default.
//! Each agent's [`RoutedSpending`] sums its calls by real model (`/router`
//! lists it), and the status line says `router → cheap (deepseek-flash)`.
//! A rule whose target is not configured or cannot be connected is off,
//! with a warning at startup.

use std::collections::BTreeMap;
use std::mem;

use bevy_ecs::message::Messages;
use bevy_ecs::system::SystemId;
use bevy_log::warn;
use rig_core::completion::UsageTotals;
use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_tui::{PickItem, PickRequest, Side, StatusItem, StatusItems, StatusSystems, Tone};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The virtual model's name, as `/model router` takes it.
pub const ROUTER: &str = "router";
/// The file in `RIG_HOME` with the rules.
const CONFIG: &str = "router.json";
/// Where the routed model is: right after the agent's model.
const ROUTE: StatusItem = StatusItem::at(Side::Left, 35, 14);

/// Adds the `router` virtual model and `/router`.
#[derive(Default)]
pub struct ModelRouterPlugin;

impl Plugin for ModelRouterPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<RouterConfig>() {
            // Only an agent the launcher started reads `RIG_HOME`, not a test.
            let config = app
                .world()
                .get_resource::<Invoked>()
                .and_then(|_| RouterConfig::read())
                .unwrap_or_default();
            app.insert_resource(config);
        }
        app.init_resource::<RouterTargets>()
            .init_resource::<ModelCommand>()
            .add_message::<PickRequest>()
            .register_required_components::<Agent, Routing>()
            .register_required_components::<Agent, RoutedSpending>()
            .add_command(
                "router",
                "Show the model router's rules and what each model it used spent",
                router,
            )
            .add_systems(Startup, connect_targets)
            .add_systems(PostUpdate, show.in_set(StatusSystems))
            .add_observer(start_turn)
            .add_observer(route)
            .add_observer(record)
            .add_observer(leave_router);
    }

    /// Wraps `/model` (of `rig-models`, if added) so it also takes
    /// `router` and lists it in its picker.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut commands = world.query_filtered::<(Entity, &Name), With<SlashCommand>>();
        let original = commands
            .iter(world)
            .find(|(_, name)| name.as_str() == "/model")
            .map(|(entity, _)| entity);
        if let Some(original) = original {
            world.entity_mut(original).insert(Disabled);
        }
        world.insert_resource(ModelCommand(original));
        app.add_command(
            "model",
            "Pick the model, set it with /model vendor/model, or route each request with /model router",
            model,
        );
    }
}

/// The router's targets and rules, from `RIG_HOME/router.json`. Without a
/// file it has the usual rules and no targets, so every rule is off.
#[derive(Resource, Clone, Debug, Deserialize)]
pub struct RouterConfig {
    /// Each target's catalog model, by the name the rules use.
    #[serde(default)]
    pub targets: BTreeMap<String, String>,
    /// The rules, first match wins.
    #[serde(default)]
    pub rules: Vec<Rule>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        let rule = |when, target: &str| Rule {
            when,
            target: target.to_owned(),
        };
        let words = ["plan", "design", "why", "debug"]
            .map(str::to_owned)
            .to_vec();
        Self {
            targets: BTreeMap::new(),
            rules: vec![
                rule(When::ContextOver(60_000), "strong"),
                rule(When::FirstRequestMentions(words), "strong"),
                rule(When::AfterReadOnlyTools, "cheap"),
                rule(When::Always, "default"),
            ],
        }
    }
}

impl RouterConfig {
    fn read() -> Option<Self> {
        let path = Home::from_env().root().join(CONFIG);
        let text = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&text)
            .inspect_err(|why| warn!("router: {} is not read: {why}", path.display()))
            .ok()
    }
}

/// One routing rule: when it matches, the request goes to `target`.
#[derive(Clone, Debug, Deserialize)]
pub struct Rule {
    /// When it matches.
    pub when: When,
    /// The target's name.
    #[serde(rename = "use")]
    pub target: String,
}

/// What a rule matches.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum When {
    /// The agent's context, as its last reply reported it, is over this
    /// many tokens.
    ContextOver(u64),
    /// The first request of a turn, whose user text contains one of these
    /// words (any case).
    FirstRequestMentions(Vec<String>),
    /// A request after tool calls that were all read-only.
    AfterReadOnlyTools,
    /// Any request.
    Always,
}

impl std::fmt::Display for When {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ContextOver(tokens) => write!(f, "context over {tokens} tokens"),
            Self::FirstRequestMentions(words) => write!(f, "turn mentions {}", words.join("|")),
            Self::AfterReadOnlyTools => f.write_str("after read-only tools"),
            Self::Always => f.write_str("otherwise"),
        }
    }
}

/// The targets that could be connected, by name. A test or a plugin may
/// insert a target before startup; startup connects the rest.
#[derive(Resource, Clone, Default)]
pub struct RouterTargets(pub BTreeMap<String, Connection>);

/// Whether the agent's requests are routed: its model is `router`. Saved
/// with the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct Routing(pub bool);

/// Where the agent's last routed request went, for the status line.
#[derive(Component, Clone, Debug)]
pub struct LastRoute(pub String);

/// The agent's model calls' usage, by the real model that answered each.
/// Saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct RoutedSpending(pub BTreeMap<String, UsageTotals>);

/// `rig-models`' `/model`, disabled and run by this plugin's.
#[derive(Resource, Default)]
struct ModelCommand(Option<Entity>);

/// Connects each target a rule uses, unless one was given, and warns of
/// each rule that is off.
fn connect_targets(
    config: Res<RouterConfig>,
    models: Res<Models>,
    mut targets: ResMut<RouterTargets>,
    mut notices: MessageWriter<Notice>,
) {
    for rule in &config.rules {
        if targets.0.contains_key(&rule.target) {
            continue;
        }
        let connected = match config.targets.get(&rule.target) {
            Some(reference) => models.connect(reference),
            None => Err(format!("`{}` is not in {CONFIG}", rule.target)),
        };
        match connected {
            Ok(connection) => {
                targets.0.insert(rule.target.clone(), connection);
            }
            Err(why) => {
                let text = format!(
                    "Router rule \"{} → {}\" is off: {why}",
                    rule.when, rule.target
                );
                warn!("{text}");
                notices.write(Notice::info(None, text));
            }
        }
    }
}

/// `/model router` routes the agent's requests; anything else is
/// `rig-models`' `/model`, whose picker gets `router` first.
fn model(In(args): In<CommandArgs>, world: &mut World) {
    let agent = args.agent;
    if args.args == ROUTER {
        choose_router(world, agent);
        return;
    }
    let Some(original) = world.resource::<ModelCommand>().0 else {
        if !args.args.is_empty() {
            world.trigger(SetModel {
                entity: agent,
                model: args.args,
            });
        }
        return;
    };
    let earlier = swap_picks(world, Messages::default());
    let ran = world.run_system_with(SystemId::<In<CommandArgs>>::from_entity(original), args);
    let mut asked = swap_picks(world, earlier);
    if let Err(why) = ran {
        world.write_message(Notice::error(agent, why.to_string()));
    }
    for mut pick in asked.drain() {
        let item = PickItem {
            label: format!("{ROUTER}  a model per request, by {CONFIG}"),
            command: format!("model {ROUTER}"),
        };
        pick.items.insert(0, item);
        world.write_message(pick);
    }
}

/// Puts `picks` in place of the world's picker requests, and returns those.
fn swap_picks(world: &mut World, picks: Messages<PickRequest>) -> Messages<PickRequest> {
    match world.get_resource_mut::<Messages<PickRequest>>() {
        Some(mut current) => mem::replace(&mut *current, picks),
        None => picks,
    }
}

/// Turns routing on for an idle agent with a reachable target.
fn choose_router(world: &mut World, agent: Entity) {
    if world.get::<ActiveTurn>(agent).is_some() {
        world.write_message(Notice::turn_running(agent));
        return;
    }
    if world.resource::<RouterTargets>().0.is_empty() {
        let why = format!("No router target can be used: set them in RIG_HOME/{CONFIG}.");
        world.write_message(Notice::error(agent, why));
        return;
    }
    world.entity_mut(agent).insert(Routing(true));
    world.write_message(Notice::info(agent, "Model: router."));
}

/// Choosing a real model leaves the router.
fn leave_router(set: On<SetModel>, mut agents: Query<&mut Routing, Without<ActiveTurn>>) {
    if let Ok(mut routing) = agents.get_mut(set.entity) {
        routing.set_if_neq(Routing(false));
    }
}

/// A routed turn of an agent without a model of its own starts on a target,
/// so the kernel can prepare its first request; [`route`] picks the rest.
fn start_turn(
    turn: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    agents: Query<(&Routing, Has<Connection>)>,
    targets: Res<RouterTargets>,
    mut commands: Commands,
) {
    let Ok(TurnOf(agent)) = turns.get(turn.entity) else {
        return;
    };
    if let Ok((Routing(true), false)) = agents.get(*agent)
        && let Some(connection) = targets.0.values().next()
    {
        commands.entity(turn.entity).insert(connection.clone());
    }
}

/// Picks the target of each request of a routed agent by the first rule
/// that matches, sends the request there, and translates the reasoning
/// setting for it.
fn route(
    mut prepare: On<PrepareRequest>,
    agents: Query<(&Routing, &LastUsage, &Effort)>,
    tools: Query<(&ToolDef, &Footprint)>,
    config: Res<RouterConfig>,
    targets: Res<RouterTargets>,
    mut commands: Commands,
) {
    let (turn, agent) = (prepare.entity, prepare.agent);
    let Ok((Routing(true), last, effort)) = agents.get(agent) else {
        return;
    };
    let read_only = |name: &str| {
        tools.iter().any(|(def, footprint)| {
            def.0.name.as_str() == name && *footprint == Footprint::ReadOnly
        })
    };
    let request = Request::of(&prepare.messages, read_only);
    let chosen = config.rules.iter().find_map(|rule| {
        let target = targets.0.get(&rule.target)?;
        request
            .matches(&rule.when, last)
            .then_some((&rule.target, target))
    });
    let Some((name, connection)) = chosen else {
        return;
    };
    let spec = &connection.spec;
    let wanted = spec.default_options(effort.0);
    let fitting = match spec.validate(&wanted) {
        Ok(()) => wanted,
        Err(_) => spec.default_options(None),
    };
    prepare.options.reasoning = fitting.reasoning;
    prepare.options.cache = fitting.cache;
    if !spec.tools {
        prepare.tools.clear();
    }
    commands.entity(turn).insert(connection.clone());
    commands
        .entity(agent)
        .insert(LastRoute(format!("{name} ({})", spec.id)));
}

/// What a request is, for the rules.
struct Request {
    /// The user text of a turn's first request, lowercased; `None` for a
    /// later request.
    first: Option<String>,
    /// Whether it follows tool calls that were all read-only.
    after_read_only: bool,
}

impl Request {
    fn of(messages: &[Message], read_only: impl Fn(&str) -> bool) -> Self {
        let mut back = messages.iter().rev();
        let (last, before) = (back.next(), back.next());
        let Some(Message::User { content }) = last else {
            return Self {
                first: None,
                after_read_only: false,
            };
        };
        let results = content
            .iter()
            .any(|item| matches!(item, UserContent::ToolResult(_)));
        let first = (!results).then(|| {
            let texts = content.iter().filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.to_lowercase()),
                _ => None,
            });
            texts.collect::<Vec<_>>().join("\n")
        });
        let after_read_only = results
            && matches!(before, Some(Message::Assistant(reply))
                if reply.tool_calls().next().is_some()
                    && reply.tool_calls().all(|call| read_only(call.function.name.as_str())));
        Self {
            first,
            after_read_only,
        }
    }

    fn matches(&self, when: &When, last: &LastUsage) -> bool {
        match when {
            When::ContextOver(tokens) => last.context().is_some_and(|used| used > *tokens),
            When::FirstRequestMentions(words) => self
                .first
                .as_ref()
                .is_some_and(|text| words.iter().any(|word| text.contains(&word.to_lowercase()))),
            When::AfterReadOnlyTools => self.after_read_only,
            When::Always => true,
        }
    }
}

/// Adds a finished model call's usage under the real model that answered
/// it: the call's connection, else its turn's, else its agent's.
fn record(
    done: On<Add<Done<ModelReply>>>,
    calls: Query<(&CallOf, &Done<ModelReply>, Option<&Connection>)>,
    turns: Query<(&TurnOf, Option<&Connection>)>,
    mut agents: Query<(Option<&Connection>, &mut RoutedSpending)>,
) {
    let Ok((&CallOf(turn), Done(Ok(response)), own)) = calls.get(done.entity) else {
        return;
    };
    let Ok((&TurnOf(agent), routed)) = turns.get(turn) else {
        return;
    };
    let Ok((agents_own, mut spent)) = agents.get_mut(agent) else {
        return;
    };
    if let Some(connection) = own.or(routed).or(agents_own) {
        let model = connection.spec.reference();
        spent.0.entry(model).or_default().record(&response.usage);
    }
}

/// `/router`: whether the agent is routed, each rule, and what each model
/// spent.
fn router(
    In(args): In<CommandArgs>,
    agents: Query<(&Routing, &RoutedSpending)>,
    config: Res<RouterConfig>,
    targets: Res<RouterTargets>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((Routing(on), RoutedSpending(spent))) = agents.get(args.agent) else {
        return;
    };
    let mut lines = vec![format!(
        "Router: {}.",
        if *on { "on" } else { "off (/model router)" }
    )];
    for rule in &config.rules {
        let target = match targets.0.get(&rule.target) {
            Some(connection) => connection.spec.reference(),
            None => "off".to_owned(),
        };
        lines.push(format!("  {} → {} ({target})", rule.when, rule.target));
    }
    for (model, totals) in spent {
        lines.push(format!("  {model}: {totals}"));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// `router → cheap (deepseek-flash)` in a routed agent's status line.
fn show(
    mut agents: Query<
        (&Routing, Option<&LastRoute>, &mut StatusItems),
        Or<(Changed<Routing>, Changed<LastRoute>)>,
    >,
) {
    for (Routing(on), last, mut items) in &mut agents {
        let text = match (on, last) {
            (false, _) => String::new(),
            (true, Some(LastRoute(route))) => format!("{ROUTER} → {route}"),
            (true, None) => ROUTER.to_owned(),
        };
        items.show(ROUTE.says(text, Tone::Dim));
    }
}
