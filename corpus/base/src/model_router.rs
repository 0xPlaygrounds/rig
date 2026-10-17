//! A virtual model, `router`, that sends each model request of an agent to
//! one of a few configured models. `/model router` (also in the `/model`
//! picker, through `rig_basics::models`' [`ModelItem`]) turns it on for the agent,
//! and choosing any other model turns it off. The rules are the `[router]`
//! table of `settings.toml`; the first that matches a request picks its
//! target, so one turn may mix models:
//!
//! ```toml
//! [router.targets]
//! cheap = "deepseek/deepseek-flash"
//! strong = "deepseek/deepseek-v4-pro"
//! default = "deepseek/deepseek-v4-pro"
//!
//! [[router.rules]]
//! when = { context_over = 60000 }
//! use = "strong"
//!
//! [[router.rules]]
//! when = { first_request_mentions = ["plan", "design", "why", "debug"] }
//! use = "strong"
//!
//! [[router.rules]]
//! when = "after_read_only_tools"
//! use = "cheap"
//!
//! [[router.rules]]
//! when = "always"
//! use = "default"
//! ```
//!
//! A routed request is a Prepare edit of its `connection`, so the kernel
//! sends it there, fits its options to that model (a reasoning setting the
//! target does not take falls back to the target's default), and the
//! effect log and `rig_basics::telemetry`'s per-model spending record it under the
//! real model. The agent's [`ModelLabel`] says `router → deepseek-flash`
//! in the status line. A rule whose target is not configured or cannot be
//! connected is off, with a warning at startup; once the catalog changes,
//! such as a provider plugin adding the model, it is connected again.
//!
//! The routing runs in [`RouteSystems`], before compaction measures the
//! request against the model it goes to. A call that takes over another
//! (`Supersedes`), such as a fallback's, keeps the model the plugin that
//! spawned it chose.

use std::collections::BTreeMap;

use bevy_log::warn;
use rig_basics::models::{ModelItem, ModelLabel};
use rig_harness::prelude::message::Message;
use rig_harness::prelude::*;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The virtual model's name, as `/model router` takes it.
pub const ROUTER: &str = "router";

/// The router's Prepare system, which sets each routed call's
/// `connection`: a plugin that edits a request for the model it goes to
/// runs after it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteSystems;

/// Adds the `router` virtual model.
#[derive(Default)]
pub struct ModelRouterPlugin;

impl Plugin for ModelRouterPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<RouterConfig>("router")
            .init_resource::<Targets>()
            .register_required_components::<Agent, Routing>()
            .add_command(
                ROUTER,
                "Route each model request by the [router] rules of settings.toml",
                router,
            )
            .configure_sets(AgentPass, RouteSystems.in_set(PrepareSystems::Edit))
            .add_systems(
                PreUpdate,
                connect_targets.run_if(resource_changed::<Models>),
            )
            .add_systems(AgentPass, route.in_set(RouteSystems))
            .add_observer(label)
            .add_observer(leave.run_if(resource_exists::<SessionRestored>));
        let item = ModelItem::new(ROUTER, "router  a model per request, by [router]", ROUTER);
        app.world_mut().spawn(item);
    }
}

/// The `[router]` table: the targets and the rules. Without one it has the
/// usual rules and no targets, so every rule is off.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouterConfig {
    /// Each target's catalog model, by the name the rules use.
    pub targets: BTreeMap<String, String>,
    /// The rules, first match wins.
    pub rules: Vec<Rule>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        let rule = |when, target: &str| Rule {
            when,
            target: target.to_owned(),
        };
        let words = ["plan", "design", "why", "debug"].map(str::to_owned);
        Self {
            targets: BTreeMap::new(),
            rules: vec![
                rule(When::ContextOver(60_000), "strong"),
                rule(When::FirstRequestMentions(words.to_vec()), "strong"),
                rule(When::AfterReadOnlyTools, "cheap"),
                rule(When::Always, "default"),
            ],
        }
    }
}

/// One routing rule: when it matches, the request goes to `target`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// The first request of a turn whose user text contains one of these
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

/// The targets that could be connected, by name.
#[derive(Resource, Clone, Default)]
pub struct Targets(pub BTreeMap<String, Connection>);

/// Whether the agent's requests are routed: its model is `router`. Saved
/// with the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct Routing(pub bool);

/// Connects each target a rule uses, whenever the catalog changed, and
/// warns of each rule that is off the first time, at startup.
fn connect_targets(
    config: Res<RouterConfig>,
    models: Res<Models>,
    mut targets: ResMut<Targets>,
    mut notices: MessageWriter<Notice>,
    mut started: Local<bool>,
) {
    let warn_off = !std::mem::replace(&mut *started, true);
    targets.0.clear();
    for rule in &config.rules {
        if targets.0.contains_key(&rule.target) {
            continue;
        }
        let connected = match config.targets.get(&rule.target) {
            Some(reference) => models.connect(reference),
            None => Err(format!("`{}` is not in [router.targets]", rule.target)),
        };
        match connected {
            Ok(connection) => {
                targets.0.insert(rule.target.clone(), connection);
            }
            Err(why) if warn_off => {
                let text = format!(
                    "Router rule \"{} → {}\" is off: {why}",
                    rule.when, rule.target
                );
                warn!("{text}");
                notices.write(Notice::info(None, text));
            }
            Err(_) => {}
        }
    }
}

/// `/router` (and `/model router`): routes the agent's requests, and says
/// by which rules.
fn router(
    In(args): In<CommandArgs>,
    config: Res<RouterConfig>,
    targets: Res<Targets>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if targets.0.is_empty() {
        let why = "No router target can be used: set them in [router.targets] of settings.toml.";
        notices.write(Notice::error(args.agent, why));
        return;
    }
    commands.entity(args.agent).insert(Routing(true));
    let mut lines = vec!["Model: router.".to_owned()];
    for rule in &config.rules {
        let model = targets.0.get(&rule.target);
        let model = model.map_or_else(|| "off".to_owned(), |target| target.spec.reference());
        lines.push(format!("  {} → {} ({model})", rule.when, rule.target));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// A routed agent's model is `router`, also once restored.
fn label(inserted: On<Insert<Routing>>, agents: Query<&Routing>, mut commands: Commands) {
    if let Ok(Routing(true)) = agents.get(inserted.entity) {
        commands
            .entity(inserted.entity)
            .insert(ModelLabel(ROUTER.to_owned()));
    }
}

/// Choosing a model, after the session was restored, leaves the router.
fn leave(inserted: On<Insert<ModelChoice>>, agents: Query<&Routing>, mut commands: Commands) {
    if let Ok(Routing(true)) = agents.get(inserted.entity) {
        commands
            .entity(inserted.entity)
            .insert(Routing(false))
            .remove::<ModelLabel>();
    }
}

/// Sends each turn request of a routed agent to the target of the first
/// rule that matches it, unless it took over another call.
fn route(
    mut calls: Query<
        (&mut ModelRequest, &CallOf, &ForAgent),
        (
            Added<Admitted>,
            ModelCalls,
            With<TurnCall>,
            Without<Supersedes>,
        ),
    >,
    agents: Query<(&Routing, &Conversation, &LastUsage, Option<&ModelLabel>)>,
    turns: Query<&Turn>,
    tools: Query<(&ToolDef, &Footprint)>,
    config: Res<RouterConfig>,
    targets: Res<Targets>,
    mut commands: Commands,
) {
    for (mut call, &CallOf(turn), &ForAgent(agent)) in &mut calls {
        let Ok((Routing(true), conversation, last, label)) = agents.get(agent) else {
            continue;
        };
        let first = turns
            .get(turn)
            .map_or(0, |turn| conversation.start_of(turn.start));
        let this_turn = conversation.messages().get(first..).unwrap_or_default();
        let read_only = |name: &str| {
            tools
                .iter()
                .any(|(def, footprint)| def.0.name.as_str() == name && footprint.0.reads_only())
        };
        let request = Request::of(this_turn, read_only);
        let chosen = config.rules.iter().find_map(|rule| {
            let target = targets.0.get(&rule.target)?;
            request.matches(&rule.when, last).then_some(target)
        });
        let Some(target) = chosen else {
            continue;
        };
        call.connection = Some(target.clone());
        let shown = ModelLabel(format!("{ROUTER} → {}", target.spec.id));
        if label != Some(&shown) {
            commands.entity(agent).insert(shown);
        }
    }
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
    /// The request after `this_turn`, the messages of its turn so far.
    fn of(this_turn: &[std::sync::Arc<Message>], read_only: impl Fn(&str) -> bool) -> Self {
        let mut replies = this_turn.iter().filter_map(|message| match &**message {
            Message::Assistant(reply) => Some(reply),
            _ => None,
        });
        let Some(reply) = replies.next_back() else {
            let text = this_turn.iter().flat_map(|message| match &**message {
                Message::User { content } => content
                    .iter()
                    .filter_map(|item| match item {
                        UserContent::Text(text) => Some(text.text.to_lowercase()),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            });
            return Self {
                first: Some(text.collect::<Vec<_>>().join("\n")),
                after_read_only: false,
            };
        };
        let mut called = reply.tool_calls().peekable();
        let after_read_only =
            called.peek().is_some() && called.all(|call| read_only(call.function.name.as_str()));
        Self {
            first: None,
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
