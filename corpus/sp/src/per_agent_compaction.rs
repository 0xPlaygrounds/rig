//! Per-agent compaction rules: `compaction.toml` (the project's
//! `.rig/compaction.toml`, then the user's `RIG_HOME/compaction.toml`)
//! holds `[[rule]]`s matched by an agent's name (`main` for the agent the
//! user talks to, a subagent's task title) or a subagent's type (`task`'s
//! `type` argument), the first match winning. A matched agent is compacted
//! by its rule's strategy when its next request passes `trigger_percent` of
//! its model's window; an agent no rule matches keeps rig-compaction's
//! built-in behaviour.
//!
//! - `summary`: rig-compaction summarizes the older turns, its knobs set
//!   from the rule ([`CompactionKnobs`]), and the summary call goes to the
//!   rule's `model`, with `prompt_file` as its system prompt.
//! - `drop-tool-output`: the tool results older than `keep_recent_turns`
//!   turns become the rule's `stub`, keeping every call and text; no model
//!   is called.
//! - `never`: a request that would not fit the window ends the turn with
//!   an error, which a subagent reports to its parent.
//!
//! Each compaction leaves an aside of kind [`RECORD`] after it, saved with
//! the conversation; `/compaction` lists the live agents with their rule,
//! fill and last compaction, and `/compact` follows the focused agent's
//! rule. The files are watched, so a change applies from each agent's next
//! request.

use std::path::{Path, PathBuf};

use rig_compaction::{Compact, CompactionCuts, CompactionKnobs, CompactionPlugin, Summarizing};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{Message, Usage};
use rig_harness::rig_core::message::ToolResultContent;
use rig_harness::rig_memory::{
    ClearToolOutputs, CompactionOverrides, Cut, SummaryLimits, SummaryPrompts, estimate_tokens,
};
use rig_harness::rig_tools::workspace::Listing as Files;
use rig_harness::settings::Settings;
use rig_subagents::{SpawnSubagent, SpawnedAgent, TASK};
use serde::{Deserialize, Serialize};

/// The rules file's name, in `RIG_HOME` and in the project's `.rig`.
pub const FILE: &str = "compaction.toml";

/// The kind of the aside each compaction of a matched agent leaves.
pub const RECORD: &str = "per-agent-compaction";

/// The plugin's name, as the author of its edits.
const NAME: &str = "per-agent-compaction";

/// What a stub says when the rule names none.
const STUB: &str = "[tool output dropped to save context]";

/// Compacts each agent by the first `compaction.toml` rule that matches it.
#[derive(Default)]
pub struct PerAgentCompactionPlugin;

/// The plugin's Prepare systems, in [`PrepareSystems::Edit`]: before
/// rig-compaction measures the request in [`PrepareSystems::Fit`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PerAgentCompactionSystems;

impl Plugin for PerAgentCompactionPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CompactionPlugin>() {
            app.add_plugins(CompactionPlugin);
        }
        app.init_resource::<CompactionRules>()
            .register_type::<SubagentType>()
            .keeps_saved::<SubagentType>()
            .configure_sets(
                AgentPass,
                PerAgentCompactionSystems.in_set(PrepareSystems::Edit),
            )
            .add_systems(
                AgentPass,
                (fit_by_rule, route_summaries)
                    .chain()
                    .in_set(PerAgentCompactionSystems),
            )
            .add_systems(Startup, watch_rules)
            .add_systems(Update, (reload_rules, record_summaries.after(AgentLoop)))
            .add_observer(rules_loaded)
            .add_observer(type_subagent);
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let mut named = world.query::<(Entity, &Name)>();
        let taken: Vec<Entity> = named
            .iter(world)
            .filter(|(_, name)| matches!(name.as_str(), "/compact" | "/compaction"))
            .map(|(entity, _)| entity)
            .collect();
        for entity in taken {
            world.entity_mut(entity).insert(Disabled);
        }
        let mut tools = world.query::<(&mut ToolDef, &mut ToolRules)>();
        for (mut def, mut rules) in tools.iter_mut(world) {
            if def.0.name.as_str() != TASK {
                continue;
            }
            if let Some(properties) = std::sync::Arc::make_mut(&mut def.0)
                .parameters
                .get_mut("properties")
                && let Some(properties) = properties.as_object_mut()
            {
                let kind = serde_json::json!({
                    "type": "string",
                    "description": "The kind of subagent, such as explorer or reviewer."
                });
                properties.insert("type".to_owned(), kind);
            }
            let rule =
                "Give `task` a `type` (explorer, reviewer, …) saying what kind of work it is.";
            rules.0.push(rule.to_owned());
        }
        app.add_command(
            "compact",
            "Compact the agent now by its compaction.toml rule; /compact <focus> for a summary",
            compact_now,
        )
        .add_command(
            "compaction",
            "List the agents with their compaction rule, fill and last compaction",
            list_agents,
        );
    }
}

/// How a rule compacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Strategy {
    /// Older turns become one summary by the rule's model.
    Summary,
    /// Older tool results become a stub.
    DropToolOutput,
    /// Nothing is compacted: a request that would not fit fails the turn.
    Never,
}

impl Strategy {
    /// Its name in `compaction.toml`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Summary => "summary",
            Self::DropToolOutput => "drop-tool-output",
            Self::Never => "never",
        }
    }
}

/// A `[[rule]]` of `compaction.toml`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    /// What `/compaction` and the records call it; by default what it matches.
    #[serde(default)]
    pub name: Option<String>,
    /// The agent name it matches: `main`, or a subagent's title.
    #[serde(default)]
    pub agent: Option<String>,
    /// The subagent type it matches.
    #[serde(default, rename = "type")]
    pub subagent_type: Option<String>,
    /// How it compacts.
    pub strategy: Strategy,
    /// The share of the model's window, in percent, past which a request
    /// is compacted.
    #[serde(default = "full")]
    pub trigger_percent: u8,
    /// For `summary`: the catalog model that writes the summary.
    #[serde(default)]
    pub model: Option<String>,
    /// For `summary`: the summarizer's system prompt, relative to the file.
    #[serde(default)]
    pub prompt_file: Option<PathBuf>,
    /// The newest turns (model replies with the results they asked for)
    /// kept as they are.
    #[serde(default = "one")]
    pub keep_recent_turns: usize,
    /// For `drop-tool-output`: what a dropped result says.
    #[serde(default)]
    pub stub: Option<String>,
    /// The text of `prompt_file`, read with the rule.
    #[serde(skip)]
    pub prompt: Option<String>,
}

fn full() -> u8 {
    100
}

fn one() -> usize {
    1
}

impl Rule {
    /// Its name, else what it matches.
    pub fn label(&self) -> String {
        match (&self.name, &self.agent, &self.subagent_type) {
            (Some(name), ..) => name.clone(),
            (None, Some(agent), _) => format!("agent={agent}"),
            (None, None, Some(kind)) => format!("type={kind}"),
            (None, None, None) => "unnamed".to_owned(),
        }
    }

    /// Whether it matches the agent `name` of the subagent type `kind`.
    fn matches(&self, name: &str, kind: Option<&str>) -> bool {
        let agent = self.agent.as_deref().is_none_or(|agent| agent == name);
        let typed = self
            .subagent_type
            .as_deref()
            .is_none_or(|wanted| kind == Some(wanted));
        agent && typed
    }

    /// The tokens past which a request of a `window`-token model is
    /// compacted.
    fn threshold(&self, window: u32) -> u64 {
        u64::from(window) * u64::from(self.trigger_percent.min(100)) / 100
    }

    /// Why the rule cannot be used, if it cannot.
    fn problem(&self) -> Option<&'static str> {
        if self.agent.is_none() && self.subagent_type.is_none() {
            return Some("it names neither an `agent` nor a `type`");
        }
        if self.strategy == Strategy::Summary && self.model.is_none() {
            return Some("a `summary` rule needs a `model`");
        }
        (self.keep_recent_turns == 0).then_some("`keep_recent_turns` must be at least 1")
    }
}

/// The rules in force: the project's, then the user's.
#[derive(Resource, Clone, Debug, Default)]
pub struct CompactionRules(pub Vec<Rule>);

impl CompactionRules {
    /// The first rule matching the agent `name` of the subagent type `kind`.
    pub fn rule_for(&self, name: &str, kind: Option<&str>) -> Option<&Rule> {
        self.0.iter().find(|rule| rule.matches(name, kind))
    }
}

/// On a subagent: its type, from `task`'s `type`, which rules match.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct SubagentType(pub String);

/// What a compaction of a matched agent did, the value of its [`RECORD`]
/// aside.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompactionRecord {
    /// The strategy.
    pub strategy: Strategy,
    /// The rule's label.
    pub rule: String,
    /// The tokens the next request would have sent.
    pub tokens: u64,
}

impl CompactionRecord {
    /// The newest record in `conversation`.
    pub fn newest(conversation: &Conversation) -> Option<Self> {
        let records = conversation.asides().rev();
        let mut mine = records.filter(|(_, aside)| aside.kind == RECORD);
        mine.find_map(|(_, aside)| serde_json::from_value(aside.value.clone()).ok())
    }

    /// Leaves it in `conversation` after its newest message.
    fn leave(&self, conversation: &mut Conversation, after: Option<MessageId>) {
        let value = serde_json::to_value(self).unwrap_or_default();
        let aside = Aside::new(RECORD, Author::plugin(NAME), value).after(after);
        conversation.add_aside(aside);
    }
}

/// An agent's name as rules match it: a subagent's title, `main` for the
/// agent nothing spawned.
fn agent_name(title: Option<&AgentTitle>, spawned: bool, name: &Name) -> String {
    match (title, spawned) {
        (Some(title), _) => title.0.clone(),
        (None, false) => "main".to_owned(),
        (None, true) => name.as_str().to_owned(),
    }
}

/// What a rule is matched by.
type Identity<'a> = (
    &'a Name,
    Option<&'a AgentTitle>,
    Has<SpawnedBy>,
    Option<&'a SubagentType>,
);

/// An agent's [`Identity`] as read.
type Who<'a> = (
    &'a Name,
    Option<&'a AgentTitle>,
    bool,
    Option<&'a SubagentType>,
);

/// The rule of the agent `who` is.
fn rule_of<'r>(rules: &'r CompactionRules, who: Who) -> Option<&'r Rule> {
    let (name, title, spawned, kind) = who;
    let name = agent_name(title, spawned, name);
    rules.rule_for(&name, kind.map(|kind| kind.0.as_str()))
}

/// The cut rig-compaction gets for keeping `turns` turns.
fn cut_name(turns: usize) -> String {
    format!("{NAME}/keep-{turns}-turns")
}

/// Where the newest `turns` turns of `messages` start: the `turns`-th
/// newest model reply, after the first `from`.
fn turns_start(messages: &[std::sync::Arc<Message>], from: usize, turns: usize) -> Option<usize> {
    let replies = messages.iter().enumerate().skip(from);
    let mut replies = replies.filter(|(_, message)| matches!(&***message, Message::Assistant(_)));
    let starts: Vec<usize> = replies.by_ref().map(|(at, _)| at).collect();
    let at = *starts.iter().rev().nth(turns.saturating_sub(1))?;
    (at > from).then_some(at)
}

/// The messages of `messages` before its newest `turns` turns whose tool
/// results are not `stub` yet, each with its results as `stub`.
fn stubbed(
    messages: &[std::sync::Arc<Message>],
    turns: usize,
    stub: &str,
) -> Vec<(usize, Message)> {
    let Some(cut) = turns_start(messages, 0, turns) else {
        return Vec::new();
    };
    let older = messages.iter().take(cut).enumerate();
    older
        .filter_map(|(at, message)| {
            let Message::User { content } = &**message else {
                return None;
            };
            let mut content = content.clone();
            let mut changed = false;
            for item in content.iter_mut() {
                if let UserContent::ToolResult(result) = item
                    && !matches!(result.content.first(), Some(only) if only.as_text() == Some(stub))
                {
                    result.content = vec![ToolResultContent::text(stub)];
                    changed = true;
                }
            }
            changed.then_some((at, Message::User { content }))
        })
        .collect()
}

/// Puts the stubs of [`stubbed`] in `conversation`, keeping ids.
fn drop_outputs(conversation: &mut Conversation, turns: usize, stub: &str) -> usize {
    let stubs = stubbed(conversation.messages(), turns, stub);
    conversation.edit_each(Author::plugin(NAME), |entry| {
        stubs
            .iter()
            .find(|(at, _)| *at == entry.index)
            .map(|(_, message)| message.clone())
    });
    stubs.len()
}

/// rig-compaction's knobs for an agent with `rule` (none: its own), on a
/// model of a `window`-token window: a `summary` rule's cut, threshold and
/// prompt; any other rule keeps rig-compaction off the agent.
fn knobs(rule: Option<&Rule>, window: Option<u32>) -> CompactionOverrides {
    let off = Some(ClearToolOutputs::new(usize::MAX));
    match rule.map(|rule| (rule, rule.strategy)) {
        None => CompactionOverrides::default(),
        Some((rule, Strategy::Summary)) => {
            let mut limits = SummaryLimits::DEFAULT;
            if let Some(window) = window {
                limits.reserve = u64::from(window) - rule.threshold(window);
                limits.summary_tokens = limits.summary_tokens.min(u64::from(window) / 4);
            }
            let prompts = rule.prompt.clone().map(|system| SummaryPrompts {
                system: system.into(),
                ..SummaryPrompts::DEFAULT
            });
            CompactionOverrides {
                limits: Some(limits),
                clearing: off,
                cut: Some(Cut::Named(cut_name(rule.keep_recent_turns))),
                prompts,
                ..CompactionOverrides::default()
            }
        }
        Some(_) => CompactionOverrides {
            clearing: off,
            per_turn: Some(0),
            ..CompactionOverrides::default()
        },
    }
}

/// Sets rig-compaction's knobs of `agent` to `wanted`, when they differ.
fn set_knobs(
    agent: Entity,
    own: Option<&CompactionKnobs>,
    wanted: CompactionOverrides,
    commands: &mut Commands,
) {
    match own {
        Some(own) if own.0 == wanted => {}
        None if wanted.is_empty() => {}
        _ => {
            commands.entity(agent).insert(CompactionKnobs(wanted));
        }
    }
}

/// Before each turn's request of a matched agent is measured: sets
/// rig-compaction's knobs from its rule, drops older tool output for a
/// `drop-tool-output` rule past its threshold, and fails the turn of a
/// `never` rule whose request would not fit the window.
#[expect(clippy::type_complexity, reason = "a stage system's queries")]
fn fit_by_rule(
    mut calls: Query<
        (&mut ModelRequest, &Adapters, &CallOf, &ForAgent),
        (Added<Admitted>, With<TurnCall>),
    >,
    mut agents: Query<(
        Identity,
        &mut Conversation,
        &mut LastUsage,
        Option<&CompactionKnobs>,
    )>,
    starts: ModelStarts,
    rules: Res<CompactionRules>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (mut model, adapters, &CallOf(turn), &ForAgent(agent)) in &mut calls {
        let Ok((who, mut conversation, mut last, own)) = agents.get_mut(agent) else {
            continue;
        };
        let rule = rule_of(&rules, who);
        let Ok((connection, sent)) = starts.sent(&model, Some(adapters), Some(agent)) else {
            continue;
        };
        let window = connection.spec.context_window;
        set_knobs(agent, own, knobs(rule, window), &mut commands);
        let (Some(rule), Some(window)) = (rule, window) else {
            continue;
        };
        let used = last
            .context()
            .unwrap_or(0)
            .max(estimate_tokens(&sent.chat_history));
        match rule.strategy {
            Strategy::Never if used > u64::from(window) => {
                let why = format!(
                    "the next request (about {used} tokens) would exceed the context limit of \
                     {window} tokens, and rule `{}` never compacts",
                    rule.label()
                );
                commands.entity(turn).insert(TurnOutcome::Failed(why));
            }
            Strategy::DropToolOutput if used > rule.threshold(window) => {
                let stub = rule.stub.as_deref().unwrap_or(STUB);
                let turns = rule.keep_recent_turns;
                if drop_outputs(&mut conversation, turns, stub) == 0 {
                    continue;
                }
                // The request's copy too.
                match &mut model.history {
                    Some(history) => {
                        let mut copy = Conversation(history.clone());
                        drop_outputs(&mut copy, turns, stub);
                        *history = copy.0;
                    }
                    None => {
                        for (at, message) in stubbed(&model.request.chat_history, turns, stub) {
                            if let Some(old) = model.request.chat_history.get_mut(at) {
                                *old = message.into();
                            }
                        }
                    }
                }
                let left = estimate_tokens(conversation.messages());
                *last = LastUsage(Some(Usage::new().total_tokens(left)));
                let record = CompactionRecord {
                    strategy: rule.strategy,
                    rule: rule.label(),
                    tokens: used,
                };
                let newest = conversation.messages().len().checked_sub(1);
                let after = newest.and_then(|at| conversation.id(at));
                record.leave(&mut conversation, after);
                let text = format!("Dropped older tool output by rule `{}`.", rule.label());
                notices.write(Notice::info(agent, text));
            }
            _ => {}
        }
    }
}

/// Sends each summary call of a `summary` agent to its rule's model.
fn route_summaries(
    mut calls: Query<(&ForAgent, &mut ModelRequest), (Added<Admitted>, With<Summarizing>)>,
    agents: Query<Identity>,
    rules: Res<CompactionRules>,
    models: Res<Models>,
    mut notices: MessageWriter<Notice>,
) {
    for (&ForAgent(agent), mut model) in &mut calls {
        let Ok(who) = agents.get(agent) else {
            continue;
        };
        let Some(rule) = rule_of(&rules, who).filter(|rule| rule.strategy == Strategy::Summary)
        else {
            continue;
        };
        let Some(name) = &rule.model else {
            continue;
        };
        match models.connect(name) {
            Ok(connection) => model.connection = Some(connection),
            Err(why) => {
                let text = format!("Rule `{}`: cannot reach `{name}`: {why}", rule.label());
                notices.write(Notice::warn(agent, text));
            }
        }
    }
}

/// Leaves a record after each summary of a matched agent.
fn record_summaries(
    mut committed: MessageReader<ConversationChanged>,
    mut agents: Query<(Identity, &mut Conversation)>,
    rules: Res<CompactionRules>,
) {
    for ConversationChanged { agent, change } in committed.read() {
        let Some(summary) = rig_compaction::summary_in(&change.change) else {
            continue;
        };
        let Ok((who, mut conversation)) = agents.get_mut(*agent) else {
            continue;
        };
        let Some(rule) = rule_of(&rules, who) else {
            continue;
        };
        let record = CompactionRecord {
            strategy: Strategy::Summary,
            rule: rule.label(),
            tokens: 0,
        };
        record.leave(&mut conversation, Some(summary));
    }
}

/// `/compact [focus]`: by the agent's rule, else rig-compaction's.
fn compact_now(
    In(args): In<CommandArgs>,
    mut agents: Query<(
        Identity,
        &mut Conversation,
        Option<&CompactionKnobs>,
        Option<&Connection>,
        Has<ActiveTurn>,
    )>,
    rules: Res<CompactionRules>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok((who, mut conversation, own, connection, busy)) = agents.get_mut(agent) else {
        return;
    };
    let rule = rule_of(&rules, who);
    let window = connection.and_then(|connection| connection.spec.context_window);
    match rule.map(|rule| (rule, rule.strategy)) {
        Some((rule, Strategy::Never)) => {
            let why = format!(
                "/compact: rule `{}` never compacts this agent",
                rule.label()
            );
            notices.write(Notice::error(agent, why));
        }
        Some(_) | None if busy => {
            notices.write(Notice::turn_running(agent));
        }
        Some((rule, Strategy::DropToolOutput)) => {
            let stub = rule.stub.as_deref().unwrap_or(STUB);
            let tokens = estimate_tokens(conversation.messages());
            if drop_outputs(&mut conversation, rule.keep_recent_turns, stub) == 0 {
                notices.write(Notice::info(agent, "No tool output to drop."));
                return;
            }
            let record = CompactionRecord {
                strategy: rule.strategy,
                rule: rule.label(),
                tokens,
            };
            let newest = conversation.messages().len().checked_sub(1);
            let after = newest.and_then(|at| conversation.id(at));
            record.leave(&mut conversation, after);
            let text = format!("Dropped older tool output by rule `{}`.", rule.label());
            notices.write(Notice::info(agent, text));
        }
        Some((_, Strategy::Summary)) | None => {
            set_knobs(agent, own, knobs(rule, window), &mut commands);
            let focus = args.args;
            commands.trigger(Compact {
                entity: agent,
                focus,
            });
        }
    }
}

/// `/compaction`: every live agent, its rule, fill and last compaction.
fn list_agents(
    In(args): In<CommandArgs>,
    agents: Query<(Identity, &Conversation, &LastUsage, Option<&Connection>), With<Agent>>,
    rules: Res<CompactionRules>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines = vec!["Compaction by agent:".to_owned()];
    for (who, conversation, last, connection) in &agents {
        let (name, title, spawned, _) = who;
        let name = agent_name(title, spawned, name);
        let rule = match rule_of(&rules, who) {
            Some(rule) => format!(
                "rule `{}` ({} at {}%)",
                rule.label(),
                rule.strategy.as_str(),
                rule.trigger_percent
            ),
            None => "built-in".to_owned(),
        };
        let used = last
            .context()
            .unwrap_or_else(|| estimate_tokens(conversation.messages()));
        let window = connection.and_then(|connection| connection.spec.context_window);
        let fill = match window {
            Some(window) if window > 0 => {
                format!("{}% of {window}", used * 100 / u64::from(window))
            }
            _ => format!("{used} tokens"),
        };
        let compacted = CompactionRecord::newest(conversation).map_or_else(
            || "never compacted".to_owned(),
            |record| format!("last: {} by `{}`", record.strategy.as_str(), record.rule),
        );
        lines.push(format!("{name}: {rule}, {fill}, {compacted}"));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// Puts each subagent's `type` on it.
fn type_subagent(
    spawned: On<Add<SpawnedAgent>>,
    requests: Query<(&SpawnSubagent, &SpawnedAgent)>,
    mut commands: Commands,
) {
    let Ok((request, &SpawnedAgent(agent))) = requests.get(spawned.entity) else {
        return;
    };
    if let Some(kind) = request.extra.get("type").and_then(|kind| kind.as_str()) {
        commands
            .entity(agent)
            .try_insert(SubagentType(kind.to_owned()));
    }
}

/// On the plugin's watches of the rules files' directories.
#[derive(Component)]
struct RulesWatch;

/// The rules read, and what was wrong with them.
struct Loaded(Vec<Rule>, Vec<String>);

/// Watches `RIG_HOME` and the project's `.rig` for the rules files.
fn watch_rules(home: Option<Res<RigHome>>, root: Option<Res<ProjectRoot>>, mut commands: Commands) {
    let dirs = [
        root.map(|root| root.0.join(".rig")),
        home.map(|home| home.0.root().to_path_buf()),
    ];
    for dir in dirs.into_iter().flatten() {
        commands.spawn((RulesWatch, FileWatch::new(dir).listing(Files::All)));
    }
}

/// Reads the rules again when a watch saw a rules file.
fn reload_rules(
    mut changes: MessageReader<FileChanges>,
    watches: Query<(), With<RulesWatch>>,
    (home, root, workspace): (
        Option<Res<RigHome>>,
        Option<Res<ProjectRoot>>,
        Option<Res<ProjectWorkspace>>,
    ),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let named = |path: &PathBuf| path.file_name().is_some_and(|name| name == FILE);
    let seen = changes
        .read()
        .filter(|change| watches.contains(change.watch))
        .any(|change| change.paths.iter().any(named));
    if !seen {
        return;
    }
    let project = root.map(|root| root.0.join(".rig").join(FILE));
    let user = home.map(|home| home.0.root().join(FILE));
    let workspace = workspace.map(|workspace| workspace.0.clone());
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let read = async move {
        let project = project.map(|path| {
            let text = workspace.and_then(|workspace| workspace.read_text(&path).ok());
            (path, text.map(|text| Settings::from_toml(&text)))
        });
        let user = user.map(|path| (path.clone(), Some(Settings::read(&path))));
        load(project.into_iter().chain(user))
    };
    commands.spawn(Running::spawn(pool, &wake, read));
}

/// The rules of `files` in order, each file's path and settings, with their
/// prompt files read, and the problems.
fn load(files: impl Iterator<Item = (PathBuf, Option<Settings>)>) -> Loaded {
    let (mut rules, mut problems) = (Vec::new(), Vec::new());
    for (path, settings) in files {
        let Some(settings) = settings else {
            continue;
        };
        let file = path.display();
        problems.extend(
            settings
                .problems()
                .iter()
                .map(|problem| format!("{file}: {problem}")),
        );
        let read: Vec<Rule> = match settings.get("rule") {
            Ok(read) => read,
            Err(why) => {
                problems.push(format!("{file}: {why}"));
                continue;
            }
        };
        let dir = path.parent().unwrap_or(Path::new("."));
        for mut rule in read {
            if let Some(problem) = rule.problem() {
                problems.push(format!("{file}: rule `{}`: {problem}", rule.label()));
                continue;
            }
            if let Some(prompt) = &rule.prompt_file {
                match std::fs::read_to_string(dir.join(prompt)) {
                    Ok(text) => rule.prompt = Some(text),
                    Err(why) => {
                        let shown = prompt.display();
                        problems.push(format!("{file}: rule `{}`: {shown}: {why}", rule.label()));
                    }
                }
            }
            rules.push(rule);
        }
    }
    Loaded(rules, problems)
}

/// Puts the rules read in force, adds the cuts they name, and shows the
/// problems.
fn rules_loaded(
    loaded: On<Insert<Done<Loaded>>>,
    mut done: Query<&mut Done<Loaded>>,
    (mut rules, mut cuts): (ResMut<CompactionRules>, ResMut<CompactionCuts>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(mut done) = done.get_mut(loaded.entity) else {
        return;
    };
    let Loaded(read, problems) = std::mem::replace(&mut done.0, Loaded(Vec::new(), Vec::new()));
    commands.entity(loaded.entity).despawn();
    for turns in read.iter().map(|rule| rule.keep_recent_turns) {
        cuts.0.add(cut_name(turns), move |messages, from, _, _| {
            turns_start(messages, from, turns)
        });
    }
    rules.0 = read;
    for problem in problems {
        notices.write(Notice::warn(None, format!("compaction rules: {problem}")));
    }
}

#[cfg(test)]
mod tests;
