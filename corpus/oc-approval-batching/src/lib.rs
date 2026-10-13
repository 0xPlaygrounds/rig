//! OC-01: batched approvals for parallel tool calls.
//!
//! A Gate system looks at every tool call of a reply as soon as it is
//! admitted, the calls that wait behind an earlier call of the reply
//! included, and holds each one its [`ApproveRules`] ask about with a hold
//! of its own ([`BatchHold`]). The held calls of one reply are rows of one
//! [`ApprovalBatch`], the dialog: front-neutral data any front shows and
//! ticks, then answers with [`ConfirmBatch`] or [`DenyBatch`]. A ticked
//! call is let go and runs; an unticked one is refused with "denied by
//! user" and the dialog's note. Calls that need no approval are never
//! held. Held calls of another reply that come while a dialog is open
//! wait in [`QueuedRows`] for the next one; each agent's rows are a group
//! of their own. An interrupt despawns the holds, as it does any hold, so
//! the kernel answers the calls as cancelled and the dialog goes once its
//! last row went. With nobody to answer, the rules' fallback applies at
//! once (deny by default).
//!
//! The rules are `approve.toml` in this plugin's user data
//! (`$RIG_HOME/data/approval-batching/approve.toml`):
//!
//! ```toml
//! fallback = "deny"   # or "allow", for a run nobody answers
//!
//! [[ask]]
//! tool = "shell"
//! argument = "command"          # the argument matched; else the summary
//! patterns = ["rm *", "git push*"]  # `*` any text, `?` one character
//! ```
//!
//! The terminal view shows the dialog over the screen ([`tui`]).

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectId;
use serde::Deserialize;

pub mod tui;

/// Where the rules are, in [`PluginData::user`].
pub const RULES_FILE: &str = "approval-batching/approve.toml";

/// What the model reads for a call the user unticked, before the note.
pub const DENIED: &str = "denied by user";

/// What the model reads for a held call nobody could answer, with the
/// `deny` fallback.
pub const HEADLESS: &str = "denied: it needs approval and nobody can answer";

/// How long a row's one-line summary is at most, in characters.
const SUMMARY: usize = 80;

/// What applies to a held call when nobody can answer.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Fallback {
    /// Refuse it.
    #[default]
    Deny,
    /// Let it run.
    Allow,
}

/// One `[[ask]]` of `approve.toml`: the calls of `tool` whose argument
/// matches one of `patterns` (every call of it with none).
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct AskRule {
    /// The tool's name.
    pub tool: String,
    /// The argument matched, a string; without it, the call's summary.
    #[serde(default)]
    pub argument: Option<String>,
    /// Wildcard patterns: `*` matches any text, `?` one character.
    #[serde(default)]
    pub patterns: Vec<String>,
}

/// `approve.toml`: which calls are asked about, and what applies when
/// nobody can answer.
#[derive(Resource, Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ApproveRules {
    /// For a run nobody answers.
    pub fallback: Fallback,
    /// The rules, any of which asks.
    pub ask: Vec<AskRule>,
}

impl ApproveRules {
    /// The rules of `text`, an `approve.toml`.
    ///
    /// # Errors
    ///
    /// When it is not such a file.
    pub fn from_toml(text: &str) -> Result<Self, String> {
        toml::from_str(text).map_err(|error| error.to_string())
    }

    /// Whether a rule asks about `call`.
    pub fn asks(&self, call: &ToolCall) -> bool {
        let arguments = &call.function.arguments;
        self.ask
            .iter()
            .filter(|rule| rule.tool == call.function.name.as_str())
            .any(|rule| {
                let text = match &rule.argument {
                    Some(name) => arguments
                        .get(name)
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned),
                    None => Some(summary(arguments)),
                };
                rule.patterns.is_empty()
                    || text.is_some_and(|text| {
                        rule.patterns.iter().any(|pattern| wildcard(pattern, &text))
                    })
            })
    }
}

/// Whether `text` matches `pattern`, where `*` matches any text and `?`
/// one character.
fn wildcard(pattern: &str, text: &str) -> bool {
    let (pattern, text): (Vec<char>, Vec<char>) =
        (pattern.chars().collect(), text.chars().collect());
    // The last `*` seen, and where in `text` it began to match.
    let (mut p, mut t, mut star): (usize, usize, Option<(usize, usize)>) = (0, 0, None);
    while t < text.len() {
        match (pattern.get(p), text.get(t)) {
            (Some('*'), _) => {
                star = Some((p, t));
                p += 1;
            }
            (Some('?'), _) => (p, t) = (p + 1, t + 1),
            (Some(a), Some(b)) if a == b => (p, t) = (p + 1, t + 1),
            _ => match star {
                Some((at, from)) => {
                    (p, t) = (at + 1, from + 1);
                    star = Some((at, from + 1));
                }
                None => return false,
            },
        }
    }
    pattern.iter().skip(p).all(|c| *c == '*')
}

/// A call's arguments in a line: a lone string argument as it is, else
/// compact JSON, cut to 80 characters.
pub fn summary(arguments: &serde_json::Map<String, serde_json::Value>) -> String {
    let lone = arguments
        .values()
        .next()
        .and_then(serde_json::Value::as_str);
    let line = match lone {
        Some(only) if arguments.len() == 1 => only.to_owned(),
        _ => serde_json::Value::Object(arguments.clone()).to_string(),
    };
    let line = line.lines().next().unwrap_or_default().to_owned();
    match line.char_indices().nth(SUMMARY) {
        Some((cut, _)) => format!("{}…", line.get(..cut).unwrap_or_default()),
        None => line,
    }
}

/// One held call, a row of a dialog.
#[derive(Clone, Debug)]
pub struct BatchRow {
    /// The call.
    pub call: Entity,
    /// The hold on it, a [`BatchHold`].
    pub hold: Entity,
    /// The tool's name.
    pub tool: String,
    /// Its arguments in a line.
    pub summary: String,
    /// Its arguments in full, pretty JSON.
    pub arguments: String,
    /// Whether it is allowed; ticked at first.
    pub checked: bool,
}

/// The held calls of one agent's reply, a group of a dialog.
#[derive(Clone, Debug)]
pub struct BatchGroup {
    /// The agent.
    pub agent: Entity,
    /// Its name, for the group's heading.
    pub agent_name: String,
    /// The model call whose reply asked for the calls.
    pub reply: Option<EffectId>,
    /// The calls, in the order they were held.
    pub rows: Vec<BatchRow>,
}

/// The dialog: every held call of the replies it shows, grouped by agent,
/// each ticked or not, and a note for the denied ones. A front shows it,
/// edits `checked` and `note`, and answers it with [`ConfirmBatch`] or
/// [`DenyBatch`].
#[derive(Component, Clone, Debug, Default)]
#[require(Name = Name::new("approval batch"))]
pub struct ApprovalBatch {
    /// The groups, an agent's reply each.
    pub groups: Vec<BatchGroup>,
    /// Appended to every denied call's result, when not empty.
    pub note: String,
}

impl ApprovalBatch {
    /// Every row, group after group.
    pub fn rows(&self) -> impl Iterator<Item = &BatchRow> {
        self.groups.iter().flat_map(|group| &group.rows)
    }

    /// Row `index` of [`rows`](Self::rows), to edit.
    pub fn row_mut(&mut self, index: usize) -> Option<&mut BatchRow> {
        self.groups
            .iter_mut()
            .flat_map(|group| &mut group.rows)
            .nth(index)
    }

    /// Ticks every row, or none.
    pub fn tick_all(&mut self, checked: bool) {
        for group in &mut self.groups {
            for row in &mut group.rows {
                row.checked = checked;
            }
        }
    }

    /// Adds `row`'s group to the dialog, or `row` to the group of its reply.
    fn add(&mut self, group: BatchGroup) {
        match self
            .groups
            .iter_mut()
            .find(|it| it.agent == group.agent && it.reply == group.reply)
        {
            Some(it) => it.rows.extend(group.rows),
            None => self.groups.push(group),
        }
    }

    /// Whether the dialog shows the reply `reply` of `agent`.
    fn shows(&self, agent: Entity, reply: Option<EffectId>) -> bool {
        self.groups
            .iter()
            .any(|group| group.agent == agent && group.reply == reply)
    }

    /// Takes out the row of `hold`; whether any row is left.
    fn forget(&mut self, hold: Entity) -> bool {
        for group in &mut self.groups {
            group.rows.retain(|row| row.hold != hold);
        }
        self.groups.retain(|group| !group.rows.is_empty());
        !self.groups.is_empty()
    }
}

/// The held calls of replies no open dialog shows, for the next dialog.
#[derive(Resource, Default, Debug)]
pub struct QueuedRows(pub ApprovalBatch);

/// On a hold this plugin put on a call, named for the fronts.
#[derive(Component, Debug)]
#[require(Name = Name::new("waiting for approval"))]
pub struct BatchHold;

/// On a call this plugin looked at: it is never looked at again.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Seen;

/// Answers the dialog as it stands: the ticked calls run, the others are
/// denied with its note.
#[derive(EntityEvent, Clone, Debug)]
pub struct ConfirmBatch {
    /// The dialog.
    pub entity: Entity,
}

/// Denies every call of the dialog, with its note.
#[derive(EntityEvent, Clone, Debug)]
pub struct DenyBatch {
    /// The dialog.
    pub entity: Entity,
}

/// This plugin's Gate system, which holds or, with nobody to answer,
/// decides the calls its rules ask about.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchSystems;

/// One dialog for every call of a reply that needs approval.
#[derive(Default)]
pub struct ApprovalBatchingPlugin;

impl Plugin for ApprovalBatchingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ApproveRules>()
            .init_resource::<QueuedRows>()
            .configure_sets(AgentPass, BatchSystems.in_set(Stage::Gate))
            .add_systems(Startup, load_rules)
            .add_systems(AgentPass, gather.in_set(BatchSystems))
            .add_observer(confirm)
            .add_observer(deny)
            .add_observer(forget);
        tui::add(app);
    }
}

/// Reads `approve.toml`, when there is one, else keeps the rules the app
/// has.
fn load_rules(
    data: Option<Res<PluginData>>,
    mut rules: ResMut<ApproveRules>,
    mut notices: MessageWriter<Notice>,
) {
    let Some(text) = data.and_then(|data| data.user.read_file(RULES_FILE).ok()) else {
        return;
    };
    match ApproveRules::from_toml(&String::from_utf8_lossy(&text)) {
        Ok(read) => *rules = read,
        Err(error) => {
            notices.write(Notice::warn(None, format!("{RULES_FILE}: {error}")));
        }
    }
}

/// Holds each call its rules ask about as soon as it is admitted, waiting
/// or not, as a row of the open dialog when it shows the call's reply,
/// else of the next one; with nobody to answer, decides it by the
/// fallback. Opens the next dialog once none is open.
fn gather(
    mut calls: Query<
        (Entity, &ToolRequest, &ForAgent, &Effect, &mut Decision),
        (
            ToolCalls,
            With<Admitted>,
            Without<Started>,
            Without<Outcome>,
            Without<Holds>,
            Without<Settled>,
            Without<SupersededBy>,
            Without<Interrupted>,
            Without<Seen>,
        ),
    >,
    (rules, answerers): (Res<ApproveRules>, Answerers),
    names: Query<&Name, With<Agent>>,
    mut open: Query<&mut ApprovalBatch>,
    mut queued: ResMut<QueuedRows>,
    mut commands: Commands,
) {
    // In the order the calls were asked for, so rows keep the reply's order.
    let mut calls: Vec<_> = calls.iter_mut().collect();
    calls.sort_unstable_by_key(|(_, _, _, effect, _)| effect.id());
    for (call, request, &ForAgent(agent), effect, mut decision) in calls {
        if decision.is_decided() || !rules.asks(&request.call) {
            continue;
        }
        commands.entity(call).insert(Seen);
        if !answerers.can_answer(Some(agent)) {
            if rules.fallback == Fallback::Deny {
                decision.decide(
                    Decider::plugin("approval-batching"),
                    Outcome::refused(HEADLESS),
                );
            }
            continue;
        }
        let hold = commands.spawn((BatchHold, HoldOf(call))).id();
        let arguments = &request.call.function.arguments;
        let group = BatchGroup {
            agent,
            agent_name: names.get(agent).map(Name::to_string).unwrap_or_default(),
            reply: effect.parent,
            rows: vec![BatchRow {
                call,
                hold,
                tool: request.call.function.name.as_str().to_owned(),
                summary: summary(arguments),
                arguments: serde_json::to_string_pretty(arguments).unwrap_or_default(),
                checked: true,
            }],
        };
        match open
            .iter_mut()
            .find(|batch| batch.shows(agent, effect.parent))
        {
            Some(mut batch) => batch.add(group),
            None => queued.0.add(group),
        }
    }
    if open.is_empty() && !queued.0.groups.is_empty() {
        commands.spawn(std::mem::take(&mut queued.0));
    }
}

/// Lets the ticked calls run and denies the others, closing the dialog.
fn confirm(
    confirmed: On<ConfirmBatch>,
    batches: Query<&ApprovalBatch>,
    calls: Query<&mut Decision, ToolCalls>,
    commands: Commands,
) {
    answer(confirmed.entity, false, batches, calls, commands);
}

/// Denies every call of the dialog, closing it.
fn deny(
    denied: On<DenyBatch>,
    batches: Query<&ApprovalBatch>,
    calls: Query<&mut Decision, ToolCalls>,
    commands: Commands,
) {
    answer(denied.entity, true, batches, calls, commands);
}

/// Closes the dialog `entity`, denying its unticked calls, or all with
/// `deny_all`, and lets go of every call.
fn answer(
    entity: Entity,
    deny_all: bool,
    batches: Query<&ApprovalBatch>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let Ok(batch) = batches.get(entity) else {
        return;
    };
    let why = match batch.note.trim() {
        "" => DENIED.to_owned(),
        note => format!("{DENIED}: {note}"),
    };
    commands.entity(entity).despawn();
    for row in batch.rows() {
        if (deny_all || !row.checked)
            && let Ok(mut decision) = calls.get_mut(row.call)
        {
            decision.decide(Decider::Person, Outcome::refused(why.clone()));
        }
        commands.entity(row.hold).try_despawn();
    }
}

/// A hold gone some other way, as an interrupt despawns them, takes its
/// row out; a dialog left without rows closes.
fn forget(
    removed: On<Remove<BatchHold>>,
    mut batches: Query<(Entity, &mut ApprovalBatch)>,
    mut queued: ResMut<QueuedRows>,
    mut commands: Commands,
) {
    queued.0.forget(removed.entity);
    for (batch, mut rows) in &mut batches {
        if !rows.forget(removed.entity) {
            commands.entity(batch).try_despawn();
        }
    }
}

#[cfg(test)]
mod tests;
