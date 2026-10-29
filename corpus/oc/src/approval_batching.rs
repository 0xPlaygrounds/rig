//! OC-01: batched approvals for parallel tool calls.
//!
//! Any plugin that wants the user to approve a tool call says so with
//! [`NeedsApproval`]; this plugin's own `approve.toml` rules do it in
//! Prepare, so they mark every call of a reply, the ones that wait behind
//! an earlier call of it included. After every gate and before the
//! harness asks about each call alone ([`ApprovalSystems`]), this plugin
//! holds every call of a reply that needs approval with a hold of its own
//! ([`BatchHold`]) and asks one question ([`Ask`]) for all of them, one
//! choice per call, grouped by agent, each ticked at first with its full
//! arguments as its [`Choice::more`], any number ticked, with a note
//! ([`Answer`]'s text). Every front shows it; in the terminal view Space
//! ticks, `a` ticks all, `n` none and `e` shows the full arguments. The
//! ticked calls are [`Approved`] and run; the others are refused with
//! "denied by user" and the note, and closing the question denies them
//! all. Calls that need no approval are never held.
//!
//! A call of a reply the open question shows joins it: a new question
//! supersedes the open one, and the fronts keep the user's ticks. A call
//! of another reply waits in [`QueuedRows`] for the next one. An
//! interrupt despawns the holds, as it does any hold, so the kernel
//! answers the calls as cancelled, and the question closes once its last
//! call went. With nobody to answer, the harness
//! refuses every call that needs approval at once; `fallback = "allow"`
//! keeps this plugin's own rules from asking then.
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

use rig_basics::approval::{ApprovalPlugin, ApprovalSystems, Approved, NeedsApproval};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectId;
use serde::Deserialize;

/// Where the rules are, in [`PluginData::user`].
pub const RULES_FILE: &str = "approval-batching/approve.toml";

/// What the model reads for a call the user did not tick, before the note.
pub const DENIED: &str = "denied by user";

/// This plugin's name, on the reasons it gives.
const NAME: &str = "approval-batching";

/// How long a row's one-line summary is at most, in characters.
const SUMMARY: usize = 80;

/// What applies to a call the rules ask about when nobody can answer.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Fallback {
    /// Refuse it, as the harness does.
    #[default]
    Deny,
    /// Let it run, unless another plugin wants it approved.
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

/// One held call, a choice of the question.
#[derive(Clone, Debug)]
pub struct BatchRow {
    /// The call.
    pub call: Entity,
    /// Its effect's id, which names its choice.
    pub id: EffectId,
    /// The hold on it, a [`BatchHold`].
    pub hold: Entity,
    /// The agent whose call it is.
    pub agent: Entity,
    /// Its name, for the call's group.
    pub agent_name: String,
    /// The model call whose reply asked for the call.
    pub reply: Option<EffectId>,
    /// The tool's name.
    pub tool: String,
    /// Its arguments in a line.
    pub summary: String,
    /// Its arguments in full, pretty JSON.
    pub arguments: String,
}

/// On the open question: the calls it asks about, its choices in order.
#[derive(Component, Clone, Debug, Default)]
pub struct ApprovalBatch {
    /// The calls, grouped by agent, each reply's in the order asked for.
    pub rows: Vec<BatchRow>,
}

impl ApprovalBatch {
    /// The question about every row: each a choice, ticked to allow it,
    /// and ticked at first.
    fn ask(&self) -> Ask {
        let title = format!("Approve {} tool calls?", self.rows.len());
        let body = "Untick the calls to deny; a note typed goes with every denied one.";
        let ask = Ask::new(title).body(body).multiple().free_text();
        self.rows.iter().fold(ask, |ask, row| {
            let label = format!("{}: {}  {}", row.agent_name, row.tool, row.summary);
            let choice = Choice::new(row.id.to_string(), label).checked();
            ask.with(choice.more(&row.arguments))
        })
    }

    /// The agent whose calls they all are, when they are one agent's.
    fn agent(&self) -> Option<Entity> {
        let agent = self.rows.first()?.agent;
        self.rows
            .iter()
            .all(|row| row.agent == agent)
            .then_some(agent)
    }

    /// Whether it asks about the reply `reply` of `agent`.
    fn shows(&self, agent: Entity, reply: Option<EffectId>) -> bool {
        self.rows
            .iter()
            .any(|row| row.agent == agent && row.reply == reply)
    }
}

/// The held calls of replies no open question shows, for the next one.
#[derive(Resource, Default, Debug)]
pub struct QueuedRows(pub Vec<BatchRow>);

/// On a hold this plugin put on a call, named for the fronts.
#[derive(Component, Debug)]
#[require(Name = Name::new("waiting for approval"))]
pub struct BatchHold;

/// This plugin's system that holds the calls needing approval: after
/// every gate and the subagents' restart check, before the harness asks
/// ([`ApprovalSystems`]). The answer is carried out by a gate.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchSystems;

/// One question for every call of a reply that needs approval.
#[derive(Default)]
pub struct ApprovalBatchingPlugin;

impl Plugin for ApprovalBatchingPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ApprovalPlugin>() {
            app.add_plugins(ApprovalPlugin);
        }
        app.init_resource::<ApproveRules>()
            .init_resource::<QueuedRows>()
            .configure_sets(
                AgentPass,
                BatchSystems
                    .after(GateSystems::After)
                    .before(ApprovalSystems),
            )
            .add_systems(Startup, load_rules)
            .add_systems(
                AgentPass,
                (
                    want.in_set(Stage::Prepare).after(PrepareSystems::Edit),
                    answered.in_set(Stage::Gate),
                    gather.in_set(BatchSystems),
                ),
            )
            .add_observer(forget);
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

/// Wants the user to approve each call admitted this pass that the rules
/// ask about, unless nobody can answer and the fallback allows it.
fn want(
    calls: Query<(Entity, &ToolRequest, &ForAgent), (ToolCalls, Added<Admitted>)>,
    (rules, answerers): (Res<ApproveRules>, Answerers),
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent)) in &calls {
        let allowed = rules.fallback == Fallback::Allow && !answerers.can_answer(Some(agent));
        if !allowed && rules.asks(&request.call) {
            let reason = NeedsApproval::add(NAME, "approve.toml asks about it.");
            commands.entity(call).queue(reason);
        }
    }
}

/// Holds each call that needs approval, waiting or not, and nobody decided,
/// as a choice of the open question when it shows the call's reply (a new
/// question supersedes it), else of the next one. Asks the next question
/// once none is open. With nobody to answer, it leaves the call to the
/// harness, which refuses it.
fn gather(
    calls: Query<
        (Entity, &ToolRequest, &ForAgent, &Effect, &Decision),
        (
            ToolCalls,
            With<NeedsApproval>,
            Without<Approved>,
            With<Admitted>,
            Without<Started>,
            Without<Outcome>,
            Without<Holds>,
            Without<Settled>,
            Without<SupersededBy>,
            Without<Interrupted>,
        ),
    >,
    answerers: Answerers,
    names: Query<&Name, With<Agent>>,
    mut open: Query<(Entity, &mut ApprovalBatch)>,
    mut queued: ResMut<QueuedRows>,
    mut commands: Commands,
) {
    // In the order the calls were asked for, so rows keep the reply's order.
    let mut calls: Vec<_> = calls.iter().collect();
    calls.sort_unstable_by_key(|(_, _, _, effect, _)| effect.id());
    let mut grown = None;
    for (call, request, &ForAgent(agent), effect, decision) in calls {
        if decision.is_decided() || !answerers.can_answer(Some(agent)) {
            continue;
        }
        let arguments = &request.call.function.arguments;
        let row = BatchRow {
            call,
            id: effect.id(),
            hold: commands.spawn((BatchHold, HoldOf(call))).id(),
            agent,
            agent_name: names.get(agent).map(Name::to_string).unwrap_or_default(),
            reply: effect.parent,
            tool: request.call.function.name.as_str().to_owned(),
            summary: summary(arguments),
            arguments: serde_json::to_string_pretty(arguments).unwrap_or_default(),
        };
        match open
            .iter_mut()
            .find(|(_, batch)| batch.shows(agent, effect.parent))
        {
            Some((question, mut batch)) => {
                batch.rows.push(row);
                grown = Some(question);
            }
            None => queued.0.push(row),
        }
    }
    let next = match (grown, open.is_empty()) {
        // A grown question is superseded by one with every call.
        (Some(question), _) => open.get_mut(question).ok().map(|(_, mut batch)| {
            commands.entity(question).remove::<ApprovalBatch>();
            (Some(question), std::mem::take(&mut batch.rows))
        }),
        (None, true) if !queued.0.is_empty() => Some((None, std::mem::take(&mut queued.0))),
        _ => None,
    };
    let Some((old, mut rows)) = next else {
        return;
    };
    rows.sort_by_key(|row| row.agent);
    let batch = ApprovalBatch { rows };
    let mut question = commands.spawn(batch.ask());
    #[cfg(feature = "tui")]
    question.insert(keys());
    // Several agents' question is the front's, not one agent's.
    if let Some(agent) = batch.agent() {
        question.insert(ForAgent(agent));
    }
    let new = question.insert(batch).id();
    if let Some(old) = old {
        commands.entity(old).insert(SupersededBy(new));
    }
}

/// The question's keys in the terminal view, as the spec names them.
#[cfg(feature = "tui")]
fn keys() -> rig_tui::ChoiceKeys {
    use rig_tui::ChoiceAction::{More, Tick, TickAll, TickNone};
    use rig_tui::crossterm::event::KeyCode::Char;
    rig_tui::ChoiceKeys(vec![
        (Char(' '), Tick),
        (Char('a'), TickAll),
        (Char('n'), TickNone),
        (Char('e'), More),
    ])
}

/// Carries out each answer: the ticked calls are approved, the others
/// refused with the note; a question closed unanswered refuses them all.
/// Every call is let go.
fn answered(
    questions: Query<(Entity, &ApprovalBatch, &Outcome), Added<Outcome>>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    for (question, batch, outcome) in &questions {
        let answer = Answer::of(outcome).unwrap_or_default();
        let why = match answer.text.as_deref().map(str::trim) {
            Some(note) if !note.is_empty() => format!("{DENIED}: {note}"),
            _ => DENIED.to_owned(),
        };
        // Before the holds go, so `forget` leaves the question alone.
        commands.entity(question).remove::<ApprovalBatch>();
        for row in &batch.rows {
            if answer.is(&row.id.to_string()) {
                commands.entity(row.call).try_insert(Approved);
            } else if let Ok(mut decision) = calls.get_mut(row.call) {
                decision.decide(Decider::Person, Outcome::refused(why.clone()));
            }
            commands.entity(row.hold).try_despawn();
        }
    }
}

/// A hold gone some other way, as an interrupt despawns them, takes its
/// row out; a question left without rows closes.
fn forget(
    removed: On<Remove<BatchHold>>,
    mut questions: Query<(Entity, &mut ApprovalBatch)>,
    mut queued: ResMut<QueuedRows>,
    mut commands: Commands,
) {
    queued.0.retain(|row| row.hold != removed.entity);
    for (question, mut batch) in &mut questions {
        let before = batch.rows.len();
        batch.rows.retain(|row| row.hold != removed.entity);
        if batch.rows.is_empty() && before > 0 {
            commands.entity(question).try_despawn();
        }
    }
}

#[cfg(test)]
mod tests;
