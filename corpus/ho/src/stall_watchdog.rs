//! Plugin corpus (held-out): H28: Stall watchdog (spec `H28-stall-watchdog.md`).
//!
//! Watches every model call and tool call while it runs, of any agent or
//! plugin, with a timer of its own per call ([`WakeAt`] at the call's next
//! deadline), so nothing runs while no call does. A model call whose
//! stream ([`StreamTiming`]) sent nothing for the model timeout is reported
//! and sent again once ([`resend`]); a second stall fails it. A tool call
//! past its soft limit is reported, shown on the status line while it
//! runs, and stopped only past its `kill_after`. Each stall is a line of
//! `<session dir>/stall-watchdog/stalls.jsonl`, which `/stalls` lists.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rig_harness::clock::ClockPlugin;
use rig_harness::prelude::*;
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};
use serde::{Deserialize, Serialize};

/// The settings table's name.
const NAME: &str = "stall_watchdog";

/// The file of a session's stalls, in a directory of the plugin's own: a
/// top-level `*.jsonl` of the session's is an agent's log.
const FILE: &str = "stall-watchdog/stalls.jsonl";

/// After the status; gone before most items when the line is narrow.
const PLACE: StatusItem = StatusItem::new("ho-stall-watchdog", Side::Left, 55, 9);

/// The `[stall_watchdog]` table of the settings, in seconds.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct StallSettings {
    /// How long a model call may send nothing.
    pub model_timeout: f64,
    /// How long a tool call runs before it is reported.
    pub tool_soft_limit: f64,
    /// Each tool's own limits, by name.
    pub tools: HashMap<String, ToolLimits>,
}

impl Default for StallSettings {
    fn default() -> Self {
        Self {
            model_timeout: 60.0,
            tool_soft_limit: 300.0,
            tools: HashMap::new(),
        }
    }
}

/// One tool's limits, in seconds.
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ToolLimits {
    /// In place of `tool_soft_limit`.
    pub soft_limit: Option<f64>,
    /// When its calls are stopped; never without one.
    pub kill_after: Option<f64>,
}

impl StallSettings {
    fn model_timeout(&self) -> Duration {
        seconds(self.model_timeout).unwrap_or(Duration::from_secs(60))
    }

    /// The soft limit and the kill time of `tool`'s calls.
    fn limits(&self, tool: &str) -> (Duration, Option<Duration>) {
        let own = self.tools.get(tool);
        let soft = own
            .and_then(|own| own.soft_limit)
            .or(Some(self.tool_soft_limit))
            .and_then(seconds)
            .unwrap_or(Duration::from_secs(300));
        (soft, own.and_then(|own| own.kill_after).and_then(seconds))
    }
}

/// `secs` as a duration, unless it is not a positive number.
fn seconds(secs: f64) -> Option<Duration> {
    Duration::try_from_secs_f64(secs)
        .ok()
        .filter(|duration| !duration.is_zero())
}

#[derive(Default)]
pub struct StallWatchdogPlugin;

impl Plugin for StallWatchdogPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ClockPlugin>() {
            app.add_plugins(ClockPlugin);
        }
        app.add_setting::<StallSettings>(NAME)
            .add_systems(PostUpdate, show_overdue.in_set(StatusSystems::Show))
            .add_observer(arm)
            .add_observer(model_due)
            .add_observer(tool_due)
            .add_observer(settled)
            .add_observer(follow_successor::<Resent>)
            .add_command("stalls", "List this session's stalled calls", list);
    }
}

/// The watchdog's timer of a running call, despawned with it.
#[derive(Component)]
#[relationship(relationship_target = WatchedBy)]
struct Watch(Entity);

/// On a watched call: its timer.
#[derive(Component)]
#[relationship_target(relationship = Watch, linked_spawn)]
struct WatchedBy(Entity);

/// On a timer whose call stalled or ran past its soft limit: its line of
/// the file, written once the call settles.
#[derive(Component)]
struct Stall {
    what: &'static str,
    name: String,
    /// How long it sent nothing; for a tool, how long it ran, at its end.
    seconds: Option<f64>,
    /// How many items its stream sent, for a model call.
    tokens: Option<u64>,
    /// How the watchdog ended it, if it did.
    ended: Option<&'static str>,
}

/// On a timer whose tool call is past its soft limit: what the status
/// line of `target` says.
#[derive(Component)]
struct Overdue {
    target: Entity,
    text: String,
}

/// On a model call sent again after a stall, and on its successors: its
/// next stall fails it.
#[derive(Component)]
struct Resent;

/// Watches each model call and tool call as it starts, woken at its first
/// deadline.
fn arm(
    started: On<Add<Started>>,
    calls: Query<(&Started, Option<&ToolRequest>), Or<(ModelCalls, ToolCalls)>>,
    settings: Res<StallSettings>,
    mut commands: Commands,
) {
    let Ok((at, tool)) = calls.get(started.entity) else {
        return;
    };
    let first = match tool {
        Some(tool) => match settings.limits(tool.call.function.name.as_str()) {
            (soft, Some(kill)) => soft.min(kill),
            (soft, None) => soft,
        },
        None => settings.model_timeout(),
    };
    commands.spawn((Watch(started.entity), WakeAt(at.at + first)));
}

/// A running model call due: one quiet for the timeout stalled. It is
/// sent again on its first stall and failed on its second.
fn model_due(
    due: On<Add<Due>>,
    timers: Query<&Watch>,
    mut calls: Query<
        (
            &Started,
            Option<&StreamTiming>,
            Option<&SentTo>,
            Option<&ForAgent>,
            Has<Resent>,
            &mut Decision,
        ),
        (
            ModelCalls,
            Without<Outcome>,
            Without<Settled>,
            Without<SupersededBy>,
        ),
    >,
    settings: Res<StallSettings>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(&Watch(call)) = timers.get(due.entity) else {
        return;
    };
    let Ok((started, streamed, sent, agent, resent, mut decision)) = calls.get_mut(call) else {
        return;
    };
    let timeout = settings.model_timeout();
    let last = streamed.map_or(started.at, |streamed| streamed.last.max(started.at));
    let quiet = SystemTime::now().duration_since(last).unwrap_or_default();
    if quiet < timeout {
        commands.entity(due.entity).insert(WakeAt(last + timeout));
        return;
    }
    let model = sent.map_or_else(|| "model".to_owned(), |sent| sent.0.spec.id.clone());
    let tokens = streamed.map_or(0, |streamed| streamed.items);
    let mut stall = Stall {
        what: "model",
        name: model.clone(),
        seconds: Some(quiet.as_secs_f64()),
        tokens: Some(tokens),
        ended: None,
    };
    let what = if resent {
        let why = format!("the model sent nothing for {}", human(quiet));
        if decision.decide(
            Decider::plugin("ho-stall-watchdog"),
            Outcome(Err(ErrorReport::new(ErrorKind::Timeout, why))),
        ) {
            stall.ended = Some("failed");
        }
        "giving up"
    } else {
        commands.entity(call).insert(Resent);
        commands.queue(resend(call));
        "sending it again"
    };
    let text = format!(
        "Model stalled {}, {model}, after {tokens} tokens: {what}.",
        human(quiet)
    );
    notices.write(Notice::warn(agent.map(|agent| agent.0), text));
    commands.entity(due.entity).insert(stall);
}

/// A running tool call due: past its soft limit it is reported and shown,
/// past its `kill_after` stopped; woken again at its next deadline.
fn tool_due(
    due: On<Add<Due>>,
    timers: Query<(&Watch, Has<Stall>)>,
    mut calls: Query<
        (
            &Started,
            &ToolRequest,
            Option<&CallOf>,
            Option<&ForAgent>,
            &mut Decision,
        ),
        (ToolCalls, Without<Outcome>, Without<Settled>),
    >,
    settings: Res<StallSettings>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((&Watch(call), reported)) = timers.get(due.entity) else {
        return;
    };
    let Ok((started, request, of, agent, mut decision)) = calls.get_mut(call) else {
        return;
    };
    let name = request.call.function.name.as_str();
    let (soft, kill) = settings.limits(name);
    let now = SystemTime::now();
    let ran = now.duration_since(started.at).unwrap_or_default();
    let agent = agent.map(|agent| agent.0);
    let mut timer = commands.entity(due.entity);
    if let Some(kill) = kill.filter(|kill| ran >= *kill) {
        let why = format!("stopped by the stall watchdog after {}", human(kill));
        if decision.decide(
            Decider::plugin("ho-stall-watchdog"),
            Outcome(Err(ErrorReport::new(ErrorKind::Timeout, why))),
        ) {
            let text = format!("{name} stopped after {}.", human(ran));
            notices.write(Notice::warn(agent, text));
            timer.insert(Stall {
                what: "tool",
                name: name.to_owned(),
                seconds: Some(ran.as_secs_f64()),
                tokens: None,
                ended: Some("killed"),
            });
        }
        return;
    }
    if ran < soft {
        timer.insert(WakeAt(started.at + soft));
        return;
    }
    if !reported {
        let text = format!("{name} running {}, past its soft limit.", human(ran));
        notices.write(Notice::info(agent, text));
        timer.insert(Stall {
            what: "tool",
            name: name.to_owned(),
            seconds: None,
            tokens: None,
            ended: None,
        });
    }
    if let Some(target) = of.map(|of| of.0).or(agent) {
        let text = format!("{name} running {}", human(ran));
        timer.insert(Overdue { target, text });
    }
    // The status shows tenths of a second, then whole seconds, then minutes.
    let tick = if ran < Duration::from_secs(60) { 1 } else { 60 };
    let next = now + Duration::from_secs(tick);
    let next = kill.map_or(next, |kill| next.min(started.at + kill));
    timer.insert(WakeAt(next));
}

/// Writes the line of each stalled call once it settles, and stops
/// watching it.
fn settled(
    settled: On<Add<Settled>>,
    calls: Query<(
        &Started,
        &WatchedBy,
        Option<&Outcome>,
        Has<SupersededBy>,
        Option<&ForAgent>,
    )>,
    stalls: Query<&Stall>,
    ids: Query<&AgentId>,
    paths: Option<Res<SessionPaths>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((started, &WatchedBy(timer), outcome, superseded, agent)) = calls.get(settled.entity)
    else {
        return;
    };
    commands.entity(timer).despawn();
    let (Ok(stall), Some(paths)) = (stalls.get(timer), paths) else {
        return;
    };
    let failed = match outcome {
        Some(Outcome(Err(report))) => Some(report.kind),
        _ => None,
    };
    let ended = match (stall.ended, failed) {
        _ if superseded => "retried",
        (Some(ended), Some(ErrorKind::Timeout)) => ended,
        (_, Some(ErrorKind::Cancelled)) => "interrupted",
        _ => "finished",
    };
    let ran = SystemTime::now()
        .duration_since(started.at)
        .unwrap_or_default();
    let entry = Entry {
        at_ms: millis(SystemTime::now()),
        agent: agent
            .and_then(|agent| ids.get(agent.0).ok())
            .map(|id| id.0.clone()),
        what: stall.what.to_owned(),
        name: stall.name.clone(),
        seconds: stall.seconds.unwrap_or(ran.as_secs_f64()),
        tokens: stall.tokens,
        ended: ended.to_owned(),
    };
    let written = serde_json::to_string(&entry)
        .map_err(std::io::Error::other)
        .and_then(|line| {
            let path = file(&paths);
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let mut file = OpenOptions::new().create(true).append(true).open(path)?;
            writeln!(file, "{line}")
        });
    if let Err(failure) = written {
        notices.write(Notice::warn(
            None,
            format!("Could not write {FILE}: {failure}."),
        ));
    }
}

/// A line of `stalls.jsonl`.
#[derive(Serialize, Deserialize)]
struct Entry {
    at_ms: u64,
    agent: Option<String>,
    /// `model` or `tool`.
    what: String,
    /// The model or tool.
    name: String,
    /// How long the model sent nothing, or the tool ran.
    seconds: f64,
    /// What a model's stream sent before it stalled.
    tokens: Option<u64>,
    /// `retried`, `failed`, `finished`, `killed` or `interrupted`.
    ended: String,
}

impl Entry {
    fn describe(&self) -> String {
        let took = human(Duration::try_from_secs_f64(self.seconds).unwrap_or_default());
        match self.tokens {
            Some(tokens) => format!(
                "model {} stalled {took} after {tokens} tokens: {}",
                self.name, self.ended
            ),
            None => format!("tool {} ran {took}: {}", self.name, self.ended),
        }
    }
}

fn file(paths: &SessionPaths) -> PathBuf {
    paths.path().join(FILE)
}

/// `/stalls`.
fn list(
    In(args): In<CommandArgs>,
    paths: Option<Res<SessionPaths>>,
    mut notices: MessageWriter<Notice>,
) {
    let Some(paths) = paths else {
        notices.write(Notice::error(
            args.agent,
            "/stalls needs a session directory.",
        ));
        return;
    };
    let text = std::fs::read_to_string(file(&paths)).unwrap_or_default();
    let lines: Vec<String> = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Entry>(line).ok())
        .map(|entry| entry.describe())
        .collect();
    let text = if lines.is_empty() {
        "No stalls in this session.".to_owned()
    } else {
        lines.join("\n")
    };
    notices.write(Notice::info(args.agent, text));
}

/// Shows the tool calls past their soft limit on their turn's (or
/// agent's) status line, and clears the item once none is.
fn show_overdue(
    overdue: Query<&Overdue>,
    mut items: Query<&mut StatusItems>,
    mut shown: Local<HashSet<Entity>>,
) {
    let mut texts: BTreeMap<Entity, Vec<&str>> = BTreeMap::new();
    for overdue in &overdue {
        texts.entry(overdue.target).or_default().push(&overdue.text);
    }
    for gone in shown.iter().filter(|target| !texts.contains_key(target)) {
        if let Ok(mut items) = items.get_mut(*gone) {
            items.show(PLACE);
        }
    }
    shown.clear();
    for (target, texts) in texts {
        shown.insert(target);
        let item = PLACE.says(texts.join(", "), Tone::Yellow);
        if let Ok(mut items) = items.get_mut(target)
            && !items.iter().any(|old| *old == item)
        {
            items.show(item);
        }
    }
}

/// `duration` as the status line says it: tenths of a second under 10 s,
/// then seconds, then minutes.
fn human(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 10.0 {
        format!("{secs:.1} s")
    } else if secs < 60.0 {
        format!("{} s", duration.as_secs())
    } else {
        format!("{} m", duration.as_secs() / 60)
    }
}

fn millis(at: SystemTime) -> u64 {
    at.duration_since(UNIX_EPOCH).map_or(0, |since| {
        u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
    })
}

#[cfg(test)]
mod tests;
