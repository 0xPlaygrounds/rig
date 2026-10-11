//! Plugin corpus (held-out): H28: Stall watchdog (spec `H28-stall-watchdog.md`).
//!
//! Watches every model call and tool call while it runs, of any agent or
//! plugin. Each model call carries a pass-through stream tap (an
//! [`Adapter`]) that notes when its stream last sent an item; a call that
//! sent nothing for the model timeout is reported, and a turn's call is
//! sent again once (a successor that `Supersedes` it), a second stall in
//! the turn failing it. A tool call past its soft limit is reported, shown
//! on the status line while it runs, and stopped only past its
//! `kill_after`. Each stall is a line of `<session dir>/stalls.jsonl`,
//! which `/stalls` lists. A running call is woken for by a [`WakeAt`] at
//! its next deadline, so nothing runs while no call does.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rig_harness::clock::ClockPlugin;
use rig_harness::prelude::*;
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::streaming::Relayed;
use rig_harness::rig_ecs::effects::Handler;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};
use serde::{Deserialize, Serialize};

/// The adapter's name, and the settings table's.
const NAME: &str = "stall_watchdog";

/// The file of a session's stalls.
const FILE: &str = "stalls.jsonl";

/// After the status; gone before most items when the line is narrow.
const PLACE: StatusItem = StatusItem::at(Side::Left, 55, 9);

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
            .register_required_components::<TurnOf, TurnStalls>()
            .add_systems(AgentPass, tap.in_set(Stage::Prepare))
            .add_systems(Update, (model_due, tool_due).before(AgentLoop))
            .add_systems(PostUpdate, show_overdue.in_set(StatusSystems))
            .add_observer(arm)
            .add_observer(settled)
            .add_command("stalls", "List this session's stalled calls", list);
    }
}

/// On each model call: when its stream last sent an item (ms since the
/// epoch, 0 before one) and how many it sent.
#[derive(Component, Clone, Default)]
struct Pulse(Arc<Beats>);

#[derive(Default)]
struct Beats {
    last_ms: AtomicU64,
    items: AtomicU64,
}

impl Pulse {
    fn beat(&self) {
        self.0
            .last_ms
            .store(millis(SystemTime::now()), Ordering::Relaxed);
        self.0.items.fetch_add(1, Ordering::Relaxed);
    }

    fn last(&self) -> Option<SystemTime> {
        let ms = self.0.last_ms.load(Ordering::Relaxed);
        (ms > 0).then(|| UNIX_EPOCH + Duration::from_millis(ms))
    }

    fn items(&self) -> u64 {
        self.0.items.load(Ordering::Relaxed)
    }
}

/// On a running call the watchdog watches: when it started.
#[derive(Component)]
struct Watched {
    started: SystemTime,
}

/// A call that stalled or ran past its soft limit, written to the file
/// once it settles.
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

/// A tool call past its soft limit: what the status line of `target` says.
#[derive(Component)]
struct Overdue {
    target: Entity,
    text: String,
}

/// On a turn: how many of its model calls stalled.
#[derive(Component, Default)]
struct TurnStalls(u32);

/// Gives each model call, as it is admitted, a tap that notes each item
/// its stream sends, whichever model it goes to.
fn tap(mut calls: Query<(Entity, &mut ModelRequest), Added<Admitted>>, mut commands: Commands) {
    for (call, mut request) in &mut calls {
        let pulse = Pulse::default();
        let beats = pulse.clone();
        request
            .adapters
            .push(Adapter::new(NAME, move |mut connection: Connection| {
                let beats = beats.clone();
                let tapped = connection.handler.0.map_stream(NAME, move |_| {
                    let beats = beats.clone();
                    move |item: Relayed| {
                        if matches!(item, Relayed::Item(_)) {
                            beats.beat();
                        }
                        vec![item]
                    }
                });
                connection.handler = Handler(tapped);
                connection
            }));
        commands.entity(call).insert(pulse);
    }
}

/// Watches each model call and tool call as it starts, woken at its first
/// deadline.
fn arm(
    started: On<Add<Started>>,
    calls: Query<(&Started, Option<&ToolRequest>), Or<(With<ModelRequest>, With<ToolRequest>)>>,
    settings: Res<StallSettings>,
    mut commands: Commands,
) {
    let Ok((at, tool)) = calls.get(started.entity) else {
        return;
    };
    let first = match tool {
        Some(tool) => settings.limits(tool.call.function.name.as_str()).0,
        None => settings.model_timeout(),
    };
    let watched = Watched { started: at.at };
    commands
        .entity(started.entity)
        .try_insert((watched, WakeAt(at.at + first)));
}

/// A running model call due: one quiet for the timeout stalled. A turn's
/// call is sent again on its first stall; any other stall fails the call.
fn model_due(
    calls: Query<
        (
            Entity,
            &Watched,
            &Pulse,
            &Effect,
            Option<&SentTo>,
            Option<&CallOf>,
            Option<&ForAgent>,
        ),
        (
            Added<Due>,
            With<ModelRequest>,
            Without<Outcome>,
            Without<Settled>,
            Without<SupersededBy>,
        ),
    >,
    mut turns: Query<&mut TurnStalls>,
    settings: Res<StallSettings>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let now = SystemTime::now();
    let timeout = settings.model_timeout();
    for (call, watched, pulse, effect, sent, of, agent) in &calls {
        let last = pulse
            .last()
            .map_or(watched.started, |last| last.max(watched.started));
        let quiet = now.duration_since(last).unwrap_or_default();
        if quiet < timeout {
            commands.entity(call).try_insert(WakeAt(last + timeout));
            continue;
        }
        let model = sent.map_or_else(|| "model".to_owned(), |sent| sent.0.spec.id.clone());
        let tokens = pulse.items();
        let agent = agent.map(|agent| agent.0);
        let retry = match (of, agent) {
            (Some(&CallOf(turn)), Some(agent)) => turns
                .get_mut(turn)
                .ok()
                .filter(|stalls| stalls.0 == 0)
                .map(|mut stalls| {
                    stalls.0 += 1;
                    (turn, agent)
                }),
            _ => None,
        };
        let what = if retry.is_some() {
            "sending it again"
        } else {
            "giving up"
        };
        let text = format!(
            "Model stalled {}, {model}, after {tokens} tokens: {what}.",
            human(quiet)
        );
        notices.write(Notice::info(agent, text));
        let mut stall = Stall {
            what: "model",
            name: model,
            seconds: Some(quiet.as_secs_f64()),
            tokens: Some(tokens),
            ended: None,
        };
        if let Some((turn, agent)) = retry {
            commands.spawn((
                TurnCall,
                Supersedes(call),
                CallOf(turn),
                ForAgent(agent),
                Effect::child_of(effect.parent),
            ));
        } else {
            stall.ended = Some("failed");
            let why = format!("the model sent nothing for {}", human(quiet));
            let failed = Outcome(Err(ErrorReport::new(ErrorKind::Timeout, why)));
            commands.entity(call).try_insert_if_new(failed);
        }
        commands.entity(call).try_insert(stall);
    }
}

/// A running tool call due: past its soft limit it is reported and shown,
/// past its `kill_after` stopped; woken again at its next deadline.
fn tool_due(
    calls: Query<
        (
            Entity,
            &Watched,
            &ToolRequest,
            Option<&CallOf>,
            Option<&ForAgent>,
            Has<Stall>,
        ),
        (Added<Due>, Without<Outcome>, Without<Settled>),
    >,
    settings: Res<StallSettings>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let now = SystemTime::now();
    for (call, watched, request, of, agent, reported) in &calls {
        let name = request.call.function.name.as_str();
        let (soft, kill) = settings.limits(name);
        let ran = now.duration_since(watched.started).unwrap_or_default();
        let agent = agent.map(|agent| agent.0);
        let mut entity = commands.entity(call);
        if let Some(kill) = kill.filter(|kill| ran >= *kill) {
            notices.write(Notice::error(
                agent,
                format!("{name} stopped after {}.", human(ran)),
            ));
            let why = format!("stopped by the stall watchdog after {}", human(kill));
            entity.try_insert_if_new(Outcome(Err(ErrorReport::new(ErrorKind::Timeout, why))));
            entity.try_insert(Stall {
                what: "tool",
                name: name.to_owned(),
                seconds: Some(ran.as_secs_f64()),
                tokens: None,
                ended: Some("killed"),
            });
            continue;
        }
        if ran < soft {
            entity.try_insert(WakeAt(watched.started + soft));
            continue;
        }
        if !reported {
            let text = format!("{name} running {}, past its soft limit.", human(ran));
            notices.write(Notice::info(agent, text));
            entity.try_insert(Stall {
                what: "tool",
                name: name.to_owned(),
                seconds: None,
                tokens: None,
                ended: None,
            });
        }
        let target = of.map(|of| of.0).or(agent).unwrap_or(call);
        let text = format!("{name} running {}", human(ran));
        entity.try_insert(Overdue { target, text });
        // The status shows whole seconds, then whole minutes.
        let tick = if ran < Duration::from_secs(60) { 1 } else { 60 };
        let mut next = now + Duration::from_secs(tick);
        if let Some(kill) = kill {
            next = next.min(watched.started + kill);
        }
        entity.try_insert(WakeAt(next));
    }
}

/// Writes each stalled call's line once it settles, and stops watching
/// every watched call.
fn settled(
    settled: On<Add<Settled>>,
    calls: Query<(
        &Watched,
        Option<&Stall>,
        Option<&Outcome>,
        Has<SupersededBy>,
        Option<&ForAgent>,
    )>,
    ids: Query<&AgentId>,
    paths: Option<Res<SessionPaths>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((watched, stall, outcome, superseded, agent)) = calls.get(settled.entity) else {
        return;
    };
    commands
        .entity(settled.entity)
        .try_remove::<(WakeAt, Due, Overdue)>();
    let (Some(stall), Some(paths)) = (stall, paths) else {
        return;
    };
    let cancelled =
        matches!(outcome, Some(Outcome(Err(report))) if report.kind == ErrorKind::Cancelled);
    let ended = match stall.ended {
        _ if superseded => "retried",
        Some(ended) => ended,
        None if cancelled => "interrupted",
        None => "finished",
    };
    let ran = SystemTime::now()
        .duration_since(watched.started)
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
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(file(&paths))?;
            writeln!(file, "{line}")
        });
    if let Err(failure) = written {
        notices.write(Notice::error(
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

/// Shows the tool calls past their soft limit on their turn's (or agent's)
/// status line, and clears the item once none is.
fn show_overdue(
    calls: Query<&Overdue>,
    mut items: Query<&mut StatusItems>,
    mut shown: Local<HashSet<Entity>>,
    mut commands: Commands,
) {
    let mut texts: BTreeMap<Entity, Vec<&str>> = BTreeMap::new();
    for overdue in &calls {
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
        match items.get_mut(target) {
            Ok(mut items) => {
                let current = items
                    .iter()
                    .find(|old| (old.side, old.order) == (PLACE.side, PLACE.order));
                if current != Some(&item) {
                    items.show(item);
                }
            }
            Err(_) => {
                let mut items = StatusItems::default();
                items.show(item);
                commands.entity(target).try_insert(items);
            }
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
