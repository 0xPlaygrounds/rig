//! `/perf`: which plugin is slowing the agent down, and what it was doing.
//!
//! What it measures, from the stack's public data:
//!
//! - **Holding up a turn.** A plugin that makes a turn wait holds an effect
//!   (`HoldOf`): a request before it goes out, a tool call before it
//!   starts, a result before it reaches the model, or the turn's end. Each
//!   hold's life is timed and charged to the plugin that spawned it
//!   (`ProvidedBy`), with what it held ("before request #2, turn 3"). A
//!   question for the user (`Ask`) is the user's time, not a plugin's.
//! - **Background work.** A plugin's own effect (`EffectRequest`) is timed
//!   from its start to its answer: as holding up the turn when it is a call
//!   of one (`CallOf`), else as background work.
//! - **Its systems and observers.** The harness times every system and
//!   observer run ([`SystemTimings`], with its `trace` feature, which this
//!   crate turns on), summed per plugin package. A frame's runs hold up the
//!   user's turn while one runs (the agent loop and the plugins share the
//!   frame), else they are background work.
//! - **Interface stalls.** A frame whose work takes over 50 ms kept the
//!   terminal from drawing and from reacting to keys. It names the plugin
//!   whose systems took most of it, with the keys the frame handled.
//!
//! A plugin that holds up one turn longer than `[perf] threshold_ms` of
//! `settings.toml` (1000 by default) is named in a notice, once a session.
//! `/perf` shows the measurements, `/perf --json` prints them as JSON, and
//! `/perf reset` clears them; they are kept in memory only.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::time::Instant;

use rig_harness::prelude::*;
use rig_harness::timings::Timing;
use serde::{Deserialize, Serialize};

/// A frame whose work takes longer than this is an interface stall.
pub const STALL: Duration = Duration::from_millis(50);

/// The stalls kept, the latest.
const STALLS_KEPT: usize = 256;

/// Who did what nothing names a plugin for.
pub const UNATTRIBUTED: &str = "unattributed";

/// The `[perf]` table of `settings.toml`: `threshold_ms = 1000`.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct PerfSettings {
    /// A plugin that holds up a single turn longer than this, in
    /// milliseconds, is named in a notice, once a session.
    pub threshold_ms: u64,
}

impl Default for PerfSettings {
    fn default() -> Self {
        Self { threshold_ms: 1000 }
    }
}

/// One plugin's measurements.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct PluginPerf {
    /// How long it held up turns, in milliseconds.
    pub held_up_ms: u64,
    /// How long its work that held nothing up took, in milliseconds.
    pub background_ms: u64,
    /// How many times it was called: holds, effects, system and observer
    /// runs.
    pub calls: u64,
    /// Its slowest single call.
    pub slowest: Option<Slowest>,
}

/// A plugin's slowest single call.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Slowest {
    /// How long it took, in milliseconds.
    pub ms: u64,
    /// What it was handling, such as `before request #3, turn 2`.
    pub handling: String,
}

/// A frame that kept the interface from drawing and reacting to keys.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Stall {
    /// How long the frame took, in milliseconds.
    pub ms: u64,
    /// The plugin whose systems took most of it, else [`UNATTRIBUTED`].
    pub plugin: String,
    /// The keys the frame handled, if any.
    pub handling: Option<String>,
}

/// What `/perf --json` prints: per plugin, by name, for the last turn and
/// the session, and the interface stalls.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct PerfReport {
    /// The last turn of an agent the user talks to (not a subagent).
    pub last_turn: BTreeMap<String, PluginPerf>,
    /// The session, since the last `/perf reset`.
    pub session: BTreeMap<String, PluginPerf>,
    /// The stalls, oldest first.
    pub stalls: Vec<Stall>,
}

/// One plugin's measurements, to the nanosecond.
#[derive(Clone, Default)]
struct Tally {
    held_up: Duration,
    background: Duration,
    calls: u64,
    slowest: Option<(Duration, String)>,
}

/// Measurements by plugin name.
type Table = BTreeMap<String, Tally>;

/// The measurements, kept in memory only.
#[derive(Resource, Default)]
pub struct Perf {
    session: Table,
    /// The turns that run, and the last one.
    turns: HashMap<Entity, Table>,
    last_turn: Option<Entity>,
    stalls: Vec<Stall>,
    /// The plugins named in a notice this session.
    noticed: HashSet<String>,
    /// The holds that hold now, and the plugin effects that run.
    open: HashMap<Entity, Open>,
    /// The system timings per package at the end of the last frame.
    systems: BTreeMap<String, Timing>,
    /// When this frame's work began.
    frame: Option<Instant>,
    /// The keys this frame handled.
    keys: Vec<String>,
}

/// A hold, or a plugin's effect, being timed.
struct Open {
    started: Instant,
    handling: String,
    /// The turn it holds up, if any.
    turn: Option<Entity>,
}

impl Perf {
    /// The measurements as `/perf --json` prints them.
    pub fn report(&self) -> PerfReport {
        let last_turn = self.last_turn.and_then(|turn| self.turns.get(&turn));
        PerfReport {
            last_turn: last_turn.map(perfs).unwrap_or_default(),
            session: perfs(&self.session),
            stalls: self.stalls.clone(),
        }
    }

    /// Forgets what was measured; what is being timed now goes on.
    pub fn reset(&mut self) {
        self.session.clear();
        self.turns.clear();
        self.stalls.clear();
    }

    /// Charges `calls` taking `took`, the slowest `slowest`, to `plugin`,
    /// holding up `turn` if any, and returns how long `plugin` held up
    /// that turn in all.
    fn record(
        &mut self,
        plugin: &str,
        (took, calls): (Duration, u64),
        slowest: Option<(Duration, &str)>,
        turn: Option<Entity>,
    ) -> Option<Duration> {
        let held_up = turn.is_some();
        let session = self.session.entry(plugin.to_owned()).or_default();
        session.add(took, calls, held_up, slowest);
        let turn = self.turns.entry(turn?).or_default();
        let in_turn = turn.entry(plugin.to_owned()).or_default();
        in_turn.add(took, calls, held_up, slowest);
        Some(in_turn.held_up)
    }

    /// The text of a notice naming `plugin` for holding up a turn by
    /// `held_up` in all, the first time it goes over the threshold.
    fn over(&mut self, plugin: &str, held_up: Duration, threshold_ms: u64) -> Option<String> {
        let ms = millis(held_up);
        if ms <= threshold_ms || plugin == UNATTRIBUTED || !self.noticed.insert(plugin.to_owned()) {
            return None;
        }
        Some(format!(
            "{plugin} held up this turn by {ms} ms (over {threshold_ms} ms); /perf shows what it \
             was doing."
        ))
    }
}

impl Tally {
    fn add(
        &mut self,
        took: Duration,
        calls: u64,
        held_up: bool,
        slowest: Option<(Duration, &str)>,
    ) {
        if held_up {
            self.held_up += took;
        } else {
            self.background += took;
        }
        self.calls += calls;
        if let Some((took, handling)) = slowest
            && self.slowest.as_ref().is_none_or(|(most, _)| took > *most)
        {
            self.slowest = Some((took, handling.to_owned()));
        }
    }
}

/// `table` in whole milliseconds.
fn perfs(table: &Table) -> BTreeMap<String, PluginPerf> {
    let perf = |tally: &Tally| PluginPerf {
        held_up_ms: millis(tally.held_up),
        background_ms: millis(tally.background),
        calls: tally.calls,
        slowest: tally.slowest.as_ref().map(|(took, handling)| Slowest {
            ms: millis(*took),
            handling: handling.clone(),
        }),
    };
    table
        .iter()
        .map(|(plugin, tally)| (plugin.clone(), perf(tally)))
        .collect()
}

impl PerfReport {
    /// The report as `/perf` shows it.
    pub fn text(&self) -> String {
        let mut text = String::new();
        table(&mut text, "Last turn", &self.last_turn);
        table(&mut text, "Session", &self.session);
        let mut by_plugin: BTreeMap<&str, Vec<&Stall>> = BTreeMap::new();
        for stall in &self.stalls {
            by_plugin.entry(&stall.plugin).or_default().push(stall);
        }
        let over = STALL.as_millis();
        let none = if by_plugin.is_empty() { " none" } else { "" };
        writeln!(text, "Interface stalls (over {over} ms):{none}").ok();
        for (plugin, stalls) in by_plugin {
            let longest = stalls.iter().max_by_key(|stall| stall.ms);
            let ms = longest.map_or(0, |stall| stall.ms);
            let handling = longest.and_then(|stall| stall.handling.as_deref());
            let handling =
                handling.map_or(String::new(), |what| format!(" while handling: {what}"));
            let count = stalls.len();
            writeln!(
                text,
                "  {plugin}: {count} stalls, longest {ms} ms{handling}"
            )
            .ok();
        }
        text
    }
}

/// Appends `plugins`, slowest first, under `title`.
fn table(text: &mut String, title: &str, plugins: &BTreeMap<String, PluginPerf>) {
    if plugins.is_empty() {
        writeln!(text, "{title}: nothing measured").ok();
        return;
    }
    writeln!(text, "{title}:").ok();
    let mut rows: Vec<(&String, &PluginPerf)> = plugins.iter().collect();
    rows.sort_by_key(|(_, perf)| std::cmp::Reverse(perf.held_up_ms + perf.background_ms));
    for (plugin, perf) in rows {
        let PluginPerf {
            held_up_ms,
            background_ms,
            calls,
            slowest,
        } = perf;
        let slowest = slowest.as_ref().map_or(String::new(), |slowest| {
            format!(", slowest {} ms {}", slowest.ms, slowest.handling)
        });
        writeln!(
            text,
            "  {plugin}: held up {held_up_ms} ms, background {background_ms} ms, {calls} \
             calls{slowest}"
        )
        .ok();
    }
}

/// `took` in whole milliseconds.
fn millis(took: Duration) -> u64 {
    u64::try_from(took.as_millis()).unwrap_or(u64::MAX)
}

/// The name of the plugin `by` names, else [`UNATTRIBUTED`].
fn plugin_name(by: Option<&ProvidedBy>, names: &Query<&Name>) -> String {
    by.and_then(|&ProvidedBy(plugin)| names.get(plugin).ok())
        .map_or_else(|| UNATTRIBUTED.to_owned(), |name| name.as_str().to_owned())
}

/// Adds `/perf`, the `[perf]` settings, and the timing.
#[derive(Default)]
pub struct PluginProfilerPlugin;

impl Plugin for PluginProfilerPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<PerfSettings>("perf")
            .init_resource::<Perf>()
            .init_resource::<SystemTimings>()
            .add_command(
                "perf",
                "Which plugin slows the agent down: /perf [--json | reset]",
                perf,
            )
            .add_systems(First, frame_began)
            .add_systems(Last, frame_ended)
            .add_observer(turn_started)
            .add_observer(hold_added)
            .add_observer(hold_removed)
            .add_observer(effect_started)
            .add_observer(effect_answered)
            .add_observer(effect_gone);
        #[cfg(feature = "tui")]
        app.add_systems(PreUpdate, keys_seen.in_set(rig_tui::InputSystems::Filter));
    }
}

/// `/perf`, `/perf --json`, `/perf reset`.
fn perf(In(args): In<CommandArgs>, mut perf: ResMut<Perf>, mut notices: MessageWriter<Notice>) {
    let notice = match args.args.trim() {
        "" => Notice::info(args.agent, perf.report().text()),
        "--json" => match serde_json::to_string_pretty(&perf.report()) {
            Ok(json) => Notice::info(args.agent, json),
            Err(error) => Notice::error(args.agent, format!("/perf --json: {error}")),
        },
        "reset" => {
            perf.reset();
            Notice::info(args.agent, "Plugin timings cleared.")
        }
        other => Notice::error(
            args.agent,
            format!("/perf takes `--json` or `reset`, not `{other}`"),
        ),
    };
    notices.write(notice);
}

/// A turn of an agent the user talks to is the last turn from now on.
fn turn_started(
    started: On<Add<Turn>>,
    turns: Query<&TurnOf>,
    subagents: Query<(), With<SpawnedBy>>,
    running: Query<(), With<TurnOf>>,
    mut perf: ResMut<Perf>,
) {
    let turn = started.entity;
    let agent = turns.get(turn).ok().map(|&TurnOf(agent)| agent);
    if agent.is_some_and(|agent| subagents.contains(agent)) {
        return;
    }
    perf.last_turn = Some(turn);
    perf.turns.insert(turn, Table::default());
    let last = perf.last_turn;
    perf.turns
        .retain(|&turn, _| running.contains(turn) || Some(turn) == last);
}

/// What a held effect is, for [`hold_added`].
type HeldEffect = (
    Option<&'static ToolRequest>,
    Has<ModelRequest>,
    Has<Outcome>,
    Option<&'static Turn>,
    Option<&'static CallOf>,
    Has<Incoming>,
    Option<&'static EffectRequest>,
);

/// Starts timing a hold: what it holds and the turn it holds up.
fn hold_added(
    added: On<Add<HoldOf>>,
    holds: Query<(&HoldOf, Has<Ask>)>,
    effects: Query<HeldEffect>,
    calls: Query<&Calls>,
    models: Query<(), With<ModelRequest>>,
    turns: Query<&Turn>,
    mut perf: ResMut<Perf>,
) {
    let Ok((&HoldOf(held), false)) = holds.get(added.entity) else {
        return;
    };
    let Ok((tool, model, answered, turn, call_of, message, effect)) = effects.get(held) else {
        return;
    };
    let (turn_entity, turn_index) = match (turn, call_of) {
        (Some(turn), _) => (Some(held), Some(turn.index)),
        (None, Some(&CallOf(turn))) => (Some(turn), turns.get(turn).ok().map(|turn| turn.index)),
        (None, None) => (None, None),
    };
    let when = if answered { "result of" } else { "before" };
    let mut handling = if turn.is_some() {
        "turn end".to_owned()
    } else if model {
        let of_turn = turn_entity.and_then(|turn| calls.get(turn).ok());
        let calls = of_turn.into_iter().flat_map(|calls| calls.iter());
        let requests = calls.filter(|&call| models.contains(call));
        let number = requests.take_while(|&call| call != held).count() + 1;
        format!("{when} request #{number}")
    } else if let Some(tool) = tool {
        format!("{when} {} call", tool.call.function.name.as_str())
    } else if message {
        "before message".to_owned()
    } else if let Some(effect) = effect {
        format!("{when} {} effect", effect.kind.name())
    } else {
        "an effect".to_owned()
    };
    if let Some(index) = turn_index {
        write!(handling, ", turn {}", index + 1).ok();
    }
    let open = Open {
        started: Instant::now(),
        handling,
        turn: turn_entity,
    };
    perf.open.insert(added.entity, open);
}

/// Charges a hold that let go to the plugin that spawned it, and names a
/// plugin that held up a turn too long, once.
fn hold_removed(
    removed: On<Remove<HoldOf>>,
    holds: Query<Option<&ProvidedBy>>,
    names: Query<&Name>,
    agents: Query<&ForAgent>,
    settings: Res<PerfSettings>,
    mut perf: ResMut<Perf>,
    mut notices: MessageWriter<Notice>,
) {
    let Some(open) = perf.open.remove(&removed.entity) else {
        return;
    };
    let plugin = plugin_name(holds.get(removed.entity).ok().flatten(), &names);
    let took = open.started.elapsed();
    let slowest = Some((took, open.handling.as_str()));
    let Some(held_up) = perf.record(&plugin, (took, 1), slowest, open.turn) else {
        return;
    };
    if let Some(text) = perf.over(&plugin, held_up, settings.threshold_ms) {
        let agent = open.turn.and_then(|turn| agents.get(turn).ok());
        notices.write(Notice::warn(agent.map(|agent| agent.0), text));
    }
}

/// Starts timing a plugin's own effect.
fn effect_started(
    started: On<Add<Started>>,
    effects: Query<(&EffectRequest, Option<&CallOf>), Without<Ask>>,
    mut perf: ResMut<Perf>,
) {
    let Ok((effect, call_of)) = effects.get(started.entity) else {
        return;
    };
    let open = Open {
        started: Instant::now(),
        handling: format!("{} effect", effect.kind.name()),
        turn: call_of.map(|&CallOf(turn)| turn),
    };
    perf.open.insert(started.entity, open);
}

/// Charges a plugin's answered effect to the plugin that spawned it.
fn effect_answered(
    answered: On<Add<Outcome>>,
    effects: Query<Option<&ProvidedBy>, With<EffectRequest>>,
    names: Query<&Name>,
    mut perf: ResMut<Perf>,
) {
    let Ok(by) = effects.get(answered.entity) else {
        return;
    };
    let Some(open) = perf.open.remove(&answered.entity) else {
        return;
    };
    let plugin = plugin_name(by, &names);
    let took = open.started.elapsed();
    let slowest = Some((took, open.handling.as_str()));
    perf.record(&plugin, (took, 1), slowest, open.turn);
}

/// Forgets a plugin effect that went unanswered.
fn effect_gone(gone: On<Remove<EffectRequest>>, mut perf: ResMut<Perf>) {
    perf.open.remove(&gone.entity);
}

/// Marks when this frame's work began.
fn frame_began(mut perf: ResMut<Perf>) {
    perf.frame = Some(Instant::now());
    perf.keys.clear();
}

/// Notes the keys the terminal view handles this frame.
#[cfg(feature = "tui")]
fn keys_seen(events: Res<rig_tui::TerminalEvents>, mut perf: ResMut<Perf>) {
    use rig_tui::crossterm::event::Event;
    for item in &events.0 {
        if let rig_tui::KeyInput::Event(Event::Key(key)) = item {
            let press = rig_tui::KeyPress::of(key);
            perf.keys.push(format!("key {press}"));
        }
    }
}

/// Charges each plugin package's system and observer runs this frame to
/// its plugin, holding up the user's last turn while it runs, and keeps a
/// frame that took too long as a stall, naming the plugin that took most
/// of it.
fn frame_ended(
    timings: Res<SystemTimings>,
    plugins: Query<(&Name, &PluginSource)>,
    turns: Query<(&Turn, &TurnOf)>,
    settings: Res<PerfSettings>,
    mut perf: ResMut<Perf>,
    mut notices: MessageWriter<Notice>,
) {
    let now = timings.by_plugin(plugins.iter().map(|(_, source)| source));
    let before = std::mem::replace(&mut perf.systems, now.clone());
    let turn = perf
        .last_turn
        .and_then(|turn| Some((turn, turns.get(turn).ok()?)));
    let keys = perf.keys.join(", ");
    let handling = match turn {
        _ if !keys.is_empty() => format!("a system, handling {keys}"),
        Some((_, (turn, _))) => format!("a system, turn {}", turn.index + 1),
        None => "a system, between turns".to_owned(),
    };
    let mut most: Option<(Duration, String)> = None;
    for (package, timing) in &now {
        let Some(plugin) = plugin_of(package, &plugins) else {
            continue;
        };
        let was = before.get(package).copied().unwrap_or_default();
        let took = timing.total.saturating_sub(was.total);
        let runs = timing.runs.saturating_sub(was.runs);
        if runs == 0 {
            continue;
        }
        let slowest = (timing.max > was.max).then_some((timing.max, handling.as_str()));
        let held = perf.record(&plugin, (took, runs), slowest, turn.map(|(turn, _)| turn));
        if let Some(held_up) = held
            && let Some(text) = perf.over(&plugin, held_up, settings.threshold_ms)
        {
            notices.write(Notice::warn(turn.map(|(_, (_, agent))| agent.0), text));
        }
        if most.as_ref().is_none_or(|(most, _)| took > *most) {
            most = Some((took, plugin));
        }
    }
    let Some(took) = perf.frame.take().map(|began| began.elapsed()) else {
        return;
    };
    if took <= STALL {
        return;
    }
    let plugin = most
        .filter(|(took, _)| *took > STALL)
        .map_or_else(|| UNATTRIBUTED.to_owned(), |(_, plugin)| plugin);
    let handling = (!keys.is_empty()).then_some(keys);
    perf.stalls.push(Stall {
        ms: millis(took),
        plugin,
        handling,
    });
    if perf.stalls.len() > STALLS_KEPT {
        perf.stalls.remove(0);
    }
}

/// The plugin a package of [`SystemTimings::by_plugin`] is: its one
/// plugin's name, else the package; none for a package no plugin comes
/// from, such as `bevy_ecs` or the harness's own crates.
fn plugin_of(package: &str, plugins: &Query<(&Name, &PluginSource)>) -> Option<String> {
    let mut named = plugins.iter().filter(|(_, source)| source.krate == package);
    let (name, _) = named.next()?;
    let one = named.next().is_none();
    Some(if one { name.as_str() } else { package }.to_owned())
}

#[cfg(all(test, feature = "tui"))]
mod tests;
