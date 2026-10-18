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
//!   observer run with the plugin that added it and the schedule it ran in
//!   ([`PluginTimings`], with rig-harness's `trace` feature, which this
//!   crate turns on). A run in `AgentPass` (a stage system, or an observer
//!   its commands set off) is work an agent's pass waited on: it holds up
//!   the user's turn while one runs. Any other run is background work.
//! - **Interface stalls.** A frame whose work takes over 50 ms kept the
//!   terminal from drawing and from reacting to keys. It names the plugin
//!   whose runs took most of it, with the keys the frame handled.
//!
//! The harness hands over a frame's runs in the next frame, so system time
//! and stalls are charged one frame late, with what the frame was doing.
//!
//! A plugin that holds up one turn longer than `[perf] threshold_ms` of
//! `settings.toml` (1000 by default) is named in a notice, once a session.
//! `/perf` shows the measurements, `/perf --json` prints them as JSON, and
//! `/perf reset` clears them; they are kept in memory only.
//!
//! It measures the app it is in: named as an entry's `plugin`, the
//! session's worker (and `/perf` reports it); named as its `view` too, the
//! terminal view's own frames and keys.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::mem;
use std::time::Instant;

use rig_harness::prelude::*;
use rig_harness::timings::SystemRun;
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
    /// How many times it was called: its holds and its effects.
    pub calls: u64,
    /// How many times its systems and observers ran.
    pub runs: u64,
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
    runs: u64,
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
    /// When this frame's work began.
    began: Option<Instant>,
    /// The keys this frame handled.
    keys: Vec<String>,
    /// The last frame, whose runs the harness hands over in this one.
    last_frame: Option<Frame>,
}

/// What a frame was doing, kept until its runs are handed over.
struct Frame {
    /// How long its work took.
    took: Option<Duration>,
    /// The keys it handled.
    keys: Vec<String>,
    /// The user's turn that ran in it, with the turn's number and agent.
    turn: Option<(Entity, u64, Entity)>,
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

    /// Charges `tally` to `plugin`, in `turn` if any, and returns how long
    /// `plugin` held up that turn in all.
    fn record(&mut self, plugin: &str, tally: &Tally, turn: Option<Entity>) -> Option<Duration> {
        self.session
            .entry(plugin.to_owned())
            .or_default()
            .add(tally);
        let turn = self.turns.entry(turn?).or_default();
        let in_turn = turn.entry(plugin.to_owned()).or_default();
        in_turn.add(tally);
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
    /// One call taking `took`, handling `handling`, holding up a turn or
    /// not.
    fn call(took: Duration, handling: String, held_up: bool) -> Self {
        let slowest = Some((took, handling));
        let mut tally = Self {
            calls: 1,
            slowest,
            ..Self::default()
        };
        *if held_up {
            &mut tally.held_up
        } else {
            &mut tally.background
        } = took;
        tally
    }

    fn add(&mut self, other: &Self) {
        self.held_up += other.held_up;
        self.background += other.background;
        self.calls += other.calls;
        self.runs += other.runs;
        if let Some((took, _)) = &other.slowest
            && self.slowest.as_ref().is_none_or(|(most, _)| took > most)
        {
            self.slowest.clone_from(&other.slowest);
        }
    }
}

/// `table` in whole milliseconds.
fn perfs(table: &Table) -> BTreeMap<String, PluginPerf> {
    let perf = |tally: &Tally| PluginPerf {
        held_up_ms: millis(tally.held_up),
        background_ms: millis(tally.background),
        calls: tally.calls,
        runs: tally.runs,
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
            runs,
            slowest,
        } = perf;
        let slowest = slowest.as_ref().map_or(String::new(), |slowest| {
            format!(", slowest {} ms {}", slowest.ms, slowest.handling)
        });
        writeln!(
            text,
            "  {plugin}: held up {held_up_ms} ms, background {background_ms} ms, {calls} \
             calls, {runs} runs{slowest}"
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
        app.add_systems(
            rig_tui::TuiKeys,
            keys_seen.in_set(rig_tui::InputSystems::Filter),
        );
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
    let tally = Tally::call(open.started.elapsed(), open.handling, open.turn.is_some());
    let Some(held_up) = perf.record(&plugin, &tally, open.turn) else {
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
    let tally = Tally::call(open.started.elapsed(), open.handling, open.turn.is_some());
    perf.record(&plugin, &tally, open.turn);
}

/// Forgets a plugin effect that went unanswered.
fn effect_gone(gone: On<Remove<EffectRequest>>, mut perf: ResMut<Perf>) {
    perf.open.remove(&gone.entity);
}

/// Marks when this frame's work began.
fn frame_began(mut perf: ResMut<Perf>) {
    perf.began = Some(Instant::now());
}

/// Notes the keys the terminal view handles this frame.
#[cfg(feature = "tui")]
fn keys_seen(events: Res<rig_tui::PendingInput>, mut perf: ResMut<Perf>) {
    use rig_tui::crossterm::event::Event;
    for item in &events.0 {
        if let rig_tui::KeyInput::Event(Event::Key(key)) = item {
            let press = rig_tui::KeyPress::of(key);
            perf.keys.push(format!("key {press}"));
        }
    }
}

/// Keeps what this frame did, and charges the last frame's system and
/// observer runs, which the harness hands over now, to their plugins: a
/// run in `AgentPass` while the user's turn ran held it up. Keeps the last
/// frame as a stall if it took too long, naming the plugin whose runs took
/// most of it.
fn frame_ended(
    timings: PluginTimings,
    plugins: Query<&Name, With<PluginSource>>,
    turns: Query<(&Turn, &TurnOf)>,
    settings: Res<PerfSettings>,
    mut perf: ResMut<Perf>,
    mut notices: MessageWriter<Notice>,
) {
    let turn = perf.last_turn.and_then(|turn| {
        let (Turn { index, .. }, &TurnOf(agent)) = turns.get(turn).ok()?;
        Some((turn, index + 1, agent))
    });
    let this = Frame {
        took: perf.began.take().map(|began| began.elapsed()),
        keys: mem::take(&mut perf.keys),
        turn,
    };
    let Some(frame) = perf.last_frame.replace(this) else {
        return;
    };
    let pass = format!("{AgentPass:?}");
    let mut charged = BTreeMap::<&str, (Tally, Option<SystemRun>)>::new();
    let runs = timings.last_frame();
    for run in &runs {
        let Some(plugin) = run.plugin.and_then(|plugin| plugins.get(plugin).ok()) else {
            continue;
        };
        let (tally, longest) = charged.entry(plugin.as_str()).or_default();
        if frame.turn.is_some() && run.schedule.as_deref() == Some(pass.as_str()) {
            tally.held_up += run.took;
        } else {
            tally.background += run.took;
        }
        tally.runs += 1;
        if longest
            .as_ref()
            .is_none_or(|longest| run.took > longest.took)
        {
            *longest = Some(run.clone());
        }
    }
    let keys = frame.keys.join(", ");
    let mut most: Option<(Duration, &str)> = None;
    for (plugin, (mut tally, longest)) in charged {
        tally.slowest = longest.map(|run| (run.took, handling(&run, &keys, frame.turn)));
        let turn = frame.turn.map(|(turn, ..)| turn);
        let held = perf.record(plugin, &tally, turn);
        if let Some(held_up) = held
            && let Some(text) = perf.over(plugin, held_up, settings.threshold_ms)
        {
            notices.write(Notice::warn(frame.turn.map(|(.., agent)| agent), text));
        }
        let took = tally.held_up + tally.background;
        if most.is_none_or(|(most, _)| took > most) {
            most = Some((took, plugin));
        }
    }
    let Some(took) = frame.took.filter(|&took| took > STALL) else {
        return;
    };
    let plugin = most.filter(|&(took, _)| took > STALL);
    perf.stalls.push(Stall {
        ms: millis(took),
        plugin: plugin.map_or(UNATTRIBUTED, |(_, plugin)| plugin).to_owned(),
        handling: (!keys.is_empty()).then_some(keys),
    });
    if perf.stalls.len() > STALLS_KEPT {
        perf.stalls.remove(0);
    }
}

/// What a system or observer run was handling: the system, its schedule,
/// and the frame's keys or turn.
fn handling(run: &SystemRun, keys: &str, turn: Option<(Entity, u64, Entity)>) -> String {
    let schedule = run.schedule.as_deref().unwrap_or("no schedule");
    let mut handling = format!("{} in {schedule}", run.system);
    if !keys.is_empty() {
        write!(handling, ", handling {keys}").ok();
    } else if let Some((_, number, _)) = turn {
        write!(handling, ", turn {number}").ok();
    }
    handling
}

#[cfg(all(test, feature = "tui"))]
mod tests;
