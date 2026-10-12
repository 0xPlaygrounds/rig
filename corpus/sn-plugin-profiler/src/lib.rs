//! `/perf`: which plugin is slowing the agent down, and what it was doing.
//!
//! What it can measure, from the stack's public data:
//!
//! - **Holding up a turn.** A plugin that makes a turn wait holds an effect
//!   (`HoldOf`): a request before it goes out, a tool call before it
//!   starts, a result before it reaches the model, or the turn's end. Each
//!   hold's life is timed and charged to the plugin that spawned it
//!   (`ProvidedBy`, which a plugin puts on what it spawns while the app
//!   runs), with what it held ("before request #2, turn 3"). A question
//!   for the user (`Ask`) is the user's time, not a plugin's.
//! - **Background work.** A plugin's own effect (`EffectRequest`) is timed
//!   from its start to its answer: as holding up the turn when it is a call
//!   of one (`CallOf`), else as background work.
//! - **Interface stalls.** A frame whose work takes over 50 ms kept the
//!   terminal from drawing and from reacting to keys. With the `tui`
//!   feature, each key binding's handler is timed and charged to the plugin
//!   that bound it; a stall it caused names it, any other is unattributed.
//!
//! What it cannot measure: the time inside another plugin's stage systems
//! and observers. Bevy times no system without its `trace` feature, the
//! harness owns the log's tracing layers, and systems are not attributed to
//! plugins, so such work shows only as an unattributed stall.
//!
//! A plugin that holds up one turn longer than `[perf] threshold_ms` of
//! `settings.toml` (1000 by default) is named in a notice, once a session.
//! `/perf` shows the measurements, `/perf --json` prints them as JSON, and
//! `/perf reset` clears them; they are kept in memory only.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::time::Instant;

use rig_harness::prelude::*;
use serde::{Deserialize, Serialize};

/// A frame whose work takes longer than this is an interface stall.
pub const STALL: Duration = Duration::from_millis(50);

/// The stalls kept, the latest.
const STALLS_KEPT: usize = 256;

/// Who did what nothing names a plugin for.
pub const UNATTRIBUTED: &str = "unattributed";

/// What `/perf` says it cannot see.
const NOT_MEASURED: &str = "Not measured: time inside other plugins' stage systems and \
                            observers, which shows only as an unattributed stall.";

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
    /// How many times it was measured: holds, effects and key presses.
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
    /// The plugin that was running, else [`UNATTRIBUTED`].
    pub plugin: String,
    /// What it was handling, when known.
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

/// Measurements by plugin name.
type Table = BTreeMap<String, PluginPerf>;

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
    frame: Option<Instant>,
    /// The key handlers that ran this frame.
    handled: Vec<Handled>,
}

/// A hold, or a plugin's effect, being timed.
struct Open {
    started: Instant,
    handling: String,
    /// The turn it holds up, if any.
    turn: Option<Entity>,
}

/// A key handler that ran this frame.
struct Handled {
    plugin: String,
    handling: String,
    took: Duration,
}

impl Perf {
    /// The measurements as `/perf --json` prints them.
    pub fn report(&self) -> PerfReport {
        let last_turn = self.last_turn.and_then(|turn| self.turns.get(&turn));
        PerfReport {
            last_turn: last_turn.cloned().unwrap_or_default(),
            session: self.session.clone(),
            stalls: self.stalls.clone(),
        }
    }

    /// Forgets what was measured; what is being timed now goes on.
    pub fn reset(&mut self) {
        self.session.clear();
        self.turns.clear();
        self.stalls.clear();
    }

    /// Charges `took` to `plugin`, holding up `turn` if any, and returns
    /// how long `plugin` held up that turn in all.
    fn record(
        &mut self,
        plugin: &str,
        took: Duration,
        handling: &str,
        turn: Option<Entity>,
    ) -> Option<u64> {
        let ms = millis(took);
        let held_up = turn.is_some();
        let session = self.session.entry(plugin.to_owned()).or_default();
        session.add(ms, held_up, handling);
        let turn = self.turns.entry(turn?).or_default();
        let in_turn = turn.entry(plugin.to_owned()).or_default();
        in_turn.add(ms, held_up, handling);
        Some(in_turn.held_up_ms)
    }
}

impl PluginPerf {
    fn add(&mut self, ms: u64, held_up: bool, handling: &str) {
        if held_up {
            self.held_up_ms += ms;
        } else {
            self.background_ms += ms;
        }
        self.calls += 1;
        if self.slowest.as_ref().is_none_or(|slowest| ms > slowest.ms) {
            let handling = handling.to_owned();
            self.slowest = Some(Slowest { ms, handling });
        }
    }
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
        if by_plugin.is_empty() {
            writeln!(text, "Interface stalls (over {over} ms): none").ok();
        } else {
            writeln!(text, "Interface stalls (over {over} ms):").ok();
        }
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
        text.push_str(NOT_MEASURED);
        text
    }
}

/// Appends `plugins`, slowest first, under `title`.
fn table(text: &mut String, title: &str, plugins: &Table) {
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
        app.add_observer(key_pressed);
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
    let started = Instant::now();
    let open = Open {
        started,
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
    let in_turn = perf.record(&plugin, took, &open.handling, open.turn);
    let Some(held_up_ms) = in_turn else {
        return;
    };
    let threshold = settings.threshold_ms;
    if held_up_ms > threshold && plugin != UNATTRIBUTED && perf.noticed.insert(plugin.clone()) {
        let agent = open
            .turn
            .and_then(|turn| agents.get(turn).ok())
            .map(|agent| agent.0);
        let text = format!(
            "{plugin} held up this turn by {held_up_ms} ms (over {threshold} ms); /perf shows \
             what it was doing."
        );
        notices.write(Notice::warn(agent, text));
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
    perf.record(&plugin, open.started.elapsed(), &open.handling, open.turn);
}

/// Forgets a plugin effect that went unanswered.
fn effect_gone(gone: On<Remove<EffectRequest>>, mut perf: ResMut<Perf>) {
    perf.open.remove(&gone.entity);
}

/// Marks when this frame's work began.
fn frame_began(mut perf: ResMut<Perf>) {
    perf.frame = Some(Instant::now());
    perf.handled.clear();
}

/// Keeps a frame whose work took too long as a stall, naming the key
/// handler that took most of it, if one did.
fn frame_ended(mut perf: ResMut<Perf>) {
    let Some(began) = perf.frame.take() else {
        return;
    };
    let took = began.elapsed();
    let handled = std::mem::take(&mut perf.handled);
    if took <= STALL {
        return;
    }
    let culprit = handled
        .into_iter()
        .filter(|handled| handled.took > STALL)
        .max_by_key(|handled| handled.took);
    let (plugin, handling) = culprit.map_or((UNATTRIBUTED.to_owned(), None), |handled| {
        (handled.plugin, Some(handled.handling))
    });
    let ms = millis(took);
    perf.stalls.push(Stall {
        ms,
        plugin,
        handling,
    });
    if perf.stalls.len() > STALLS_KEPT {
        perf.stalls.remove(0);
    }
}

/// Times a key binding's handler: this observer runs before the binding's
/// own (Bevy runs an event's global observers before its target's), and
/// the command it queues runs after every observer of the press.
#[cfg(feature = "tui")]
fn key_pressed(
    pressed: On<rig_tui::KeyPressed>,
    bindings: Query<(&rig_tui::KeyBinding, Option<&ProvidedBy>)>,
    names: Query<&Name>,
    mut commands: Commands,
) {
    let Ok((binding, by)) = bindings.get(pressed.entity) else {
        return;
    };
    let plugin = plugin_name(by, &names);
    let handling = format!("key {}", binding.keys);
    let started = Instant::now();
    commands.queue(move |world: &mut World| {
        let took = started.elapsed();
        if let Some(mut perf) = world.get_resource_mut::<Perf>() {
            perf.record(&plugin, took, &handling, None);
            perf.handled.push(Handled {
                plugin,
                handling,
                took,
            });
        }
    });
}

#[cfg(test)]
mod tests;
