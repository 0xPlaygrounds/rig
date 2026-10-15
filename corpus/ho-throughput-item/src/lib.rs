//! Live model speed in the status line: while a turn's model call streams,
//! its agent's item shows the time to the first token and the tokens per
//! second (`TTFT 1.2 s · 54 tok/s`), or `waiting 3 s` before the first
//! token, and keeps the last call's numbers, dimmed, once it ends.
//! `/speed` lists every model's calls, median and p90 time to first token
//! and median tokens per second, kept with the session.
//!
//! Tokens are estimated as a quarter of the characters streamed (text,
//! reasoning and tool arguments, from `StreamDelta`), and corrected to the
//! provider's output tokens when its usage arrives. Times are the kernel's
//! own stamps on the call: `Started` and `StreamTiming`.

use std::collections::BTreeMap;
use std::time::SystemTime;

use rig_harness::prelude::*;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

/// After the agent's status, before its subagents at work.
const SPEED: StatusItem = StatusItem::new("ho-throughput-item", Side::Left, 55, 9);
/// How often the live item may change.
const EVERY: Duration = Duration::from_millis(250);
/// The samples kept per model for the medians.
const SAMPLES: usize = 500;

/// The Review system that adds up what each model call streamed, for
/// others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThroughputSystems;

#[derive(Default)]
pub struct ThroughputPlugin;

impl Plugin for ThroughputPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SpeedTable::default())
            .register_required_components::<Agent, LiveSpeed>()
            .register_required_components::<ModelRequest, Tally>()
            .configure_sets(AgentPass, ThroughputSystems.in_set(Stage::Review))
            .add_systems(AgentPass, tally_deltas.in_set(ThroughputSystems))
            .add_systems(PostUpdate, show.in_set(StatusSystems::Show))
            .add_observer(went_live)
            .add_observer(answered)
            .add_observer(settled)
            .add_command(
                "speed",
                "Time to first token and tokens per second, per model",
                speed,
            );
    }
}

/// Each model's speed, kept with the session.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct SpeedTable(pub BTreeMap<String, ModelSpeed>);

/// One model's calls and their speed.
#[derive(Reflect, Clone, Debug, Default, PartialEq)]
pub struct ModelSpeed {
    /// The calls answered.
    pub calls: u64,
    /// The attempts that failed, counted apart.
    pub failed: u64,
    /// The output tokens of the calls answered.
    pub output_tokens: u64,
    /// The latest calls' times to first token.
    pub first_token: Vec<Duration>,
    /// The latest calls' tokens per second, from the first token on.
    pub tokens_per_second: Vec<f64>,
}

impl ModelSpeed {
    /// A call answered with `tokens`, and its numbers if it streamed.
    fn record(&mut self, done: Option<Done>, tokens: u64) {
        self.calls += 1;
        self.output_tokens += tokens;
        if let Some(done) = done {
            keep_latest(&mut self.first_token, done.first_token);
        }
        if let Some(rate) = done.and_then(|done| done.rate) {
            keep_latest(&mut self.tokens_per_second, rate);
        }
    }
}

fn keep_latest<T>(samples: &mut Vec<T>, sample: T) {
    samples.push(sample);
    if samples.len() > SAMPLES {
        samples.remove(0);
    }
}

/// On every model call: the characters of the text, reasoning and tool
/// arguments it streamed so far, and when its outcome came.
#[derive(Component, Default)]
struct Tally {
    chars: u64,
    ended: Option<SystemTime>,
}

impl Tally {
    /// The tokens streamed, estimated.
    fn tokens(&self) -> u64 {
        self.chars / 4
    }
}

/// An agent's item: its turn's call streaming now, and the numbers of its
/// last call answered.
#[derive(Component, Default)]
struct LiveSpeed {
    call: Option<Entity>,
    last: Option<Done>,
    /// When the item last changed, on Bevy's clock; `None` to change it
    /// now.
    shown: Option<Duration>,
}

/// A finished call's numbers.
#[derive(Clone, Copy, Debug)]
struct Done {
    first_token: Duration,
    rate: Option<f64>,
}

fn chars(text: &str) -> u64 {
    u64::try_from(text.chars().count()).unwrap_or(u64::MAX)
}

/// Adds what each call streamed to its tally, in the pass it arrived, so
/// it is all there when the call settles.
fn tally_deltas(mut deltas: MessageReader<StreamDelta>, mut calls: Query<&mut Tally>) {
    for delta in deltas.read() {
        let text = match &delta.event {
            StreamEvent::Text { text, .. } | StreamEvent::Reasoning { text, .. } => text,
            StreamEvent::Arguments { json, .. } => json,
            _ => continue,
        };
        if let Ok(mut tally) = calls.get_mut(delta.effect) {
            tally.chars += chars(text);
        }
    }
}

/// A turn's call that started is its agent's live call; the loop stays
/// awake for its item while it runs.
fn went_live(
    started: On<Add<Started>>,
    calls: Query<&ForAgent, (With<TurnCall>, With<Tally>)>,
    mut agents: Query<&mut LiveSpeed>,
    mut commands: Commands,
) {
    let call = started.entity;
    let Ok(&ForAgent(agent)) = calls.get(call) else {
        return;
    };
    if let Ok(mut live) = agents.get_mut(agent) {
        live.call = Some(call);
        live.shown = None;
        commands.entity(call).insert(KeepAwake(EVERY));
    }
}

/// When each call's outcome came.
fn answered(answered: On<Add<Outcome>>, mut calls: Query<&mut Tally>) {
    if let Ok(mut tally) = calls.get_mut(answered.entity) {
        tally.ended = Some(SystemTime::now());
    }
}

/// Adds each settled call to its model's speed, or to its failed attempts,
/// and a turn's call to its agent's item.
fn settled(
    settled: On<Add<Settled>>,
    calls: Query<(
        &Tally,
        &Started,
        Option<&StreamTiming>,
        &SentTo,
        &Outcome,
        Option<&ForAgent>,
        Has<TurnCall>,
    )>,
    mut agents: Query<&mut LiveSpeed>,
    mut table: ResMut<SpeedTable>,
    mut commands: Commands,
) {
    let call = settled.entity;
    let Ok((tally, started, first, sent, outcome, agent, turn)) = calls.get(call) else {
        return;
    };
    commands.entity(call).remove::<KeepAwake>();
    let speed = table.0.entry(sent.0.spec.reference()).or_default();
    let done = match outcome
        .completion()
        .filter(|response| response.error.is_none())
    {
        Some(response) => {
            let tokens = response
                .usage
                .output_tokens
                .unwrap_or_else(|| tally.tokens());
            let done = first.map(|first| finished(started, first, tally.ended, tokens));
            speed.record(done, tokens);
            done
        }
        None => {
            speed.failed += 1;
            None
        }
    };
    let live = agent
        .filter(|_| turn)
        .and_then(|agent| agents.get_mut(agent.0).ok());
    if let Some(mut live) = live
        && live.call == Some(call)
    {
        live.call = None;
        live.last = done.or(live.last);
        live.shown = None;
    }
}

/// The numbers of a call that started at `started`, streamed its first
/// item at `first` and `tokens` tokens until `ended`.
fn finished(
    started: &Started,
    first: &StreamTiming,
    ended: Option<SystemTime>,
    tokens: u64,
) -> Done {
    let first_token = first.first.duration_since(started.at).unwrap_or_default();
    let streaming = ended.and_then(|ended| ended.duration_since(first.first).ok());
    Done {
        first_token,
        rate: streaming.and_then(|streaming| rate(tokens, streaming)),
    }
}

fn rate(tokens: u64, over: Duration) -> Option<f64> {
    (!over.is_zero()).then(|| tokens as f64 / over.as_secs_f64())
}

/// `TTFT 1.2 s · 54 tok/s`.
fn label(done: Done) -> String {
    let rate = done
        .rate
        .map_or_else(|| "–".to_owned(), |rate| format!("{rate:.0}"));
    format!(
        "TTFT {:.1} s · {rate} tok/s",
        done.first_token.as_secs_f64()
    )
}

/// Each agent's item: its live call at most every `EVERY`, its last call
/// dimmed once it ended.
fn show(
    mut agents: Query<(&mut LiveSpeed, &mut StatusItems)>,
    calls: Query<(&Tally, &Started, Option<&StreamTiming>)>,
    time: Res<Time<Real>>,
) {
    let now = time.elapsed();
    for (mut live, mut items) in &mut agents {
        let due = live
            .shown
            .is_none_or(|shown| now.saturating_sub(shown) >= EVERY);
        if !due || (live.call.is_none() && live.shown.is_some()) {
            continue;
        }
        let item = match live.call.and_then(|call| calls.get(call).ok()) {
            Some((_, started, None)) => {
                let waited = SystemTime::now()
                    .duration_since(started.at)
                    .unwrap_or_default();
                SPEED.says(format!("waiting {} s", waited.as_secs()), Tone::Yellow)
            }
            Some((tally, started, Some(first))) => {
                let done = finished(started, first, Some(SystemTime::now()), tally.tokens());
                SPEED.says(label(done), Tone::Plain)
            }
            None => SPEED.says(live.last.map(label).unwrap_or_default(), Tone::Dim),
        };
        items.show(item);
        live.shown = Some(now);
    }
}

/// `/speed`: each model's calls, median and p90 time to first token, and
/// median tokens per second.
fn speed(In(args): In<CommandArgs>, table: Res<SpeedTable>, mut notices: MessageWriter<Notice>) {
    if table.0.is_empty() {
        notices.write(Notice::info(args.agent, "No model calls yet."));
        return;
    }
    let lines: Vec<String> = table
        .0
        .iter()
        .map(|(model, speed)| {
            let first: Vec<f64> = speed
                .first_token
                .iter()
                .map(Duration::as_secs_f64)
                .collect();
            let seconds =
                |at| percentile(&first, at).map_or("–".to_owned(), |s| format!("{s:.1} s"));
            let rate = percentile(&speed.tokens_per_second, 0.5)
                .map_or("–".to_owned(), |rate| format!("{rate:.0}"));
            let failed = match speed.failed {
                0 => String::new(),
                failed => format!(", {failed} failed"),
            };
            format!(
                "{model}: {} calls{failed} · TTFT median {} p90 {} · median {rate} tok/s",
                speed.calls,
                seconds(0.5),
                seconds(0.9),
            )
        })
        .collect();
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

/// The `at` percentile of `samples` (nearest rank), if any.
fn percentile(samples: &[f64], at: f64) -> Option<f64> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (at * sorted.len() as f64).ceil() as usize;
    sorted.get(rank.saturating_sub(1)).copied()
}

#[cfg(test)]
mod tests;
