//! What the timelines show: a turn's summary, its chart in the terminal,
//! and its Chrome trace, and the `/timeline` command.

use std::collections::HashMap;
use std::path::PathBuf;

use rig_harness::prelude::*;
use serde_json::{Value, json};

use super::{PhaseKind, Recorder, Span, SpanKind, Timelines, TurnTimeline};

/// How wide the chart's bars are, in columns.
const BAR: usize = 48;
/// How wide its labels are, in columns.
const LABEL: usize = 26;
/// How deep subagent turns are drawn and exported, at most.
const MAX_DEPTH: usize = 4;

/// Where a turn's wall time went. Each moment counts once, for what was
/// most in the way: the user, then a tool, then the model, else other
/// (plugin work, the harness, nothing).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    /// The turn's length.
    pub wall_ms: f64,
    /// Waiting on the model: for its first item and while it streamed.
    pub model_ms: f64,
    /// Running tool calls.
    pub tool_ms: f64,
    /// Waiting for the user's answers.
    pub user_ms: f64,
    /// The rest.
    pub other_ms: f64,
    /// Time during which two or more things ran at once.
    pub parallel_ms: f64,
}

impl Summary {
    /// The summary of `turn`.
    pub fn of(turn: &TurnTimeline) -> Self {
        let mut user = Vec::new();
        let mut tool = Vec::new();
        let mut model = Vec::new();
        let mut plugin = Vec::new();
        for span in &turn.spans {
            let phases = |kinds: &[PhaseKind]| -> Vec<(f64, f64)> {
                let mut phases = span.phases.iter();
                let within = phases.by_ref().filter(|phase| kinds.contains(&phase.kind));
                within.map(|phase| (phase.start_ms, phase.end_ms)).collect()
            };
            match span.kind {
                SpanKind::User => user.push((span.start_ms, span.end_ms)),
                SpanKind::Tool => tool.extend(phases(&[PhaseKind::Running])),
                SpanKind::Model => {
                    model.extend(phases(&[PhaseKind::FirstToken, PhaseKind::Streaming]));
                }
                SpanKind::Plugin => plugin.push((span.start_ms, span.end_ms)),
            }
        }
        let mut cuts: Vec<f64> = [&user, &tool, &model, &plugin]
            .into_iter()
            .flatten()
            .flat_map(|&(start, end)| [start, end])
            .chain([0.0, turn.wall_ms])
            .filter(|at| (0.0..=turn.wall_ms).contains(at))
            .collect();
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let mut summary = Self {
            wall_ms: turn.wall_ms,
            ..Self::default()
        };
        let during = |intervals: &[(f64, f64)], at: f64| {
            let all = intervals.iter();
            all.filter(|&&(start, end)| start <= at && at < end).count()
        };
        for pair in cuts.windows(2) {
            let [from, to] = pair else { continue };
            let (length, at) = (to - from, (from + to) / 2.0);
            let (users, tools, models) = (during(&user, at), during(&tool, at), during(&model, at));
            let busy = users + tools + models + during(&plugin, at);
            let bucket = match (users, tools, models) {
                (1.., _, _) => &mut summary.user_ms,
                (_, 1.., _) => &mut summary.tool_ms,
                (_, _, 1..) => &mut summary.model_ms,
                _ => &mut summary.other_ms,
            };
            *bucket += length;
            if busy >= 2 {
                summary.parallel_ms += length;
            }
        }
        summary
    }

    /// The share of the wall time during which two or more things ran at
    /// once, from 0 to 1.
    pub fn parallel_share(&self) -> f64 {
        if self.wall_ms > 0.0 {
            self.parallel_ms / self.wall_ms
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "model {} · tools {} · waiting on you {} · other {} · 2+ at once {:.0}% of {}",
            duration(self.model_ms),
            duration(self.tool_ms),
            duration(self.user_ms),
            duration(self.other_ms),
            self.parallel_share() * 100.0,
            duration(self.wall_ms),
        )
    }
}

/// `ms` for a reader: `120ms`, `4.2s`, `3m05s`, `1h02m`.
fn duration(ms: f64) -> String {
    let ms = ms.max(0.0);
    let seconds = (ms / 1000.0) as u64;
    match seconds {
        _ if ms < 1000.0 => format!("{ms:.0}ms"),
        0..60 => format!("{:.1}s", ms / 1000.0),
        60..3600 => format!("{}m{:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h{:02}m", seconds / 3600, seconds / 60 % 60),
    }
}

/// A row of the chart: a label, where its bar's parts lie in the turn
/// shown, and a note after the bar.
struct Row {
    depth: usize,
    label: String,
    parts: Vec<(f64, f64, char)>,
    note: String,
}

impl Row {
    fn end(&self) -> f64 {
        self.parts
            .iter()
            .map(|&(_, end, _)| end)
            .fold(0.0, f64::max)
    }
}

fn phase_mark(kind: PhaseKind) -> char {
    match kind {
        PhaseKind::Queued | PhaseKind::Waiting => '·',
        PhaseKind::Held => '░',
        PhaseKind::FirstToken => '▒',
        PhaseKind::Streaming | PhaseKind::Running => '█',
    }
}

/// The rows of `turn`, shifted by `offset`, its subagents' turns under the
/// calls that started them: one row each, or all of their rows when
/// `expand`.
fn rows_of(
    timelines: &Timelines,
    turn: &TurnTimeline,
    (offset, depth, expand): (f64, usize, bool),
    rows: &mut Vec<Row>,
) {
    let mut roots: Vec<&Span> = turn
        .spans
        .iter()
        .filter(|span| span.parent.is_none())
        .collect();
    roots.sort_by(|a, b| a.start_ms.total_cmp(&b.start_ms));
    let mut stack: Vec<(&Span, usize)> =
        roots.into_iter().rev().map(|span| (span, depth)).collect();
    while let Some((span, at)) = stack.pop() {
        let parts = match span.kind {
            SpanKind::User => vec![(span.start_ms, span.end_ms, '▓')],
            _ if span.phases.is_empty() => vec![(span.start_ms, span.end_ms, '░')],
            _ => span
                .phases
                .iter()
                .map(|phase| (phase.start_ms, phase.end_ms, phase_mark(phase.kind)))
                .collect(),
        };
        let parts = parts
            .into_iter()
            .map(|(start, end, mark)| (start + offset, end + offset, mark));
        rows.push(Row {
            depth: at,
            label: span.label.clone(),
            parts: parts.collect(),
            note: duration(span.duration_ms()),
        });
        for sub in timelines.nested_under(turn, span.id) {
            let start = offset + sub.under.as_ref().map_or(0.0, |under| under.offset_ms);
            rows.push(Row {
                depth: at + 1,
                label: format!("↳ {} turn {}", sub.name, sub.index),
                parts: vec![(start, start + sub.wall_ms, '▬')],
                note: format!("{}, {}", duration(sub.wall_ms), sub.outcome),
            });
            if expand && at < MAX_DEPTH {
                rows_of(timelines, sub, (start, at + 2, expand), rows);
            }
        }
        let mut children: Vec<&Span> = turn
            .spans
            .iter()
            .filter(|child| child.parent == Some(span.id))
            .collect();
        children.sort_by(|a, b| b.start_ms.total_cmp(&a.start_ms));
        stack.extend(children.into_iter().map(|child| (child, at + 1)));
    }
    for gap in &turn.suspended {
        rows.push(Row {
            depth,
            label: "suspended".to_owned(),
            parts: vec![(gap.at_ms + offset, gap.at_ms + offset, '‖')],
            note: format!("slept {}", duration(gap.for_ms)),
        });
    }
}

/// The step between the axis's marks that makes at most five of them.
fn axis_step(total: f64) -> f64 {
    const STEPS: [f64; 21] = [
        1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1e3, 2e3, 5e3, 1e4, 15e3, 3e4, 6e4,
        12e4, 3e5, 6e5, 18e5, 36e5,
    ];
    let fits = STEPS.into_iter().find(|step| total / step <= 5.0);
    fits.unwrap_or_else(|| (total / 5.0 / 36e5).ceil() * 36e5)
}

fn fit(text: &str, width: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    match chars.len() > width {
        true => {
            let kept: String = chars.iter().take(width.saturating_sub(1)).collect();
            format!("{kept}…")
        }
        false => format!("{text:width$}"),
    }
}

/// `turn` as a chart: one row per span under a time axis, its subagents'
/// turns one row each, or drawn in full when `expand`, then its summary.
pub fn render(timelines: &Timelines, turn: &TurnTimeline, expand: bool) -> String {
    let mut rows = Vec::new();
    rows_of(timelines, turn, (0.0, 0, expand), &mut rows);
    let total = rows
        .iter()
        .map(Row::end)
        .fold(turn.wall_ms, f64::max)
        .max(1.0);
    let column = |ms: f64| ((ms / total * BAR as f64) as usize).min(BAR);
    let mut lines = vec![format!(
        "Turn {} of {}: {}, {}",
        turn.index,
        turn.name,
        duration(turn.wall_ms),
        turn.outcome
    )];
    let step = axis_step(total);
    let mut marks: Vec<char> = vec![' '; BAR + 12];
    let mut axis: Vec<char> = vec!['·'; BAR];
    let mut free = 0;
    for tick in (0..)
        .map(|k| f64::from(k) * step)
        .take_while(|at| *at <= total)
    {
        let at = column(tick);
        if let Some(cell) = axis.get_mut(at) {
            *cell = '•';
        }
        let name: Vec<char> = duration(tick).chars().collect();
        if at >= free && at + name.len() <= marks.len() {
            for (cell, mark) in marks.iter_mut().skip(at).zip(&name) {
                *cell = *mark;
            }
            free = at + name.len() + 1;
        }
    }
    let pad = " ".repeat(LABEL + 1);
    lines.push(format!(
        "{pad}{}",
        marks.iter().collect::<String>().trim_end()
    ));
    lines.push(format!("{pad}{}", axis.iter().collect::<String>()));
    for row in &rows {
        let mut bar = vec![' '; BAR];
        for &(start, end, mark) in &row.parts {
            let from = column(start).min(BAR - 1);
            let to = column(end).max(from + 1);
            for cell in bar.iter_mut().take(to).skip(from) {
                *cell = mark;
            }
        }
        let label = format!("{}{}", "  ".repeat(row.depth), row.label);
        let bar: String = bar.into_iter().collect();
        lines.push(format!("{} {bar} {}", fit(&label, LABEL), row.note));
    }
    lines.push(Summary::of(turn).to_string());
    lines.push(
        "█ running or streaming  ▒ waiting for the model's first token  ░ held  · waiting  \
         ▓ waiting on you  ▬ a subagent's turn"
            .to_owned(),
    );
    lines.join("\n")
}

/// `turns` in Chrome's trace event format: a complete (`X`) event for each
/// turn and for each of its spans, each agent's spans on lanes of its own
/// that never overlap, in microseconds since the Unix epoch.
pub fn chrome_trace(turns: &[&TurnTimeline]) -> Value {
    let mut events = vec![json!({
        "name": "process_name", "ph": "M", "pid": 1, "args": { "name": "rig" },
    })];
    let mut agents: Vec<&str> = Vec::new();
    // Each agent's lanes: when the last span on each ends.
    let mut lanes: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut spans: Vec<(&TurnTimeline, &Span, f64)> = Vec::new();
    for turn in turns {
        if !agents.contains(&turn.agent.as_str()) {
            agents.push(&turn.agent);
            let tid = agents.len() * 1000;
            let name = format!("{} turns", turn.name);
            events.push(json!({
                "name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": { "name": name },
            }));
        }
        let base = turn.began_unix_ms as f64 * 1000.0;
        let tid = agents
            .iter()
            .position(|agent| *agent == turn.agent)
            .unwrap_or_default();
        events.push(json!({
            "name": format!("turn {} ({})", turn.index, turn.name), "cat": "turn", "ph": "X",
            "ts": base, "dur": turn.wall_ms * 1000.0, "pid": 1, "tid": (tid + 1) * 1000,
            "args": { "outcome": turn.outcome, "agent": turn.agent },
        }));
        spans.extend(
            turn.spans
                .iter()
                .map(|span| (*turn, span, base + span.start_ms * 1000.0)),
        );
    }
    spans.sort_by(|a, b| a.2.total_cmp(&b.2));
    for (turn, span, ts) in spans {
        let agent = agents
            .iter()
            .position(|agent| *agent == turn.agent)
            .unwrap_or_default();
        let ends = lanes.entry(&turn.agent).or_default();
        let dur = span.duration_ms() * 1000.0;
        let lane = match ends.iter().position(|end| *end <= ts) {
            Some(free) => free,
            None => {
                ends.push(ts);
                ends.len() - 1
            }
        };
        if let Some(end) = ends.get_mut(lane) {
            *end = ts + dur;
        }
        let phases: Vec<Value> = span
            .phases
            .iter()
            .map(|phase| json!([phase.kind, phase.start_ms, phase.end_ms]))
            .collect();
        events.push(json!({
            "name": span.label, "cat": span.kind, "ph": "X", "ts": ts, "dur": dur,
            "pid": 1, "tid": (agent + 1) * 1000 + lane + 1,
            "args": { "turn": turn.index, "phases_ms": phases },
        }));
    }
    json!({ "traceEvents": events, "displayTimeUnit": "ms" })
}

/// `turn` and every subagent turn nested under it, at most `MAX_DEPTH`
/// deep.
fn with_nested<'a>(timelines: &'a Timelines, turn: &'a TurnTimeline) -> Vec<&'a TurnTimeline> {
    let mut found = vec![turn];
    let mut level = vec![turn];
    for _ in 0..MAX_DEPTH {
        let next: Vec<&TurnTimeline> = level
            .iter()
            .flat_map(|turn| {
                turn.spans
                    .iter()
                    .flat_map(|span| timelines.nested_under(turn, span.id))
            })
            .collect();
        found.extend(&next);
        level = next;
    }
    found
}

/// What `/timeline` was asked for.
#[derive(Default)]
struct Asked {
    all: bool,
    expand: bool,
    session: bool,
    export: Option<String>,
    nth: Option<usize>,
}

fn parse(args: &str) -> Result<Asked, String> {
    let mut asked = Asked::default();
    let mut words = args.split_whitespace();
    while let Some(word) = words.next() {
        match word {
            "all" => asked.all = true,
            "expand" | "+" => asked.expand = true,
            "--session" => asked.session = true,
            "export" => match words.next() {
                Some(path) => asked.export = Some(path.to_owned()),
                None => return Err("/timeline export needs a file, such as trace.json".to_owned()),
            },
            n => match n.parse::<usize>() {
                Ok(n) if n > 0 => asked.nth = Some(n),
                _ => return Err(format!("/timeline: what is `{n}`?")),
            },
        }
    }
    Ok(asked)
}

/// `/timeline [N|all] [expand]` and `/timeline export FILE [N] [--session]`.
pub(crate) fn timeline(
    In(args): In<CommandArgs>,
    timelines: Res<Timelines>,
    recorder: Res<Recorder>,
    ids: Query<&AgentId>,
    root: Option<Res<ProjectRoot>>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let asked = match parse(&args.args) {
        Ok(asked) => asked,
        Err(why) => {
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let running = recorder.running();
    let Ok(id) = ids.get(agent) else {
        return;
    };
    let mut mine: Vec<&TurnTimeline> = timelines.of_agent(&id.0).collect();
    mine.extend(running.iter().filter(|turn| turn.agent == id.0));
    mine.sort_by_key(|turn| turn.index);
    if asked.all {
        let lines: Vec<String> = mine
            .iter()
            .map(|turn| {
                let summary = Summary::of(turn);
                format!(
                    "turn {:>3}  {:>7}  {:<8}  {summary}",
                    turn.index,
                    duration(turn.wall_ms),
                    turn.outcome
                )
            })
            .collect();
        let text = match lines.is_empty() {
            true => "No turn recorded yet.".to_owned(),
            false => lines.join("\n"),
        };
        notices.write(Notice::info(agent, text));
        return;
    }
    let nth = asked.nth.unwrap_or(1);
    let turn = mine
        .len()
        .checked_sub(nth)
        .and_then(|at| mine.get(at).copied());
    if let Some(path) = asked.export {
        let turns: Vec<&TurnTimeline> = match (asked.session, turn) {
            (true, _) => timelines.turns.iter().chain(&running).collect(),
            (false, Some(turn)) => with_nested(&timelines, turn),
            (false, None) => Vec::new(),
        };
        if turns.is_empty() {
            notices.write(Notice::error(agent, "No such turn recorded."));
            return;
        }
        let spans: usize = turns.iter().map(|turn| turn.spans.len()).sum();
        let path = root.map_or_else(|| PathBuf::from(&path), |root| root.0.join(&path));
        let written = serde_json::to_vec_pretty(&chrome_trace(&turns))
            .map_err(|error| error.to_string())
            .and_then(|bytes| std::fs::write(&path, bytes).map_err(|error| error.to_string()));
        match written {
            Ok(()) => notices.write(Notice::info(
                agent,
                format!(
                    "Wrote {} turns, {spans} spans, to {}: open it in https://ui.perfetto.dev or \
                     chrome://tracing.",
                    turns.len(),
                    path.display()
                ),
            )),
            Err(error) => {
                notices.write(Notice::error(agent, format!("{}: {error}", path.display())))
            }
        };
        return;
    }
    match turn {
        Some(turn) => notices.write(Notice::info(agent, render(&timelines, turn, asked.expand))),
        None => notices.write(Notice::info(agent, "No such turn recorded.")),
    };
}
