//! The metrics the app records and the scrape thread renders, in the
//! Prometheus text format (version 0.0.4).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::SystemTime;

/// The most distinct values one label takes; later ones are `other`.
pub const MAX_LABEL_VALUES: usize = 100;

/// A label's values, bounded to [`MAX_LABEL_VALUES`].
#[derive(Default)]
struct Bounded(BTreeSet<String>);

impl Bounded {
    /// `value`, or `other` once the label has its maximum of others.
    fn fold(&mut self, value: &str) -> String {
        if self.0.contains(value) || self.0.len() < MAX_LABEL_VALUES {
            self.0.insert(value.to_owned());
            value.to_owned()
        } else {
            "other".to_owned()
        }
    }
}

/// A label set: names and values, the `session` label left to rendering.
pub type Labels = Vec<(&'static str, String)>;

/// A metric family: its name, type and help.
struct Family {
    name: &'static str,
    kind: &'static str,
    help: &'static str,
}

/// Every family, in the order they are rendered.
const FAMILIES: &[Family] = &[
    family(MODEL_REQUESTS, "counter", "Model requests sent, by model."),
    family(
        INPUT_TOKENS,
        "counter",
        "Input tokens the providers reported, cached ones included, by model.",
    ),
    family(
        OUTPUT_TOKENS,
        "counter",
        "Output tokens the providers reported, by model.",
    ),
    family(
        CACHED_TOKENS,
        "counter",
        "Input tokens read from a provider's cache, by model.",
    ),
    family(
        COST,
        "counter",
        "Known cost of model requests in USD, by model.",
    ),
    family(TOOL_CALLS, "counter", "Tool calls settled, by tool."),
    family(
        TOOL_ERRORS,
        "counter",
        "Tool calls that failed to execute, by tool.",
    ),
    family(
        PLUGIN_ERRORS,
        "counter",
        "Errors the process logged, by the plugin that logged them.",
    ),
    family(
        RETRIES,
        "counter",
        "Failed model calls the agent retried, by model.",
    ),
    family(
        TURN_DURATION,
        "histogram",
        "How long turns took, in seconds.",
    ),
    family(
        FIRST_TOKEN,
        "histogram",
        "Time from sending a model request to its first streamed item, in seconds, by model.",
    ),
    family(
        TOOL_DURATION,
        "histogram",
        "How long tool calls ran, in seconds, by tool.",
    ),
    family(AGENTS_BUSY, "gauge", "Agents with a running turn."),
    family(
        SUBAGENTS_RUNNING,
        "gauge",
        "Spawned agents with a running turn.",
    ),
    family(
        APPROVALS_WAITING,
        "gauge",
        "Approval questions waiting for the user.",
    ),
    family(
        CONTEXT_FILL,
        "gauge",
        "The share of its model's input limit an agent's context fills, by agent.",
    ),
    family(
        SINCE_ACTIVITY,
        "gauge",
        "Seconds since an agent's last model or tool activity, by agent.",
    ),
];

const fn family(name: &'static str, kind: &'static str, help: &'static str) -> Family {
    Family { name, kind, help }
}

/// `rig_model_requests_total`.
pub const MODEL_REQUESTS: &str = "rig_model_requests_total";
/// `rig_input_tokens_total`.
pub const INPUT_TOKENS: &str = "rig_input_tokens_total";
/// `rig_output_tokens_total`.
pub const OUTPUT_TOKENS: &str = "rig_output_tokens_total";
/// `rig_cached_input_tokens_total`.
pub const CACHED_TOKENS: &str = "rig_cached_input_tokens_total";
/// `rig_cost_usd_total`.
pub const COST: &str = "rig_cost_usd_total";
/// `rig_tool_calls_total`.
pub const TOOL_CALLS: &str = "rig_tool_calls_total";
/// `rig_tool_errors_total`.
pub const TOOL_ERRORS: &str = "rig_tool_errors_total";
/// `rig_plugin_errors_total`.
pub const PLUGIN_ERRORS: &str = "rig_plugin_errors_total";
/// `rig_retries_total`.
pub const RETRIES: &str = "rig_retries_total";
/// `rig_turn_duration_seconds`.
pub const TURN_DURATION: &str = "rig_turn_duration_seconds";
/// `rig_model_time_to_first_token_seconds`.
pub const FIRST_TOKEN: &str = "rig_model_time_to_first_token_seconds";
/// `rig_tool_call_duration_seconds`.
pub const TOOL_DURATION: &str = "rig_tool_call_duration_seconds";
/// `rig_agents_busy`.
pub const AGENTS_BUSY: &str = "rig_agents_busy";
/// `rig_subagents_running`.
pub const SUBAGENTS_RUNNING: &str = "rig_subagents_running";
/// `rig_approvals_waiting`.
pub const APPROVALS_WAITING: &str = "rig_approvals_waiting";
/// `rig_context_fill_ratio`.
pub const CONTEXT_FILL: &str = "rig_context_fill_ratio";
/// `rig_seconds_since_last_activity`.
pub const SINCE_ACTIVITY: &str = "rig_seconds_since_last_activity";

/// The bucket bounds of each histogram, in seconds.
fn buckets(name: &str) -> &'static [f64] {
    match name {
        TURN_DURATION => &[
            0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0,
        ],
        FIRST_TOKEN => &[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        _ => &[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
    }
}

/// A histogram's cumulative bucket counts, sum and count.
#[derive(Clone, Default)]
struct Histogram {
    cumulative: Vec<u64>,
    sum: f64,
    count: u64,
}

/// What the process recorded, shared by the app and the scrape thread.
#[derive(Default)]
pub struct Registry {
    session: String,
    counters: BTreeMap<(&'static str, Labels), f64>,
    histograms: BTreeMap<(&'static str, Labels), Histogram>,
    gauges: BTreeMap<(&'static str, Labels), f64>,
    /// Each agent's last activity, by its label.
    activity: BTreeMap<String, SystemTime>,
    labels: BTreeMap<&'static str, Bounded>,
}

impl Registry {
    /// `value` as a value of the label `name`, bounded.
    pub fn label(&mut self, name: &'static str, value: &str) -> (&'static str, String) {
        (name, self.labels.entry(name).or_default().fold(value))
    }

    /// Sets the `session` label of every series.
    pub fn set_session(&mut self, session: &str) {
        session.clone_into(&mut self.session);
    }

    /// Adds `by` to the counter `name` with `labels`.
    pub fn add(&mut self, name: &'static str, labels: Labels, by: f64) {
        *self.counters.entry((name, labels)).or_default() += by;
    }

    /// Observes `seconds` in the histogram `name` with `labels`.
    pub fn observe(&mut self, name: &'static str, labels: Labels, seconds: f64) {
        let bounds = buckets(name);
        let histogram = self.histograms.entry((name, labels)).or_default();
        histogram.cumulative.resize(bounds.len(), 0);
        for (count, bound) in histogram.cumulative.iter_mut().zip(bounds) {
            if seconds <= *bound {
                *count += 1;
            }
        }
        histogram.sum += seconds;
        histogram.count += 1;
    }

    /// Replaces every gauge with `gauges`.
    pub fn set_gauges(&mut self, gauges: impl IntoIterator<Item = ((&'static str, Labels), f64)>) {
        self.gauges = gauges.into_iter().collect();
    }

    /// Notes activity of the agent labelled `agent` at `at`.
    pub fn active(&mut self, agent: String, at: SystemTime) {
        self.activity.insert(agent, at);
    }

    /// Forgets the activity of agents not in `agents`.
    pub fn keep_activity(&mut self, agents: &BTreeSet<String>) {
        self.activity.retain(|agent, _| agents.contains(agent));
    }

    /// The text a scrape at `now` returns.
    pub fn render(&self, now: SystemTime) -> String {
        let mut out = String::new();
        for family in FAMILIES {
            let _ = writeln!(out, "# HELP {} {}", family.name, family.help);
            let _ = writeln!(out, "# TYPE {} {}", family.name, family.kind);
            for (key, value) in self.counters.iter().chain(&self.gauges) {
                if key.0 == family.name {
                    self.sample(&mut out, family.name, &key.1, None, *value);
                }
            }
            if family.name == SINCE_ACTIVITY {
                for (agent, at) in &self.activity {
                    let since = now.duration_since(*at).unwrap_or_default().as_secs_f64();
                    let labels = vec![("agent", agent.clone())];
                    self.sample(&mut out, family.name, &labels, None, since);
                }
            }
            for (key, histogram) in &self.histograms {
                if key.0 != family.name {
                    continue;
                }
                let bucket = format!("{}_bucket", family.name);
                for (count, bound) in histogram.cumulative.iter().zip(buckets(family.name)) {
                    let le = Some(bound.to_string());
                    self.sample(&mut out, &bucket, &key.1, le, *count as f64);
                }
                let all = Some("+Inf".to_owned());
                self.sample(&mut out, &bucket, &key.1, all, histogram.count as f64);
                let sum = format!("{}_sum", family.name);
                self.sample(&mut out, &sum, &key.1, None, histogram.sum);
                let count = format!("{}_count", family.name);
                self.sample(&mut out, &count, &key.1, None, histogram.count as f64);
            }
        }
        out
    }

    /// One sample line, with the session label first and `le` last.
    fn sample(
        &self,
        out: &mut String,
        name: &str,
        labels: &Labels,
        le: Option<String>,
        value: f64,
    ) {
        let mut pairs = vec![format!("session=\"{}\"", escape(&self.session))];
        pairs.extend(
            labels
                .iter()
                .map(|(label, value)| format!("{label}=\"{}\"", escape(value))),
        );
        pairs.extend(le.map(|le| format!("le=\"{le}\"")));
        let _ = writeln!(out, "{name}{{{}}} {value}", pairs.join(","));
    }
}

/// A label value escaped as the text format wants it.
fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// The [`Registry`], shared with the scrape thread.
#[derive(bevy_ecs::resource::Resource, Clone, Default)]
pub struct SharedRegistry(pub Arc<Mutex<Registry>>);

impl SharedRegistry {
    /// The registry, locked; a poisoned lock is taken over, since every
    /// update leaves it whole.
    pub fn lock(&self) -> MutexGuard<'_, Registry> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
