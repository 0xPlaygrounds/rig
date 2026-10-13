//! How long each system ran, as data: [`SystemTimings`], read per plugin
//! through [`PluginTimings`]. It is filled when the harness is built with
//! its `trace` feature, which turns on Bevy's spans around every system and
//! schedule run; a plugin crate turns it on by depending on rig-harness
//! with `features = ["trace"]`. The spans are info level, so a `RUST_LOG`
//! that sets `bevy_ecs` below info leaves it empty. A run is charged to the
//! plugin that added the system or observer ([`PluginSystems`],
//! [`ProvidedBy`]), and kept with the schedule it ran in: a run in
//! `AgentPass` (a stage system, or an observer its commands set off) is
//! work an agent's pass waited on.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! /// The plugins whose systems took longest, slowest first.
//! fn slowest(timings: PluginTimings, names: Query<&Name>) -> Vec<String> {
//!     let mut totals: Vec<_> = timings.totals().into_iter().collect();
//!     totals.sort_by_key(|(_, timing)| std::cmp::Reverse(timing.total));
//!     let mut lines = Vec::new();
//!     for ((plugin, schedule), timing) in totals {
//!         let plugin = plugin.and_then(|plugin| names.get(plugin).ok());
//!         let plugin = plugin.map_or("rig", Name::as_str);
//!         let schedule = schedule.unwrap_or_default();
//!         lines.push(format!("{plugin} in {schedule}: {:?}", timing.total));
//!     }
//!     lines
//! }
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::mem;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use bevy_ecs::observer::Observer;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_log::tracing::Subscriber;
use bevy_log::tracing::field::{Field, Visit};
use bevy_log::tracing::span::{Attributes, Id};
use bevy_log::tracing_subscriber::layer::{Context, Layer};
use bevy_log::tracing_subscriber::registry::LookupSpan;

use crate::{PluginSystems, ProvidedBy};

/// The name Bevy gives the span around a system's run.
const SYSTEM: &str = "system";

/// The name Bevy gives the span around a schedule's run.
const SCHEDULE: &str = "schedule";

/// How long each system ran, by the system's name, such as
/// `rig_tui::input::read_input`, and the schedule it ran in: totals so far,
/// and each run of the last whole frame. Clones share the timings; the log
/// fills them from other threads as systems run.
#[derive(Resource, Clone, Default)]
pub struct SystemTimings(Arc<Mutex<Timings>>);

/// What [`SystemTimings`] holds.
#[derive(Default)]
struct Timings {
    /// Per system and schedule.
    totals: BTreeMap<(String, Option<String>), Timing>,
    /// The runs of this frame so far, in order.
    frame: Vec<(String, Option<String>, Duration)>,
    /// The runs of the last whole frame, in order.
    last: Vec<(String, Option<String>, Duration)>,
    /// The labels of the schedules running now, the innermost last: they
    /// run one at a time, nested only inside exclusive systems.
    schedules: Vec<String>,
}

/// How often a system, or the systems of a plugin, ran and for how long.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timing {
    /// How many runs.
    pub runs: u64,
    /// Their time together.
    pub total: Duration,
    /// The longest run.
    pub max: Duration,
}

impl Timing {
    fn add(&mut self, took: Duration) {
        self.runs += 1;
        self.total += took;
        self.max = self.max.max(took);
    }
}

impl SystemTimings {
    fn lock(&self) -> MutexGuard<'_, Timings> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Each system's timing so far, by its name and its schedule's label.
    pub fn systems(&self) -> BTreeMap<(String, Option<String>), Timing> {
        self.lock().totals.clone()
    }

    /// Forgets every timing, such as to measure from now on.
    pub fn clear(&self) {
        let mut timings = self.lock();
        timings.totals.clear();
        timings.frame.clear();
        timings.last.clear();
    }

    /// Ends the frame: its runs become the last whole frame's.
    fn next_frame(&self) {
        let mut timings = self.lock();
        let frame = mem::take(&mut timings.frame);
        timings.last = frame;
    }

    fn record(&self, system: &str, took: Duration) {
        let mut timings = self.lock();
        let schedule = timings.schedules.last().cloned();
        let key = (system.to_owned(), schedule.clone());
        timings.totals.entry(key).or_default().add(took);
        timings.frame.push((system.to_owned(), schedule, took));
    }
}

/// Ends each frame of [`SystemTimings`], in `First`.
pub(crate) fn next_frame(timings: Res<SystemTimings>) {
    timings.next_frame();
}

/// [`SystemTimings`] per plugin: each run charged to the plugin that added
/// its system ([`PluginSystems`]) or observer ([`ProvidedBy`]); `None` for
/// the harness's own and Bevy's.
#[derive(SystemParam)]
pub struct PluginTimings<'w, 's> {
    timings: Res<'w, SystemTimings>,
    systems: Option<Res<'w, PluginSystems>>,
    observers: Query<'w, 's, (&'static Observer, &'static ProvidedBy)>,
}

/// One system run of the last whole frame, from [`PluginTimings::last_frame`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemRun {
    /// The system's name.
    pub system: String,
    /// The plugin that added it; `None` for the harness's own and Bevy's.
    pub plugin: Option<Entity>,
    /// The label of the schedule it ran in, such as `AgentPass`.
    pub schedule: Option<String>,
    /// How long it ran.
    pub took: Duration,
}

impl PluginTimings<'_, '_> {
    /// The timings so far per plugin and schedule.
    pub fn totals(&self) -> BTreeMap<(Option<Entity>, Option<String>), Timing> {
        let plugin = self.plugin_of();
        let mut totals = BTreeMap::<_, Timing>::new();
        for ((system, schedule), timing) in self.timings.lock().totals.iter() {
            let total = totals.entry((plugin(system, schedule), schedule.clone()));
            let total = total.or_default();
            total.runs += timing.runs;
            total.total += timing.total;
            total.max = total.max.max(timing.max);
        }
        totals
    }

    /// Each system run of the last whole frame, in order.
    pub fn last_frame(&self) -> Vec<SystemRun> {
        let plugin = self.plugin_of();
        let last = self.timings.lock().last.clone();
        let runs = last.into_iter().map(|(system, schedule, took)| SystemRun {
            plugin: plugin(&system, &schedule),
            system,
            schedule,
            took,
        });
        runs.collect()
    }

    /// The plugin a run of a system in a schedule is charged to.
    fn plugin_of(&self) -> impl Fn(&str, &Option<String>) -> Option<Entity> + '_ {
        let mut observers = HashMap::<String, Option<Entity>>::new();
        for (observer, &ProvidedBy(plugin)) in &self.observers {
            let by = observers.entry(observer.system_name().to_string());
            let by = by.or_insert(Some(plugin));
            if *by != Some(plugin) {
                *by = None;
            }
        }
        move |system, schedule| {
            let added = schedule.as_ref().and_then(|schedule| {
                let systems = self.systems.as_ref()?;
                systems.plugin(system, schedule)
            });
            added.or_else(|| observers.get(system).copied().flatten())
        }
    }
}

/// The log layer that times each system span from entering it to leaving
/// it into its [`SystemTimings`].
pub(crate) struct TimeSystems(pub(crate) SystemTimings);

/// A system span's name, and when it was entered while it runs.
struct Running {
    system: String,
    since: Option<Instant>,
}

/// A schedule span's label.
struct Scheduled(String);

/// Reads the `name` field of a system or schedule span.
#[derive(Default)]
struct SpanName(String);

impl Visit for SpanName {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "name" {
            value.clone_into(&mut self.0);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        if field.name() == "name" && self.0.is_empty() {
            self.0 = format!("{value:?}");
        }
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for TimeSystems {
    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
        let kind = attributes.metadata().name();
        if kind != SYSTEM && kind != SCHEDULE {
            return;
        }
        let mut name = SpanName::default();
        attributes.record(&mut name);
        let Some(span) = context.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        if kind == SYSTEM {
            extensions.insert(Running {
                system: name.0,
                since: None,
            });
        } else {
            extensions.insert(Scheduled(name.0));
        }
    }

    fn on_enter(&self, id: &Id, context: Context<'_, S>) {
        let Some(span) = context.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        if let Some(running) = extensions.get_mut::<Running>() {
            running.since = Some(Instant::now());
        } else if let Some(Scheduled(label)) = extensions.get_mut::<Scheduled>() {
            self.0.lock().schedules.push(label.clone());
        }
    }

    fn on_exit(&self, id: &Id, context: Context<'_, S>) {
        let Some(span) = context.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        if let Some(running) = extensions.get_mut::<Running>()
            && let Some(since) = running.since.take()
        {
            self.0.record(&running.system, since.elapsed());
        } else if extensions.get_mut::<Scheduled>().is_some() {
            self.0.lock().schedules.pop();
        }
    }
}

#[cfg(test)]
mod tests;
