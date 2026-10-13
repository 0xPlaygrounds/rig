//! How long each system ran, as data: [`SystemTimings`]. It is filled when
//! the harness is built with its `trace` feature, which turns on Bevy's
//! span around every system run; a plugin crate turns it on by depending
//! on rig-harness with `features = ["trace"]`. The spans are info level,
//! so a `RUST_LOG` that sets `bevy_ecs` below info leaves it empty.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! /// The plugin packages that took longest, slowest first.
//! fn slowest(timings: Res<SystemTimings>, plugins: Query<&PluginSource>) -> Vec<String> {
//!     let mut packages: Vec<_> = timings.by_plugin(&plugins).into_iter().collect();
//!     packages.sort_by_key(|(_, timing)| std::cmp::Reverse(timing.total));
//!     packages.into_iter().map(|(package, timing)| format!("{package}: {:?}", timing.total)).collect()
//! }
//! ```

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use bevy_log::tracing::Subscriber;
use bevy_log::tracing::field::{Field, Visit};
use bevy_log::tracing::span::{Attributes, Id};
use bevy_log::tracing_subscriber::layer::{Context, Layer};
use bevy_log::tracing_subscriber::registry::LookupSpan;

use crate::PluginSource;

/// The name Bevy gives the span around a system's run.
const SYSTEM: &str = "system";

/// How long each system ran so far, by the system's name, such as
/// `rig_tui::input::read_input`. Clones share the timings; the log fills
/// them from another thread as systems run.
#[derive(Resource, Clone, Default)]
pub struct SystemTimings(Arc<Mutex<BTreeMap<String, Timing>>>);

/// How often a system, or the systems of a package, ran and for how long.
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
    fn add(&mut self, other: Self) {
        self.runs += other.runs;
        self.total += other.total;
        self.max = self.max.max(other.max);
    }
}

impl SystemTimings {
    fn lock(&self) -> MutexGuard<'_, BTreeMap<String, Timing>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Each system's timing so far, by its name.
    pub fn systems(&self) -> BTreeMap<String, Timing> {
        self.lock().clone()
    }

    /// Forgets every timing, such as to measure from now on.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// The timings summed per package: a system counts for the package of
    /// `plugins` (such as every [`PluginSource`] in the world) whose crate
    /// holds it, by the crate its name starts with (`rig_tui` for the
    /// package `rig-tui`), else for that crate, such as `bevy_ecs`.
    pub fn by_plugin<'a>(
        &self,
        plugins: impl IntoIterator<Item = &'a PluginSource>,
    ) -> BTreeMap<String, Timing> {
        let packages: BTreeMap<String, &str> = plugins
            .into_iter()
            .map(|source| (source.krate.replace('-', "_"), source.krate.as_str()))
            .collect();
        let mut summed = BTreeMap::<String, Timing>::new();
        for (system, timing) in self.lock().iter() {
            let krate = crate_of(system);
            let package = packages.get(krate).copied().unwrap_or(krate);
            summed.entry(package.to_owned()).or_default().add(*timing);
        }
        summed
    }

    fn record(&self, system: &str, took: Duration) {
        let mut timings = self.lock();
        let timing = match timings.get_mut(system) {
            Some(timing) => timing,
            None => timings.entry(system.to_owned()).or_default(),
        };
        timing.add(Timing {
            runs: 1,
            total: took,
            max: took,
        });
    }
}

/// The crate a system's name starts with, such as `rig_tui` in
/// `rig_tui::input::read_input` or `<rig_tui::View as ..>::draw`.
fn crate_of(system: &str) -> &str {
    let path = system.trim_start_matches('<');
    path.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .next()
        .unwrap_or(path)
}

/// The log layer that times each system span from entering it to leaving
/// it into its [`SystemTimings`].
pub(crate) struct TimeSystems(pub(crate) SystemTimings);

/// A system span's name, and when it was entered while it runs.
struct Running {
    system: String,
    since: Option<Instant>,
}

/// Reads the `name` field of a system span.
#[derive(Default)]
struct SystemName(String);

impl Visit for SystemName {
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
        if attributes.metadata().name() != SYSTEM {
            return;
        }
        let mut name = SystemName::default();
        attributes.record(&mut name);
        if let Some(span) = context.span(id) {
            span.extensions_mut().insert(Running {
                system: name.0,
                since: None,
            });
        }
    }

    fn on_enter(&self, id: &Id, context: Context<'_, S>) {
        if let Some(span) = context.span(id)
            && let Some(running) = span.extensions_mut().get_mut::<Running>()
        {
            running.since = Some(Instant::now());
        }
    }

    fn on_exit(&self, id: &Id, context: Context<'_, S>) {
        if let Some(span) = context.span(id)
            && let Some(running) = span.extensions_mut().get_mut::<Running>()
            && let Some(since) = running.since.take()
        {
            self.0.record(&running.system, since.elapsed());
        }
    }
}

#[cfg(test)]
mod tests;
