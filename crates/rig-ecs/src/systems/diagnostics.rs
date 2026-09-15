//! The runtime's measurements as `bevy_diagnostic` diagnostics: what
//! `LogDiagnosticsPlugin` prints and an inspector plots. One system at the
//! end of every pass; the exact counters stay where they are
//! ([`AssemblyStats`], the world's queries) — these are their time series.

use bevy_app::App;
use bevy_diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic};
use bevy_ecs::{prelude::*, query::QueryFilter};

use crate::{
    agent::{Failed, Run, Settled, content::cache::AssemblyStats},
    bus::{InFlight, PendingEffect},
};

/// Effects in flight at the end of the pass.
pub const EFFECTS_IN_FLIGHT: DiagnosticPath = DiagnosticPath::const_new("rig/effects/in_flight");
/// Effects pending dispatch at the end of the pass.
pub const EFFECTS_PENDING: DiagnosticPath = DiagnosticPath::const_new("rig/effects/pending");
/// Runs neither settled nor failed.
pub const RUNS_LIVE: DiagnosticPath = DiagnosticPath::const_new("rig/runs/live");
/// Requests folded, cumulative.
pub const ASSEMBLIES: DiagnosticPath = DiagnosticPath::const_new("rig/assembly/assemblies");
/// Utterance renders, cumulative.
pub const RENDERS: DiagnosticPath = DiagnosticPath::const_new("rig/assembly/renders");
/// Utterance cache hits, cumulative.
pub const CACHE_HITS: DiagnosticPath = DiagnosticPath::const_new("rig/assembly/cache_hits");

/// Register every diagnostic.
pub fn register(app: &mut App) {
    app.register_diagnostic(Diagnostic::new(EFFECTS_IN_FLIGHT))
        .register_diagnostic(Diagnostic::new(EFFECTS_PENDING))
        .register_diagnostic(Diagnostic::new(RUNS_LIVE))
        .register_diagnostic(Diagnostic::new(ASSEMBLIES))
        .register_diagnostic(Diagnostic::new(RENDERS))
        .register_diagnostic(Diagnostic::new(CACHE_HITS));
}

/// An effect pending dispatch.
#[derive(QueryFilter)]
pub struct PendingDispatch {
    _pending: With<PendingEffect>,
    _not_taken: Without<InFlight>,
    _not_answered: Without<crate::bus::EffectOutcome>,
}

/// A run neither settled nor failed.
#[derive(QueryFilter)]
pub struct LiveRun {
    _run: With<Run>,
    _not_settled: Without<Settled>,
    _not_failed: Without<Failed>,
}

/// `RigSet::Settle`: one measurement per diagnostic per pass.
pub fn measure(
    mut diagnostics: Diagnostics,
    in_flight: Query<(), With<InFlight>>,
    pending: Query<(), PendingDispatch>,
    live: Query<(), LiveRun>,
    stats: Option<Res<AssemblyStats>>,
) {
    diagnostics.add_measurement(&EFFECTS_IN_FLIGHT, || in_flight.iter().count() as f64);
    diagnostics.add_measurement(&EFFECTS_PENDING, || pending.iter().count() as f64);
    diagnostics.add_measurement(&RUNS_LIVE, || live.iter().count() as f64);
    if let Some(stats) = stats {
        diagnostics.add_measurement(&ASSEMBLIES, || stats.assemblies as f64);
        diagnostics.add_measurement(&RENDERS, || stats.renders as f64);
        diagnostics.add_measurement(&CACHE_HITS, || stats.hits as f64);
    }
}
