//! The plugin, the schedule and its sets, and the run to quiescence.

use bevy_app::{App, Plugin, Update};
use bevy_ecs::{
    prelude::*,
    schedule::{LogLevel, ScheduleBuildSettings, ScheduleLabel},
};
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::serve::ServingPolicy;

use super::{
    collect::{collect_streams, collect_tasks, settle},
    dispatch::dispatch,
    effect::{IdCounter, SeqCounter},
    handlers::{HandlerTable, unbound},
    record::{record_bound, record_cancelled},
};

/// The schedule the bus runs in, to quiescence, once per `Update`. Users add
/// their systems here, ordered against [`BusSet`]s.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RigSchedule;

/// The four sets of one pass of [`RigSchedule`], in this order.
///
/// | set | true before | written during |
/// |---|---|---|
/// | `Gate` | pending effects are as spawned | a user system patches a `PendingEffect`, denies one (`EffectOutcome(Err(..))`), or holds one (`Held`) |
/// | `Dispatch` | every un-held, un-answered `PendingEffect` is a candidate | the plugin takes them in `Seq` order: `Issued`, `InFlight`, `Serving`/`Streaming`/`Asked`; a record opens |
/// | `Collect` | handlers may have finished or streamed | the plugin writes `Streamed`, `EffectOutcome`; the record closes; `InFlight` goes |
/// | `Judge` | outcomes of this pass have landed and are recorded | a user system may rewrite an `EffectOutcome` before anything after `Judge` reads it |
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BusSet {
    /// Before dispatch: patch, deny, hold.
    Gate,
    /// The plugin takes pending effects.
    Dispatch,
    /// The plugin lands what finished.
    Collect,
    /// After the record: replace.
    Judge,
}

/// The world's [`ServingPolicy`], as a resource.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy(pub ServingPolicy);

/// Set by a plugin system that moved an effect between states this pass;
/// the runner loops [`RigSchedule`] while it is set.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct Progress(pub bool);

impl Progress {
    /// Note progress.
    pub fn mark(&mut self) {
        self.0 = true;
    }
}

/// How many effects `Dispatch` has taken this tick, against
/// [`ServingPolicy::command_capacity`]: the per-tick intake bound. Reset by
/// the runner at the start of every tick.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct Intake(pub usize);

/// Passes of [`RigSchedule`] one tick may run before the runner stops and
/// warns: a diagnostic, never a hang.
pub const QUIESCENCE_CAP: usize = 64;

/// The bus in a world. Adds [`RigSchedule`] with its sets and systems, the
/// counters, the handler table and the cancel observer, and the exclusive
/// runner in `Update`.
///
/// The task pool: `build` calls `IoTaskPool::get_or_init`, so the plugin
/// works with or without `bevy_app`'s `TaskPoolPlugin` (when that plugin
/// is added first its pool wins; when this one is first, a default pool
/// is made and `TaskPoolPlugin` finds it initialised).
#[derive(Debug, Clone)]
pub struct BusPlugin {
    /// The serving policy: intake per tick, stream buffer, serial keys.
    pub policy: ServingPolicy,
    /// Ambiguity detection on the schedule: `Warn` by default; the crate's
    /// tests build with `Error`.
    pub ambiguity: LogLevel,
}

impl Default for BusPlugin {
    fn default() -> Self {
        Self {
            policy: ServingPolicy::default(),
            ambiguity: LogLevel::Warn,
        }
    }
}

impl BusPlugin {
    /// The plugin under `policy`.
    pub fn with_policy(policy: ServingPolicy) -> Self {
        Self {
            policy,
            ambiguity: LogLevel::Warn,
        }
    }

    /// Build the schedule with ambiguity detection at `level`.
    pub fn ambiguity_detection(mut self, level: LogLevel) -> Self {
        self.ambiguity = level;
        self
    }
}

impl Plugin for BusPlugin {
    fn build(&self, app: &mut App) {
        IoTaskPool::get_or_init(TaskPool::default);
        app.insert_resource(Policy(self.policy))
            .init_resource::<SeqCounter>()
            .init_resource::<IdCounter>()
            .init_resource::<Progress>()
            .init_resource::<Intake>()
            .init_non_send::<HandlerTable>();
        app.add_observer(unbound)
            .add_observer(record_cancelled)
            .add_observer(record_bound);
        let mut schedule = Schedule::new(RigSchedule);
        schedule.set_build_settings(ScheduleBuildSettings {
            ambiguity_detection: self.ambiguity,
            ..Default::default()
        });
        schedule.configure_sets(
            (
                BusSet::Gate,
                BusSet::Dispatch,
                BusSet::Collect,
                BusSet::Judge,
            )
                .chain(),
        );
        schedule.add_systems(dispatch.in_set(BusSet::Dispatch));
        schedule.add_systems(
            (collect_tasks, collect_streams, settle)
                .chain()
                .in_set(BusSet::Collect),
        );
        app.add_schedule(schedule);
        app.add_systems(Update, run_to_quiescence);
    }
}

/// The runner: reset the tick's intake, then run [`RigSchedule`] while a
/// plugin system reports [`Progress`], at most [`QUIESCENCE_CAP`] passes.
pub fn run_to_quiescence(world: &mut World) {
    world.resource_mut::<Intake>().0 = 0;
    for pass in 0..QUIESCENCE_CAP {
        world.resource_mut::<Progress>().0 = false;
        world.run_schedule(RigSchedule);
        if !world.resource::<Progress>().0 {
            return;
        }
        if pass + 1 == QUIESCENCE_CAP {
            tracing::warn!(
                target: "rig_ecs::bus",
                cap = QUIESCENCE_CAP,
                "RigSchedule reached the quiescence cap in one tick; the rest waits for the next"
            );
        }
    }
}
