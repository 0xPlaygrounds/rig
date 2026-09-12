//! The bus's installation into a world, the schedule and its sets, and the
//! run to quiescence.

use bevy_ecs::{
    prelude::*,
    schedule::{LogLevel, ScheduleBuildSettings, ScheduleLabel},
};
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::serve::ServingPolicy;

use super::{
    collect::{collect_streams, collect_tasks, settle},
    dispatch::dispatch,
    effect::{IdCounter, SeqCounter, WorldOutcomeCounter},
    handlers::{HandlerTable, unbound},
    record::{DeliveryBatch, begin_delivery_pass, record_bound, record_cancelled, record_outcome},
};

/// The schedule the bus runs in, to quiescence, once per host tick. Users add
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
/// the runner at the start of every tick, and at the start of every pass a
/// host runs itself (each such pass is that host's tick).
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct Intake(pub usize);

/// Passes of [`RigSchedule`] one tick may run before the runner stops and
/// warns: a diagnostic, never a hang.
pub const QUIESCENCE_CAP: usize = 64;

/// The bus's configuration: [`install`](Self::install) adds [`RigSchedule`]
/// with its sets and systems, the counters, the handler table and the
/// observers to a world. It does not schedule the runner: the host calls
/// [`run_to_quiescence`] once per tick from the schedule or loop it owns.
///
/// The task pool: `install` calls `IoTaskPool::get_or_init`, so the bus
/// works with or without a host that initialises the pools itself (a pool
/// initialised first wins; otherwise a default pool is made and a later
/// initialiser finds it in place).
#[derive(Debug, Clone)]
pub struct Bus {
    /// The serving policy: intake per tick, serial keys and bounded delivery.
    /// `stream_capacity` supplies shared queue slots, clamped to at least one;
    /// the single sender has one additional reserved slot.
    pub policy: ServingPolicy,
    /// Ambiguity detection on the schedule: `Warn` by default; the crate's
    /// tests build with `Error`.
    pub ambiguity: LogLevel,
}

impl Default for Bus {
    fn default() -> Self {
        Self {
            policy: ServingPolicy::default(),
            ambiguity: LogLevel::Warn,
        }
    }
}

impl Bus {
    /// The bus under `policy`.
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

    /// Install the bus into `world`. Installing twice panics: the schedule
    /// and its resources exist once per world.
    pub fn install(&self, world: &mut World) {
        assert!(
            !world.contains_resource::<Policy>(),
            "the bus is already installed in this world"
        );
        IoTaskPool::get_or_init(TaskPool::default);
        world.insert_resource(Policy(self.policy));
        world.init_resource::<SeqCounter>();
        world.init_resource::<IdCounter>();
        world.init_resource::<Progress>();
        world.init_resource::<super::collect::CollectionBudget>();
        world.init_resource::<Intake>();
        world.init_resource::<DeliveryBatch>();
        world.init_resource::<WorldOutcomeCounter>();
        world.init_non_send::<HandlerTable>();
        world.init_non_send::<super::effect::Executions>();
        world.add_observer(super::effect::drop_execution);
        world.add_observer(unbound);
        world.add_observer(record_outcome);
        world.add_observer(record_cancelled);
        world.add_observer(super::record::witness_cancelled);
        world.add_observer(record_bound);
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
        schedule.add_systems(begin_delivery_pass.before(BusSet::Gate));
        #[cfg(feature = "replay")]
        schedule.add_systems(
            super::delivery::collect_replayed
                .in_set(BusSet::Collect)
                .before(collect_tasks),
        );
        schedule.add_systems(
            (
                collect_tasks,
                collect_streams,
                super::collect::collect_world,
                settle,
            )
                .chain()
                .in_set(BusSet::Collect),
        );
        world.init_resource::<Schedules>();
        world.resource_mut::<Schedules>().insert(schedule);
    }
}

/// Install the bus under `policy` with default ambiguity detection: the
/// one-call form of [`Bus::install`].
pub fn install_bus(world: &mut World, policy: ServingPolicy) {
    Bus::with_policy(policy).install(world);
}

/// The runner: reset the tick's intake, then run [`RigSchedule`] while a
/// plugin system reports [`Progress`], at most [`QUIESCENCE_CAP`] passes.
pub fn run_to_quiescence(world: &mut World) {
    world.resource_mut::<Intake>().0 = 0;
    {
        let mut budget = world.resource_mut::<super::collect::CollectionBudget>();
        budget.in_runner = true;
        budget.remaining = super::collect::STREAM_WORK_PER_TICK;
    }
    for pass in 0..QUIESCENCE_CAP {
        world.resource_mut::<Progress>().0 = false;
        world.run_schedule(RigSchedule);
        #[cfg(feature = "replay")]
        super::delivery::diagnose_idle_replay(world);
        if !world.resource::<Progress>().0
            || world
                .resource::<super::collect::CollectionBudget>()
                .remaining
                == 0
        {
            break;
        }
        if pass + 1 == QUIESCENCE_CAP {
            log::warn!(
                target: "rig_ecs::bus",
                "RigSchedule reached the quiescence cap ({QUIESCENCE_CAP}) in one tick; the rest waits for the next"
            );
        }
    }
    world
        .resource_mut::<super::collect::CollectionBudget>()
        .in_runner = false;
}

#[cfg(test)]
mod tests;
