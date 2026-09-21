//! Bus scheduling, serving policy, and wake-driven app execution.
//!
//! ```
//! let wake = rig_ecs::bus::Wake::default();
//! wake.signal();
//! assert!(wake.take());
//! ```

use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use bevy_app::{App, AppExit, MainScheduleOrder, Plugin, PluginsState, Update};
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
    handlers::{HandlerIndex, WorldKinds},
    record::{DeliveryBatch, begin_delivery_pass, record_bound, record_cancelled, record_outcome},
};

/// The schedule the bus runs in, once per app update, after `Update`.
/// Users add their systems here, ordered against [`BusSet`]s.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RigSchedule;

/// Runs after [`RigSchedule`] and its deferred commands in every app update.
/// Replay adapters can install idle-pass diagnosis here.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RigEnd;

/// Ordered sets of [`RigSchedule`]: gate pending effects, dispatch, collect and
/// record answers, then apply policy replacements.
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

/// A shared signal requesting another app update, raised by completed tasks,
/// stream delivery, or host systems needing a later pass.
/// Each signal advances [`Wake::generation`] and wakes blocking and async waiters;
/// signalling requires only shared access to the resource.
#[derive(Resource, Clone, Default)]
pub struct Wake(Arc<WakeInner>);

#[derive(Default)]
struct WakeInner {
    raised: AtomicBool,
    generation: AtomicU64,
    waker: futures::task::AtomicWaker,
    lock: Mutex<()>,
    condvar: Condvar,
}

impl Wake {
    /// Raise the signal.
    pub fn signal(&self) {
        self.0.generation.fetch_add(1, Ordering::AcqRel);
        self.0.raised.store(true, Ordering::Release);
        self.0.waker.wake();
        let guard = self
            .0
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.0.condvar.notify_all();
        drop(guard);
    }

    /// Return the wrapping count of signals raised.
    pub fn generation(&self) -> u64 {
        self.0.generation.load(Ordering::Acquire)
    }

    /// Take the signal: whether it was raised since the last take.
    pub fn take(&self) -> bool {
        self.0.raised.swap(false, Ordering::AcqRel)
    }

    /// Block until the signal is raised or `timeout` elapses, then take it.
    /// Returns whether it was raised.
    pub fn wait(&self, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut guard = self
            .0
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while !self.0.raised.load(Ordering::Acquire) {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            let (next, _) = self
                .0
                .condvar
                .wait_timeout(guard, deadline - now)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard = next;
        }
        drop(guard);
        self.take()
    }

    /// Resolve when the signal is raised, taking it.
    pub fn woken(&self) -> impl Future<Output = ()> + Send + '_ {
        std::future::poll_fn(move |cx| {
            if self.take() {
                return std::task::Poll::Ready(());
            }
            self.0.waker.register(cx.waker());
            if self.take() {
                std::task::Poll::Ready(())
            } else {
                std::task::Poll::Pending
            }
        })
    }
}

/// A runner that updates the app whenever [`Wake`] is raised, or every
/// `idle` at most, until the app asks to exit. Install with
/// `app.set_runner(woken_runner(..))`; [`BusPlugin`] installs it by default.
pub fn woken_runner(idle: Duration) -> impl FnOnce(App) -> AppExit {
    move |mut app: App| {
        if app.plugins_state() == PluginsState::Ready {
            app.finish();
            app.cleanup();
        }
        let wake = app.world().resource::<Wake>().clone();
        loop {
            app.update();
            if let Some(exit) = app.should_exit() {
                return exit;
            }
            wake.wait(idle);
        }
    }
}

/// The bus: [`RigSchedule`] with its sets and systems after `Update`, the
/// counters, the handler table, the observers, the [`Wake`] and its runner.
///
/// The task pool: `build` calls `IoTaskPool::get_or_init`, so the bus works
/// with or without a host that initialises the pools itself.
#[derive(Debug, Clone)]
pub struct BusPlugin {
    /// The serving policy: intake per tick, serial keys and bounded delivery.
    /// `stream_capacity` supplies shared queue slots, clamped to at least one;
    /// the single sender has one additional reserved slot.
    pub policy: ServingPolicy,
    /// Schedule ambiguity detection level, defaulting to `Warn`.
    pub ambiguity: LogLevel,
    /// How long the default runner waits for a wake before updating anyway.
    pub idle: Duration,
}

impl Default for BusPlugin {
    fn default() -> Self {
        Self {
            policy: ServingPolicy::default(),
            ambiguity: LogLevel::Warn,
            idle: Duration::from_millis(100),
        }
    }
}

impl BusPlugin {
    /// The bus under `policy`.
    pub fn with_policy(policy: ServingPolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    /// Build the schedule with ambiguity detection at `level`.
    #[must_use = "the setting applies to the returned value"]
    pub fn ambiguity_detection(mut self, level: LogLevel) -> Self {
        self.ambiguity = level;
        self
    }

    /// Install resources, observers, [`RigSchedule`], and [`RigEnd`] in `world`.
    /// Panics if the bus is already installed. Does not configure an app runner;
    /// callers driving a bare world must run the schedules themselves.
    pub fn install(&self, world: &mut World) {
        assert!(
            !world.contains_resource::<Policy>(),
            "the bus is already installed in this world"
        );
        IoTaskPool::get_or_init(TaskPool::default);
        // A zero intake bound would leave pending effects permanently undispatched.
        world.insert_resource(Policy(ServingPolicy {
            command_capacity: self.policy.command_capacity.max(1),
            ..self.policy
        }));
        world.init_resource::<Wake>();
        world.init_resource::<SeqCounter>();
        world.init_resource::<IdCounter>();
        world.init_resource::<DeliveryBatch>();
        world.init_resource::<WorldOutcomeCounter>();
        world.init_resource::<HandlerIndex>();
        world.init_resource::<WorldKinds>();
        world.init_non_send::<super::effect::Executions>();
        world.init_non_send::<super::handlers::HandlerTable>();
        world.add_observer(super::effect::drop_execution);
        world.add_observer(super::handlers::unbound);
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
        world
            .resource_mut::<Schedules>()
            .insert(Schedule::new(RigEnd));
    }
}

impl Plugin for BusPlugin {
    fn build(&self, app: &mut App) {
        self.install(app.world_mut());
        let mut order = app.world_mut().resource_mut::<MainScheduleOrder>();
        order.insert_after(Update, RigSchedule);
        order.insert_after(RigSchedule, RigEnd);
        app.set_runner(woken_runner(self.idle));
    }
}

#[cfg(test)]
mod tests;
