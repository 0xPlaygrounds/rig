//! The bus as a `bevy_app` plugin: the schedule and its sets, the policy,
//! and the wake that lets a host run the app only when a task finished.

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

/// The schedule after [`RigSchedule`] in every app update: the pass is over
/// and every command of it applied. The bus diagnoses an idle replay here;
/// nothing else runs in it.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RigEnd;

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

/// The signal that another app update is worth running: every task the bus
/// spawns raises it when it finishes or delivers a stream item, and a host
/// system raises it when it did something a later pass must see (a policy
/// still deliberating, a world change made outside a tick). A runner waits
/// on it ([`Wake::wait`] blocking, [`Wake::woken`] async) instead of
/// spinning; [`woken_runner`] is that runner. Every raise also advances
/// [`Wake::generation`], the activity counter a system reads to tell an
/// idle pass from a busy one. Raising it is a read of the resource:
/// systems that raise it never conflict.
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

    /// How many times the signal was raised: unchanged between two reads
    /// means nothing happened in between.
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
    /// Ambiguity detection on the schedule: `Warn` by default; the crate's
    /// tests build with `Error`.
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

    /// The world half of the plugin: the resources, observers and
    /// [`RigSchedule`] in `world`'s `Schedules`. [`Plugin::build`] adds
    /// this, then places the schedule after `Update` and sets the runner;
    /// a test that drives `RigSchedule` itself needs only this.
    pub fn install(&self, world: &mut World) {
        assert!(
            !world.contains_resource::<Policy>(),
            "the bus is already installed in this world"
        );
        IoTaskPool::get_or_init(TaskPool::default);
        // A tick takes at least one effect: a zero intake bound would leave
        // every pending effect pending forever with no error, no record and
        // no witness event. rig-agent's driver clamps the same field.
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
