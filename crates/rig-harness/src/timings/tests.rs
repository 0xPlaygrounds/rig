use std::time::Duration;

use bevy_log::tracing::{info_span, subscriber};
use bevy_log::tracing_subscriber::Registry;
use bevy_log::tracing_subscriber::layer::SubscriberExt;

use super::{SystemTimings, TimeSystems};

/// Every run of a system span is timed under the system's name; other
/// spans are not timed, and clearing forgets them all.
#[test]
fn each_run_of_a_system_is_timed() {
    let timings = SystemTimings::default();
    let logged = Registry::default().with(TimeSystems(timings.clone()));
    subscriber::with_default(logged, || {
        let system = info_span!(parent: None, "system", name = "corpus_profiled::busy");
        for _ in 0..2 {
            let _running = system.enter();
            std::thread::sleep(Duration::from_millis(2));
        }
        let _other = info_span!("calculate conflicting systems").entered();
    });
    let systems = timings.systems();
    let busy = systems.get(&("corpus_profiled::busy".to_owned(), None));
    assert_eq!(busy.map(|busy| busy.runs), Some(2), "{systems:?}");
    assert!(busy.is_some_and(|busy| busy.max >= Duration::from_millis(2)));
    assert_eq!(systems.len(), 1, "{systems:?}");
    timings.clear();
    assert!(
        timings.systems().is_empty(),
        "clearing forgets every timing"
    );
}

#[cfg(feature = "trace")]
mod trace {
    use bevy_app::prelude::*;
    use bevy_ecs::prelude::*;
    use bevy_ecs::schedule::{ScheduleLabel, SingleThreadedExecutor};
    use bevy_ecs::system::RunSystemOnce;
    use bevy_log::tracing::subscriber;
    use bevy_log::tracing_subscriber::Registry;
    use bevy_log::tracing_subscriber::layer::SubscriberExt;

    use crate::timings::{PluginTimings, SystemRun, SystemTimings, TimeSystems};
    use crate::{PluginEntity, load};

    /// A schedule an exclusive `Update` system runs, as `AgentPass` is.
    #[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
    struct Pass;

    fn busy() {}

    fn idle() {}

    /// Adds `busy` to `Update`.
    #[derive(Default)]
    struct Busy;

    impl Plugin for Busy {
        fn build(&self, app: &mut App) {
            app.add_systems(Update, busy);
        }
    }

    /// Adds `idle` to `Pass`, and the system running `Pass` to `Update`.
    #[derive(Default)]
    struct Passes;

    impl Plugin for Passes {
        fn build(&self, app: &mut App) {
            app.add_systems(Pass, idle)
                .add_systems(Update, |world: &mut World| {
                    world.run_schedule(Pass);
                });
        }
    }

    /// An app loading `Busy` and `Passes` from one package, its schedules
    /// on this thread, run for a frame under `timings`. Bevy makes a
    /// system's span when the system is added, so the app is built under
    /// the log too, as the harness's log comes before every plugin.
    fn timed(timings: &SystemTimings) -> App {
        let logged = Registry::default().with(TimeSystems(timings.clone()));
        let app = subscriber::with_default(logged, || {
            let mut app = App::new();
            load::<Busy>(&mut app, "corpus-two", "path corpus/two");
            load::<Passes>(&mut app, "corpus-two", "path corpus/two");
            for label in [Update.intern(), Pass.intern()] {
                app.edit_schedule(label, |schedule| {
                    schedule.set_executor(SingleThreadedExecutor::new());
                });
            }
            app.world_mut().run_schedule(Update);
            app
        });
        timings.next_frame();
        app
    }

    /// Two plugins of one package each get their own row, with the
    /// schedule their system ran in, nested ones too.
    #[test]
    fn each_plugin_of_a_package_is_timed_apart_by_schedule() {
        let timings = SystemTimings::default();
        let mut app = timed(&timings);
        app.insert_resource(timings.clone());
        let world = app.world_mut();
        let busy = world.resource::<PluginEntity<Busy>>().entity();
        let passes = world.resource::<PluginEntity<Passes>>().entity();
        let totals = world.run_system_once(|timings: PluginTimings| timings.totals());
        let totals = totals.unwrap_or_default();
        let runs = |plugin, schedule: &str| {
            let key = (Some(plugin), Some(schedule.to_owned()));
            totals.get(&key).map(|timing| timing.runs)
        };
        assert_eq!(runs(busy, "Update"), Some(1), "{totals:?}");
        assert_eq!(runs(passes, "Update"), Some(1), "{totals:?}");
        assert_eq!(runs(passes, "Pass"), Some(1), "{totals:?}");
    }

    /// The last whole frame lists each run once, with its plugin and
    /// schedule.
    #[test]
    fn the_last_frame_lists_each_run() {
        let timings = SystemTimings::default();
        let mut app = timed(&timings);
        app.insert_resource(timings.clone());
        let world = app.world_mut();
        let passes = world.resource::<PluginEntity<Passes>>().entity();
        let last = world.run_system_once(|timings: PluginTimings| timings.last_frame());
        let last = last.unwrap_or_default();
        let idle: Vec<&SystemRun> = last
            .iter()
            .filter(|run| run.system.ends_with("::idle"))
            .collect();
        assert_eq!(idle.len(), 1, "{last:?}");
        let run = idle
            .first()
            .map(|run| (run.plugin, run.schedule.as_deref()));
        assert_eq!(run, Some((Some(passes), Some("Pass"))));
    }
}
