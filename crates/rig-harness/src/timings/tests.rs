use std::time::Duration;

use bevy_log::tracing::{info_span, subscriber};
use bevy_log::tracing_subscriber::Registry;
use bevy_log::tracing_subscriber::layer::SubscriberExt;

use super::{SystemTimings, TimeSystems};
use crate::PluginSource;

/// Every run of a system span is timed under the system's name, and summed
/// under the plugin package its crate is; other spans are not timed.
#[test]
fn each_run_of_a_system_is_timed_and_summed_under_its_plugin() {
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
    let busy = systems.get("corpus_profiled::busy").copied();
    assert_eq!(busy.map(|busy| busy.runs), Some(2), "{systems:?}");
    assert!(busy.is_some_and(|busy| busy.max >= Duration::from_millis(2)));
    assert_eq!(systems.len(), 1, "{systems:?}");
    let source = PluginSource {
        krate: "corpus-profiled".to_owned(),
        source: "path corpus/profiled".to_owned(),
    };
    let packages = timings.by_plugin([&source]);
    assert_eq!(packages.get("corpus-profiled").copied(), busy);
}

/// Built with `trace`, Bevy's own span around each system run is timed:
/// a system of this crate counts for the package `rig-harness`.
#[cfg(feature = "trace")]
#[test]
fn bevy_systems_are_timed_under_the_trace_feature() {
    use bevy_ecs::prelude::*;
    use bevy_ecs::schedule::{Schedule, SingleThreadedExecutor};

    fn busy() {}

    let timings = SystemTimings::default();
    let logged = Registry::default().with(TimeSystems(timings.clone()));
    subscriber::with_default(logged, || {
        let mut world = World::new();
        let mut schedule = Schedule::default();
        schedule.set_executor(SingleThreadedExecutor::new());
        schedule.add_systems(busy);
        schedule.run(&mut world);
    });
    let source = PluginSource {
        krate: "rig-harness".to_owned(),
        source: "path crates/rig-harness".to_owned(),
    };
    let packages = timings.by_plugin([&source]);
    let runs = packages.get("rig-harness").map(|timing| timing.runs);
    assert_eq!(runs, Some(1), "{:?}", timings.systems());
}
