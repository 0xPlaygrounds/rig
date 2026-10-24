//! Opt-in provider HTTP recording in the session's `traffic.jsonl`, using
//! the cassette engine's recorder and scrubber. Enable it with
//! `--traffic-log` or `[traffic] log = true` in settings.toml.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_traffic::TrafficPlugin;
//!
//! let mut app = App::new();
//! app.insert_resource(Settings::from_toml("[traffic]\nlog = true"))
//!     .add_plugins(TrafficPlugin);
//! ```

use std::io::{self, Write};
use std::sync::{Arc, OnceLock};

use bevy_app::AppExit;
use bevy_log::warn;
use rig_cassette::traffic::TrafficLog;
use rig_ecs::WriteLogs;
use rig_ecs::session_log::SessionStore;
use rig_harness::prelude::*;
use rig_memory::Files;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The file of the session's store the exchanges are written to.
pub const TRAFFIC_LOG: &str = "traffic.jsonl";

/// The command-line flag that turns the log on.
const FLAG: &str = "traffic-log";

/// Records exchanges through the models' HTTP transport in the session's [`TRAFFIC_LOG`] when
/// [`TrafficSettings::log`] or `--traffic-log` turns it on.
#[derive(Default)]
pub struct TrafficPlugin;

/// `[traffic]` in `settings.toml`.
#[derive(Resource, Deserialize, Default, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct TrafficSettings {
    /// Whether every session records its provider traffic.
    pub log: bool,
}

/// The log the models' transport writes to, once the session's store is
/// open.
#[derive(Resource, Clone, Default)]
struct Traffic(Arc<OnceLock<TrafficLog>>);

impl Plugin for TrafficPlugin {
    fn build(&self, app: &mut App) {
        app.add_flag(Flag::new(
            FLAG,
            "Record model provider HTTP requests and replies in the session's traffic.jsonl",
        ))
        .add_setting::<TrafficSettings>("traffic");
        let world = app.world();
        let flagged = world
            .get_resource::<Invoked>()
            .is_some_and(|invoked| invoked.flag(FLAG));
        let set = world
            .get_resource::<TrafficSettings>()
            .is_some_and(|settings| settings.log);
        if !flagged && !set {
            return;
        }
        let traffic = Traffic::default();
        let opened = traffic.0.clone();
        // Models connect after the app is built, when the log is open.
        app.world_mut()
            .get_resource_or_init::<Models>()
            .0
            .wrap_http(move |http| match opened.get() {
                Some(log) => log.wrap_http(http),
                None => http,
            });
        app.insert_resource(traffic)
            .add_systems(Last, flush.in_set(WriteLogs).run_if(on_message::<AppExit>));
    }

    fn finish(&self, app: &mut App) {
        let world = app.world();
        let (Some(traffic), Some(SessionStore(store))) = (
            world.get_resource::<Traffic>(),
            world.get_resource::<SessionStore>(),
        ) else {
            return;
        };
        let store: Arc<dyn Files> = store.clone();
        let file = StoreFile(store);
        match TrafficLog::new(file) {
            Ok(log) => {
                traffic.0.set(log).ok();
            }
            Err(failure) => warn!("the traffic log could not start: {failure}"),
        }
    }
}

/// [`TRAFFIC_LOG`] in the session's store, appended to.
struct StoreFile(Arc<dyn Files>);

impl Write for StoreFile {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.append_file(TRAFFIC_LOG, bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Waits for the queued exchanges to reach the session store before exit.
fn flush(traffic: Res<Traffic>) {
    if let Some(log) = traffic.0.get() {
        log.flush();
    }
}
