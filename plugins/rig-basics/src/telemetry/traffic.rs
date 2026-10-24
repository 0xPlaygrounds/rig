//! The traffic log: with `--traffic-log`, or `log = true` under `[traffic]`
//! in `settings.toml`, every HTTP exchange with a model provider is written
//! to the session's [`TRAFFIC_LOG`] as rig-core's
//! [`TrafficLog`](rig_core::traffic::TrafficLog) records it: the request,
//! the reply's status, headers and body or streamed events with their
//! times, the error, and the effect id of the model call, all with secrets
//! removed. Off, it adds nothing to a request.
//!
//! It wraps the models' transport while the plugin is built, so a wrapper
//! a plugin listed before it adds is inside it and one listed after it is
//! outside: the log shows each request as it leaves the wrappers outside.

use std::sync::{Arc, OnceLock};

use bevy_app::AppExit;
use bevy_log::warn;
use rig_core::traffic::TrafficLog;
use rig_ecs::WriteLogs;
use rig_harness::prelude::*;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The file in the session's directory the exchanges are written to.
pub const TRAFFIC_LOG: &str = "traffic.jsonl";

/// The command-line flag that turns the log on.
const FLAG: &str = "traffic-log";

/// Records every provider exchange in the session's [`TRAFFIC_LOG`] when
/// [`TrafficSettings::log`] or `--traffic-log` turns it on.
#[derive(Default)]
pub struct TrafficLogPlugin;

/// `[traffic]` in `settings.toml`.
#[derive(Resource, Deserialize, Default, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct TrafficSettings {
    /// Whether every session records its provider traffic.
    pub log: bool,
}

/// The log the models' transport writes to, once the session's file is
/// open.
#[derive(Resource, Clone, Default)]
struct Traffic(Arc<OnceLock<TrafficLog>>);

impl Plugin for TrafficLogPlugin {
    fn build(&self, app: &mut App) {
        app.add_flag(Flag::new(
            FLAG,
            "Record every model provider request and reply in the session's traffic.jsonl",
        ))
        .add_setting::<TrafficSettings>("traffic");
        let world = app.world();
        let flagged = world
            .get_resource::<Invoked>()
            .is_some_and(|invoked| invoked.flag(FLAG));
        if !flagged && !world.resource::<TrafficSettings>().log {
            return;
        }
        let traffic = Traffic::default();
        let opened = traffic.0.clone();
        // Models connect after the app is built, when the file is open.
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
        let (Some(traffic), Some(paths)) = (
            world.get_resource::<Traffic>(),
            world.get_resource::<SessionPaths>(),
        ) else {
            return;
        };
        let path = paths.path().join(TRAFFIC_LOG);
        match TrafficLog::create(&path) {
            Ok(log) => {
                traffic.0.set(log).ok();
            }
            Err(failure) => warn!(
                "the traffic log could not open {}: {failure}",
                path.display()
            ),
        }
    }
}

/// Writes what is still queued, the exchanges the exit stopped included,
/// before the process ends.
fn flush(traffic: Res<Traffic>) {
    if let Some(log) = traffic.0.get() {
        log.flush();
    }
}
