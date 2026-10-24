//! The traffic log: with `--traffic-log`, or `log = true` under `[traffic]`
//! in `settings.toml`, every HTTP exchange with a model provider is written
//! to the session's [`TRAFFIC_LOG`] as rig-core's
//! [`TrafficLog`](rig_core::traffic::TrafficLog) records it: the request
//! as sent, the reply's status, headers and body or streamed chunks with
//! their times, the error, and the effect id of the model call, all with
//! secrets removed. Off, it adds nothing to a request.

use bevy_app::AppExit;
use bevy_log::warn;
use rig_core::traffic::TrafficLog;
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

/// The log the models' transport writes to, flushed when the app exits.
#[derive(Resource)]
struct Traffic(TrafficLog);

impl Plugin for TrafficLogPlugin {
    fn build(&self, app: &mut App) {
        app.add_flag(Flag::new(
            FLAG,
            "Record every model provider request and reply in the session's traffic.jsonl",
        ))
        .add_setting::<TrafficSettings>("traffic");
    }

    fn finish(&self, app: &mut App) {
        let world = app.world();
        let flagged = world
            .get_resource::<Invoked>()
            .is_some_and(|invoked| invoked.flag(FLAG));
        let on = flagged || world.resource::<TrafficSettings>().log;
        let Some(paths) = world.get_resource::<SessionPaths>().filter(|_| on) else {
            return;
        };
        let path = paths.path().join(TRAFFIC_LOG);
        let log = match TrafficLog::create(&path) {
            Ok(log) => log,
            Err(failure) => {
                warn!(
                    "the traffic log could not open {}: {failure}",
                    path.display()
                );
                return;
            }
        };
        let wrapping = log.clone();
        app.world_mut()
            .get_resource_or_init::<Models>()
            .0
            .wrap_http(move |http| wrapping.wrap(http));
        app.insert_resource(Traffic(log))
            .add_systems(Last, flush.run_if(on_message::<AppExit>));
    }
}

/// Writes what is still queued before the process ends.
fn flush(traffic: Res<Traffic>) {
    traffic.0.flush();
}
