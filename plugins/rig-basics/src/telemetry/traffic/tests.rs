use std::fs;
use std::path::PathBuf;

use rig_core::http_client::{Bytes, DynHttpClient, HttpClientExt, Request};
use rig_core::test_utils::RecordingHttpClient;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::scratch_home;
use serde_json::{Value, json};

use super::*;

const KEY: &str = "a-key-of-no-shape-1234";

/// Sends one request through the models' transport of an app built with
/// `settings` in a home named `name`, then exits the app. Returns whether
/// the request was answered, and the path of the session's traffic log.
fn one_request(name: &str, settings: &str) -> (bool, PathBuf) {
    let dir = scratch_home(name).session(&SessionId::generate());
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(SessionPaths(dir.clone()))
        .add_plugins(TrafficLogPlugin);
    app.finish();
    let http = app
        .world_mut()
        .get_resource_or_init::<Models>()
        .0
        .transport()
        .wrap(DynHttpClient::new(RecordingHttpClient::new(
            "{\"ok\":true}",
        )));
    let answered = Request::post("https://example.com/v1/messages")
        .header("x-api-key", KEY)
        .body(Bytes::from_static(b"{\"model\":\"m\"}"))
        .ok()
        .and_then(|request| bevy_tasks::block_on(http.send::<_, Bytes>(request)).ok())
        .and_then(|response| bevy_tasks::block_on(response.into_body()).ok())
        .is_some();
    app.world_mut().write_message(AppExit::Success);
    app.update();
    (answered, dir.path().join(TRAFFIC_LOG))
}

#[test]
fn the_setting_writes_each_exchange_to_the_session() {
    let (answered, path) = one_request("traffic-on", "[traffic]\nlog = true");
    let written = fs::read_to_string(path).unwrap_or_default();
    let entry: Value = serde_json::from_str(written.trim()).unwrap_or_default();
    let at = |pointer: &str| entry.pointer(pointer).cloned();

    assert!(answered);
    assert_eq!(at("/request/body/model"), Some(json!("m")), "{written}");
    assert_eq!(at("/request/headers/x-api-key"), Some(json!("[REDACTED]")));
    assert_eq!(at("/response/body/ok"), Some(json!(true)));
    assert_eq!(at("/end"), Some(json!("complete")));
    assert!(!written.contains(KEY));
}

#[test]
fn off_nothing_is_written() {
    let (answered, path) = one_request("traffic-off", "");

    assert!(answered);
    assert!(!path.exists());
}
