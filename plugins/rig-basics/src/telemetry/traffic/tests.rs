use std::fs;

use rig_core::http_client::{Bytes, DynHttpClient, HttpClientExt, Request};
use rig_core::test_utils::RecordingHttpClient;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::scratch_home;
use serde_json::{Value, json};

use super::*;

const KEY: &str = "a-key-of-no-shape-1234";

/// Sends one request through the models' transport of an app built with
/// `settings`, then exits it, and returns the session's traffic log.
fn traffic_after_one_request(settings: &str) -> Option<String> {
    let home = scratch_home(&format!("traffic-{}", settings.len()));
    let dir = home.session(&SessionId::generate());
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
    let request = Request::post("https://example.com/v1/messages")
        .header("x-api-key", KEY)
        .body(Bytes::from_static(b"{\"model\":\"m\"}"))
        .ok()?;
    let response = bevy_tasks::block_on(http.send::<_, Bytes>(request)).ok()?;
    bevy_tasks::block_on(response.into_body()).ok()?;
    app.world_mut().write_message(AppExit::Success);
    app.update();
    fs::read_to_string(dir.path().join(TRAFFIC_LOG)).ok()
}

#[test]
fn the_setting_writes_each_exchange_to_the_session_and_off_writes_nothing() {
    let written = traffic_after_one_request("[traffic]\nlog = true").unwrap_or_default();
    let entry: Value = serde_json::from_str(written.trim()).unwrap_or_default();
    let at = |pointer: &str| entry.pointer(pointer).cloned();
    assert_eq!(at("/request/body/model"), Some(json!("m")), "{written}");
    assert_eq!(at("/request/headers/x-api-key"), Some(json!("[REDACTED]")));
    assert_eq!(at("/response/body/ok"), Some(json!(true)));
    assert!(!written.contains(KEY));

    assert_eq!(traffic_after_one_request(""), None);
}
