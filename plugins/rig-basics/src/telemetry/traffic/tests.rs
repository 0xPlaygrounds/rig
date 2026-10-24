use rig_core::http_client::{Bytes, DynHttpClient, HttpClientExt, Request};
use rig_core::test_utils::RecordingHttpClient;
use rig_harness_test_support::MemoryStore;
use rig_memory::Files;
use serde_json::{Value, json};

use super::*;

const KEY: &str = "a-key-of-no-shape-1234";

/// Sends one request through the models' transport of an app on `store`
/// built with `settings`, then exits the app. Returns whether the request
/// was answered.
fn one_request(store: &MemoryStore, settings: &str) -> bool {
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(SessionStore::new(store.clone()))
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
    answered
}

#[test]
fn the_setting_writes_each_exchange_to_the_session() {
    let store = MemoryStore::default();
    let answered = one_request(&store, "[traffic]\nlog = true");
    let written = store.read_file(TRAFFIC_LOG).unwrap_or_default();
    let written = String::from_utf8_lossy(&written);
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
    let store = MemoryStore::default();
    let answered = one_request(&store, "");

    assert!(answered);
    assert!(store.read_file(TRAFFIC_LOG).is_err());
}
