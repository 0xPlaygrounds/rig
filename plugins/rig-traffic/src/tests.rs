use rig_core::http_client::{Bytes, DynHttpClient, HttpClientExt, Request};
use rig_core::test_utils::RecordingHttpClient;
use rig_harness::harness_protocol::Invocation;
use rig_harness_test_support::MemoryStore;
use serde_json::{Value, json};

use super::*;

const KEY: &str = "a-key-of-no-shape-1234";

fn app(settings: &str, flagged: bool, store: Option<&MemoryStore>) -> App {
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(Invoked {
            args: Invocation {
                flags: if flagged {
                    vec![(FLAG.to_owned(), None)]
                } else {
                    Vec::new()
                },
                ..Default::default()
            },
            ..Default::default()
        });
    if let Some(store) = store {
        app.insert_resource(SessionStore::new(store.clone()));
    }
    app.add_plugins(TrafficPlugin);
    app
}

fn request(app: &mut App) -> bool {
    let http = app
        .world_mut()
        .get_resource_or_init::<Models>()
        .0
        .transport()
        .wrap(DynHttpClient::new(RecordingHttpClient::new(
            "{\"ok\":true}",
        )));
    Request::post("https://example.com/v1/messages")
        .header("x-api-key", KEY)
        .body(Bytes::from_static(b"{\"model\":\"m\"}"))
        .ok()
        .and_then(|request| bevy_tasks::block_on(http.send::<_, Bytes>(request)).ok())
        .and_then(|response| bevy_tasks::block_on(response.into_body()).ok())
        .is_some()
}

fn exit(app: &mut App) {
    app.world_mut().write_message(AppExit::Success);
    app.update();
}

fn records(settings: &str, flagged: bool) {
    let store = MemoryStore::default();
    let mut app = app(settings, flagged, Some(&store));
    assert!(
        app.world()
            .get_resource::<Traffic>()
            .is_some_and(|traffic| traffic.0.get().is_none())
    );
    app.finish();
    assert!(request(&mut app));
    assert!(request(&mut app));
    exit(&mut app);
    let bytes = store.read_file(TRAFFIC_LOG).unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes);
    let entries: Vec<Value> = text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    assert_eq!(entries.len(), 2, "{text}");
    for entry in entries {
        assert_eq!(entry.pointer("/request/body/model"), Some(&json!("m")));
        assert_eq!(
            entry.pointer("/request/headers/x-api-key"),
            Some(&json!("[REDACTED]"))
        );
        assert_eq!(entry.pointer("/response/body/ok"), Some(&json!(true)));
        assert_eq!(entry.get("end"), Some(&json!("complete")));
    }
    assert!(!text.contains(KEY), "{text}");
}

#[test]
fn the_setting_records_each_exchange_and_exit_flushes_it() {
    records("[traffic]\nlog = true", false);
}

#[test]
fn the_flag_records_even_when_the_setting_is_false() {
    records("[traffic]\nlog = false", true);
}

#[test]
fn off_declares_the_flag_but_installs_no_recorder_or_transport() {
    let store = MemoryStore::default();
    let mut app = app("", false, Some(&store));
    app.finish();
    assert!(!app.world().contains_resource::<Traffic>());
    assert!(!app.world().contains_resource::<Models>());
    let mut flags = app.world_mut().query::<&Flag>();
    assert!(flags.iter(app.world()).any(|flag| flag.name == FLAG));
    assert!(request(&mut app));
    exit(&mut app);
    assert!(store.read_file(TRAFFIC_LOG).is_err());
}

#[test]
fn without_a_store_the_enabled_plugin_leaves_requests_usable() {
    let mut app = app("[traffic]\nlog = true", false, None);
    app.finish();
    assert!(
        app.world()
            .get_resource::<Traffic>()
            .is_some_and(|traffic| traffic.0.get().is_none())
    );
    assert!(request(&mut app));
    exit(&mut app);
}
