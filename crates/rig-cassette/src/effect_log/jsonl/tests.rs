use std::sync::Arc;

use rig_core::effect::{EffectKind, HandlerKey, Outcome};

use super::*;

fn record(id: u64, key: &str) -> EffectRecord {
    EffectRecord {
        stream_origin: None,
        tool_output: None,
        parent: None,
        scope: Some(Arc::from("agent")),
        id: EffectId::from_raw(id),
        key: HandlerKey::from(key),
        kind: EffectKind::Custom {
            kind: Arc::from("test"),
            payload: serde_json::json!({ "ask": id }),
        },
        outcome: Ok(Outcome::Custom {
            payload: serde_json::json!({ "n": id }),
        }),
        events: None,
    }
}

#[test]
fn appended_logs_read_back_as_one_with_merged_headers() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    let mut writer = Writer::new(&path);
    writer
        .append(&EffectLog::from_records(vec![record(1, "model")]))
        .expect("first append");
    let second = EffectLog::from_records(vec![record(2, "model"), record(3, "tool")]);
    writer.append(&second).expect("second append");
    // Same header, no records: nothing to write.
    writer
        .append(&EffectLog {
            header: second.header.clone(),
            records: Vec::new(),
        })
        .expect("empty append");

    let text = std::fs::read_to_string(&path).expect("log text");
    assert_eq!(
        text.lines()
            .filter(|line| line.starts_with("{\"header\""))
            .count(),
        2
    );
    assert_eq!(text.lines().count(), 5);

    let log = read(&path).expect("log");
    let ids: Vec<u64> = log
        .records
        .iter()
        .map(|record| record.id.as_u64())
        .collect();
    assert_eq!(ids, [1, 2, 3]);
    assert!(
        log.header
            .signature
            .contains_key(&HandlerKey::from("model"))
    );
    assert!(log.header.signature.contains_key(&HandlerKey::from("tool")));
    assert_eq!(last_id(&path).expect("tail"), Some(EffectId::from_raw(3)));
}

#[test]
fn a_log_without_records_has_no_last_id() {
    let dir = assert_fs::TempDir::new().expect("scratch directory");
    let path = dir.path().join("effects.jsonl");
    Writer::new(&path)
        .append(&EffectLog::default())
        .expect("header only");
    assert_eq!(last_id(&path).expect("tail"), None);
}
