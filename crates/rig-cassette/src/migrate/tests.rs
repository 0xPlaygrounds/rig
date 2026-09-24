use serde_json::{Value, json};

use super::{ECS_CHECKPOINT_FORMAT, MigrateError, Migration, migrate};
use crate::effect_log::EffectLog;

/// A format-0 log as the recorder wrote it before the format marker.
const V0_LOG: &str = include_str!("v0_effect_log.json");
/// The same log in the current format, written from the listed format
/// changes rather than produced by the migration.
const CURRENT_LOG: &str = include_str!("current_effect_log.json");

fn parse(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

#[test]
fn a_format_zero_log_migrates_to_the_current_form() {
    let (migrated, migration) = migrate(parse(V0_LOG)).unwrap();
    assert_eq!(migration, Migration::EffectLog { from: 0 });
    assert_eq!(migrated, parse(CURRENT_LOG));
}

#[test]
fn migrating_a_current_log_changes_nothing() {
    let current = parse(CURRENT_LOG);
    let (again, migration) = migrate(current.clone()).unwrap();
    assert_eq!(migration, Migration::Current);
    assert_eq!(again, current);
}

#[test]
fn a_format_zero_log_is_refused_until_it_is_migrated() {
    let error = serde_json::from_value::<EffectLog>(parse(V0_LOG)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("effect log is format 0"), "{message}");
    assert!(message.contains("rig-migrate"), "{message}");

    let (migrated, _) = migrate(parse(V0_LOG)).unwrap();
    let log: EffectLog = serde_json::from_value(migrated.clone()).unwrap();
    assert_eq!(log.records.len(), 2);
    assert_eq!(serde_json::to_value(&log).unwrap(), migrated);
}

#[test]
fn a_newer_log_is_refused_by_the_migration() {
    let mut log = parse(CURRENT_LOG);
    log["header"]["format"] = json!(99);
    let error = migrate(log).unwrap_err();
    assert!(
        matches!(error, MigrateError::Newer { found: 99, .. }),
        "{error}"
    );
}

#[test]
fn an_unrecognized_document_is_refused() {
    let error = migrate(json!({ "records": [] })).unwrap_err();
    assert!(matches!(error, MigrateError::Unrecognized), "{error}");
}

/// A format-2 checkpoint holding one answered completion, as rig-ecs wrote
/// it: the rig-core outcome still spells absent fields as `null`.
fn v2_checkpoint() -> Value {
    json!({
        "format": 2,
        "entities": [{
            "rig_ecs::bus::effect::EffectOutcome": {
                "Ok": {
                    "outcome": "completion",
                    "choice": [{ "type": "text", "text": "ready" }],
                    "usage": {},
                    "message_id": null,
                    "response_id": null,
                    "finish_reason": null,
                    "provider": "mock",
                    "model": null,
                    "raw": { "message_id": null }
                }
            },
            "rig_ecs::agent::MaxTokens": null
        }],
        "counters": { "next_run": 1, "next_id": 1 },
        "binaries": []
    })
}

#[test]
fn a_format_two_checkpoint_migrates_its_rig_core_values() {
    let (migrated, migration) = migrate(v2_checkpoint()).unwrap();
    assert_eq!(migration, Migration::Checkpoint { from: 2 });
    assert_eq!(
        migrated,
        json!({
            "format": ECS_CHECKPOINT_FORMAT,
            "entities": [{
                "rig_ecs::bus::effect::EffectOutcome": {
                    "Ok": {
                        "outcome": "completion",
                        "choice": [{ "type": "text", "text": "ready" }],
                        "end": {
                            "meta": {
                                "provider": "mock",
                                "usage": {},
                                "raw": { "message_id": null }
                            }
                        }
                    }
                },
                "rig_ecs::agent::MaxTokens": null
            }],
            "counters": { "next_run": 1, "next_id": 1 },
            "binaries": []
        })
    );
    let (again, migration) = migrate(migrated.clone()).unwrap();
    assert_eq!(migration, Migration::Current);
    assert_eq!(again, migrated);
}

#[test]
fn an_empty_identifier_migrates_to_absent() {
    let mut log = parse(V0_LOG);
    let completion = &mut log["records"][1]["outcome"]["Ok"];
    completion["message_id"] = json!("");
    completion["response_id"] = json!("");
    completion["model"] = json!("");
    let reranked = &mut log["records"][0]["outcome"]["Ok"];
    reranked["model"] = json!("");
    assert!(serde_json::from_value::<EffectLog>(migrate(log.clone()).unwrap().0).is_ok());

    let (migrated, _) = migrate(log).unwrap();
    let mut expected = parse(CURRENT_LOG);
    expected["records"][0]["outcome"]["Ok"]["meta"]
        .as_object_mut()
        .unwrap()
        .remove("model");
    assert_eq!(migrated, expected);
}

#[test]
fn a_format_zero_record_without_tool_output_is_refused() {
    let mut log = parse(V0_LOG);
    log["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("tool_output");
    let error = migrate(log).unwrap_err();
    assert!(
        matches!(&error, MigrateError::Invalid { path, .. } if path == "/records/0/tool_output"),
        "{error}"
    );
}

/// A format-1 run as rig-agent wrote it: absent message fields spelled
/// `null`, and a completion call with an empty identifier.
fn v1_run() -> Value {
    json!({
        "format": 1,
        "max_turns": 1,
        "chat_history": null,
        "new_messages": [
            { "role": "assistant", "id": null, "content": [{ "type": "text", "text": "hi" }] }
        ],
        "completion_calls": [
            { "call_index": 0, "usage": {}, "message_id": "", "response_id": "resp_1", "raw": null }
        ],
        "state": "PreparingRequest"
    })
}

#[test]
fn a_format_one_run_migrates_its_ids_and_messages() {
    let (migrated, migration) = migrate(v1_run()).unwrap();
    assert_eq!(migration, Migration::AgentRun { from: 1 });
    assert_eq!(migrated["format"], json!(super::AGENT_RUN_FORMAT));
    assert_eq!(
        migrated["new_messages"],
        json!([{ "role": "assistant", "content": [{ "type": "text", "text": "hi" }] }])
    );
    assert_eq!(
        migrated["completion_calls"],
        json!([{ "call_index": 0, "usage": {}, "response_id": "resp_1", "raw": null }])
    );
    let (again, migration) = migrate(migrated.clone()).unwrap();
    assert_eq!(migration, Migration::Current);
    assert_eq!(again, migrated);
}
