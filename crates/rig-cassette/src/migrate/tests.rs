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
                        "usage": {},
                        "provider": "mock",
                        "raw": { "message_id": null }
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
