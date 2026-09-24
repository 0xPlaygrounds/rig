//! Offline migration of persisted effect logs and ECS checkpoints to the
//! formats this rig reads.
//!
//! [`migrate`] recognizes an effect log (`{"header", "records"}`) or a rig-ecs
//! checkpoint (`{"format", "entities", …}`) and rewrites an older format to
//! the current one. A document already in the current format is returned
//! unchanged, so running a migration twice is a no-op. Older formats exist
//! only here: the runtime types read the current format alone. The
//! `rig-migrate` binary (feature `migrate`) applies this to files in place.
//!
//! ```
//! use rig_cassette::migrate::{Migration, migrate};
//!
//! let old = serde_json::json!({
//!     "header": { "handlers": [], "signature": {}, "hooks": [], "required": {} },
//!     "records": []
//! });
//! let (current, migration) = migrate(old)?;
//! assert_eq!(migration, Migration::EffectLog { from: 0 });
//! assert_eq!(current["header"]["format"], 1);
//! # Ok::<(), rig_cassette::migrate::MigrateError>(())
//! ```

use rig_core::completion::{AssistantContent, Message};
use rig_core::effect::{EffectKind, HandlerDescriptor, Outcome};
use rig_core::error::ErrorReport;
use rig_core::streaming::StreamEvent;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::effect_log::{EffectLog, LOG_FORMAT};

/// The rig-ecs checkpoint format this migration produces.
pub const ECS_CHECKPOINT_FORMAT: u32 = 3;

/// What [`migrate`] did to a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Migration {
    /// An effect log was rewritten from format `from` to [`LOG_FORMAT`].
    EffectLog {
        /// The format the log was in.
        from: u32,
    },
    /// A rig-ecs checkpoint was rewritten from format `from` to
    /// [`ECS_CHECKPOINT_FORMAT`].
    Checkpoint {
        /// The format the checkpoint was in.
        from: u32,
    },
    /// The document was already current and was returned unchanged.
    Current,
}

/// Why a document could not be migrated.
#[derive(Debug, thiserror::Error)]
pub enum MigrateError {
    /// The document is neither an effect log nor a rig-ecs checkpoint.
    #[error("not an effect log or a rig-ecs checkpoint")]
    Unrecognized,
    /// The document names a format newer than this migration knows.
    #[error("{artifact} format {found} is newer than this rig reads ({current})")]
    Newer {
        /// The artifact kind.
        artifact: &'static str,
        /// The format the document names.
        found: u64,
        /// The format this rig reads.
        current: u32,
    },
    /// A value inside the document does not have the shape its format
    /// defines.
    #[error("{path}: {source}")]
    Invalid {
        /// Where the value sits in the document.
        path: String,
        /// What was wrong with it.
        source: serde_json::Error,
    },
}

/// Rewrite `document` to the current format of the artifact it is.
///
/// Returns the current document and what was done. A current document is
/// returned as given. Errors name the value that has no valid migration.
pub fn migrate(document: Value) -> Result<(Value, Migration), MigrateError> {
    if document.get("header").is_some() && document.get("records").is_some() {
        effect_log(document)
    } else if document.get("entities").is_some() && document.get("format").is_some() {
        checkpoint(document)
    } else {
        Err(MigrateError::Unrecognized)
    }
}

/// Rewrite an effect log to [`LOG_FORMAT`].
fn effect_log(mut log: Value) -> Result<(Value, Migration), MigrateError> {
    let from = log
        .pointer("/header/format")
        .map_or(Some(0), Value::as_u64)
        .ok_or_else(|| invalid_format("/header/format"))?;
    match from {
        from if from == u64::from(LOG_FORMAT) => return Ok((log, Migration::Current)),
        0 => {}
        found => {
            return Err(MigrateError::Newer {
                artifact: "effect log",
                found,
                current: LOG_FORMAT,
            });
        }
    }
    if let Some(header) = log.get_mut("header").and_then(Value::as_object_mut) {
        header.insert("format".to_owned(), Value::from(LOG_FORMAT));
    }
    // Every domain value now reads through its current type and writes the
    // current form: absent optional fields are omitted.
    let log: EffectLog = typed(log, "")?;
    let log = serde_json::to_value(&log).map_err(|source| MigrateError::Invalid {
        path: String::new(),
        source,
    })?;
    Ok((log, Migration::EffectLog { from: 0 }))
}

/// Rewrite a rig-ecs checkpoint to [`ECS_CHECKPOINT_FORMAT`].
fn checkpoint(mut checkpoint: Value) -> Result<(Value, Migration), MigrateError> {
    let from = checkpoint
        .get("format")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid_format("/format"))?;
    match from {
        from if from == u64::from(ECS_CHECKPOINT_FORMAT) => {
            return Ok((checkpoint, Migration::Current));
        }
        2 => {}
        found => {
            return Err(MigrateError::Newer {
                artifact: "rig-ecs checkpoint",
                found,
                current: ECS_CHECKPOINT_FORMAT,
            });
        }
    }
    if let Some(format) = checkpoint.get_mut("format") {
        *format = Value::from(ECS_CHECKPOINT_FORMAT);
    }
    if let Some(entities) = checkpoint.get_mut("entities").and_then(Value::as_array_mut) {
        for (index, entity) in entities.iter_mut().enumerate() {
            if let Some(components) = entity.as_object_mut() {
                components_to_current(components, &format!("/entities/{index}"))?;
            }
        }
    }
    let from = u32::try_from(from).unwrap_or_default();
    Ok((checkpoint, Migration::Checkpoint { from }))
}

/// Rewrite the rig-core values each checkpointed component holds, by the
/// component's type path.
fn components_to_current(
    components: &mut Map<String, Value>,
    entity: &str,
) -> Result<(), MigrateError> {
    for (component, value) in components.iter_mut() {
        let path = format!("{entity}/{component}");
        match component.as_str() {
            "rig_ecs::bus::effect::PendingEffect" => {
                retype::<EffectKind>(value, "kind", &path)?;
            }
            "rig_ecs::bus::handlers::Bound" => {
                retype::<HandlerDescriptor>(value, "descriptor", &path)?;
            }
            "rig_ecs::bus::effect::EffectOutcome" => {
                *value = typed_value::<Result<Outcome, ErrorReport>>(value.take(), &path)?;
            }
            "rig_ecs::bus::effect::Streamed" => {
                retype_each::<StreamEvent>(value, "events", &path)?;
                retype::<Option<Result<Outcome, ErrorReport>>>(value, "outcome", &path)?;
                retype::<Vec<(usize, ErrorReport)>>(value, "errors", &path)?;
            }
            "rig_ecs::agent::Outputs" => {
                retype_each::<AssistantContent>(value, "content", &path)?;
            }
            "rig_ecs::agent::InvalidCall" => {
                retype_each::<AssistantContent>(value, "prefix", &path)?;
            }
            "rig_ecs::agent::Reprompt" => {
                *value = typed_value::<Message>(value.take(), &path)?;
            }
            "rig_ecs::agent::content::parts::ContentPart" => {
                content_part_to_current(value, &path)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// A content part's rig-core payload, for the variants that hold one.
fn content_part_to_current(part: &mut Value, path: &str) -> Result<(), MigrateError> {
    let Some(variants) = part.as_object_mut() else {
        return Ok(());
    };
    for (variant, payload) in variants.iter_mut() {
        let path = format!("{path}/{variant}");
        *payload = match variant.as_str() {
            "Text" => typed_value::<rig_core::message::Text>(payload.take(), &path)?,
            "ToolCall" => typed_value::<rig_core::message::ToolCall>(payload.take(), &path)?,
            "Reasoning" => typed_value::<rig_core::message::Reasoning>(payload.take(), &path)?,
            _ => continue,
        };
    }
    Ok(())
}

/// Rewrite `value[field]`, when present, through its current type `T`.
fn retype<T>(value: &mut Value, field: &str, path: &str) -> Result<(), MigrateError>
where
    T: Serialize + DeserializeOwned,
{
    if let Some(inner) = value.get_mut(field) {
        *inner = typed_value::<T>(inner.take(), &format!("{path}/{field}"))?;
    }
    Ok(())
}

/// Rewrite every element of the array `value[field]`, when present, through
/// its current type `T`.
fn retype_each<T>(value: &mut Value, field: &str, path: &str) -> Result<(), MigrateError>
where
    T: Serialize + DeserializeOwned,
{
    if let Some(items) = value.get_mut(field).and_then(Value::as_array_mut) {
        for (index, item) in items.iter_mut().enumerate() {
            *item = typed_value::<T>(item.take(), &format!("{path}/{field}/{index}"))?;
        }
    }
    Ok(())
}

/// Read `value` as `T`.
fn typed<T: DeserializeOwned>(value: Value, path: &str) -> Result<T, MigrateError> {
    serde_json::from_value(value).map_err(|source| MigrateError::Invalid {
        path: path.to_owned(),
        source,
    })
}

/// Read `value` as `T` and write it back in `T`'s current form.
fn typed_value<T>(value: Value, path: &str) -> Result<Value, MigrateError>
where
    T: Serialize + DeserializeOwned,
{
    serde_json::to_value(typed::<T>(value, path)?).map_err(|source| MigrateError::Invalid {
        path: path.to_owned(),
        source,
    })
}

/// The error for a format marker that is not a number.
fn invalid_format(path: &str) -> MigrateError {
    MigrateError::Invalid {
        path: path.to_owned(),
        source: serde::de::Error::custom("the format is not a non-negative integer"),
    }
}

#[cfg(test)]
mod tests;
