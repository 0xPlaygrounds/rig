//! Offline migration of persisted effect logs and ECS checkpoints to the
//! formats this rig reads.
//!
//! [`migrate`] recognizes an effect log (`{"header", "records"}`), a rig-ecs
//! checkpoint (`{"format", "entities", …}`) or a rig-agent run (`{"format",
//! "max_turns", "state", …}`) and rewrites an older format to the current
//! one. A document already in the current format is returned
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

/// The rig-agent run format this migration produces.
pub const AGENT_RUN_FORMAT: u32 = 2;

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
    /// A rig-agent run was rewritten from format `from` to
    /// [`AGENT_RUN_FORMAT`].
    AgentRun {
        /// The format the run was in.
        from: u32,
    },
    /// The document was already current and was returned unchanged.
    Current,
}

/// Why a document could not be migrated.
#[derive(Debug, thiserror::Error)]
pub enum MigrateError {
    /// The document is not an effect log, a rig-ecs checkpoint or a rig-agent
    /// run.
    #[error("not an effect log, a rig-ecs checkpoint or a rig-agent run")]
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
    } else if ["format", "max_turns", "state"]
        .iter()
        .all(|key| document.get(key).is_some())
    {
        agent_run(document)
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
        if let Some(errors) = header
            .get_mut("stream_errors")
            .and_then(Value::as_object_mut)
        {
            for error in errors
                .values_mut()
                .filter_map(Value::as_array_mut)
                .flatten()
            {
                if let Some(report) = error.get_mut("error") {
                    report_to_current(report);
                }
            }
        }
    }
    if let Some(records) = log.get_mut("records").and_then(Value::as_array_mut) {
        for (index, record) in records.iter_mut().enumerate() {
            // Format 0 required the field: a record without it lost what the
            // tool published, and reading it as "nothing published" would
            // invent that answer.
            if record.get("tool_output").is_none() {
                return Err(MigrateError::Invalid {
                    path: format!("/records/{index}/tool_output"),
                    source: serde::de::Error::missing_field("tool_output"),
                });
            }
            if let Some(outcome) = record.get_mut("outcome") {
                answer_to_current(outcome);
            }
            if let Some(events) = record.get_mut("events").and_then(Value::as_array_mut) {
                events.iter_mut().for_each(event_to_current);
            }
        }
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

/// Rewrite a rig-agent run to [`AGENT_RUN_FORMAT`]: empty identifiers in its
/// completion calls become absent and its messages take their current form.
/// Every other value loads as written.
fn agent_run(mut run: Value) -> Result<(Value, Migration), MigrateError> {
    let from = run
        .get("format")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid_format("/format"))?;
    match from {
        from if from == u64::from(AGENT_RUN_FORMAT) => return Ok((run, Migration::Current)),
        1 => {}
        found => {
            return Err(MigrateError::Newer {
                artifact: "rig-agent run",
                found,
                current: AGENT_RUN_FORMAT,
            });
        }
    }
    if let Some(format) = run.get_mut("format") {
        *format = Value::from(AGENT_RUN_FORMAT);
    }
    calls_to_current(&mut run);
    retype_messages(&mut run, "chat_history", "")?;
    retype_messages(&mut run, "new_messages", "")?;
    if let Some(done) = run.pointer_mut("/state/Done") {
        calls_to_current(done);
        retype_messages(done, "messages", "/state/Done")?;
    }
    let from = u32::try_from(from).unwrap_or_default();
    Ok((run, Migration::AgentRun { from }))
}

/// The completion calls a run or its final response records, with empty
/// identifiers absent.
fn calls_to_current(holder: &mut Value) {
    if let Some(calls) = holder
        .get_mut("completion_calls")
        .and_then(Value::as_array_mut)
    {
        for call in calls {
            drop_empty(call, &["message_id", "response_id", "provider_request_id"]);
        }
    }
}

/// Rewrite the message list `holder[field]`, when present, through
/// [`Message`].
fn retype_messages(holder: &mut Value, field: &str, path: &str) -> Result<(), MigrateError> {
    if holder.get(field).is_some_and(Value::is_null) {
        return Ok(());
    }
    retype_each::<Message>(holder, field, path)
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
                answer_to_current(value);
                *value = typed_value::<Result<Outcome, ErrorReport>>(value.take(), &path)?;
            }
            "rig_ecs::bus::effect::Streamed" => {
                if let Some(events) = value.get_mut("events").and_then(Value::as_array_mut) {
                    events.iter_mut().for_each(event_to_current);
                }
                if let Some(outcome) = value
                    .get_mut("outcome")
                    .filter(|outcome| !outcome.is_null())
                {
                    answer_to_current(outcome);
                }
                if let Some(errors) = value.get_mut("errors").and_then(Value::as_array_mut) {
                    for error in errors {
                        if let Some(report) = error.get_mut(1) {
                            report_to_current(report);
                        }
                    }
                }
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

/// A format-0 answer, `{"Ok": outcome}` or `{"Err": report}`, in the
/// current shape.
fn answer_to_current(answer: &mut Value) {
    if let Some(outcome) = answer.get_mut("Ok") {
        outcome_to_current(outcome);
    }
    if let Some(report) = answer.get_mut("Err") {
        report_to_current(report);
    }
}

/// A format-0 outcome in the current shape.
fn outcome_to_current(outcome: &mut Value) {
    match outcome.get("outcome").and_then(Value::as_str) {
        Some("completion") => {
            drop_empty(
                outcome,
                &["message_id", "response_id", "provider_request_id", "model"],
            );
            if let Some(fields) = outcome.as_object_mut() {
                let mut end = Map::new();
                move_fields(fields, &mut end, END_FIELDS);
                nest_meta(fields, &mut end);
                fields.insert("end".to_owned(), Value::Object(end));
            }
        }
        Some("embeddings") => {
            if let Some(response) = outcome.get_mut("response") {
                envelope(response, "embeddings");
            }
        }
        Some("reranked") => envelope(outcome, "results"),
        _ => {}
    }
}

/// The fields of a format-0 response that are the provider's metadata.
const META_FIELDS: &[&str] = &[
    "provider",
    "model",
    "response_id",
    "provider_request_id",
    "usage",
    "raw",
];

/// The fields of a format-0 completion that say how it ended, beside its
/// metadata.
const END_FIELDS: &[&str] = &["finish_reason", "message_id", "reasoning_issuer"];

/// Move each of `keys` present in `from` into `to`.
fn move_fields(from: &mut Map<String, Value>, to: &mut Map<String, Value>, keys: &[&str]) {
    for key in keys {
        if let Some(value) = from.remove(*key) {
            to.insert((*key).to_owned(), value);
        }
    }
}

/// Move the metadata fields of `from` under `to["meta"]`.
fn nest_meta(from: &mut Map<String, Value>, to: &mut Map<String, Value>) {
    let mut meta = Map::new();
    move_fields(from, &mut meta, META_FIELDS);
    to.insert("meta".to_owned(), Value::Object(meta));
}

/// A format-0 modality response in the current shape: its `output_key`
/// field becomes `output`, and the metadata fields move under `meta`.
/// Fields the envelope does not name (an outcome's tag) stay in place.
fn envelope(response: &mut Value, output_key: &str) {
    drop_empty(response, &["response_id", "provider_request_id", "model"]);
    let Some(fields) = response.as_object_mut() else {
        return;
    };
    let output = fields.remove(output_key).unwrap_or(Value::Null);
    let mut nested = Map::new();
    nest_meta(fields, &mut nested);
    fields.insert("output".to_owned(), output);
    fields.extend(nested);
}

/// A format-0 stream event in the current shape: a terminal record's
/// metadata moves under `meta`.
fn event_to_current(event: &mut Value) {
    if event.get("event").and_then(Value::as_str) == Some("final") {
        drop_empty(
            event,
            &["message_id", "response_id", "provider_request_id", "model"],
        );
        if let Some(fields) = event.as_object_mut() {
            let mut nested = Map::new();
            nest_meta(fields, &mut nested);
            fields.extend(nested);
        }
    }
}

/// A format-0 error report in the current shape.
fn report_to_current(report: &mut Value) {
    drop_empty(report, &["request_id"]);
    if let Some(response) = report.get_mut("provider_response") {
        drop_empty(response, &["provider_request_id"]);
    }
}

/// Remove each of `keys` whose value is the empty string: format 0 read an
/// empty identifier as absent, and the current format refuses one.
fn drop_empty(object: &mut Value, keys: &[&str]) {
    if let Some(fields) = object.as_object_mut() {
        for key in keys {
            if fields.get(*key).and_then(Value::as_str) == Some("") {
                fields.remove(*key);
            }
        }
    }
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
