//! Replaying a recorded [`EffectLog`] as a handler.

use std::{
    collections::VecDeque,
    sync::{Mutex, PoisonError},
};

use crate::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{
        EffectFamily, EffectKind, EffectLog, EffectRecord, EmbedModality, FamilyDescriptor,
        HandlerDescriptor, HandlerKey,
    },
    error::{ErrorKind, ErrorReport},
};

use super::{OutcomeSink, Serve};

/// A handler that answers dispatches from a recorded log instead of a
/// provider: the replay half of record/replay.
///
/// One replayer serves one key. It answers that key's records in recorded
/// order, checking each incoming effect's family against the record's; a
/// divergence (a different family, or more dispatches than records) fails
/// the dispatch with an `Internal` report naming the position, never with a
/// guess. Register one per key with [`EffectLogReplayer::register_all`].
pub struct EffectLogReplayer {
    key: HandlerKey,
    family: EffectFamily,
    descriptor: HandlerDescriptor,
    records: Mutex<VecDeque<EffectRecord>>,
}

impl EffectLogReplayer {
    /// A replayer for `key`, holding that key's records from `log` in
    /// order. `None` when the log has no record for the key — there is
    /// nothing to describe the handler by.
    pub fn for_key(log: &EffectLog, key: &HandlerKey) -> Option<Self> {
        // Dispatch order, whatever order the log was assembled in: ids are
        // minted at dispatch and strictly increasing.
        let mut records: Vec<EffectRecord> = log
            .iter()
            .filter(|record| &record.key == key)
            .cloned()
            .collect();
        records.sort_by_key(|record| record.id);
        let records: VecDeque<EffectRecord> = records.into();
        let first = records.front()?;
        let family = first.kind.family();
        let descriptor = HandlerDescriptor {
            key: key.clone(),
            family: describe(key, &first.kind, log),
        };
        Some(Self {
            key: key.clone(),
            family,
            descriptor,
            records: Mutex::new(records),
        })
    }

    /// Every key the log mentions, in first-appearance order, each with its
    /// replayer.
    pub fn for_log(log: &EffectLog) -> Vec<Self> {
        let mut keys: Vec<HandlerKey> = Vec::new();
        for record in log {
            if !keys.contains(&record.key) {
                keys.push(record.key.clone());
            }
        }
        keys.iter()
            .filter_map(|key| Self::for_key(log, key))
            .collect()
    }

    /// Register a replayer for every key in `log` on `driver`.
    pub fn register_all(log: &EffectLog, driver: &mut super::BusDriver) -> Result<(), ErrorReport> {
        for replayer in Self::for_log(log) {
            let key = replayer.key.clone();
            driver.register_erased(key, super::ErasedHandler::new(replayer))?;
        }
        Ok(())
    }

    /// The key this replayer serves.
    pub fn key(&self) -> &HandlerKey {
        &self.key
    }

    /// Records not yet replayed.
    pub fn remaining(&self) -> usize {
        self.records
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// Why `got` is not the effect that was recorded — never a guess: a family
/// change, a different tool name or arguments, or any other difference in
/// the payload.
fn divergence(recorded: &EffectKind, got: &EffectKind) -> Option<String> {
    if recorded.family() != got.family() {
        return Some(format!(
            "family {:?} was recorded, {:?} arrived",
            recorded.family(),
            got.family()
        ));
    }
    if let (
        EffectKind::ToolCall {
            name: recorded_name,
            args: recorded_args,
            ..
        },
        EffectKind::ToolCall { name, args, .. },
    ) = (recorded, got)
    {
        if recorded_name != name {
            return Some(format!(
                "tool `{recorded_name}` was recorded, `{name}` arrived"
            ));
        }
        if recorded_args != args {
            return Some(format!(
                "arguments differ for `{name}`: recorded `{recorded_args}`, arrived `{args}`"
            ));
        }
        return None;
    }
    let (Ok(recorded), Ok(got)) = (serde_json::to_value(recorded), serde_json::to_value(got))
    else {
        return Some("the payload could not be compared".to_owned());
    };
    if recorded == got {
        return None;
    }
    Some(first_difference(&recorded, &got, "payload"))
}

/// The path of the first differing field between two serialized payloads.
fn first_difference(recorded: &serde_json::Value, got: &serde_json::Value, path: &str) -> String {
    match (recorded, got) {
        (serde_json::Value::Object(recorded), serde_json::Value::Object(got)) => {
            let mut keys: Vec<&String> = recorded.keys().chain(got.keys()).collect();
            keys.sort();
            keys.dedup();
            for key in keys {
                match (recorded.get(key), got.get(key)) {
                    (Some(a), Some(b)) if a == b => {}
                    (Some(a), Some(b)) => return first_difference(a, b, &format!("{path}.{key}")),
                    (Some(_), None) => {
                        return format!("{path}.{key} was recorded but did not arrive");
                    }
                    (None, Some(_)) => return format!("{path}.{key} arrived but was not recorded"),
                    (None, None) => {}
                }
            }
            format!("{path} differs")
        }
        (serde_json::Value::Array(recorded), serde_json::Value::Array(got)) => {
            if recorded.len() != got.len() {
                return format!(
                    "{path} has {} recorded elements, {} arrived",
                    recorded.len(),
                    got.len()
                );
            }
            for (index, (a, b)) in recorded.iter().zip(got).enumerate() {
                if a != b {
                    return first_difference(a, b, &format!("{path}[{index}]"));
                }
            }
            format!("{path} differs")
        }
        _ => format!("{path} differs: recorded `{recorded}`, arrived `{got}`"),
    }
}

/// The descriptor a replayer advertises under `key`. A tool's definition is
/// taken from the recorded completion requests that advertised it, so a
/// replayed run builds byte-identical requests — the replayer must look to
/// the model exactly like the tool it stands in for, or the next completion
/// dispatch diverges.
fn describe(key: &HandlerKey, kind: &EffectKind, log: &EffectLog) -> FamilyDescriptor {
    match kind {
        EffectKind::Completion { .. } => FamilyDescriptor::Completion {
            model: ModelRef::new(format!("replay:{key}")),
            capabilities: ProviderCapabilities::default(),
        },
        EffectKind::ToolCall { name, .. } => {
            let advertised = log.iter().find_map(|record| match &record.kind {
                EffectKind::Completion { request, .. } => request
                    .tools
                    .iter()
                    .find(|tool| &tool.name == name)
                    .cloned(),
                _ => None,
            });
            match advertised {
                Some(tool) => FamilyDescriptor::Tool {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.parameters,
                    embedding: None,
                },
                None => FamilyDescriptor::Tool {
                    name: name.clone(),
                    description: format!("replayed from the effect log under `{key}`"),
                    parameters: serde_json::json!({"type": "object"}),
                    embedding: None,
                },
            }
        }
        EffectKind::Embed { inputs } => FamilyDescriptor::Embed {
            model: format!("replay:{key}"),
            dims: None,
            max_documents: usize::MAX,
            modality: match inputs {
                crate::effect::EmbedInputs::Texts(_) => EmbedModality::Text,
                crate::effect::EmbedInputs::Images(_) => EmbedModality::Image,
            },
        },
        EffectKind::Memory { .. } => FamilyDescriptor::Memory {},
        EffectKind::Retrieve { .. } => FamilyDescriptor::Retrieve {},
        EffectKind::Custom { .. } => FamilyDescriptor::Custom {
            kind: kind.name().to_owned(),
        },
    }
}

impl Serve for EffectLogReplayer {
    /// A replayer answers whatever its log holds.
    type Family = crate::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let next = self
            .records
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front();
        {
            let outcome = match next {
                None => Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!(
                        "replay divergence: `{}` received a `{}` dispatch after its log ran out",
                        self.key,
                        kind.name()
                    ),
                )),
                Some(record) => match divergence(&record.kind, &kind) {
                    Some(what) => Err(ErrorReport::new(
                        ErrorKind::Internal,
                        format!(
                            "replay divergence: `{}` recorded {} ({}) but received {}: {what}",
                            self.key,
                            record.kind.name(),
                            record.id,
                            kind.name()
                        ),
                    )),
                    None => record.outcome,
                },
            };
            debug_assert_eq!(self.family, self.descriptor.family.family());
            sink.resolve(outcome).await;
        }
    }
}
