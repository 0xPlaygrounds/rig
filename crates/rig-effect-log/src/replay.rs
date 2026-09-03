//! Replaying a recorded [`EffectLog`] as a handler.

use std::{
    collections::VecDeque,
    sync::{Mutex, PoisonError},
};

use rig_core::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{
        EffectFamily, EffectKind, EffectRecord, EmbedModality, FamilyDescriptor, HandlerDescriptor,
        HandlerKey,
    },
    error::{ErrorKind, ErrorReport},
};

use rig_core::serve::{OutcomeSink, Serve};

use super::{EFFECT_LOG_FORMAT, EffectLog};

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
    /// order. A key the header's required row names but the log never
    /// dispatched — a tool the program advertised and the model never
    /// called — is served too, from its advertised definition, and answers
    /// any dispatch with a divergence. `None` when neither the records nor
    /// the required row know the key — there is nothing to describe the
    /// handler by.
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
        let (family, described) = match records.front() {
            Some(first) => (first.kind.family(), describe(key, &first.kind, log)),
            None => {
                let family = *log.header.required.get(key)?;
                (family, describe_required(key, family, log)?)
            }
        };
        let descriptor = HandlerDescriptor {
            key: key.clone(),
            family: described,
        };
        Some(Self {
            key: key.clone(),
            family,
            descriptor,
            records: Mutex::new(records),
        })
    }

    /// Every key the log mentions, in first-appearance order, then every
    /// key the required row names that no record does, each with its
    /// replayer.
    pub fn for_log(log: &EffectLog) -> Vec<Self> {
        let mut keys: Vec<HandlerKey> = Vec::new();
        for record in log {
            if !keys.contains(&record.key) {
                keys.push(record.key.clone());
            }
        }
        for key in log.header.required.keys() {
            if !keys.contains(key) {
                keys.push(key.clone());
            }
        }
        keys.iter()
            .filter_map(|key| Self::for_key(log, key))
            .collect()
    }

    /// Register a replayer for every key in `log` on `driver`. Refuses a
    /// log of another format, and a log whose signature names a family its
    /// records do not answer — before the first dispatch, not at the record
    /// where it would have diverged.
    pub fn register_all(
        log: &EffectLog,
        driver: &mut rig_bus::BusDriver,
    ) -> Result<(), ErrorReport> {
        Self::check_header(log)?;
        for replayer in Self::for_log(log) {
            let key = replayer.key.clone();
            driver.register_erased(key, rig_core::serve::ErasedHandler::new(replayer))?;
        }
        Ok(())
    }

    /// The header checks a replay makes before any dispatch: the format is
    /// this crate's, and every key the signature names is answered by
    /// records of that family.
    pub fn check_header(log: &EffectLog) -> Result<(), ErrorReport> {
        if log.header.format != EFFECT_LOG_FORMAT {
            return Err(ErrorReport::new(
                ErrorKind::Internal,
                format!(
                    "replay refused: the log is format {}, this rig reads format {}",
                    log.header.format, EFFECT_LOG_FORMAT
                ),
            ));
        }
        for (key, family) in &log.header.signature {
            let recorded = log
                .records
                .iter()
                .find(|record| &record.key == key)
                .map(|record| record.kind.family());
            match recorded {
                Some(recorded) if recorded == *family => {}
                Some(recorded) => {
                    return Err(ErrorReport::new(
                        ErrorKind::Internal,
                        format!(
                            "replay refused: the signature says `{key}` serves {family}, its records are {recorded}"
                        ),
                    ));
                }
                None => {}
            }
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
        // Name and args are the readable fast path; the dispatch context is
        // part of the effect too (a tool answers differently under a
        // different context), so it is compared as data like every other
        // family's payload.
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

/// The descriptor a replayer advertises for a required key the log never
/// dispatched to: the one the header's handler table recorded for it when
/// it was installed — a retrievable tool the index never named, a route
/// the hook never selected — else, for a log with no handler table, a tool
/// from the definition the recorded requests hold, or a model, memory or
/// retrieval index by its family alone. The replayer answers any dispatch
/// to it with a divergence.
fn describe_required(
    key: &HandlerKey,
    family: EffectFamily,
    log: &EffectLog,
) -> Option<FamilyDescriptor> {
    if let Some(installed) = log
        .header
        .handlers
        .iter()
        .find(|descriptor| &descriptor.key == key && descriptor.family.family() == family)
    {
        return Some(installed.family.clone());
    }
    match family {
        EffectFamily::Tool => {
            let name = key
                .as_str()
                .rsplit_once("tool:")
                .map(|(_, rest)| rest.split_once('#').map_or(rest, |(name, _)| name))?;
            let advertised = advertised_tool(name, log)?;
            Some(FamilyDescriptor::Tool {
                name: advertised.name,
                description: advertised.description,
                parameters: advertised.parameters,
                embedding: None,
            })
        }
        EffectFamily::Completion => Some(FamilyDescriptor::Completion {
            model: ModelRef::new(format!("replay:{key}")),
            capabilities: ProviderCapabilities::default(),
        }),
        EffectFamily::Memory => Some(FamilyDescriptor::Memory {}),
        EffectFamily::Retrieve => Some(FamilyDescriptor::Retrieve {}),
        // An embedding or rerank descriptor names a modality or a document
        // cap the row does not carry; a custom kind its label. None of
        // them is in an agent's required row.
        EffectFamily::Embed | EffectFamily::Rerank | EffectFamily::Custom => None,
    }
}

/// The definition of the tool `name` as some completion request in `log`
/// advertised it.
fn advertised_tool(name: &str, log: &EffectLog) -> Option<rig_core::completion::ToolDefinition> {
    log.iter().find_map(|record| match &record.kind {
        EffectKind::Completion { request, .. } => {
            request.tools.iter().find(|tool| tool.name == name).cloned()
        }
        _ => None,
    })
}

fn describe(key: &HandlerKey, kind: &EffectKind, log: &EffectLog) -> FamilyDescriptor {
    match kind {
        EffectKind::Completion { .. } => FamilyDescriptor::Completion {
            model: ModelRef::new(format!("replay:{key}")),
            capabilities: ProviderCapabilities::default(),
        },
        EffectKind::ToolCall { name, .. } => match advertised_tool(name, log) {
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
        },
        EffectKind::Embed { inputs } => FamilyDescriptor::Embed {
            model: format!("replay:{key}"),
            dims: None,
            max_documents: usize::MAX,
            modality: match inputs {
                rig_core::effect::EmbedInputs::Texts(_) => EmbedModality::Text,
                rig_core::effect::EmbedInputs::Images(_) => EmbedModality::Image,
            },
        },
        EffectKind::Rerank { .. } => FamilyDescriptor::Rerank {
            model: format!("replay:{key}"),
            max_documents: usize::MAX,
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
    type Family = rig_core::effect::family::Dynamic;

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
                    ErrorKind::Divergence,
                    format!(
                        "replay divergence: `{}` received a `{}` dispatch after its log ran out",
                        self.key,
                        kind.name()
                    ),
                )),
                Some(record) => match divergence(&record.kind, &kind) {
                    Some(what) => Err(ErrorReport::new(
                        ErrorKind::Divergence,
                        format!(
                            "replay divergence: `{}` recorded {} ({}) but received {}: {what}",
                            self.key,
                            record.kind.name(),
                            record.id,
                            kind.name()
                        ),
                    )),
                    None => {
                        // A stream recorded verbatim replays verbatim: the
                        // consumer sees the original delta boundaries, not
                        // the fold re-emitted. A stream that ended in an
                        // error — the consumer's cancel, the provider's
                        // refusal — has its events and then the error, which
                        // is the record's outcome; a success has its
                        // terminal among its events.
                        if let (Some(events), true) = (record.events, sink.is_stream()) {
                            let mut sink = sink;
                            for event in events {
                                if sink.send(Ok(event)).await.is_err() {
                                    return;
                                }
                            }
                            if record.outcome.is_err() {
                                sink.resolve(record.outcome).await;
                            }
                            return;
                        }
                        record.outcome
                    }
                },
            };
            debug_assert_eq!(self.family, self.descriptor.family.family());
            sink.resolve(outcome).await;
        }
    }
}
