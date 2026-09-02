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

use super::{Handler, HandlerFuture, OutcomeSink};

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
        let records: VecDeque<EffectRecord> = log
            .iter()
            .filter(|record| &record.key == key)
            .cloned()
            .collect();
        let first = records.front()?;
        let family = first.kind.family();
        let descriptor = HandlerDescriptor {
            key: key.clone(),
            family: describe(key, &first.kind),
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
    pub fn register_all(log: &EffectLog, driver: &mut super::BusDriver) {
        for replayer in Self::for_log(log) {
            let key = replayer.key.clone();
            driver.register_erased(key, super::ErasedHandler::new(replayer));
        }
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

fn describe(key: &HandlerKey, kind: &EffectKind) -> FamilyDescriptor {
    match kind {
        EffectKind::Completion { .. } => FamilyDescriptor::Completion {
            model: ModelRef::new(format!("replay:{key}")),
            capabilities: ProviderCapabilities::default(),
        },
        EffectKind::ToolCall { name, .. } => FamilyDescriptor::Tool {
            name: name.clone(),
            description: format!("replayed from the effect log under `{key}`"),
            parameters: serde_json::json!({"type": "object"}),
            embedding: None,
        },
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

impl Handler for EffectLogReplayer {
    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }

    fn handle(&self, kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        let next = self
            .records
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front();
        Box::pin(async move {
            let outcome = match next {
                None => Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!(
                        "replay divergence: `{}` received a `{}` dispatch after its log ran out",
                        self.key,
                        kind.name()
                    ),
                )),
                Some(record) if record.kind.family() != kind.family() => Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!(
                        "replay divergence: `{}` recorded a `{}` effect ({}) but received `{}`",
                        self.key,
                        record.kind.name(),
                        record.id,
                        kind.name()
                    ),
                )),
                Some(record) => record.outcome,
            };
            debug_assert_eq!(self.family, self.descriptor.family.family());
            sink.resolve(outcome).await;
        })
    }
}
