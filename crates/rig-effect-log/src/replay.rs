//! Replaying a recorded [`EffectLog`] as a handler.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Mutex, PoisonError},
};

use rig_core::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{
        EffectFamily, EffectId, EffectKind, EffectRecord, EmbedModality, FamilyDescriptor,
        HandlerDescriptor, HandlerKey,
    },
    error::{ErrorKind, ErrorReport},
};

use rig_core::serve::{OutcomeSink, Serve};

use super::{EFFECT_LOG_FORMAT, EffectLog, stable_hash};

/// How a replayer compares an incoming request with the record's: by the
/// whole payload as data (the divergence names the first differing JSON
/// pointer), or by [`stable_hash`] of each (the divergence names the two
/// hashes). A replayer's mode, never a log's: the records hold the request
/// either way, and the mode decides what a replay refuses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestCheck {
    /// Compare the payloads as data.
    #[default]
    Payload,
    /// Compare their stable hashes.
    Hash,
}

/// A handler that answers dispatches from a recorded log instead of a
/// provider: the replay half of record/replay.
///
/// One replayer serves one key, in one of two modes. **By position**
/// ([`for_key`](Self::for_key)): it answers that key's records in recorded
/// order, checking each incoming effect's family against the record's; a
/// divergence (a different family, or more dispatches than records) fails
/// the dispatch with a report naming the position, never with a guess —
/// the mode for a runtime that mints its own ids. **By id**
/// ([`for_key_by_id`](Self::for_key_by_id)): it answers each dispatch
/// with the record of the dispatch's own id, whatever order the dispatches
/// arrive in — the mode for a world that re-issues effects under their
/// recorded ids, where the replayer is then a pure function of the log and
/// the id. Register one per key with [`EffectLogReplayer::register_all`].
pub struct EffectLogReplayer {
    key: HandlerKey,
    family: EffectFamily,
    descriptor: HandlerDescriptor,
    records: Mutex<Records>,
    check: RequestCheck,
}

/// The records a replayer answers from, in its mode.
enum Records {
    /// In recorded order, front first.
    ByPosition(VecDeque<EffectRecord>),
    /// By the dispatch's id.
    ById(BTreeMap<EffectId, EffectRecord>),
}

impl Records {
    fn len(&self) -> usize {
        match self {
            Self::ByPosition(records) => records.len(),
            Self::ById(records) => records.len(),
        }
    }
}

impl EffectLogReplayer {
    /// A replayer for `key`, holding that key's records from `log` in
    /// order. A key the header's required row names but the log never
    /// dispatched — a tool the program advertised and the model never
    /// called — is served too, from its advertised definition, and answers
    /// any dispatch with a divergence. Refused by name when neither the
    /// records nor the required row know the key, or when the row names a
    /// key of a family only the handler table can describe and the table
    /// has no entry for it — there is nothing to describe the handler by.
    pub fn for_key(log: &EffectLog, key: &HandlerKey) -> Result<Self, ErrorReport> {
        let records = Self::records_of(log, key);
        let by_position: VecDeque<EffectRecord> = records.into();
        let first = by_position.front().cloned();
        Self::describing(log, key, first, Records::ByPosition(by_position))
    }

    /// A replayer for `key` answering by id: see the type's docs. Described
    /// as [`for_key`](Self::for_key) describes.
    pub fn for_key_by_id(log: &EffectLog, key: &HandlerKey) -> Result<Self, ErrorReport> {
        let records = Self::records_of(log, key);
        let first = records.first().cloned();
        let by_id: BTreeMap<EffectId, EffectRecord> = records
            .into_iter()
            .map(|record| (record.id, record))
            .collect();
        Self::describing(log, key, first, Records::ById(by_id))
    }

    /// `key`'s records in dispatch order, whatever order the log was
    /// assembled in: ids are minted at dispatch and strictly increasing.
    fn records_of(log: &EffectLog, key: &HandlerKey) -> Vec<EffectRecord> {
        let mut records: Vec<EffectRecord> = log
            .iter()
            .filter(|record| &record.key == key)
            .cloned()
            .collect();
        records.sort_by_key(|record| record.id);
        records
    }

    fn describing(
        log: &EffectLog,
        key: &HandlerKey,
        first: Option<EffectRecord>,
        records: Records,
    ) -> Result<Self, ErrorReport> {
        let (family, described) = match first {
            Some(first) => (first.kind.family(), describe(key, &first.kind, log)),
            // No record: the handler table is the header's first source —
            // a key the host served that nothing dispatched to, or that a
            // layer denied every dispatch to, is described from it; a key
            // the required row names is described for its family; anything
            // else is refused by name.
            None => match log
                .header
                .handlers
                .iter()
                .find(|installed| &installed.key == key)
            {
                Some(installed) => (installed.family.family(), installed.family.clone()),
                None => {
                    let family = *log.header.required.get(key).ok_or_else(|| {
                        ErrorReport::new(
                            ErrorKind::HandlerUnavailable,
                            format!(
                                "`{key}` has no records in the log, no entry in its handler table and no place in its required row: nothing describes it"
                            ),
                        )
                    })?;
                    (family, describe_required(key, family, log)?)
                }
            },
        };
        let descriptor = HandlerDescriptor {
            key: key.clone(),
            family: described,
            layers: Vec::new(),
        };
        Ok(Self {
            key: key.clone(),
            family,
            descriptor,
            records: Mutex::new(records),
            check: RequestCheck::Payload,
        })
    }

    /// The same replayer comparing requests as `check` says.
    pub fn checking(mut self, check: RequestCheck) -> Self {
        self.check = check;
        self
    }

    /// The mode this replayer compares requests in.
    pub const fn check(&self) -> RequestCheck {
        self.check
    }

    /// Every key the log mentions, in first-appearance order, then every
    /// key the required row names that no record does, each with its
    /// replayer.
    pub fn for_log(log: &EffectLog) -> Result<Vec<Self>, ErrorReport> {
        Self::keys_of(log)
            .iter()
            .map(|key| Self::for_key(log, key))
            .collect()
    }

    /// [`for_log`](Self::for_log) with every replayer answering by id.
    pub fn for_log_by_id(log: &EffectLog) -> Result<Vec<Self>, ErrorReport> {
        Self::keys_of(log)
            .iter()
            .map(|key| Self::for_key_by_id(log, key))
            .collect()
    }

    fn keys_of(log: &EffectLog) -> Vec<HandlerKey> {
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
        keys
    }

    /// Register a replayer for every key in `log` on `driver`. Refuses a
    /// log of another format, and a log whose signature names a family its
    /// records do not answer — before the first dispatch, not at the record
    /// where it would have diverged.
    #[cfg(feature = "bus")]
    pub fn register_all(
        log: &EffectLog,
        driver: &mut rig_bus::BusDriver,
    ) -> Result<(), ErrorReport> {
        Self::register_all_checking(log, driver, RequestCheck::Payload)
    }

    /// [`register_all`](Self::register_all) with every replayer comparing
    /// requests as `check` says.
    #[cfg(feature = "bus")]
    pub fn register_all_checking(
        log: &EffectLog,
        driver: &mut rig_bus::BusDriver,
        check: RequestCheck,
    ) -> Result<(), ErrorReport> {
        Self::check_header(log)?;
        for replayer in Self::for_log(log)? {
            let key = replayer.key.clone();
            driver.register_erased(
                key,
                rig_core::serve::ErasedHandler::new(replayer.checking(check)),
            )?;
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

/// [`divergence`] under `check`: by payload, or by the stable hashes of
/// the two payloads, reported as the pair.
fn divergence_under(
    check: RequestCheck,
    recorded: &EffectKind,
    got: &EffectKind,
) -> Option<String> {
    match check {
        RequestCheck::Payload => divergence(recorded, got),
        RequestCheck::Hash => {
            let (Ok(recorded), Ok(got)) = (stable_hash(recorded), stable_hash(got)) else {
                return Some("the payload could not be hashed".to_owned());
            };
            (recorded != got)
                .then(|| format!("hash {recorded:#018x} was recorded, {got:#018x} arrived"))
        }
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
        // Name and args are the readable fast path; the rest of the payload
        // is compared as data like every other family's (the tool context is
        // not on the wire: it travels beside the sink).
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
) -> Result<FamilyDescriptor, ErrorReport> {
    if let Some(installed) = log
        .header
        .handlers
        .iter()
        .find(|descriptor| &descriptor.key == key && descriptor.family.family() == family)
    {
        return Ok(installed.family.clone());
    }
    let gap = |what: &str| {
        ErrorReport::new(
            ErrorKind::HandlerUnavailable,
            format!(
                "the required key `{key}` ({family}) cannot be described: {what}; the log's handler table has no entry for it"
            ),
        )
    };
    match family {
        EffectFamily::Tool => {
            let parts = key.parts();
            let name = match parts.kind.as_deref() {
                Some("tool") => parts.label.as_ref(),
                _ => return Err(gap("the key does not name a tool")),
            };
            let advertised = advertised_tool(name, log)
                .ok_or_else(|| gap("no recorded request advertises the tool"))?;
            Ok(FamilyDescriptor::Tool {
                name: advertised.name,
                description: advertised.description,
                parameters: advertised.parameters,
                embedding: None,
            })
        }
        EffectFamily::Completion => Ok(FamilyDescriptor::Completion {
            model: ModelRef::new(format!("replay:{key}")),
            capabilities: ProviderCapabilities::default(),
        }),
        EffectFamily::Memory => Ok(FamilyDescriptor::Memory {}),
        EffectFamily::Retrieve => Ok(FamilyDescriptor::Retrieve {}),
        // An embedding or rerank descriptor names a modality or a document
        // cap the row does not carry; a custom kind its label. Only the
        // handler table has them, and a log without it is refused by name.
        EffectFamily::Embed | EffectFamily::Rerank | EffectFamily::Custom => Err(gap(
            "a descriptor of this family is not derivable from the row",
        )),
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
        let (next, missing) = {
            let mut records = self.records.lock().unwrap_or_else(PoisonError::into_inner);
            match &mut *records {
                Records::ByPosition(records) => (
                    records.pop_front(),
                    format!(
                        "replay divergence: `{}` received a `{}` dispatch after its log ran out",
                        self.key,
                        kind.name()
                    ),
                ),
                Records::ById(records) => (
                    records.remove(&sink.id()),
                    format!(
                        "replay divergence: `{}` received a `{}` dispatch as {}, and the log has no record of that id",
                        self.key,
                        kind.name(),
                        sink.id()
                    ),
                ),
            }
        };
        {
            let outcome = match next {
                None => Err(ErrorReport::new(ErrorKind::Divergence, missing)),
                Some(record) => match divergence_under(self.check, &record.kind, &kind) {
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
