//! Handlers that replay recorded effect requests, outcomes, and stream items.
//!
//! ```
//! use rig_cassette::effect_log::{EffectLog, EffectLogReplayer};
//! let handlers = EffectLogReplayer::for_log(&EffectLog::default())?;
//! assert!(handlers.is_empty());
//! # Ok::<(), rig_core::error::ErrorReport>(())
//! ```

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex, PoisonError},
};

use rig_core::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{
        EffectFamily, EffectId, EffectKind, EffectRecord, EmbedModality, FamilyDescriptor,
        HandlerDescriptor, HandlerKey,
    },
    error::{ErrorKind, ErrorReport},
};

use rig_core::serve::{Dispatch, Reply, Serve};
use serde_json::Value;

use super::log::canonical_tool_args;
use super::{EffectLog, canonical_local_ids, stable_hash};

/// Selects request comparison by payload, by [`stable_hash`], or none. The
/// comparison leaves out what a new session issues afresh: rig's tool call
/// ids, and a completion's prompt-cache key. Payload diagnostics name the
/// first differing field path; hash diagnostics give both hashes. Records
/// keep the full request in every mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestCheck {
    /// Compare the payloads as data.
    #[default]
    Payload,
    /// Compare their stable hashes: as strict as `Payload`, reported as
    /// the two hashes.
    Hash,
    /// Compare nothing: answer in recorded order, for a host whose requests
    /// carry other values that differ in every session. It verifies no
    /// request at all, so a host opts into it by name.
    Skip,
}

/// A shared signal for refusals a replayer makes, kept apart from the error
/// outcomes the log recorded: a host's scheduler can stop waiting on a
/// trace that cannot happen without mistaking a recorded failure for a new
/// refusal.
#[derive(Clone, Debug, Default)]
pub struct ReplayRefusals(Arc<Mutex<Option<ErrorReport>>>);

impl ReplayRefusals {
    /// Take the first pending refusal, if a replayer refused a dispatch.
    pub fn take(&self) -> Option<ErrorReport> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).take()
    }

    fn report(&self, report: &ErrorReport) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert_with(|| report.clone());
    }
}

/// Replays one handler key's recorded dispatches, checking incoming
/// requests and returning divergence errors for missing or mismatched
/// records.
///
/// [`for_key`](Self::for_key) answers in recorded order, for a runtime
/// that mints new ids. [`for_key_by_id`](Self::for_key_by_id) answers each
/// dispatch with the record of its id, whatever order dispatches arrive
/// in, for a runtime that issues the recorded ids again (such as one
/// resumed from a [`Checkpoint`](super::Checkpoint)).
pub struct EffectLogReplayer {
    key: HandlerKey,
    family: EffectFamily,
    descriptor: HandlerDescriptor,
    records: Mutex<VecDeque<EffectRecord>>,
    stream_errors: BTreeMap<EffectId, Vec<super::RecordedStreamError>>,
    check: RequestCheck,
    by_id: bool,
    refusals: Option<ReplayRefusals>,
}

impl EffectLogReplayer {
    /// Build an ordered replayer for `key`, validating the log header.
    /// Required keys without records return divergence on dispatch. Returns an
    /// error if the key is unknown or lacks enough metadata to describe it.
    pub fn for_key(log: &EffectLog, key: &HandlerKey) -> Result<Self, ErrorReport> {
        Self::check_header(log)?;
        // `key`'s records in the recorder's observed serve order. An ID may
        // be reserved by a dispatch future long before that future is polled.
        let records: VecDeque<EffectRecord> = log
            .iter()
            .filter(|record| &record.key == key)
            .cloned()
            .collect();
        // A replayer supplies recorded data beneath the application's layers.
        // Preserve the recorded family (including model identity/capabilities),
        // but let the caller reapply executable middleware.
        let (family, described) = if let Some(installed) = log
            .header
            .handlers
            .iter()
            .find(|installed| &installed.key == key)
        {
            (installed.family.family(), installed.family.clone())
        } else if let Some(first) = records.front() {
            (first.kind.family(), describe(key, &first.kind, log))
        } else {
            let family = std::iter::once(&log.header.required)
                .chain(log.header.programs.values().map(|program| &program.required))
                .find_map(|row| row.get(key)).copied().ok_or_else(|| {
                    ErrorReport::new(ErrorKind::HandlerUnavailable, format!(
                        "`{key}` has no records in the log, no entry in its handler table and no place in its required row: nothing describes it"
                    ))
                })?;
            (family, describe_required(key, family, log)?)
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
            stream_errors: log
                .header
                .stream_errors
                .iter()
                .filter(|(id, _)| {
                    log.records
                        .iter()
                        .any(|record| record.id == **id && &record.key == key)
                })
                .map(|(id, errors)| (*id, errors.clone()))
                .collect(),
            check: RequestCheck::Payload,
            by_id: false,
            refusals: None,
        })
    }

    /// A replayer for `key` answering by id: see the type's docs. Described
    /// as [`for_key`](Self::for_key) describes.
    pub fn for_key_by_id(log: &EffectLog, key: &HandlerKey) -> Result<Self, ErrorReport> {
        Ok(Self {
            by_id: true,
            ..Self::for_key(log, key)?
        })
    }

    /// The same replayer comparing requests as `check` says.
    #[must_use = "the setting applies to the returned value"]
    pub fn checking(mut self, check: RequestCheck) -> Self {
        self.check = check;
        self
    }

    /// The same replayer also signalling to `refusals` each dispatch it
    /// refuses: a missing record, a request that diverges, or no slot to
    /// publish a recorded tool output into. A recorded error outcome is
    /// replayed, never signalled.
    #[must_use = "the setting applies to the returned value"]
    pub fn reporting_refusals(mut self, refusals: ReplayRefusals) -> Self {
        self.refusals = Some(refusals);
        self
    }

    fn refuse(&self, report: ErrorReport) -> ErrorReport {
        if let Some(refusals) = &self.refusals {
            refusals.report(&report);
        }
        report
    }

    /// The mode this replayer compares requests in.
    pub const fn check(&self) -> RequestCheck {
        self.check
    }

    /// Build replayers in record first-appearance order, followed by missing keys
    /// from the required and per-program rows. Returns an error for invalid
    /// headers or keys that cannot be described.
    pub fn for_log(log: &EffectLog) -> Result<Vec<Self>, ErrorReport> {
        Self::check_header(log)?;
        Self::keys_of(log)
            .iter()
            .map(|key| Self::for_key(log, key))
            .collect()
    }

    /// [`for_log`](Self::for_log) with every replayer answering by id.
    pub fn for_log_by_id(log: &EffectLog) -> Result<Vec<Self>, ErrorReport> {
        Self::check_header(log)?;
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
        for key in std::iter::once(&log.header.required)
            .chain(
                log.header
                    .programs
                    .values()
                    .map(|program| &program.required),
            )
            .flat_map(|row| row.keys())
        {
            if !keys.contains(key) {
                keys.push(key.clone());
            }
        }
        keys
    }

    /// Validate stream error positions, consistent key families and
    /// descriptors, and that every program's required key is described.
    /// Returns an error for invalid or conflicting metadata.
    pub fn check_header(log: &EffectLog) -> Result<(), ErrorReport> {
        for (id, errors) in &log.header.stream_errors {
            let events = log
                .records
                .iter()
                .find(|record| record.id == *id && record.kind.streams())
                .and_then(|record| record.events.as_ref())
                .ok_or_else(|| {
                    ErrorReport::new(
                        ErrorKind::Divergence,
                        format!("stream errors for {id} require a kept stream record"),
                    )
                })?;
            let total = events.len().checked_add(errors.len()).ok_or_else(|| {
                ErrorReport::new(ErrorKind::Divergence, "stream item count overflow")
            })?;
            if errors.iter().any(|error| error.item >= total)
                || errors
                    .windows(2)
                    .any(|pair| matches!(pair, [first, second] if first.item >= second.item))
            {
                return Err(ErrorReport::new(
                    ErrorKind::Divergence,
                    format!("invalid stream error positions for {id}"),
                ));
            }
        }
        let mut families = BTreeMap::new();
        let rows = [&log.header.signature, &log.header.required]
            .into_iter()
            .chain(
                log.header
                    .programs
                    .values()
                    .map(|program| &program.required),
            );
        let entries = rows
            .flat_map(|row| row.into_iter().map(|(key, family)| (key, *family)))
            .chain(
                log.header
                    .handlers
                    .iter()
                    .map(|descriptor| (&descriptor.key, descriptor.family.family())),
            )
            .chain(
                log.records
                    .iter()
                    .map(|record| (&record.key, record.kind.family())),
            );
        for (key, family) in entries {
            if let Some(previous) = families.insert(key, family)
                && previous != family
            {
                return Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!(
                        "replay refused: conflicting families for `{key}`: {previous} and {family}"
                    ),
                ));
            }
        }
        let mut descriptors = BTreeMap::new();
        for descriptor in &log.header.handlers {
            if let Some(previous) = descriptors.insert(&descriptor.key, descriptor)
                && previous != descriptor
            {
                return Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!(
                        "replay refused: conflicting descriptors for `{}`",
                        descriptor.key
                    ),
                ));
            }
        }
        for (scope, program) in &log.header.programs {
            if let Some(key) = program
                .required
                .keys()
                .find(|key| !descriptors.contains_key(key))
            {
                return Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!(
                        "replay refused: scope `{scope}` requires `{key}` but its recorded descriptor is missing"
                    ),
                ));
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

/// An effect a replay compares with its records under a [`RequestCheck`]:
/// its canonical form ([`canonical`]) is made once, whatever it is
/// compared with.
pub(super) struct Incoming<'a> {
    kind: &'a EffectKind,
    compared: Compared,
}

/// What of an incoming effect a [`RequestCheck`] compares.
enum Compared {
    Payload(serde_json::Result<Value>),
    /// The stable hash of the canonical form, `None` when it has none.
    Hash(Option<u64>),
    Nothing,
}

impl<'a> Incoming<'a> {
    pub(super) fn new(check: RequestCheck, kind: &'a EffectKind) -> Self {
        let compared = match check {
            RequestCheck::Payload => Compared::Payload(canonical(kind)),
            RequestCheck::Hash => Compared::Hash(hash(canonical(kind))),
            RequestCheck::Skip => Compared::Nothing,
        };
        Self { kind, compared }
    }

    /// How it differs from `recorded`, or `None` when it matches or the
    /// check compares nothing.
    pub(super) fn divergence(&self, recorded: &EffectKind) -> Option<String> {
        match &self.compared {
            Compared::Payload(got) => divergence(recorded, canonical(recorded), self.kind, got),
            Compared::Hash(got) => match (hash(canonical(recorded)), got) {
                (Some(recorded), Some(got)) => (recorded != *got)
                    .then(|| format!("hash {recorded:#018x} was recorded, {got:#018x} arrived")),
                _ => Some("the payload could not be hashed".to_owned()),
            },
            Compared::Nothing => None,
        }
    }
}

/// The [`stable_hash`] of a canonical form, if it has one.
fn hash(canonical: serde_json::Result<Value>) -> Option<u64> {
    canonical.and_then(|value| stable_hash(&value)).ok()
}

/// The effect as a replay compares it: serialized, with its rig-issued
/// call ids renamed in order of appearance, a tool call's arguments in
/// canonical key order, and a completion's prompt-cache key left out, so a
/// replay does not diverge on key order or on the ids and cache keys a new
/// session issues afresh alone.
fn canonical(kind: &EffectKind) -> serde_json::Result<Value> {
    let value = match kind {
        EffectKind::Completion { request, stream } if request.options.cache_key.is_some() => {
            // Copies pointers: the messages and tools are shared.
            let mut request = rig_core::completion::CompletionRequest::clone(request);
            request.options.cache_key = None;
            serde_json::to_value(EffectKind::Completion {
                request: std::sync::Arc::new(request),
                stream: *stream,
            })?
        }
        _ => serde_json::to_value(kind)?,
    };
    let mut value = canonical_local_ids(value);
    if let (EffectKind::ToolCall { .. }, Some(Value::String(args))) = (kind, value.get_mut("args"))
    {
        *args = canonical_tool_args(args);
    }
    Ok(value)
}

/// A diagnostic for how `got` differs from `recorded`, by their canonical
/// forms, or `None` when they match.
fn divergence(
    recorded_kind: &EffectKind,
    recorded: serde_json::Result<Value>,
    got_kind: &EffectKind,
    got: &serde_json::Result<Value>,
) -> Option<String> {
    if recorded_kind.family() != got_kind.family() {
        return Some(format!(
            "family {:?} was recorded, {:?} arrived",
            recorded_kind.family(),
            got_kind.family()
        ));
    }
    let (Ok(recorded), Ok(got)) = (recorded, got) else {
        return Some("the payload could not be compared".to_owned());
    };
    if recorded == *got {
        return None;
    }
    if let EffectKind::ToolCall { .. } = got_kind {
        // The tool and its arguments first; the live tool context travels
        // beside the sink and is not on the wire.
        let text = |value: &Value, field: &str| {
            value
                .get(field)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let (recorded_name, name) = (text(&recorded, "name"), text(got, "name"));
        if recorded_name != name {
            return Some(format!(
                "tool `{recorded_name}` was recorded, `{name}` arrived"
            ));
        }
        let (recorded_args, args) = (text(&recorded, "args"), text(got, "args"));
        if recorded_args != args {
            return Some(format!(
                "arguments differ for `{name}`: recorded `{recorded_args}`, arrived `{args}`"
            ));
        }
    }
    Some(first_difference(&recorded, got, "payload"))
}

/// The path of the first differing field between two serialized payloads.
fn first_difference(recorded: &Value, got: &Value, path: &str) -> String {
    match (recorded, got) {
        (Value::Object(recorded), Value::Object(got)) => {
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
        (Value::Array(recorded), Value::Array(got)) => {
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

/// Return a recorded or inferred descriptor for an undispatched required key.
/// Returns an error when the family needs unavailable metadata or a tool has no
/// advertised definition.
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
                name: advertised.name.into(),
                description: advertised.description,
                parameters: advertised.parameters,
                embedding: None,
            })
        }
        EffectFamily::Completion => Ok(FamilyDescriptor::Completion {
            model: ModelRef::new(format!("replay:{key}")),
            capabilities: ProviderCapabilities::default(),
        }),
        EffectFamily::Retrieve => Ok(FamilyDescriptor::Retrieve {}),
        // These families require modality, document limits, or custom labels
        // absent from the effect row.
        EffectFamily::Embed | EffectFamily::Rerank | EffectFamily::Custom => Err(gap(
            "a descriptor of this family is not derivable from the row",
        )),
    }
}

/// The definition of the tool `name` as some completion request in `log`
/// advertised it.
fn advertised_tool(name: &str, log: &EffectLog) -> Option<rig_core::completion::ToolDefinition> {
    log.iter().find_map(|record| match &record.kind {
        EffectKind::Completion { request, .. } => request
            .tools
            .iter()
            .find(|tool| tool.name == name)
            .map(|tool| (**tool).clone()),
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
                name: tool.name.into(),
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

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let next = {
            let mut records = self.records.lock().unwrap_or_else(PoisonError::into_inner);
            if self.by_id {
                let id = dispatch.id();
                (records.iter().position(|record| record.id == id))
                    .and_then(|at| records.remove(at))
            } else {
                records.pop_front()
            }
        };
        {
            let outcome = match next {
                None if self.by_id => Err(self.refuse(ErrorReport::new(
                    ErrorKind::Divergence,
                    format!(
                        "replay divergence: `{}` received a `{}` dispatch as {}, and the log has no record of that id",
                        self.key,
                        kind.name(),
                        dispatch.id()
                    ),
                ))),
                None => Err(self.refuse(ErrorReport::new(
                    ErrorKind::Divergence,
                    format!(
                        "replay divergence: `{}` received a `{}` dispatch after its log ran out",
                        self.key,
                        kind.name()
                    ),
                ))),
                Some(record) => {
                    match Incoming::new(self.check, &kind).divergence(&record.kind) {
                        Some(what) => Err(self.refuse(ErrorReport::new(
                            ErrorKind::Divergence,
                            format!(
                                "replay divergence: `{}` recorded {} ({}) but received {}: {what}",
                                self.key,
                                record.kind.name(),
                                record.id,
                                kind.name()
                            ),
                        ))),
                        None => {
                            if let Some(output) = record.published {
                                let Some(published) =
                                    dispatch.scope::<rig_core::tool::PublishedContext>()
                                else {
                                    return Reply::Outcome(Err(self.refuse(ErrorReport::new(
                                        ErrorKind::Divergence,
                                        "replay requires a tool-result publication slot",
                                    ))));
                                };
                                let context = dispatch
                                    .scope::<rig_core::tool::ToolContext>()
                                    .map(|context| context.for_dispatch())
                                    .unwrap_or_default();
                                published.publish(context.with_result_context(output));
                            }
                            // Preserve original item positions; the stream then ends with
                            // the recorded response, or with the recorded failure when no
                            // kept item reproduced it.
                            if let (Some(events), true) = (record.events, dispatch.is_stream()) {
                                let errors = self
                                    .stream_errors
                                    .get(&record.id)
                                    .cloned()
                                    .unwrap_or_default();
                                // A reply that ended well ended before any error it
                                // went on to carry: an earlier error would have been
                                // its outcome.
                                let (mut done, ending) = match record.outcome {
                                    Ok(rig_core::effect::Answered::Completion(response)) => {
                                        (Some(response), None)
                                    }
                                    Err(error)
                                        if errors.is_empty()
                                            && error != rig_core::serve::stream_truncated() =>
                                    {
                                        (None, Some(Err(error)))
                                    }
                                    Ok(_) | Err(_) => (None, None),
                                };
                                let items = events.into_items();
                                let events = items.len();
                                let mut replayed = Vec::with_capacity(events + errors.len() + 2);
                                replayed.extend(record.stream_origin.map(|origin| {
                                    Ok(rig_core::streaming::Relayed::Origin(origin))
                                }));
                                let mut errors = errors.into_iter().peekable();
                                let mut items = items.into_iter();
                                for position in 0..events + errors.len() {
                                    if errors.peek().is_some_and(|error| error.item == position) {
                                        if position >= events
                                            && let Some(response) = done.take()
                                        {
                                            replayed.push(Ok(rig_core::streaming::Relayed::Done(
                                                response,
                                            )));
                                        }
                                        replayed
                                            .extend(errors.next().map(|error| Err(error.error)));
                                    } else {
                                        replayed.extend(items.next().map(|item| {
                                            Ok(rig_core::streaming::Relayed::Item(item))
                                        }));
                                    }
                                }
                                replayed.extend(done.map(|response| {
                                    Ok(rig_core::streaming::Relayed::Done(response))
                                }));
                                replayed.extend(ending);
                                let items = replayed;
                                return Reply::Stream(Box::pin(futures::stream::iter(items)));
                            }
                            record.outcome
                        }
                    }
                }
            };
            debug_assert_eq!(self.family, self.descriptor.family.family());
            Reply::Outcome(outcome)
        }
    }
}
