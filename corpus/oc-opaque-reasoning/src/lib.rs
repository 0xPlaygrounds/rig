//! OC-23: a provider whose reasoning comes back as an opaque blob that must
//! be sent back unchanged.
//!
//! The providers listed in the `[opaque_reasoning]` settings table become
//! catalog vendors, reached on an Anthropic Messages, OpenAI Responses or
//! OpenAI Chat endpoint at their `base_url`, with the key in `key_env`:
//!
//! ```toml
//! [[opaque_reasoning.provider]]
//! name = "thinker"
//! base_url = "https://llm.example.com"
//! api = "anthropic"            # or "openai-responses", "openai-chat"
//! key_env = "THINKER_API_KEY"  # none for a local endpoint
//! models = ["deep-1"]
//! # reasoning_field = "reasoning_content"  # Chat: where it takes reasoning text back
//! ```
//!
//! rig-core's wires keep each reasoning item's blob (a thinking block's
//! signature, a redacted block's data, a Responses item's
//! `encrypted_content`) verbatim as the block's provider item, stream only
//! its readable summary, and replay the item in place to the model that
//! made it and as plain reasoning text (or nothing) to any other. This
//! plugin adds what the library cannot know:
//!
//! - a blob whose message was edited, or whose earlier history changed,
//!   is dropped from the request instead of being sent stale: each blob's
//!   message is anchored, as it is committed, by a fingerprint of the
//!   conversation up to it, kept as an aside, and a request whose history
//!   no longer matches it goes without the blob;
//! - a request the provider rejects over a reasoning blob is sent again
//!   once, without any blob, with a notice;
//! - the transcript and the effect log show `[encrypted reasoning, N bytes]`
//!   in place of the blob, which stays in the saved conversation so a
//!   resumed session sends it again.

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{
    Catalog, Compat, ModelFacts, ModelSpec, Provide, ReasoningSupport, Transport,
};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::message::{Fingerprint, Message as ChatMessage};
use rig_harness::rig_core::providers::anthropic::wire::{AnthropicConfig, compatible};
use rig_harness::rig_core::providers::openai::wire::{Auth, Dialect, OpenAIConfig, Route};
use rig_harness::rig_core::providers::registry::{ConnectError, ProviderConfig, ProviderId};
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_ecs::{StopTurns, WriteJournal};
use serde::Deserialize;
use serde_json::{Value, json};

/// The plugin's name, as its author on the conversation's asides.
pub const NAME: &str = "opaque-reasoning";

/// The kind of the aside that anchors a blob's message: `{"prefix": …}`,
/// the fingerprint of the conversation up to and including it when it
/// was committed.
pub const ANCHOR: &str = "opaque-reasoning.anchor";

/// The keys of a reasoning item that hold its blob.
const BLOB_KEYS: [&str; 3] = ["signature", "data", "encrypted_content"];

/// The item types whose blob keys are blobs.
const REASONING_TYPES: [&str; 3] = ["thinking", "redacted_thinking", "reasoning"];

/// Registers the configured providers and keeps their reasoning blobs
/// correct across tool calls, resumes, model switches and history edits.
#[derive(Default)]
pub struct OpaqueReasoningPlugin;

/// The plugin's stage systems, in `Stage::Prepare` after every phase, so
/// they see each request as every other plugin left it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpaqueReasoningSystems;

impl Plugin for OpaqueReasoningPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<OpaqueReasoning>("opaque_reasoning")
            .configure_sets(
                AgentPass,
                OpaqueReasoningSystems
                    .in_set(Stage::Prepare)
                    .after(PrepareSystems::Fit),
            )
            .add_systems(
                AgentPass,
                (
                    (anchor_blobs, drop_stale_blobs)
                        .chain()
                        .in_set(OpaqueReasoningSystems),
                    retry_without_blobs.in_set(ReviewSystems::Act),
                ),
            )
            .add_systems(Last, hide_blobs.after(StopTurns).before(WriteJournal))
            .add_systems(Startup, report_problems);
    }

    /// Adds the providers once every plugin's `build` ran.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let entries = world
            .get_resource::<OpaqueReasoning>()
            .map(|settings| settings.provider.clone())
            .unwrap_or_default();
        let mut problems = Vec::new();
        let mut models = world.get_resource_or_init::<Models>();
        for entry in entries {
            let name = entry.name.clone();
            match Gateway::new(entry) {
                Ok(gateway) => {
                    for spec in &gateway.models {
                        models.0.catalog_mut().insert(spec.clone());
                    }
                    models.0.provide(gateway.id, gateway);
                }
                Err(why) => problems.push(format!("[opaque_reasoning] `{name}` skipped: {why}.")),
            }
        }
        world.insert_resource(Problems(problems));
    }
}

/// The `[opaque_reasoning]` settings table.
#[derive(Resource, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct OpaqueReasoning {
    /// The providers.
    pub provider: Vec<ProviderEntry>,
}

/// One provider of the table.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProviderEntry {
    /// The vendor its models are filed under: `<name>/<model>`.
    pub name: String,
    /// Its API root.
    pub base_url: String,
    /// The wire it speaks.
    pub api: Api,
    /// The variable holding its key; none for an endpoint that needs none.
    pub key_env: Option<String>,
    /// Its model ids.
    pub models: Vec<String>,
    /// On Chat, the assistant message field it takes reasoning text back
    /// in, such as `reasoning_content`.
    pub reasoning_field: Option<String>,
}

/// The wire a provider speaks.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Api {
    /// Anthropic Messages: thinking blocks with a signature, redacted
    /// thinking with its data.
    Anthropic,
    /// OpenAI Responses: reasoning items with an id, a summary and
    /// `encrypted_content`.
    OpenaiResponses,
    /// OpenAI Chat Completions.
    OpenaiChat,
}

/// What in the table could not be used.
#[derive(Resource, Default)]
struct Problems(Vec<String>);

fn report_problems(problems: Option<Res<Problems>>, mut notices: MessageWriter<Notice>) {
    for problem in problems.iter().flat_map(|problems| &problems.0) {
        notices.write(Notice::error(None, problem.clone()));
    }
}

/// How a configured provider's models are reached.
struct Gateway {
    id: ProviderId,
    entry: ProviderEntry,
    models: Vec<ModelSpec>,
}

impl Gateway {
    fn new(entry: ProviderEntry) -> Result<Self, String> {
        let id = ProviderId::declare(&entry.name).map_err(|why| why.to_string())?;
        if !id.is_declared() {
            return Err("the name is a built-in provider's".to_owned());
        }
        let compat = entry
            .reasoning_field
            .as_ref()
            .map(|field| Compat::default().with_reasoning_field(field));
        let models = entry
            .models
            .iter()
            .map(|model| {
                let spec = ModelSpec::new(id, model)
                    .with_tools(true)
                    .with_reasoning(ReasoningSupport::Unknown { default: None });
                match &compat {
                    Some(compat) => spec.with_compat(compat.clone()),
                    None => spec,
                }
            })
            .collect();
        Ok(Self { id, entry, models })
    }

    /// The provider's wire at its URL, with `key`.
    fn config(&self, key: &str) -> ProviderConfig {
        let vendor = self.id.vendor();
        match self.entry.api {
            Api::Anthropic => ProviderConfig::Anthropic(
                AnthropicConfig::with_key(&compatible(vendor, "", "", None), key)
                    .with_base_url(&self.entry.base_url),
            ),
            Api::OpenaiResponses | Api::OpenaiChat => {
                let route = match self.entry.api {
                    Api::OpenaiResponses => Route::Responses,
                    _ => Route::Chat,
                };
                let mut config = OpenAIConfig::with_key(&Dialect::gateway(vendor, "", ""), key);
                config.auth = Auth::OptionalBearer;
                ProviderConfig::OpenAi(config.with_base_url(&self.entry.base_url).with_route(route))
            }
        }
    }
}

impl Provide for Gateway {
    fn handler(
        &self,
        spec: &ModelSpec,
        catalog: &Catalog,
        transport: &Transport,
    ) -> Result<ErasedHandler, ConnectError> {
        let key = match &self.entry.key_env {
            None => String::new(),
            Some(name) => std::env::var(name)
                .ok()
                .filter(|key| !key.is_empty())
                .ok_or_else(|| ConnectError::MissingKey {
                    vendor: self.id.vendor().to_owned(),
                    tried: vec![name.clone().into()],
                })?,
        };
        let http = transport.wrap(rig_reqwest::shared());
        let facts = ModelFacts::new(spec.clone()).with_catalog(catalog);
        Ok(self
            .config(&key)
            .completion_handler(&spec.reference(), &spec.id, http, facts))
    }
}

/// The bytes of a reasoning item's blob, when it holds one.
fn blob_bytes(item: &Value) -> usize {
    BLOB_KEYS
        .iter()
        .filter_map(|key| item.get(key)?.as_str())
        .map(str::len)
        .sum()
}

/// The bytes of the blobs `message` carries.
pub fn blobs_in(message: &ChatMessage) -> usize {
    let ChatMessage::Assistant(turn) = message else {
        return 0;
    };
    turn.content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Reasoning(reasoning) => reasoning.native.as_ref(),
            _ => None,
        })
        .map(|native| blob_bytes(&native.item))
        .sum()
}

/// Takes every blob out of `message`: its reasoning then replays from its
/// text, or not at all to the model that made it. Whether it had any.
pub fn drop_blobs(message: &mut ChatMessage) -> bool {
    let ChatMessage::Assistant(turn) = message else {
        return false;
    };
    let mut dropped = false;
    for block in &mut turn.content {
        if let AssistantContent::Reasoning(reasoning) = block
            && reasoning
                .native
                .as_ref()
                .is_some_and(|native| blob_bytes(&native.item) > 0)
        {
            reasoning.native = None;
            dropped = true;
        }
    }
    dropped
}

/// The fingerprint of `messages[..=i]`, for each `i`.
fn prefixes(messages: &[ChatMessage]) -> Vec<Fingerprint> {
    messages
        .iter()
        .scan(Fingerprint::of(&()), |prefix, message| {
            *prefix = Fingerprint::of(&(*prefix, message));
            Some(*prefix)
        })
        .collect()
}

/// Anchors each message with blobs as it is committed, and notes its blobs'
/// size in the transcript.
fn anchor_blobs(mut committed: MessageReader<Committed>, mut agents: Query<&mut Conversation>) {
    for Committed { agent, change } in committed.read() {
        let Change::Push { entries } = &change.change else {
            continue;
        };
        let Ok(mut conversation) = agents.get_mut(*agent) else {
            continue;
        };
        for entry in entries {
            let bytes = blobs_in(&entry.message);
            let Some(id) = entry.id.filter(|_| bytes > 0) else {
                continue;
            };
            let prefix = conversation
                .index_of(id)
                .and_then(|at| conversation.messages().get(..=at))
                .and_then(|upto| prefixes(upto).last().copied());
            let Some(prefix) = prefix else { continue };
            let author = Author::plugin(NAME);
            let anchor = Aside::new(ANCHOR, author.clone(), json!({ "prefix": prefix }));
            conversation.add_aside(anchor.after(Some(id)));
            let title = format!("[encrypted reasoning, {bytes} bytes]");
            conversation.add_aside(Note::new(title, "").aside(author, Some(id)));
        }
    }
}

/// The anchors of `conversation`: each blob message's id and prefix.
fn anchors(conversation: &Conversation) -> Vec<(MessageId, Fingerprint)> {
    conversation
        .asides()
        .filter(|(_, aside)| aside.kind == ANCHOR)
        .filter_map(|(_, aside)| {
            let prefix = serde_json::from_value(aside.value.get("prefix")?.clone()).ok()?;
            Some((aside.after?, prefix))
        })
        .collect()
}

/// Drops from each request the blobs of a message that was edited, or
/// whose earlier history changed, since it was committed.
fn drop_stale_blobs(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, ModelCalls)>,
    agents: Query<&Conversation>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let Ok(anchors) = agents.get(agent).map(anchors) else {
            continue;
        };
        let Some(history) = call.history.as_ref().filter(|_| !anchors.is_empty()) else {
            continue;
        };
        let now = prefixes(history.messages());
        let stale: Vec<usize> = history
            .entries()
            .filter(|entry| blobs_in(entry.message) > 0)
            .filter(|entry| {
                anchors
                    .iter()
                    .find(|(id, _)| *id == entry.id)
                    .is_some_and(|(_, prefix)| now.get(entry.index) != Some(prefix))
            })
            .map(|entry| entry.index)
            .collect();
        if stale.is_empty() {
            continue;
        }
        let Some(history) = call.history.as_mut() else {
            continue;
        };
        for at in stale {
            if let Some(mut entry) = history.entry(at) {
                drop_blobs(&mut entry.message);
                history.splice(at..=at, [entry], Author::plugin(NAME));
            }
        }
    }
}

/// On a model call sent again without its blobs: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct BlobsDropped;

/// Whether `report` is a provider rejecting a request over a reasoning
/// blob. Providers say so only in their error text.
fn rejects_blob(report: &ErrorReport) -> bool {
    let client_error = report
        .http_status
        .is_some_and(|status| (400..500).contains(&status));
    if report.kind != ErrorKind::ProviderResponse || !client_error {
        return false;
    }
    let text = serde_json::to_string(report)
        .unwrap_or_default()
        .to_ascii_lowercase();
    [
        "signature",
        "encrypted_content",
        "encrypted content",
        "invalid reasoning",
    ]
    .iter()
    .any(|word| text.contains(word))
}

/// Sends a model call the provider rejected over a reasoning blob again,
/// once, with every blob taken out.
fn retry_without_blobs(
    failed: Query<
        (
            Entity,
            &Outcome,
            &ModelRequest,
            &Effect,
            &ForAgent,
            Option<&CallOf>,
            Has<TurnCall>,
        ),
        (
            Unreviewed,
            ModelCalls,
            Without<SupersededBy>,
            Without<BlobsDropped>,
        ),
    >,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, outcome, request, effect, &ForAgent(agent), of, turn_call) in &failed {
        let Err(report) = &outcome.0 else { continue };
        if !rejects_blob(report) {
            continue;
        }
        let mut retry = request.clone();
        let mut dropped = false;
        for message in &mut retry.request.chat_history {
            dropped |= drop_blobs(message);
        }
        if !dropped {
            continue;
        }
        let mut successor = commands.spawn((
            retry,
            ForAgent(agent),
            Effect::child_of(effect.parent),
            BlobsDropped,
        ));
        if turn_call {
            successor.insert(TurnCall);
        }
        if let Some(&CallOf(turn)) = of {
            successor.insert(CallOf(turn));
        }
        successor.add_related::<SupersededBy>(&[call]);
        let why = report.message.lines().next().unwrap_or_default();
        notices.write(Notice::warn(
            agent,
            format!("The provider rejected a reasoning blob ({why}); sent again without blobs."),
        ));
    }
}

/// `value`, a record's JSON, with every reasoning blob replaced by its
/// size.
fn hide(value: &mut Value) -> bool {
    let mut hidden = false;
    if let Value::Object(fields) = value {
        let reasoning = fields
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| REASONING_TYPES.contains(&kind));
        for (key, field) in fields.iter_mut() {
            let blob =
                key == "encrypted_content" || (reasoning && BLOB_KEYS.contains(&key.as_str()));
            match field {
                Value::String(text) if blob && !text.is_empty() => {
                    *text = format!("[encrypted reasoning, {} bytes]", text.len());
                    hidden = true;
                }
                other => hidden |= hide(other),
            }
        }
    } else if let Value::Array(items) = value {
        for item in items {
            hidden |= hide(item);
        }
    }
    hidden
}

/// Takes every reasoning blob out of the effect records before they are
/// written to the log.
fn hide_blobs(mut records: MessageMutator<EffectRecorded>) {
    for EffectRecorded(record) in records.read() {
        let Ok(mut json) = serde_json::to_value(&*record) else {
            continue;
        };
        if hide(&mut json)
            && let Ok(clean) = serde_json::from_value(json)
        {
            *record = clean;
        }
    }
}

#[cfg(test)]
mod tests;
