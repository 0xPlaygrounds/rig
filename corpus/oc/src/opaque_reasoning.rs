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
//! made it. This plugin adds the rest:
//!
//! - every request asks for other models' reasoning to go as reasoning
//!   where its model reads it (`foreign_reasoning`), as plain text
//!   otherwise, never with its blob;
//! - a blob whose message was edited, whose earlier history changed, or
//!   that was never anchored, is left out instead of being sent stale:
//!   each blob's message is anchored, as it is committed, by a fingerprint
//!   of the conversation up to it, kept as an aside;
//! - a request the provider rejects over a reasoning blob is sent again
//!   once, without any blob, with a notice;
//! - the transcript and the effect log show `[encrypted reasoning, N bytes]`
//!   in place of the blob, which stays in the saved conversation so a
//!   resumed session sends it again.

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::{
    Catalog, Compat, ModelFacts, ModelSpec, Provide, ReasoningSupport, Transport,
};
use rig_harness::rig_core::effect::{Answered as EffectOutcome, EffectKind};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::message::{Fingerprint, Message as ChatMessage, Native};
use rig_harness::rig_core::providers::anthropic::wire::{AnthropicConfig, compatible};
use rig_harness::rig_core::providers::openai::wire::{Auth, Dialect, OpenAIConfig, Route};
use rig_harness::rig_core::providers::registry::{ConnectError, ProviderConfig, ProviderId};
use rig_harness::rig_core::serve::ErasedHandler;
use rig_harness::rig_ecs::{StopTurnsOnExit, WriteLogs};
use serde::Deserialize;
use serde_json::{Value, json};

/// The plugin's name, as its author on the conversation's asides.
pub const NAME: &str = "opaque-reasoning";

/// The kind of the aside that anchors a blob's message: `{"prefix": …}`,
/// the fingerprint of the conversation up to and including it when it
/// was committed.
pub const ANCHOR: &str = "opaque-reasoning.anchor";

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
                    (anchor_blobs, prepare_requests)
                        .chain()
                        .in_set(OpaqueReasoningSystems),
                    retry_without_blobs.in_set(ReviewSystems::Act),
                ),
            )
            .add_observer(follow_successor::<WithoutBlobs>)
            .add_systems(Last, hide_blobs.after(StopTurnsOnExit).before(WriteLogs))
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
    pub api: Wire,
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
pub enum Wire {
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
            Wire::Anthropic => ProviderConfig::Anthropic(
                AnthropicConfig::with_key(&compatible(vendor, "", "", None), key)
                    .with_base_url(&self.entry.base_url),
            ),
            Wire::OpenaiResponses | Wire::OpenaiChat => {
                let route = match self.entry.api {
                    Wire::OpenaiResponses => Route::Responses,
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

/// What a log or an export shows in place of a blob of `bytes` bytes.
fn placeholder(bytes: usize) -> String {
    format!("[encrypted reasoning, {bytes} bytes]")
}

/// The provider item of a reasoning block.
fn native(block: &AssistantContent) -> Option<&Native> {
    match block {
        AssistantContent::Reasoning(reasoning) => reasoning.native.as_ref(),
        _ => None,
    }
}

/// The provider item of any block, since a wire may sign text and calls
/// too.
fn any_native(block: &AssistantContent) -> Option<&Native> {
    match block {
        AssistantContent::Text(text) => text.native.as_ref(),
        AssistantContent::ToolCall(call) => call.native.as_ref(),
        AssistantContent::Reasoning(reasoning) => reasoning.native.as_ref(),
        AssistantContent::Image(image) => image.native.as_ref(),
        _ => None,
    }
}

/// The bytes of the blobs `message` carries.
pub fn blobs_in(message: &ChatMessage) -> usize {
    let ChatMessage::Assistant(turn) = message else {
        return 0;
    };
    turn.content
        .iter()
        .filter_map(native)
        .filter_map(Native::reasoning_blob)
        .map(|blob| blob.value.len())
        .sum()
}

/// `message` with every reasoning item that holds a blob taken out, so its
/// reasoning replays from its text, or not at all to the model that made
/// it; `None` when it holds no blob.
pub fn without_blobs(message: &ChatMessage) -> Option<ChatMessage> {
    if blobs_in(message) == 0 {
        return None;
    }
    let ChatMessage::Assistant(turn) = message else {
        return None;
    };
    let mut turn = turn.clone();
    for block in &mut turn.content {
        if let AssistantContent::Reasoning(reasoning) = block
            && reasoning
                .native
                .as_ref()
                .is_some_and(|native| native.reasoning_blob().is_some())
        {
            reasoning.native = None;
        }
    }
    Some(ChatMessage::Assistant(turn))
}

/// The fingerprint of `messages[..=i]`, for each `i`.
fn prefixes(messages: &[std::sync::Arc<ChatMessage>]) -> Vec<Fingerprint> {
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
fn anchor_blobs(
    mut committed: MessageReader<ConversationChanged>,
    mut agents: Query<&mut Conversation>,
) {
    for ConversationChanged { agent, change } in committed.read() {
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
            conversation.add_aside(Note::new(placeholder(bytes), "").aside(author, Some(id)));
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

/// On a model call that goes without any blob: one the provider rejected
/// over a blob, sent again. Once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct WithoutBlobs;

/// Lets each request send other models' reasoning as reasoning where its
/// model reads it, and takes out the blobs of a message that was edited,
/// whose earlier history changed, or that was never anchored, since it was
/// committed; every blob from a call marked [`WithoutBlobs`].
fn prepare_requests(
    mut calls: Query<
        (&mut ModelRequest, &ForAgent, Has<WithoutBlobs>),
        (Added<Admitted>, ModelCalls),
    >,
    agents: Query<&Conversation>,
) {
    for (mut call, &ForAgent(agent), without) in &mut calls {
        let call = &mut *call;
        call.request_mut().foreign_reasoning = true;
        let Some(history) = call.history.as_mut() else {
            continue;
        };
        let anchors = agents.get(agent).map(anchors).unwrap_or_default();
        let now = prefixes(history.messages());
        history.edit_each(Author::plugin(NAME), |entry| {
            let anchored = anchors
                .iter()
                .any(|(id, prefix)| *id == entry.id && now.get(entry.index) == Some(prefix));
            if anchored && !without {
                return None;
            }
            without_blobs(entry.message)
        });
    }
}

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
/// once, without any blob.
fn retry_without_blobs(
    failed: Query<
        (Entity, &Outcome, &ModelRequest, &ForAgent),
        (
            ReadyToSettle,
            ModelCalls,
            Without<SupersededBy>,
            Without<WithoutBlobs>,
        ),
    >,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, outcome, sent, &ForAgent(agent)) in &failed {
        let Err(report) = &outcome.0 else { continue };
        let carried = sent.request.chat_history.iter().any(|m| blobs_in(m) > 0);
        if !carried || !rejects_blob(report) {
            continue;
        }
        commands.entity(call).insert(WithoutBlobs);
        commands.queue(resend(call));
        let why = report.message.lines().next().unwrap_or_default();
        notices.write(Notice::warn(
            agent,
            format!("The provider rejected a reasoning blob ({why}); sent again without blobs."),
        ));
    }
}

/// Puts the placeholder in place of `native`'s blob.
fn hide_item(native: &mut Native) {
    let Some(blob) = native.reasoning_blob() else {
        return;
    };
    let (key, shown) = (blob.key.to_owned(), placeholder(blob.value.len()));
    if let Some(value) = native.item.pointer_mut(&key) {
        *value = Value::String(shown);
    }
}

/// Puts the placeholder in place of every string of `document`, a
/// provider's reply, that is one of `blobs`.
fn hide_in_document(document: &mut Value, blobs: &[String]) {
    match document {
        Value::String(text) if blobs.contains(text) => {
            *document = Value::String(placeholder(text.len()));
        }
        Value::Object(fields) => fields
            .values_mut()
            .for_each(|field| hide_in_document(field, blobs)),
        Value::Array(items) => items
            .iter_mut()
            .for_each(|item| hide_in_document(item, blobs)),
        _ => {}
    }
}

/// Hides the blobs of `content`.
fn hide_in(content: &mut [AssistantContent]) {
    for block in content {
        if let AssistantContent::Reasoning(reasoning) = block
            && let Some(native) = reasoning.native.as_mut()
        {
            hide_item(native);
        }
    }
}

/// Puts the placeholder in place of every reasoning blob of the model
/// calls' records before they are written to the log.
fn hide_blobs(mut records: MessageMutator<EffectRecorded>) {
    for EffectRecorded(record) in records.read() {
        if let EffectKind::Completion { request, .. } = &mut record.kind {
            for message in &mut std::sync::Arc::make_mut(request).chat_history {
                if let ChatMessage::Assistant(turn) = std::sync::Arc::make_mut(message) {
                    hide_in(&mut turn.content);
                }
            }
        }
        if let Ok(EffectOutcome::Completion(response)) = &mut record.outcome {
            let response = std::sync::Arc::make_mut(response);
            let blobs: Vec<String> = response
                .choice
                .iter()
                .filter_map(any_native)
                .filter_map(Native::reasoning_blob)
                .map(|blob| blob.value.to_owned())
                .collect();
            hide_in(&mut response.choice);
            hide_in_document(&mut response.raw, &blobs);
        }
    }
}

#[cfg(test)]
mod tests;
