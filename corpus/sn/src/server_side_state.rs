//! SN-23: a provider that keeps the conversation on its server.
//!
//! Some APIs, such as OpenAI's Responses API with `previous_response_id`,
//! keep each conversation on the server: a request that names the last
//! reply's id needs only what was added since. This plugin sends a turn's
//! request that way to the models of the [`StatefulVendors`] whenever it
//! safely can, and the whole conversation whenever it cannot.
//!
//! Each agent keeps its own chain, [`ServerChain`], saved with the session:
//! the id of the last reply its model sent and fingerprints of what the
//! server holds under it (the system prompt, the tools, every message sent
//! and the reply). A system wraps the handler of a turn's model call
//! ([`Adapted`]), for that call only, with the chain as it stands. The
//! wrapper runs inside the outermost adapters, such as secret redaction,
//! so it compares the request as they changed it with the chain, and
//! uploads only the messages after the reply with the reply's id, or the
//! whole request with the reason. A server that no longer knows the id
//! (`previous_response_not_found`) gets the whole request at once, in the
//! same call, so the user never sees the failure. What was sent comes back
//! to the app as a fact the call publishes ([`ChainReport`]), which moves
//! the agent's chain on.
//!
//! The reasoning a model returns, opaque encrypted items included, is part
//! of its reply in the conversation, so a full send carries it back.
//! `/state` says, for the agent it is typed for, the chain's id, how many
//! requests went incrementally and in full, and why the last full one was.

use std::sync::Arc;

use futures::StreamExt;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_harness::rig_core::effect::{Answered as Answer, EffectKind, HandlerDescriptor, family};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::message::{Fingerprint, Message};
use rig_harness::rig_core::providers::registry::ProviderId;
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_harness::rig_core::streaming::Relayed;
use rig_harness::rig_core::tool::{ContextValue, PublishedContext};
use rig_harness::rig_ecs::effects::Handler;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The adapter's and the handler layer's name.
pub const NAME: &str = "server-state";

/// The error code a server answers a previous response id it does not
/// know (any more) with, as OpenAI's Responses API does.
pub const NOT_FOUND: &str = "previous_response_not_found";

/// The request parameter naming the previous response.
pub const PREVIOUS: &str = "previous_response_id";

/// Sends each turn's request to a stateful server incrementally when it
/// safely can.
#[derive(Default)]
pub struct ServerSideStatePlugin;

/// The plugin's stage system, an ordinary adapter: it wraps only the
/// handler of the [`Adapted`] connection, inside the outermost adapters
/// such as a replay's, which then see the whole request. It sees every
/// splice compaction made before the request, as conversation changes are
/// announced only between stages.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServerStateSystems;

impl Plugin for ServerSideStatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StatefulVendors>()
            .register_required_components::<Agent, ServerChain>()
            .configure_sets(
                AgentPass,
                ServerStateSystems.in_set(PrepareSystems::Adapt),
            )
            .add_systems(AgentPass, chain_calls.in_set(ServerStateSystems))
            .add_observer(move_chain)
            .add_command(
                "state",
                "The server-side chain of this agent: its id, and requests sent incrementally and in full",
                state,
            );
    }
}

/// The vendors whose models keep conversations on their server, answer
/// with a response id and take `previous_response_id`: OpenAI's by
/// default. Insert another before the plugin to change them.
#[derive(Resource, Clone, Debug)]
pub struct StatefulVendors(pub Vec<ProviderId>);

impl Default for StatefulVendors {
    fn default() -> Self {
        Self(ProviderId::catalog("openai").into_iter().collect())
    }
}

/// What the server holds under a response id: fingerprints of what the
/// request that made it sent, and of its reply.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    /// The reply's id.
    pub response_id: String,
    /// The model it came from, `vendor/model`.
    pub model: String,
    /// The system prompt sent.
    pub system: String,
    /// The tools offered.
    pub tools: String,
    /// The messages sent, the system prompt left out.
    pub prefix: String,
    /// How many messages that was.
    pub len: usize,
    /// The reply's content.
    pub reply: String,
}

/// An agent's chain on the server, saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default)]
pub struct ServerChain {
    /// What the server holds for the agent, once a reply had an id.
    pub link: Option<Link>,
    /// The requests sent incrementally.
    pub incremental: u32,
    /// The requests sent in full.
    pub full: u32,
    /// Why the last request sent in full was.
    pub last_full: Option<String>,
    /// Why the next request may have to be sent in full, such as the
    /// conversation condensed since the last one.
    pub note: Option<String>,
}

impl ServerChain {
    /// The chain moved on by what a call sent.
    fn advance(&mut self, report: ChainReport) {
        self.link = report.link;
        self.note = None;
        if report.incremental {
            self.incremental += 1;
        } else {
            self.full += 1;
            self.last_full = report.reason;
        }
    }

    /// What `/state` says.
    pub fn describe(&self) -> String {
        let id = self
            .link
            .as_ref()
            .map_or("none", |link| link.response_id.as_str());
        let last = self
            .last_full
            .as_ref()
            .map(|reason| format!(" (last: {reason})"))
            .unwrap_or_default();
        format!(
            "chain: {id}; incremental: {}, full: {}{last}",
            self.incremental, self.full
        )
    }
}

/// What one model call sent, published by the call for the app.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainReport {
    /// What the server holds after the reply, when it had an id.
    pub link: Option<Link>,
    /// Whether the request went incrementally.
    pub incremental: bool,
    /// Why it went in full.
    pub reason: Option<String>,
    /// The messages uploaded, the system prompt left out.
    pub uploaded: usize,
    /// Whether an incremental attempt was refused first.
    pub retried: bool,
}

impl ContextValue for ChainReport {
    const KEY: &'static str = "server_chain";
}

/// The chain a call starts from, as its handler gets it.
#[derive(Clone, Debug, Default)]
struct ChainHint {
    link: Option<Link>,
    note: Option<String>,
}

/// Notes what each splice did to a chained conversation, then gives each
/// turn's model call its agent's chain, applied to the model it finally
/// goes to if that model keeps state.
fn chain_calls(
    mut committed: MessageReader<ConversationChanged>,
    mut calls: Query<(&ForAgent, &mut Adapted), (Added<Admitted>, With<TurnCall>, ModelCalls)>,
    mut chains: Query<&mut ServerChain, With<Conversation>>,
    vendors: Res<StatefulVendors>,
) {
    for change in committed.read() {
        let Change::Splice { .. } = &change.change.change else {
            continue;
        };
        let Ok(mut chain) = chains.get_mut(change.agent) else {
            continue;
        };
        if chain.link.is_none() {
            continue;
        }
        let note = if rig_basics::compaction::summary_in(&change.change.change).is_some() {
            "conversation condensed"
        } else {
            "conversation edited"
        };
        chain.note = Some(note.to_owned());
    }
    for (&ForAgent(agent), mut adapted) in &mut calls {
        let Ok(chain) = chains.get(agent) else {
            continue;
        };
        let Adapted(connection) = &mut *adapted;
        if !vendors.0.contains(&connection.spec.provider) {
            continue;
        }
        let chained = Chained {
            inner: connection.handler.0.clone(),
            model: connection.spec.reference(),
            hint: ChainHint {
                link: chain.link.clone(),
                note: chain.note.clone(),
            },
        };
        connection.handler = Handler(ErasedHandler::new(chained));
    }
}

/// Moves the agent's chain on by what its turn's model call sent.
fn move_chain(
    settled: On<Add<Settled>>,
    calls: Query<(&Published, &ForAgent), With<TurnCall>>,
    mut chains: Query<&mut ServerChain>,
) {
    let Ok((published, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(report)) = published.0.get::<ChainReport>() else {
        return;
    };
    if let Ok(mut chain) = chains.get_mut(agent) {
        chain.advance(report);
    }
}

/// `/state`.
fn state(
    In(args): In<CommandArgs>,
    chains: Query<&ServerChain>,
    mut notices: MessageWriter<Notice>,
) {
    let text = chains.get(args.agent).map_or_else(
        |_| "No server-side chain.".to_owned(),
        ServerChain::describe,
    );
    notices.write(Notice::info(args.agent, text));
}

/// A stateful model's handler for one call, which uploads only what the
/// server lacks when the chain it was given still holds.
struct Chained {
    inner: ErasedHandler,
    model: String,
    hint: ChainHint,
}

impl Serve for Chained {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        let mut descriptor = self.inner.descriptor();
        descriptor.layers.insert(0, NAME.to_owned());
        descriptor
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let EffectKind::Completion { request, stream } = kind else {
            return self.inner.handle(kind, dispatch).await;
        };
        let reason = match plan(&self.hint, &request, &self.model) {
            Ok((previous, from)) => {
                let upload = upload(&request, &previous, from);
                let sent = Sent {
                    incremental: true,
                    reason: None,
                    uploaded: body(&upload).len(),
                    retried: false,
                };
                match self.open(upload.into(), stream, &dispatch).await {
                    Ok(reply) => return self.tell(reply, request, sent, &dispatch),
                    Err(report) if report.code.as_deref() == Some(NOT_FOUND) => {
                        "previous response not found".to_owned()
                    }
                    Err(report) => return Reply::Outcome(Err(report)),
                }
            }
            Err(reason) => reason,
        };
        let retried = reason == "previous response not found";
        let sent = Sent {
            incremental: false,
            uploaded: body(&request).len(),
            reason: Some(reason),
            retried,
        };
        match self.open(request.clone(), stream, &dispatch).await {
            Ok(reply) => self.tell(reply, request, sent, &dispatch),
            Err(report) => Reply::Outcome(Err(report)),
        }
    }
}

/// How a request went.
struct Sent {
    incremental: bool,
    reason: Option<String>,
    uploaded: usize,
    retried: bool,
}

impl Chained {
    /// The model's reply to `request`, or its failure before it said
    /// anything. The inner dispatch has the call's scopes but not its
    /// recorder: the record is of the logical request and the answer this
    /// handler gives.
    async fn open(
        &self,
        request: Arc<CompletionRequest>,
        stream: bool,
        dispatch: &Dispatch,
    ) -> Result<Reply, ErrorReport> {
        let mut inner = Dispatch::new(dispatch.id(), dispatch.is_stream());
        for scope in dispatch.scopes() {
            inner = inner.with_scope(scope);
        }
        if let Some(context) = dispatch.adapter_context() {
            inner = inner.with_adapter_context(context);
        }
        let kind = EffectKind::Completion { request, stream };
        match self.inner.handle(kind, inner).await {
            Reply::Outcome(Err(report)) => Err(report),
            Reply::Outcome(answer) => Ok(Reply::Outcome(answer)),
            Reply::Stream(mut events) => {
                let mut head = Vec::new();
                loop {
                    match events.next().await {
                        Some(Err(report)) => return Err(report),
                        Some(Ok(item @ Relayed::Origin(_))) => head.push(Ok(item)),
                        Some(item) => {
                            head.push(item);
                            break;
                        }
                        None => break,
                    }
                }
                let events = futures::stream::iter(head).chain(events);
                Ok(Reply::Stream(Box::pin(events)))
            }
        }
    }

    /// `reply`, publishing what was sent once the response is in.
    fn tell(
        &self,
        reply: Reply,
        request: Arc<CompletionRequest>,
        sent: Sent,
        dispatch: &Dispatch,
    ) -> Reply {
        let published = dispatch.scope::<PublishedContext>();
        let model = self.model.clone();
        let publish = move |response: &CompletionResponse| {
            let report = ChainReport {
                link: link_after(&request, &model, response),
                incremental: sent.incremental,
                reason: sent.reason.clone(),
                uploaded: sent.uploaded,
                retried: sent.retried,
            };
            if let Some(published) = &published {
                published.insert(report).ok();
            }
        };
        match reply {
            Reply::Outcome(Ok(Answer::Completion(response))) => {
                publish(&response);
                Reply::Outcome(Ok(Answer::Completion(response)))
            }
            Reply::Stream(events) => Reply::Stream(Box::pin(events.map(move |item| {
                if let Ok(Relayed::Done(response)) = &item {
                    publish(response);
                }
                item
            }))),
            other => other,
        }
    }
}

/// The fingerprint of `value`, as text.
fn fingerprint(value: &impl Serialize) -> String {
    serde_json::to_value(Fingerprint::of(value))
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// The request's system prompt.
fn system(request: &CompletionRequest) -> Vec<&Message> {
    let system = |message: &&Message| matches!(message, Message::System { .. });
    request
        .chat_history
        .iter()
        .map(|message| &**message)
        .filter(system)
        .collect()
}

/// The request's messages but its system prompt.
fn body(request: &CompletionRequest) -> Vec<&Message> {
    let other = |message: &&Message| !matches!(message, Message::System { .. });
    request
        .chat_history
        .iter()
        .map(|message| &**message)
        .filter(other)
        .collect()
}

/// The previous response id and the first message to upload, or why the
/// whole request goes.
fn plan(
    hint: &ChainHint,
    request: &CompletionRequest,
    model: &str,
) -> Result<(String, usize), String> {
    let Some(link) = &hint.link else {
        return Err(hint
            .note
            .clone()
            .unwrap_or_else(|| "no previous response".to_owned()));
    };
    if link.model != model {
        return Err("model changed".to_owned());
    }
    if fingerprint(&system(request)) != link.system {
        return Err("system prompt changed".to_owned());
    }
    if fingerprint(&request.tools) != link.tools {
        return Err("tools changed".to_owned());
    }
    let changed = || {
        hint.note
            .clone()
            .unwrap_or_else(|| "earlier conversation changed".to_owned())
    };
    let body = body(request);
    let known = body.get(..link.len).ok_or_else(changed)?;
    if fingerprint(&known) != link.prefix {
        return Err(changed());
    }
    let Some(Message::Assistant(reply)) = body.get(link.len) else {
        return Err(changed());
    };
    let id = reply
        .origin
        .as_ref()
        .and_then(|origin| origin.response_id.as_deref());
    if id != Some(link.response_id.as_str()) || fingerprint(&reply.content) != link.reply {
        return Err(changed());
    }
    if body.len() <= link.len + 1 {
        return Err("nothing new to send".to_owned());
    }
    Ok((link.response_id.clone(), link.len + 1))
}

/// `request` with its system prompt and the messages from the body's
/// `from`th on, continuing `previous`.
fn upload(request: &CompletionRequest, previous: &str, from: usize) -> CompletionRequest {
    let mut upload = request.clone();
    let mut at = 0;
    upload.chat_history.retain(|message| {
        if matches!(&**message, Message::System { .. }) {
            return true;
        }
        at += 1;
        at > from
    });
    let mut params = match upload.additional_params.take() {
        Some(serde_json::Value::Object(params)) => params,
        _ => serde_json::Map::new(),
    };
    params.insert(PREVIOUS.to_owned(), previous.into());
    upload.additional_params = Some(serde_json::Value::Object(params));
    upload
}

/// What the server holds once it answered `request` with `response`;
/// `None` for a reply without an id, or one that is never replayed.
fn link_after(
    request: &CompletionRequest,
    model: &str,
    response: &CompletionResponse,
) -> Option<Link> {
    let response_id = response.origin.response_id.clone()?;
    if response.error.is_some() || response.aborted.is_some() {
        return None;
    }
    let body = body(request);
    Some(Link {
        response_id,
        model: model.to_owned(),
        system: fingerprint(&system(request)),
        tools: fingerprint(&request.tools),
        prefix: fingerprint(&body),
        len: body.len(),
        reply: fingerprint(&response.choice),
    })
}

/// The previous response id an upload names, for a server to read.
pub fn previous_of(request: &CompletionRequest) -> Option<&str> {
    request.additional_params.as_ref()?.get(PREVIOUS)?.as_str()
}
