//! Plugin corpus: 60. Reasoning tag splitter (spec `60-think-tag-splitter.md`).
//!
//! Open reasoning models behind plain OpenAI-compatible endpoints send their
//! thinking inline, as `<think>…</think>` in the answer text. This plugin
//! wraps the model handler of every [`Connection`] in a [`Serve`] layer that
//! rewrites the reply stream: text inside the tags becomes reasoning
//! fragments, the rest stays answer text, and the response the stream ends
//! with holds the reasoning as a reasoning block and the answer without
//! tags. Everything downstream (the kernel's [`Partial`], the fronts, the
//! conversation, other plugins) sees only the split reply. The effect log
//! keeps the provider's raw reply, since it records the inner handler.
//!
//! Requests to a model that sent tags carry no earlier reasoning back unless
//! [`ThinkTags::resend_reasoning`] is set.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::Poll;

use futures::{Stream, StreamExt};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionResponse;
use rig_harness::rig_core::effect::{EffectKind, HandlerDescriptor, Outcome, family};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::message::{Reasoning, Text};
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_harness::rig_core::streaming::{Item, Part, Relayed, StreamEvent, StreamEvents};
use rig_harness::rig_ecs::effects::Handler;

#[cfg(test)]
mod tests;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
/// The layer's name in the handler's descriptor; it also marks a handler
/// already wrapped.
const LAYER: &str = "think-tags";

/// Which models have their `<think>` tags split, and what is sent back.
#[derive(Resource, Clone, Debug)]
pub struct ThinkTags {
    /// Catalog references (`vendor/model`) whose replies are always split.
    pub models: Vec<String>,
    /// Also split a reply of any other model that starts with `<think>`.
    pub detect: bool,
    /// Send earlier reasoning back to a model that sent tags.
    pub resend_reasoning: bool,
}

impl Default for ThinkTags {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            detect: true,
            resend_reasoning: false,
        }
    }
}

/// The models whose replies held tags so far, written by the reply streams
/// as they find them.
#[derive(Resource, Clone, Default)]
struct Detected(Arc<Mutex<HashSet<String>>>);

impl Detected {
    fn note(&self, model: &str) {
        let mut models = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        models.insert(model.to_owned());
    }

    fn has(&self, model: &str) -> bool {
        let models = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        models.contains(model)
    }
}

#[derive(Default)]
pub struct ThinkTagSplitterPlugin;

impl Plugin for ThinkTagSplitterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ThinkTags>()
            .init_resource::<Detected>()
            .add_observer(wrap)
            .add_observer(drop_reasoning);
    }
}

/// Wraps each connection inserted, on an agent, a turn or a call, unless
/// it is wrapped already or no model of it is split.
fn wrap(
    inserted: On<Insert<Connection>>,
    mut connections: Query<&mut Connection>,
    tags: Res<ThinkTags>,
    detected: Res<Detected>,
) {
    let Ok(mut connection) = connections.get_mut(inserted.entity) else {
        return;
    };
    let model = connection.spec.reference();
    let always = tags.models.contains(&model);
    let inner = &connection.handler.0;
    if !(always || tags.detect) || inner.descriptor().layers.iter().any(|l| l == LAYER) {
        return;
    }
    let split = SplitThinkTags {
        inner: inner.clone(),
        split: Arc::new(Split {
            model,
            always,
            detected: detected.clone(),
        }),
    };
    connection.handler = Handler(ErasedHandler::new(split));
}

/// Leaves earlier reasoning out of a request to a model that sends tags.
fn drop_reasoning(
    mut prepare: On<PrepareRequest>,
    connections: Query<&Connection>,
    tags: Res<ThinkTags>,
    detected: Res<Detected>,
) {
    let turn = prepare.entity;
    let agent = prepare.agent;
    let Ok(connection) = connections.get(turn).or_else(|_| connections.get(agent)) else {
        return;
    };
    let model = connection.spec.reference();
    if tags.resend_reasoning || !(tags.models.contains(&model) || detected.has(&model)) {
        return;
    }
    for message in &mut prepare.messages {
        if let message::Message::Assistant(answer) = message {
            answer
                .content
                .retain(|block| !matches!(block, AssistantContent::Reasoning(_)));
        }
    }
}

/// A model handler whose replies have their `<think>` tags split out.
struct SplitThinkTags {
    inner: ErasedHandler,
    split: Arc<Split>,
}

/// How the replies of one connection's model are split.
struct Split {
    model: String,
    always: bool,
    detected: Detected,
}

impl Split {
    fn splitter(&self) -> Splitter {
        Splitter::new(self.always)
    }

    /// `response` with each text block split into reasoning and answer.
    fn response(&self, mut response: CompletionResponse) -> CompletionResponse {
        let mut choice = Vec::with_capacity(response.choice.len());
        for block in response.choice {
            let AssistantContent::Text(text) = &block else {
                choice.push(block);
                continue;
            };
            let mut splitter = self.splitter();
            let mut pieces = splitter.push(&text.text);
            pieces.extend(splitter.finish());
            if !splitter.split() {
                choice.push(block);
                continue;
            }
            self.detected.note(&self.model);
            let (reasoning, answer) = joined(pieces);
            if !reasoning.is_empty() {
                choice.push(AssistantContent::Reasoning(Reasoning::new(reasoning)));
            }
            if !answer.is_empty() {
                choice.push(AssistantContent::Text(Text::new(answer)));
            }
        }
        response.choice = choice;
        response
    }

    fn stream(self: Arc<Self>, mut inner: StreamEvents) -> StreamEvents {
        let mut state = StreamState {
            splitter: self.splitter(),
            answer: String::new(),
            part: None,
        };
        let mut queue: VecDeque<Result<Relayed, ErrorReport>> = VecDeque::new();
        let mut ended = false;
        let stream = futures::stream::poll_fn(move |cx| {
            loop {
                if let Some(item) = queue.pop_front() {
                    return Poll::Ready(Some(item));
                }
                if ended {
                    return Poll::Ready(None);
                }
                match inner.poll_next_unpin(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Some(item)) => self.relay(&mut state, item, &mut queue),
                    Poll::Ready(None) => {
                        ended = true;
                        // A reply cut short: what was held back still goes.
                        if let Some(part) = state.part {
                            let pieces = state.splitter.finish();
                            queue.extend(events(part, pieces, &mut state.answer));
                        }
                    }
                }
            }
        });
        boxed(stream)
    }

    /// Queues what `item` becomes once split.
    fn relay(
        &self,
        state: &mut StreamState,
        item: Result<Relayed, ErrorReport>,
        queue: &mut VecDeque<Result<Relayed, ErrorReport>>,
    ) {
        match item {
            Ok(Relayed::Item(Item::Event(StreamEvent::Text { part, text }))) => {
                state.part = Some(part);
                let pieces = state.splitter.push(&text);
                if state.splitter.split() {
                    self.detected.note(&self.model);
                }
                queue.extend(events(part, pieces, &mut state.answer));
            }
            Ok(Relayed::Item(Item::Event(StreamEvent::End {
                part,
                content: AssistantContent::Text(text),
            }))) => {
                let pieces = state.splitter.finish();
                queue.extend(events(part, pieces, &mut state.answer));
                let answer = std::mem::take(&mut state.answer);
                let content = match state.splitter.split() {
                    true => AssistantContent::Text(Text::new(answer)),
                    false => AssistantContent::Text(text),
                };
                state.splitter = self.splitter();
                state.part = None;
                let end = StreamEvent::End { part, content };
                queue.push_back(Ok(Relayed::Item(Item::Event(end))));
            }
            Ok(Relayed::Done(response)) => {
                let response = self.response(*response);
                queue.push_back(Ok(Relayed::Done(Box::new(response))));
            }
            other => queue.push_back(other),
        }
    }
}

/// The split state of one reply stream.
struct StreamState {
    splitter: Splitter,
    /// The answer text of the text part open now.
    answer: String,
    /// The text part open now.
    part: Option<Part>,
}

impl Serve for SplitThinkTags {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        let mut descriptor = self.inner.descriptor();
        descriptor.layers.insert(0, LAYER.to_owned());
        descriptor
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        match self.inner.handle(kind, dispatch).await {
            Reply::Stream(stream) => Reply::Stream(self.split.clone().stream(stream)),
            Reply::Outcome(Ok(Outcome::Completion(response))) => {
                Reply::Outcome(Ok(Outcome::Completion(self.split.response(response))))
            }
            other => other,
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn boxed(
    stream: impl Stream<Item = Result<Relayed, ErrorReport>> + Send + 'static,
) -> StreamEvents {
    Box::pin(stream)
}

#[cfg(target_family = "wasm")]
fn boxed(stream: impl Stream<Item = Result<Relayed, ErrorReport>> + 'static) -> StreamEvents {
    Box::pin(stream)
}

/// The stream events of `pieces` on the text `part`, adding the answer
/// text to `answer`. Reasoning goes out on the text's part: rig-core lets
/// only itself number a new part.
fn events(
    part: Part,
    pieces: Vec<Piece>,
    answer: &mut String,
) -> Vec<Result<Relayed, ErrorReport>> {
    let mut events = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let event = match piece {
            Piece::Text(text) => {
                answer.push_str(&text);
                StreamEvent::Text { part, text }
            }
            Piece::Reasoning(text) => StreamEvent::Reasoning { part, text },
        };
        events.push(Ok(Relayed::Item(Item::Event(event))));
    }
    events
}

/// The reasoning and the answer of `pieces`.
fn joined(pieces: Vec<Piece>) -> (String, String) {
    let (mut reasoning, mut answer) = (String::new(), String::new());
    for piece in pieces {
        match piece {
            Piece::Text(text) => answer.push_str(&text),
            Piece::Reasoning(text) => reasoning.push_str(&text),
        }
    }
    (reasoning, answer)
}

/// A fragment of a split reply.
#[derive(Debug, PartialEq, Eq)]
enum Piece {
    Text(String),
    Reasoning(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Not known yet whether the reply starts with `<think>`.
    Undecided,
    Split,
    Pass,
}

/// Splits streamed text at `<think>` tags, a tag split across fragments
/// too: the text that may still start a tag is held back.
struct Splitter {
    mode: Mode,
    /// Inside `<think>…</think>`.
    inside: bool,
    /// Text held back: a tag may start in it.
    held: String,
    /// Before any answer text, or right after `</think>`: the answer's
    /// leading whitespace is dropped.
    trim: bool,
}

impl Splitter {
    fn new(always: bool) -> Self {
        Self {
            mode: if always { Mode::Split } else { Mode::Undecided },
            inside: false,
            held: String::new(),
            trim: always,
        }
    }

    /// Whether this reply is split.
    fn split(&self) -> bool {
        self.mode == Mode::Split
    }

    /// The pieces `text` completes.
    fn push(&mut self, text: &str) -> Vec<Piece> {
        self.held.push_str(text);
        if self.mode == Mode::Undecided {
            let start = self.held.trim_start();
            if start.starts_with(OPEN) {
                self.mode = Mode::Split;
                self.trim = true;
            } else if OPEN.starts_with(start) {
                return Vec::new();
            } else {
                self.mode = Mode::Pass;
            }
        }
        let mut pieces = Vec::new();
        if self.mode == Mode::Pass {
            let held = std::mem::take(&mut self.held);
            self.emit(held, &mut pieces);
            return pieces;
        }
        loop {
            let tag = if self.inside { CLOSE } else { OPEN };
            if let Some(at) = self.held.find(tag) {
                let rest = self.held.split_off(at);
                let before = std::mem::replace(&mut self.held, rest);
                self.emit(before, &mut pieces);
                self.held.drain(..tag.len());
                self.inside = !self.inside;
                self.trim = !self.inside;
                continue;
            }
            // Hold back the longest end of the text that starts the tag.
            let keep = (1..tag.len())
                .rev()
                .find(|&n| tag.get(..n).is_some_and(|start| self.held.ends_with(start)))
                .unwrap_or(0);
            let rest = self.held.split_off(self.held.len().saturating_sub(keep));
            let ready = std::mem::replace(&mut self.held, rest);
            self.emit(ready, &mut pieces);
            return pieces;
        }
    }

    /// The pieces held back at the end of the reply: an unclosed `<think>`
    /// is reasoning.
    fn finish(&mut self) -> Vec<Piece> {
        if self.mode == Mode::Undecided {
            self.mode = Mode::Pass;
        }
        let mut pieces = Vec::new();
        let held = std::mem::take(&mut self.held);
        self.emit(held, &mut pieces);
        pieces
    }

    fn emit(&mut self, text: String, pieces: &mut Vec<Piece>) {
        if self.inside {
            if !text.is_empty() {
                pieces.push(Piece::Reasoning(text));
            }
            return;
        }
        let text = match self.trim {
            true => text.trim_start().to_owned(),
            false => text,
        };
        if !text.is_empty() {
            self.trim = false;
            pieces.push(Piece::Text(text));
        }
    }
}
