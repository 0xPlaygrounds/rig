//! Plugin corpus: 60. Reasoning tag splitter (spec `60-think-tag-splitter.md`).
//!
//! Open reasoning models behind plain OpenAI-compatible endpoints send their
//! thinking inline, as `<think>…</think>` in the answer text. For each model
//! call of such a model this plugin, in Prepare, puts a stream transform in
//! front of the call's handler, so the call's [`Partial`] (what fronts and
//! other plugins follow) streams the thinking as reasoning and the rest as
//! text. In Review it splits the response the call ended with the same way,
//! so the conversation keeps the reasoning as reasoning and the answer
//! without tags; the effect log keeps the provider's reply beside it. A
//! request to such a model sends no earlier reasoning back unless
//! `resend_reasoning` is set.
//!
//! Settings, `[think_tags]` in `RIG_HOME/settings.toml`:
//!
//! ```toml
//! [think_tags]
//! models = ["deepseek/deepseek-r1"]  # always split
//! detect = true                      # also any reply that starts with <think>
//! resend_reasoning = false
//! ```

use std::collections::{HashMap, HashSet};

use futures::StreamExt;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{EffectKind, HandlerDescriptor, Outcome as Answered, family};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::message::{Reasoning, Text};
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_harness::rig_core::streaming::{Item, Part, PartKind, Relayed, StreamEvent, StreamEvents};
use rig_harness::rig_ecs::effects::Handler;
use serde::Deserialize;

#[cfg(test)]
mod tests;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Which models have their `<think>` tags split, and what is sent back:
/// the `[think_tags]` settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ThinkTags {
    /// Catalog references (`vendor/model`) whose replies are always split.
    pub models: Vec<String>,
    /// Also split a reply of any other model that starts with `<think>`.
    pub detect: bool,
    /// Send earlier reasoning back to a model that sends tags.
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

/// The models not listed in [`ThinkTags::models`] whose replies started
/// with `<think>` this session.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Saved)]
struct Detected(HashSet<String>);

/// The plugin's Prepare and Review systems: a plugin that routes a call to
/// another model orders itself before them, one that reviews a reply's
/// text after them.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SplitThinkTags;

#[derive(Default)]
pub struct ThinkTagSplitterPlugin;

impl Plugin for ThinkTagSplitterPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ThinkTags>("think_tags")
            .init_resource::<Detected>()
            .add_systems(
                AgentPass,
                (
                    prepare.in_set(Stage::Prepare),
                    split_replies.in_set(Stage::Review),
                )
                    .in_set(SplitThinkTags),
            );
    }
}

impl ThinkTags {
    /// Whether `model`'s replies are split from their first byte.
    fn always(&self, model: &str) -> bool {
        self.models.iter().any(|listed| listed == model)
    }
}

/// Puts the stream split in front of each new model call that may send
/// tags, and leaves earlier reasoning out of a request to a model that
/// sends them.
fn prepare(
    mut calls: Query<(&mut ModelRequest, Option<&ForAgent>), Added<Admitted>>,
    agents: Query<&Connection, With<Agent>>,
    tags: Res<ThinkTags>,
    detected: Res<Detected>,
) {
    for (mut call, agent) in &mut calls {
        let own = agent.and_then(|&ForAgent(agent)| agents.get(agent).ok());
        let Some(mut connection) = call.connection.as_ref().or(own).cloned() else {
            continue;
        };
        let model = connection.spec.reference();
        let always = tags.always(&model);
        if !(always || tags.detect) {
            continue;
        }
        let inner = connection.handler.0;
        connection.handler = Handler(ErasedHandler::new(SplitStream { inner, always }));
        call.connection = Some(connection);
        if tags.resend_reasoning || !(always || detected.0.contains(&model)) {
            continue;
        }
        for message in &mut call.request.chat_history {
            if let message::Message::Assistant(answer) = message {
                answer
                    .content
                    .retain(|block| !matches!(block, AssistantContent::Reasoning(_)));
            }
        }
    }
}

/// Splits the text of each model reply not settled yet, once: the stream
/// was split as it came, the response it ended with is the provider's.
fn split_replies(
    calls: Query<(Entity, &ModelRequest, &Outcome), Unreviewed>,
    tags: Res<ThinkTags>,
    mut detected: ResMut<Detected>,
    mut commands: Commands,
) {
    for (call, request, outcome) in &calls {
        let (Some(connection), Some(response)) = (&request.connection, outcome.completion()) else {
            continue;
        };
        let model = connection.spec.reference();
        let always = tags.always(&model);
        if !(always || tags.detect) {
            continue;
        }
        let mut split = false;
        let mut choice = Vec::with_capacity(response.choice.len());
        for block in &response.choice {
            let AssistantContent::Text(text) = block else {
                choice.push(block.clone());
                continue;
            };
            let mut splitter = Splitter::new(always);
            let mut pieces = splitter.push(&text.text);
            pieces.extend(splitter.finish());
            if !splitter.split() {
                choice.push(block.clone());
                continue;
            }
            split = true;
            choice.extend(blocks(pieces));
        }
        if choice == response.choice {
            continue;
        }
        if split && !always && !detected.0.contains(&model) {
            detected.0.insert(model);
        }
        let mut response = response.clone();
        response.choice = choice;
        commands
            .entity(call)
            .insert(Outcome(Ok(Answered::Completion(response))));
    }
}

/// A model handler whose streamed text has its `<think>` tags split out.
struct SplitStream {
    inner: ErasedHandler,
    /// Split from the first byte, not only a reply that starts with a tag.
    always: bool,
}

impl Serve for SplitStream {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        let mut descriptor = self.inner.descriptor();
        descriptor.layers.insert(0, "think-tags".to_owned());
        descriptor
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        match self.inner.handle(kind, dispatch).await {
            Reply::Stream(events) => {
                let mut parts = Parts::new(self.always);
                let split = events.flat_map(move |item| futures::stream::iter(parts.relay(item)));
                let split: StreamEvents = Box::pin(split);
                Reply::Stream(split)
            }
            other => other,
        }
    }
}

/// The parts of one reply stream, renumbered as text parts split.
struct Parts {
    always: bool,
    /// The next part number to give out.
    next: u32,
    /// The number given to each part passed on.
    renumbered: HashMap<Part, Part>,
    /// The text parts being split.
    texts: HashMap<Part, TextPart>,
}

/// A text part being split, and the part its pieces go to now.
struct TextPart {
    splitter: Splitter,
    open: Option<(Part, PartKind, String)>,
}

impl Parts {
    fn new(always: bool) -> Self {
        Self {
            always,
            next: 0,
            renumbered: HashMap::new(),
            texts: HashMap::new(),
        }
    }

    fn fresh(&mut self) -> Part {
        let part = Part::new(self.next);
        self.next = self.next.saturating_add(1);
        part
    }

    /// What `item` becomes once split.
    fn relay(&mut self, item: Result<Relayed, ErrorReport>) -> Vec<Result<Relayed, ErrorReport>> {
        let Ok(Relayed::Item(Item::Event(event))) = item else {
            return vec![item];
        };
        self.event(event)
            .into_iter()
            .map(|event| Ok(Relayed::Item(Item::Event(event))))
            .collect()
    }

    fn event(&mut self, event: StreamEvent) -> Vec<StreamEvent> {
        match event {
            StreamEvent::Start {
                kind: PartKind::Text,
                ..
            } => Vec::new(),
            StreamEvent::Text { part, text } => {
                let pieces = self.text(part).splitter.push(&text);
                self.pieces(part, pieces, false)
            }
            StreamEvent::End {
                part,
                content: AssistantContent::Text(_),
            } => {
                let pieces = self.text(part).splitter.finish();
                self.pieces(part, pieces, true)
            }
            StreamEvent::Start { part, kind, name } => {
                let fresh = self.fresh();
                self.renumbered.insert(part, fresh);
                vec![StreamEvent::Start {
                    part: fresh,
                    kind,
                    name,
                }]
            }
            other => {
                let part = other.part();
                let fresh = match self.renumbered.get(&part) {
                    Some(&fresh) => fresh,
                    None => self.fresh(),
                };
                if matches!(other, StreamEvent::End { .. }) {
                    self.renumbered.remove(&part);
                } else {
                    self.renumbered.insert(part, fresh);
                }
                vec![renumber(other, fresh)]
            }
        }
    }

    fn text(&mut self, part: Part) -> &mut TextPart {
        let always = self.always;
        self.texts.entry(part).or_insert_with(|| TextPart {
            splitter: Splitter::new(always),
            open: None,
        })
    }

    /// The events of `pieces` of the text `part`, each kind in a part of
    /// its own; `end` ends the last one and the text part.
    fn pieces(&mut self, part: Part, pieces: Vec<Piece>, end: bool) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let mut open = self.texts.get_mut(&part).and_then(|text| text.open.take());
        for Piece { kind, text } in pieces {
            let (part, _, so_far) = match &mut open {
                Some(open) if open.1 == kind => open,
                _ => {
                    events.extend(open.take().map(ended));
                    let fresh = self.fresh();
                    let name = None;
                    events.push(StreamEvent::Start {
                        part: fresh,
                        kind,
                        name,
                    });
                    open.insert((fresh, kind, String::new()))
                }
            };
            so_far.push_str(&text);
            events.push(match kind {
                PartKind::Reasoning => StreamEvent::Reasoning { part: *part, text },
                _ => StreamEvent::Text { part: *part, text },
            });
        }
        if end {
            events.extend(open.map(ended));
            self.texts.remove(&part);
        } else if let Some(text) = self.texts.get_mut(&part) {
            text.open = open;
        }
        events
    }
}

/// The end of a split-out part that holds `text`.
fn ended((part, kind, text): (Part, PartKind, String)) -> StreamEvent {
    let content = content(kind, text);
    StreamEvent::End { part, content }
}

/// A block of `kind` that holds `text`.
fn content(kind: PartKind, text: String) -> AssistantContent {
    match kind {
        PartKind::Reasoning => AssistantContent::Reasoning(Reasoning::new(text)),
        _ => AssistantContent::Text(Text::new(text)),
    }
}

/// `event` on `part`.
fn renumber(event: StreamEvent, part: Part) -> StreamEvent {
    match event {
        StreamEvent::Start { kind, name, .. } => StreamEvent::Start { part, kind, name },
        StreamEvent::Text { text, .. } => StreamEvent::Text { part, text },
        StreamEvent::Reasoning { text, .. } => StreamEvent::Reasoning { part, text },
        StreamEvent::Arguments { json, .. } => StreamEvent::Arguments { part, json },
        StreamEvent::End { content, .. } => StreamEvent::End { part, content },
    }
}

/// The blocks of `pieces`: one per run of a kind.
fn blocks(pieces: Vec<Piece>) -> Vec<AssistantContent> {
    let mut runs: Vec<(PartKind, String)> = Vec::new();
    for Piece { kind, text } in pieces {
        match runs.last_mut() {
            Some((last, so_far)) if *last == kind => so_far.push_str(&text),
            _ => runs.push((kind, text)),
        }
    }
    runs.into_iter()
        .map(|(kind, text)| content(kind, text))
        .collect()
}

/// A fragment of a split reply: [`PartKind::Text`] or
/// [`PartKind::Reasoning`].
struct Piece {
    kind: PartKind,
    text: String,
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

    /// Whether this text is split.
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

    /// The pieces held back at the end of the text: an unclosed `<think>`
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
                let kind = PartKind::Reasoning;
                pieces.push(Piece { kind, text });
            }
            return;
        }
        let text = match self.trim {
            true => text.trim_start().to_owned(),
            false => text,
        };
        if !text.is_empty() {
            self.trim = false;
            let kind = PartKind::Text;
            pieces.push(Piece { kind, text });
        }
    }
}
