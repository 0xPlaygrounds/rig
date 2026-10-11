//! Plugin corpus: 60. Reasoning tag splitter (spec `60-think-tag-splitter.md`).
//!
//! Open reasoning models behind plain OpenAI-compatible endpoints send their
//! thinking inline, as `<think>…</think>` in the answer text. This plugin
//! puts an [`Adapter`] on every model call. Start applies it to the model
//! the call finally goes to, whatever routed it there; for a listed model,
//! or any model while `detect` is on, it rewrites the reply stream so the
//! thinking streams as reasoning and the rest as text, and the response the
//! reply ends with is split the same way. The kernel, the fronts and other
//! plugins only ever see the split reply, and so does the conversation.
//! Reasoning that no provider sent as reasoning (the split-out kind) is left
//! out of what those models are sent unless `resend_reasoning` is set.
//!
//! Settings, `[think_tags]` in `RIG_HOME/settings.toml`:
//!
//! ```toml
//! [think_tags]
//! models = ["deepseek/deepseek-r1"]  # always split
//! detect = true                      # also any reply that starts with <think>
//! resend_reasoning = false
//! ```

use std::collections::HashMap;

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{EffectId, EffectKind, Outcome as Answered};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::message::{Reasoning, Text};
use rig_harness::rig_core::serve::{Decision, Intercept, Verdict};
use rig_harness::rig_core::streaming::{Item, Part, PartKind, Relayed, StreamEvent};
use rig_harness::rig_ecs::effects::Handler;
use serde::Deserialize;

#[cfg(test)]
mod tests;

const NAME: &str = "think-tags";
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
    /// Send split-out reasoning back to the models that are split.
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

#[derive(Default)]
pub struct ThinkTagSplitterPlugin;

impl Plugin for ThinkTagSplitterPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ThinkTags>("think_tags")
            .add_systems(AgentPass, adapt.in_set(Stage::Prepare));
    }
}

/// Puts the split on each new model call, for whichever model it goes to.
fn adapt(mut calls: Query<&mut ModelRequest, Added<Admitted>>, tags: Res<ThinkTags>) {
    if !tags.detect && tags.models.is_empty() {
        return;
    }
    for mut call in &mut calls {
        let tags = tags.clone();
        call.adapters
            .push(Adapter::new(NAME, move |mut connection: Connection| {
                let model = connection.spec.reference();
                let always = tags.models.contains(&model);
                if !(always || tags.detect) {
                    return connection;
                }
                let split = move |_| {
                    let mut parts = Parts::new(always);
                    move |relayed| parts.relay(relayed)
                };
                let mut handler = connection.handler.0.map_stream(NAME, split);
                if !tags.resend_reasoning {
                    handler = handler.layered(LeaveOutSplitReasoning);
                }
                connection.handler = Handler(handler);
                connection
            }));
    }
}

/// Leaves reasoning no provider item stands behind, which is what the split
/// makes, out of the assistant messages a request sends.
struct LeaveOutSplitReasoning;

impl Intercept for LeaveOutSplitReasoning {
    fn name(&self) -> String {
        format!("{NAME}-history")
    }

    async fn before(&self, _: EffectId, kind: &EffectKind) -> Decision {
        let EffectKind::Completion { request, stream } = kind else {
            return Decision::Proceed;
        };
        let split = |block: &AssistantContent| matches!(block, AssistantContent::Reasoning(reasoning) if reasoning.native.is_none());
        let answers = || {
            request
                .chat_history
                .iter()
                .filter_map(|message| match message {
                    message::Message::Assistant(answer) => Some(answer),
                    _ => None,
                })
        };
        if !answers().any(|answer| answer.content.iter().any(split)) {
            return Decision::Proceed;
        }
        let mut request = request.clone();
        for message in &mut request.chat_history {
            if let message::Message::Assistant(answer) = message {
                answer.content.retain(|block| !split(block));
            }
        }
        let stream = *stream;
        Decision::Patch(EffectKind::Completion { request, stream })
    }

    async fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<Answered, ErrorReport>,
    ) -> Verdict {
        Verdict::Keep
    }
}

/// One reply being split: its parts renumbered as text parts split, so the
/// rewritten events are a valid transcript of the rewritten response.
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

    /// What `relayed` becomes once split.
    fn relay(&mut self, relayed: Relayed) -> Vec<Relayed> {
        match relayed {
            Relayed::Item(Item::Event(event)) => self
                .event(event)
                .into_iter()
                .map(|event| Relayed::Item(Item::Event(event)))
                .collect(),
            Relayed::Done(mut response) => {
                response.choice = split_choice(&response.choice, self.always);
                vec![Relayed::Done(response)]
            }
            other => vec![other],
        }
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

/// A response's blocks, each text block split as its stream was.
fn split_choice(choice: &[AssistantContent], always: bool) -> Vec<AssistantContent> {
    let mut split = Vec::with_capacity(choice.len());
    for block in choice {
        let AssistantContent::Text(text) = block else {
            split.push(block.clone());
            continue;
        };
        let mut splitter = Splitter::new(always);
        let mut pieces = splitter.push(&text.text);
        pieces.extend(splitter.finish());
        if splitter.mode != Mode::Split {
            split.push(block.clone());
            continue;
        }
        let mut runs: Vec<(PartKind, String)> = Vec::new();
        for Piece { kind, text } in pieces {
            match runs.last_mut() {
                Some((last, so_far)) if *last == kind => so_far.push_str(&text),
                _ => runs.push((kind, text)),
            }
        }
        split.extend(runs.into_iter().map(|(kind, text)| content(kind, text)));
    }
    split
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
