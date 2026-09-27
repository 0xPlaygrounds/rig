//! What a streaming consumer reads: the parts of a completion as they start,
//! grow and finish, projected from the canonical events.

use std::collections::VecDeque;

use crate::completion::CompletionResponse;
use crate::error::RigError;
use crate::message::AssistantContent;
use crate::operation::CompletionFold;

use super::{BlockKind, Delta, StreamEvent};

/// What a streaming consumer reads, in the order the parts of the response
/// start, grow and finish, then exactly one of [`Update::Done`] and
/// [`Update::Failed`], last.
///
/// `index` is the part's position in [`Update::Done`]'s `choice`. For every
/// index, `Start` comes first, the `Delta`s concatenate to the finished text,
/// reasoning text or argument JSON, and the part's last `End` equals
/// `choice[index]`. Parts interleave. A tool call's position in `choice` is
/// fixed when it ends, so its `Start`, its one argument `Delta` and its `End`
/// arrive together then; text and reasoning stream as they arrive. A part the
/// provider extends after it ended (more text under a reused block) ends
/// again with the whole part.
///
/// A failed stream ends with [`Update::Failed`] at its first error. Every
/// part that ended before it has its `End`, and its `partial` holds those
/// parts in index order. A part still open at the failure gets no `End` and
/// is not in `partial`; a part after it keeps the index it was given, which
/// counts the open part. A failure the driver reads from the transport
/// closes the open text and reasoning first, so they end and are in
/// `partial`. A relayed stream receives the origin's closes after its error,
/// so its open parts do not end.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    /// A part began.
    Start {
        /// The part's position in the response's `choice`.
        index: usize,
        /// What kind of part.
        part: PartKind,
    },
    /// Text, reasoning text, or a tool call's argument JSON fragment.
    Delta {
        /// The part this fragment extends.
        index: usize,
        /// The fragment.
        text: String,
    },
    /// A part finished.
    End {
        /// The part's position in the response's `choice`.
        index: usize,
        /// The finished part.
        part: AssistantContent,
    },
    /// The response, once the stream completed. Last.
    Done(CompletionResponse),
    /// The reply failed. `partial` is what arrived before the failure: every
    /// part that ended, and the usage reported so far. Last, like `Done`.
    Failed {
        /// Why the reply failed: the first error the stream carried, or the
        /// truncation of a stream that ended without its terminal record.
        error: RigError,
        /// What arrived before the failure, as
        /// [`CompletionStream::partial`](super::CompletionStream::partial)
        /// returned it then.
        partial: CompletionResponse,
    },
}

/// What kind of part an [`Update::Start`] began.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum PartKind {
    /// Text.
    Text,
    /// Reasoning.
    Reasoning,
    /// A tool call to `name`.
    ToolCall {
        /// The tool the call names.
        name: String,
    },
    /// An image.
    Image,
}

/// Where the projection stands on one slot of the fold's `choice`.
#[derive(Debug, Default)]
struct Slot {
    kind: Option<PartKind>,
    /// Whether the part will be in `choice`: it has content, or it is a
    /// kind that is never dropped once placed.
    certain: bool,
    /// The slot ended with nothing, so it is not in `choice`.
    empty: bool,
    started: bool,
    /// Fragments not yet emitted, and the text already emitted.
    pending: Vec<String>,
    emitted: String,
    /// An end not yet emitted.
    end: Option<AssistantContent>,
    /// Some of the part's fragments were read from the stream itself, so
    /// the rest of its text is sent at its end.
    whole: bool,
    /// The part as the fold held it when its end was last emitted.
    ended_as: Option<AssistantContent>,
}

/// What [`super::Streamed::updates`] has projected so far, kept on the
/// stream so a later call continues it.
#[derive(Debug, Default)]
pub(crate) struct Projection {
    pub(crate) projector: Projector,
    /// Updates projected and not yet read.
    pub(crate) queue: VecDeque<Update>,
    /// The terminal update is queued or read: nothing follows it.
    pub(crate) ended: bool,
    /// The reply's first error, read by the projection or from the stream
    /// itself: the projection ends with it as [`Update::Failed`] once what
    /// it holds is sent.
    pub(crate) failure: Option<RigError>,
    /// Events were read from the stream itself since the last projection.
    pub(crate) stale: bool,
}

impl Slot {
    /// The part ended as `block` before this projection saw it.
    fn settle(&mut self, block: &AssistantContent) {
        self.kind.get_or_insert_with(|| kind_of(block));
        self.certain = true;
        self.end = Some(block.clone());
    }
}

/// Projects the canonical completion events the fold has absorbed into
/// [`Update`]s. It reads each part's slot from the fold, so its indices are
/// the fold's positions; a part is held until every slot before it is known
/// to be in `choice` or not.
#[derive(Debug, Default)]
pub(crate) struct Projector {
    slots: Vec<Slot>,
    /// The issuer `choice` stamps on reasoning, known at the terminal
    /// record; a reasoning part's end waits for it.
    issuer: Option<String>,
}

impl Projector {
    /// Project `event`, which `fold` has just absorbed.
    pub(crate) fn push(
        &mut self,
        event: &StreamEvent,
        fold: &CompletionFold,
        out: &mut VecDeque<Update>,
    ) {
        match event {
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Text { additional_params },
            } => {
                if let Some(slot) = self.slot(fold, id, PartKind::Text) {
                    slot.certain |= additional_params.is_some();
                }
            }
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Reasoning { .. },
            } => {
                if let Some(slot) = self.slot(fold, id, PartKind::Reasoning) {
                    slot.certain = true;
                }
            }
            StreamEvent::BlockDelta { id, delta } => match delta {
                Delta::Text { text } => {
                    if let Some(slot) = self.slot(fold, id, PartKind::Text)
                        && !text.is_empty()
                    {
                        slot.certain = true;
                        slot.pending.push(text.clone());
                    }
                }
                Delta::TextMeta { .. } => {
                    if let Some(slot) = self.slot(fold, id, PartKind::Text) {
                        slot.certain = true;
                    }
                }
                Delta::Reasoning { text } => {
                    if let Some(slot) = self.slot(fold, id, PartKind::Reasoning) {
                        slot.certain = true;
                        if !text.is_empty() {
                            slot.pending.push(text.clone());
                        }
                    }
                }
                // A call's arguments are its finished JSON, sent at its end.
                Delta::ToolName { .. } | Delta::ToolArguments { .. } => {}
            },
            StreamEvent::BlockEnd {
                id,
                block: Some(block),
                ..
            } => {
                if let Some(slot) = self.slot(fold, id, kind_of(block)) {
                    slot.certain = true;
                    slot.end = Some(block.clone());
                }
            }
            StreamEvent::BlockEnd {
                id, block: None, ..
            } => {
                if let Some(index) = fold.slot(id)
                    && fold.block(index).is_none()
                    && let Some(slot) = self.slots.get_mut(index)
                    && !slot.certain
                {
                    slot.empty = true;
                }
            }
            StreamEvent::Final(terminal) => {
                self.issuer
                    .get_or_insert_with(|| terminal.issuer().to_owned());
            }
            StreamEvent::BlockStart { .. } | StreamEvent::Unknown(_) => {}
        }
        self.flush(out);
    }

    /// Catch up with events read from the stream itself: every part the
    /// fold placed that is not finished here is sent whole at its end.
    pub(crate) fn catch_up(&mut self, fold: &CompletionFold, out: &mut VecDeque<Update>) {
        for (index, slot) in self.slots.iter_mut().enumerate() {
            match (
                slot.end.as_ref().or(slot.ended_as.as_ref()),
                fold.block(index),
            ) {
                // Ended here, and extended since (a late signature).
                (Some(ended), Some(block)) if ended != block => slot.end = Some(block.clone()),
                (Some(_), _) => {}
                (None, block) => {
                    slot.whole = true;
                    if let Some(block) = block {
                        slot.settle(block);
                    }
                }
            }
        }
        while self.slots.len() < fold.placed() {
            let mut slot = Slot {
                whole: true,
                ..Slot::default()
            };
            if let Some(block) = fold.block(self.slots.len()) {
                slot.settle(block);
            }
            self.slots.push(slot);
        }
        self.flush(out);
    }

    /// The stream ended or failed: emit what is held, stamping reasoning
    /// with the issuer the fold names when no terminal record arrived. A slot still
    /// in doubt is settled by the fold: it is in `choice` exactly when it
    /// ended with content.
    pub(crate) fn finish(&mut self, fold: &CompletionFold, out: &mut VecDeque<Update>) {
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.certain {
                continue;
            }
            match fold.block(index) {
                Some(block) => slot.settle(block),
                None => slot.empty = true,
            }
        }
        if self.issuer.is_none() {
            self.issuer = Some(
                fold.reasoning_issuer()
                    .unwrap_or(fold.provider())
                    .to_owned(),
            );
        }
        self.flush(out);
    }

    /// The projection's slot for block `id`, which the fold has placed, with
    /// every slot the fold placed before it tracked too.
    fn slot(
        &mut self,
        fold: &CompletionFold,
        id: &super::BlockId,
        kind: PartKind,
    ) -> Option<&mut Slot> {
        let index = fold.slot(id)?;
        while self.slots.len() <= index {
            // A slot placed while nobody projected (its events were read
            // from the stream itself) is caught up from the fold when it
            // has ended.
            let mut slot = Slot::default();
            if let Some(block) = fold.block(self.slots.len()) {
                slot.settle(block);
            }
            self.slots.push(slot);
        }
        let slot = self.slots.get_mut(index)?;
        slot.kind.get_or_insert(kind);
        Some(slot)
    }

    /// Emit what every slot whose position is known has waiting, in slot
    /// order, stopping at the first slot not yet known to be in `choice`.
    fn flush(&mut self, out: &mut VecDeque<Update>) {
        let mut index = 0;
        let issuer = self.issuer.clone();
        for slot in &mut self.slots {
            if slot.empty && !slot.certain {
                continue;
            }
            if !slot.certain {
                break;
            }
            let Some(kind) = slot.kind.clone() else {
                break;
            };
            if !slot.started {
                let part = match (&kind, &slot.end) {
                    (PartKind::ToolCall { .. }, Some(AssistantContent::ToolCall(call))) => {
                        PartKind::ToolCall {
                            name: call.function.name.clone(),
                        }
                    }
                    _ => kind,
                };
                out.push_back(Update::Start { index, part });
                slot.started = true;
            }
            if slot.whole {
                slot.pending.clear();
            }
            for text in slot.pending.drain(..) {
                slot.emitted.push_str(&text);
                out.push_back(Update::Delta { index, text });
            }
            let issuer = issuer.as_deref();
            let held = matches!(slot.end, Some(AssistantContent::Reasoning(_))) && issuer.is_none();
            if let Some(part) = slot.end.take_if(|_| !held) {
                slot.ended_as = Some(part.clone());
                let part = match (part, issuer) {
                    (AssistantContent::Reasoning(reasoning), Some(issuer))
                        if reasoning.provider.is_none() =>
                    {
                        AssistantContent::Reasoning(reasoning.with_provider(issuer))
                    }
                    (part, _) => part,
                };
                if let Some(finished) = finished_text(&part) {
                    match finished.strip_prefix(slot.emitted.as_str()) {
                        Some(rest) if !rest.is_empty() => {
                            out.push_back(Update::Delta {
                                index,
                                text: rest.to_owned(),
                            });
                            slot.emitted = finished;
                        }
                        Some(_) => {}
                        None => tracing::debug!(
                            index,
                            "a part's end restated text its fragments did not send"
                        ),
                    }
                }
                out.push_back(Update::End { index, part });
            }
            index += 1;
        }
    }
}

/// The kind of part `block` is.
fn kind_of(block: &AssistantContent) -> PartKind {
    match block {
        AssistantContent::Text(_) => PartKind::Text,
        AssistantContent::Reasoning(_) => PartKind::Reasoning,
        AssistantContent::ToolCall(call) => PartKind::ToolCall {
            name: call.function.name.clone(),
        },
        AssistantContent::Image(_) => PartKind::Image,
    }
}

/// The text a finished part's deltas concatenate to: its text, its reasoning
/// text, or its argument JSON. An image has none.
fn finished_text(part: &AssistantContent) -> Option<String> {
    match part {
        AssistantContent::Text(text) => Some(text.text.clone()),
        AssistantContent::Reasoning(reasoning) => {
            Some(crate::completion::request::reasoning_text(reasoning))
        }
        AssistantContent::ToolCall(call) => Some(call.function.arguments.to_string()),
        AssistantContent::Image(_) => None,
    }
}

#[cfg(test)]
mod tests;
