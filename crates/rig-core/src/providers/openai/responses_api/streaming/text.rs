//! Responses-specific reconciliation before Rig's append-only text accumulator.
use super::super::{
    AssistantContent, OPENAI_RESPONSES_EXTRAS_KEY, OutputMessage, stamp_phase, text_block,
};
use crate::completion::CompletionError;
use crate::message::AdditionalParams;
use crate::operation::AdapterOutput;
use crate::streaming::{BlockId, Delta, StreamEvent, SyntheticIds};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};

/// Internal envelope-repair marker, converted to an unknown position by assembly.
pub(super) const MISSING_INDEX: u64 = u64::MAX;

fn inconsistent(detail: &str) -> CompletionError {
    CompletionError::ResponseError(format!("Inconsistent Responses text metadata: {detail}"))
}

#[derive(Debug, Default)]
pub(super) struct TextParts {
    parts: Vec<TextPart>,
    items: HashMap<String, HashMap<u64, usize>>,
    slots: HashMap<(u64, u64), usize>,
    ids: Option<SyntheticIds>,
    active: Option<usize>,
}

#[derive(Debug)]
struct TextPart {
    key: BlockId,
    item_id: Option<String>,
    output: Option<u64>,
    content: u64,
    started: bool,
    delivered: bool,
    extras: Map<String, Value>,
    annotations: BTreeMap<u64, Value>,
    emitted: usize,
    unindexed: Vec<Value>,
}

impl TextParts {
    pub(super) fn interrupt(&mut self) {
        if let Some(part) = self
            .active
            .take()
            .and_then(|index| self.parts.get_mut(index))
        {
            part.started = false;
        }
    }

    fn resolve(
        &mut self,
        output: u64,
        item: Option<&str>,
        content: u64,
    ) -> Result<&mut TextPart, CompletionError> {
        let output = (output != MISSING_INDEX).then_some(output);
        let item = item.filter(|id| !id.is_empty());
        let named = item.and_then(|id| self.items.get(id)?.get(&content).copied());
        let slot = output.and_then(|slot| self.slots.get(&(slot, content)).copied());
        // Explicit identities outrank positional aliases. A genuine slot can
        // correlate changing gateway ids; an invented replay slot cannot.
        let mut matched = named.or(slot);
        if matched.is_none() {
            let mut candidates = self
                .parts
                .iter()
                .enumerate()
                .filter(|(_, part)| {
                    part.content == content
                        && (part.output.is_none() || output.is_none())
                        && (part.item_id.is_none() || item.is_none())
                })
                .map(|(index, _)| index);
            matched = candidates.next();
            if candidates.next().is_some() {
                return Err(inconsistent("ambiguous text content-part identity"));
            }
        }
        let index = matched.unwrap_or_else(|| {
            let index = self.parts.len();
            let key = if content == 0 {
                item.map(BlockId::wire)
                    .unwrap_or_else(|| self.ids.get_or_insert_with(SyntheticIds::text).mint())
            } else {
                self.ids.get_or_insert_with(SyntheticIds::text).mint()
            };
            self.parts.push(TextPart {
                key,
                item_id: item.map(str::to_owned),
                output,
                content,
                started: false,
                delivered: false,
                extras: Map::new(),
                annotations: BTreeMap::new(),
                emitted: 0,
                unindexed: Vec::new(),
            });
            index
        });
        if self.active != Some(index) {
            self.interrupt();
            self.active = Some(index);
        }
        let part = self
            .parts
            .get_mut(index)
            .ok_or_else(|| inconsistent("missing text content part"))?;
        if let Some(item) = item
            && named.is_none()
        {
            self.items
                .entry(item.to_owned())
                .or_default()
                .entry(content)
                .or_insert(index);
            if part.item_id.is_none() {
                part.item_id = Some(item.to_owned());
            }
        }
        if let Some(output) = output {
            self.slots.insert((output, content), index);
            part.output = Some(output);
        }
        Ok(part)
    }

    pub(super) fn delta(
        &mut self,
        output: u64,
        item: Option<&str>,
        content: u64,
        delta: String,
        out: &mut AdapterOutput,
    ) -> Result<(), CompletionError> {
        let part = self.resolve(output, item, content)?;
        part.emit(Delta::Text { text: delta }, out);
        part.delivered = true;
        Ok(())
    }

    pub(super) fn annotation(
        &mut self,
        output: u64,
        item: Option<&str>,
        content: u64,
        annotation_index: u64,
        annotation: Value,
        out: &mut AdapterOutput,
    ) -> Result<(), CompletionError> {
        let part = self.resolve(output, item, content)?;
        if annotation_index == MISSING_INDEX {
            part.unindexed.push(annotation);
        } else {
            if part
                .annotations
                .get(&annotation_index)
                .is_some_and(|old| old != &annotation)
            {
                return Err(inconsistent("conflicting annotation at the same index"));
            }
            part.annotations.insert(annotation_index, annotation);
            part.emit_contiguous(out)?;
        }
        Ok(())
    }

    pub(super) fn message(
        &mut self,
        output: u64,
        message: &OutputMessage,
        deliver_text: bool,
        out: &mut AdapterOutput,
    ) -> Result<(), CompletionError> {
        for (content, wire_text) in message.content.iter().enumerate() {
            self.content(
                output,
                Some(&message.id),
                content as u64,
                wire_text.clone(),
                message.phase.as_deref(),
                deliver_text,
                out,
            )?;
        }
        Ok(())
    }

    pub(super) fn content(
        &mut self,
        output: u64,
        item: Option<&str>,
        content: u64,
        wire_text: AssistantContent,
        phase: Option<&str>,
        deliver_text: bool,
        out: &mut AdapterOutput,
    ) -> Result<(), CompletionError> {
        let part = self.resolve(output, item, content)?;
        // Inspect before text_block filters empty arrays: [] is a complete
        // snapshot too, and cannot erase an emitted citation.
        if let AssistantContent::OutputText(text) = &wire_text
            && let Some(Value::Array(annotations)) = text.extras.get("annotations")
        {
            for (&index, value) in &part.annotations {
                if usize::try_from(index)
                    .ok()
                    .and_then(|index| annotations.get(index))
                    != Some(value)
                {
                    return Err(inconsistent(
                        "completed annotations contradict streamed annotations",
                    ));
                }
            }
            part.annotations = annotations
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, value)| (index as u64, value))
                .collect();
            // A complete array supplies positions omitted by replay deltas.
            part.unindexed.clear();
        }
        let mut text = text_block(wire_text);
        stamp_phase(&mut text, phase);
        if deliver_text && !part.delivered {
            part.emit(Delta::Text { text: text.text }, out);
            part.delivered = true;
        }
        if let Some(params) = text.additional_params
            && let Some(extras) = params.wire_extras(OPENAI_RESPONSES_EXTRAS_KEY)
        {
            part.publish_extras(extras, out)?;
            if let Some(Value::Array(annotations)) = extras.get("annotations") {
                part.emitted = annotations.len();
            }
        }
        Ok(())
    }

    pub(super) fn flush(&mut self, out: &mut AdapterOutput) -> Result<(), CompletionError> {
        for part in &mut self.parts {
            let mut remaining: Vec<_> = part
                .annotations
                .range(part.emitted as u64..)
                .map(|(_, value)| value.clone())
                .collect();
            remaining.append(&mut part.unindexed);
            if !remaining.is_empty() {
                let mut annotations = part
                    .extras
                    .get("annotations")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                annotations.extend(remaining);
                part.publish_extras(
                    &Map::from_iter([("annotations".into(), Value::Array(annotations))]),
                    out,
                )?;
            }
            part.annotations.clear();
            part.emitted = 0;
        }
        Ok(())
    }
}

impl TextPart {
    fn emit(&mut self, delta: Delta, out: &mut AdapterOutput) {
        if !self.started {
            out.text_start(self.key.clone(), None);
            self.started = true;
        }
        // Explicit identity routes late metadata and resumed text to the same
        // block. Only switching blocks needs another start, not each fragment.
        out.push(Ok(StreamEvent::BlockDelta {
            id: self.key.clone(),
            delta,
        }));
    }

    fn emit_contiguous(&mut self, out: &mut AdapterOutput) -> Result<(), CompletionError> {
        let mut end = self.emitted;
        while self.annotations.contains_key(&(end as u64)) {
            end += 1;
        }
        if end == self.emitted {
            return Ok(());
        }
        let annotations = (0..end)
            .filter_map(|index| self.annotations.get(&(index as u64)).cloned())
            .collect();
        self.publish_extras(
            &Map::from_iter([("annotations".into(), Value::Array(annotations))]),
            out,
        )?;
        self.emitted = end;
        Ok(())
    }

    fn publish_extras(
        &mut self,
        extras: &Map<String, Value>,
        out: &mut AdapterOutput,
    ) -> Result<(), CompletionError> {
        let mut delta = Map::new();
        for (key, incoming) in extras {
            if let Some(value) = metadata_delta(self.extras.get(key), incoming)? {
                delta.insert(key.clone(), value);
            }
        }
        // Mirror the accumulator's recursive merge, including retained fields
        // omitted from a later snapshot.
        if let Some(params) = AdditionalParams::new(delta.clone()) {
            let mut state = AdditionalParams::new(std::mem::take(&mut self.extras));
            match state.as_mut() {
                Some(state) => state.merge(params),
                None => state = Some(params),
            }
            if let Some(state) = state {
                self.extras = state.as_map().clone();
            }
        }
        if let Some(params) = AdditionalParams::from_entries(
            (!delta.is_empty()).then_some((OPENAI_RESPONSES_EXTRAS_KEY, Value::Object(delta))),
        ) {
            self.emit(
                Delta::TextMeta {
                    additional_params: params,
                },
                out,
            );
        }
        Ok(())
    }
}

/// Compute only what the existing recursive, array-appending merge can add.
/// Missing snapshot fields do not delete previously emitted metadata.
fn metadata_delta(
    previous: Option<&Value>,
    incoming: &Value,
) -> Result<Option<Value>, CompletionError> {
    if previous == Some(incoming) {
        return Ok(None);
    }
    match (previous, incoming) {
        (Some(Value::Array(old)), Value::Array(new)) => {
            if !new.starts_with(old) {
                return Err(inconsistent("completed metadata changes an emitted array"));
            }
            Ok((new.len() > old.len())
                .then(|| Value::Array(new.iter().skip(old.len()).cloned().collect())))
        }
        (Some(Value::Object(old)), Value::Object(new)) => {
            let mut delta = Map::new();
            for (key, value) in new {
                if let Some(value) = metadata_delta(old.get(key), value)? {
                    delta.insert(key.clone(), value);
                }
            }
            Ok((!delta.is_empty()).then_some(Value::Object(delta)))
        }
        (Some(Value::Array(_)), _) | (Some(Value::Object(_)), _) => Err(inconsistent(
            "completed metadata changes an emitted value's type",
        )),
        _ => Ok(Some(incoming.clone())),
    }
}
