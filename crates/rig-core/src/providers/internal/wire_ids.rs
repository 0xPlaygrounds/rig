//! How a request spells its tool calls' ids on a wire that requires one. A
//! provider's id is sent as it is; an id rig issued is spelled as a
//! request-local alias, `tool-<n>` in order of first appearance, that no
//! provider id of the request uses. The spelling is deterministic, so the
//! same history always encodes to the same bytes.
//!
//! ```
//! use rig_core::message::{CallId, Message, ToolName};
//! use rig_core::providers::internal::wire_ids::WireIds;
//!
//! let call = CallId::from_wire("");
//! let history = vec![Message::tool_result(call, ToolName::new("add")?, "5")];
//! let ids = WireIds::new(&history);
//! assert_eq!(ids.get(0, 0), Some("tool-0"));
//! # Ok::<(), rig_core::message::EmptyToolName>(())
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::error::EncodeError;
use crate::message::{AssistantContent, CallId, LocalCallId, Message, UserContent};

/// The wire spelling of every tool call and result of one request's
/// history, by message and content position.
#[derive(Debug, Default)]
pub struct WireIds {
    ids: BTreeMap<(usize, usize), String>,
    by_call: HashMap<CallId, String>,
}

/// Converted messages carried a different number of tool calls and results
/// than their source.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("tool id slot count at message {message}: expected {expected}, got {actual}")]
pub struct WireSlotCount {
    message: usize,
    expected: usize,
    actual: usize,
}

impl WireIds {
    /// The spelling of every call and result in `history`.
    pub fn new(history: &[Message]) -> Self {
        Self::with_reserved(history, std::iter::empty())
    }

    /// The spelling of every call and result in `history` for `model` on
    /// `target`: a provider id as it stands, and an id rig issued as its
    /// `tool-<n>` alias passed through the target's id rules, so every id a
    /// request sends is one the wire accepts.
    pub fn for_target(
        history: &[Message],
        target: &dyn crate::completion::ReplayTarget,
        model: &str,
    ) -> Self {
        Self::spelled(history, std::iter::empty(), |alias| {
            target.normalize_tool_call_id(alias, model, None)
        })
    }

    /// The wire spelling of `call`, when it is in the history.
    pub fn of(&self, call: &CallId) -> Option<&str> {
        self.by_call.get(call).map(String::as_str)
    }

    /// [`Self::new`], with provider handles carried outside tool calls (such
    /// as Anthropic server tools) that no alias may take.
    pub fn with_reserved(history: &[Message], reserved: impl IntoIterator<Item = String>) -> Self {
        Self::spelled(history, reserved, str::to_owned)
    }

    fn spelled(
        history: &[Message],
        reserved: impl IntoIterator<Item = String>,
        normalize: impl Fn(&str) -> String,
    ) -> Self {
        let occurrences: Vec<((usize, usize), &CallId)> = history
            .iter()
            .enumerate()
            .flat_map(|(message, entry)| {
                let ids: Vec<(usize, &CallId)> = match entry {
                    Message::Assistant(turn) => turn
                        .content
                        .iter()
                        .enumerate()
                        .filter_map(|(index, part)| match part {
                            AssistantContent::ToolCall(call) => Some((index, &call.id)),
                            _ => None,
                        })
                        .collect(),
                    Message::User { content } => content
                        .iter()
                        .enumerate()
                        .filter_map(|(index, part)| match part {
                            UserContent::ToolResult(result) => Some((index, &result.call)),
                            _ => None,
                        })
                        .collect(),
                    Message::System { .. } => Vec::new(),
                };
                ids.into_iter()
                    .map(move |(content, id)| ((message, content), id))
            })
            .collect();
        let mut used: HashSet<String> = occurrences
            .iter()
            .filter_map(|(_, id)| id.provider().map(|provider| provider.as_str().to_owned()))
            .chain(reserved)
            .collect();
        let mut aliases: HashMap<&LocalCallId, String> = HashMap::new();
        let mut next = 0usize;
        let mut ids = BTreeMap::new();
        let mut by_call = HashMap::new();
        for (position, id) in occurrences {
            let spelled = match id {
                CallId::Provider(provider) => provider.as_str().to_owned(),
                CallId::Local(local) => aliases
                    .entry(local)
                    .or_insert_with(|| {
                        loop {
                            let candidate = normalize(&format!("tool-{next}"));
                            next += 1;
                            if used.insert(candidate.clone()) {
                                break candidate;
                            }
                        }
                    })
                    .clone(),
            };
            by_call.insert(id.clone(), spelled.clone());
            ids.insert(position, spelled);
        }
        Self { ids, by_call }
    }

    /// Convert each message of `history` with `convert` and spell the tool
    /// ids of what it becomes. `slots` yields one converted item's id fields
    /// in the source message's tool-content order.
    pub fn convert<T, E>(
        history: Vec<Message>,
        mut convert: impl FnMut(Message) -> Result<Vec<T>, E>,
        mut slots: impl FnMut(&mut T) -> Vec<&mut String>,
    ) -> Result<Vec<T>, EncodeError>
    where
        EncodeError: From<E>,
    {
        let ids = Self::new(&history);
        let mut converted = Vec::new();
        for (position, message) in history.into_iter().enumerate() {
            let mut items = convert(message)?;
            ids.apply(position, items.iter_mut().flat_map(&mut slots))
                .map_err(EncodeError::request)?;
            converted.extend(items);
        }
        Ok(converted)
    }

    /// The spelling for the call or result at this position of the history,
    /// `None` for other content.
    pub fn get(&self, message: usize, content: usize) -> Option<&str> {
        self.ids.get(&(message, content)).map(String::as_str)
    }

    /// Assign one converted message's id slots, in the source message's
    /// tool-content order. Conversion may split or drop text and reasoning
    /// but keeps every call and result, so the counts must agree.
    pub fn apply<'a>(
        &self,
        message: usize,
        slots: impl IntoIterator<Item = &'a mut String>,
    ) -> Result<(), WireSlotCount> {
        let planned: Vec<_> = self
            .ids
            .range((message, 0)..=(message, usize::MAX))
            .map(|(_, id)| id)
            .collect();
        let slots: Vec<_> = slots.into_iter().collect();
        if planned.len() != slots.len() {
            return Err(WireSlotCount {
                message,
                expected: planned.len(),
                actual: slots.len(),
            });
        }
        for (slot, id) in slots.into_iter().zip(planned) {
            slot.clone_from(id);
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
pub(crate) mod tests;
