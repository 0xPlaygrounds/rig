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

use crate::message::{AssistantContent, CallId, LocalCallId, Message, UserContent};

/// The wire spelling of every tool call and result of one request's
/// history, by message and content position.
#[derive(Debug)]
pub struct WireIds {
    ids: BTreeMap<(usize, usize), String>,
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

    /// [`Self::new`], with provider handles carried outside tool calls (such
    /// as Anthropic server tools) that no alias may take.
    pub fn with_reserved(history: &[Message], reserved: impl IntoIterator<Item = String>) -> Self {
        let occurrences: Vec<((usize, usize), &CallId)> = history
            .iter()
            .enumerate()
            .flat_map(|(message, entry)| {
                let ids: Vec<(usize, &CallId)> = match entry {
                    Message::Assistant { content, .. } => content
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
            .filter_map(|(_, id)| id.provider().map(|provider| provider.call_id.clone()))
            .chain(reserved)
            .collect();
        let mut aliases: HashMap<&LocalCallId, String> = HashMap::new();
        let mut next = 0usize;
        let mut ids = BTreeMap::new();
        for (position, id) in occurrences {
            let spelled = match id {
                CallId::Provider(provider) => provider.call_id.clone(),
                CallId::Local(local) => aliases
                    .entry(local)
                    .or_insert_with(|| {
                        loop {
                            let candidate = format!("tool-{next}");
                            next += 1;
                            if used.insert(candidate.clone()) {
                                break candidate;
                            }
                        }
                    })
                    .clone(),
            };
            ids.insert(position, spelled);
        }
        Self { ids }
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
