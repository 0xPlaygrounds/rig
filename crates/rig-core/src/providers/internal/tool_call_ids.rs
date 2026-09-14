//! Request-local wire identities for protocols requiring tool call IDs.
//!
//! The sidecar is indexed by message and content position, not durable local
//! identity: a later completed turn can legitimately reuse a generated ID.
//! Planning never changes the transcript or claims synthetic provider provenance.

use std::collections::{BTreeMap, HashSet};

use crate::message::{AssistantContent, Message, ToolCall, UserContent};

/// A transcript cannot be correlated unambiguously on a required-ID wire.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ToolCallIdError {
    /// More than one outstanding call has the same local correlation identity.
    #[error("ambiguous tool call identity at message {message}, content {content}")]
    Ambiguous { message: usize, content: usize },
    /// Local and provider references identify different calls.
    #[error("conflicting tool call references at message {message}, content {content}")]
    Conflict { message: usize, content: usize },
    /// A result repeats a previously consumed reference.
    #[error("duplicate tool result at message {message}, content {content}")]
    DuplicateResult { message: usize, content: usize },
    /// Missing history provides neither a preceding call nor a provider handle.
    #[error("local-only orphan tool result at message {message}, content {content}")]
    OrphanResult { message: usize, content: usize },
    /// An internal correlation reference no longer points to a call occurrence.
    #[error("missing planned call at message {message}, content {content}")]
    MissingOccurrence { message: usize, content: usize },
    /// Wire conversion did not retain one ID slot per original call/result.
    #[error("tool ID slot count at message {message}: expected {expected}, got {actual}")]
    WireSlotCount {
        message: usize,
        expected: usize,
        actual: usize,
    },
}

/// Immutable wire-ID assignments for the exact history supplied to [`Self::new`].
///
/// Genuine handles are preserved, including sequential reuse after a result.
/// Synthetic handles are unique throughout the request and cannot collide with
/// any genuine handle, even one first appearing in a later message.
#[derive(Debug)]
pub struct ToolCallIds {
    ids: BTreeMap<(usize, usize), String>,
}

struct Occurrence<'a> {
    call: &'a ToolCall,
    position: (usize, usize),
    result: Option<(usize, usize)>,
    provider: Option<String>,
}

impl ToolCallIds {
    /// Pair chronological call/result occurrences, then allocate required wire IDs.
    /// Provider-known orphan results support server-retained partial history.
    /// Item IDs and tool names never serve as call-ID aliases.
    pub fn new(history: &[Message]) -> Result<Self, ToolCallIdError> {
        Self::with_reserved(history, std::iter::empty())
    }

    /// Also reserve genuine call references carried by provider-specific content
    /// outside core tool calls, such as Anthropic server tools. These handles are
    /// never paired or rewritten; they only exclude synthetic alias candidates.
    /// Do not supply unrelated item/reasoning IDs from other wire namespaces.
    pub fn with_reserved(
        history: &[Message],
        reserved: impl IntoIterator<Item = String>,
    ) -> Result<Self, ToolCallIdError> {
        let mut occurrences: Vec<Occurrence<'_>> = Vec::new();
        let mut orphans = Vec::new();
        let mut answered_local = HashSet::new();
        let mut ids = BTreeMap::new();
        for (message, entry) in history.iter().enumerate() {
            match entry {
                Message::Assistant { content, .. } => {
                    for (content, part) in content.iter().enumerate() {
                        let AssistantContent::ToolCall(call) = part else {
                            continue;
                        };
                        let provider = call.provider.as_ref().map(|id| id.call_id.clone());
                        if occurrences.iter().any(|previous| {
                            previous.result.is_none()
                                && (previous.call.id == call.id
                                    || (provider.is_some() && previous.provider == provider))
                        }) {
                            return Err(ToolCallIdError::Ambiguous { message, content });
                        }
                        occurrences.push(Occurrence {
                            call,
                            position: (message, content),
                            result: None,
                            provider,
                        });
                    }
                }
                Message::User { content } => {
                    for (content, part) in content.iter().enumerate() {
                        let UserContent::ToolResult(result) = part else {
                            continue;
                        };
                        let provider = result.provider.as_ref().map(|id| &id.call_id);
                        let local: Vec<_> = occurrences
                            .iter()
                            .enumerate()
                            .filter(|(_, call)| {
                                call.result.is_none() && call.call.id == result.call
                            })
                            .map(|(index, _)| index)
                            .collect();
                        let wire: Vec<_> = occurrences
                            .iter()
                            .enumerate()
                            .filter(|(_, call)| {
                                call.result.is_none()
                                    && provider.is_some()
                                    && call.provider.as_ref() == provider
                            })
                            .map(|(index, _)| index)
                            .collect();
                        if local.len() > 1 || wire.len() > 1 {
                            return Err(ToolCallIdError::Ambiguous { message, content });
                        }
                        if let (Some(local), Some(wire)) = (local.first(), wire.first())
                            && local != wire
                        {
                            return Err(ToolCallIdError::Conflict { message, content });
                        }
                        // A recycled provider handle cannot redirect a stale
                        // local result to a different, newly outstanding call.
                        if local.is_empty() && answered_local.contains(&result.call) {
                            return Err(ToolCallIdError::DuplicateResult { message, content });
                        }
                        if let Some(index) = local.first().or(wire.first()).copied() {
                            let call = occurrences
                                .get_mut(index)
                                .ok_or(ToolCallIdError::MissingOccurrence { message, content })?;
                            if let (Some(existing), Some(incoming)) = (&call.provider, provider)
                                && existing != incoming
                            {
                                return Err(ToolCallIdError::Conflict { message, content });
                            }
                            if call.provider.is_none() {
                                call.provider = provider.cloned();
                            }
                            call.result = Some((message, content));
                            answered_local.insert(call.call.id.clone());
                            answered_local.insert(result.call.clone());
                        } else {
                            if occurrences.iter().any(|call| {
                                call.result.is_some()
                                    && (call.call.id == result.call
                                        || (provider.is_some()
                                            && call.provider.as_ref() == provider))
                            }) || orphans.iter().any(|(local, wire)| {
                                local == &result.call || provider == Some(wire)
                            }) {
                                return Err(ToolCallIdError::DuplicateResult { message, content });
                            }
                            let Some(provider) = provider else {
                                return Err(ToolCallIdError::OrphanResult { message, content });
                            };
                            ids.insert((message, content), provider.clone());
                            answered_local.insert(result.call.clone());
                            orphans.push((result.call.clone(), provider.clone()));
                        }
                    }
                }
                Message::System { .. } => {}
            }
        }
        // Provenance may first appear on a result. Validate the original active
        // intervals after resolving both legs, not just when admitting calls.
        for (index, call) in occurrences.iter().enumerate() {
            if call.provider.is_some()
                && occurrences.iter().take(index).any(|previous| {
                    previous.provider == call.provider
                        && previous.result.is_none_or(|end| call.position < end)
                })
            {
                let (message, content) = call.position;
                return Err(ToolCallIdError::Ambiguous { message, content });
            }
        }
        // Resolve result-only provenance before reserving IDs: the assistant
        // echo must carry the same genuine handle even when only its result did.
        let mut used: HashSet<String> = occurrences
            .iter()
            .filter_map(|call| call.provider.clone())
            .chain(ids.values().cloned())
            .chain(reserved)
            .collect();
        let mut next = 0;
        for call in occurrences {
            let id = if let Some(provider) = call.provider {
                provider
            } else {
                let hint = call.call.id.wire_hint().into_owned();
                if !hint.is_empty() && used.insert(hint.clone()) {
                    hint
                } else {
                    loop {
                        let candidate = format!("tool-{next}");
                        next += 1;
                        if used.insert(candidate.clone()) {
                            break candidate;
                        }
                    }
                }
            };
            if let Some(result) = call.result {
                ids.insert(result, id.clone());
            }
            ids.insert(call.position, id);
        }
        Ok(Self { ids })
    }

    /// The wire handle for a call or result at this original content position.
    /// Returns `None` for other content or positions outside the supplied history.
    pub fn get(&self, message: usize, content: usize) -> Option<&str> {
        self.ids.get(&(message, content)).map(String::as_str)
    }

    /// Assign a converted message's required wire slots in original tool-content
    /// order. Text/reasoning may expand or disappear, but tool occurrences must
    /// remain one-to-one. Call before combining different source messages.
    /// Validates the complete slot count before changing any wire field.
    pub fn apply<'a>(
        &self,
        message: usize,
        slots: impl IntoIterator<Item = &'a mut String>,
    ) -> Result<(), ToolCallIdError> {
        let planned: Vec<_> = self
            .ids
            .range((message, 0)..=(message, usize::MAX))
            .map(|(_, id)| id)
            .collect();
        let slots: Vec<_> = slots.into_iter().collect();
        if planned.len() != slots.len() {
            return Err(ToolCallIdError::WireSlotCount {
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
pub(crate) mod tests;
