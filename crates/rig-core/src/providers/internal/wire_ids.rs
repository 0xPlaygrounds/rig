//! How a request spells its tool calls' ids on a wire that requires one. A
//! provider's id is sent as it is; an id rig issued is spelled as a
//! request-local alias, `tool-<n>` in order of first appearance, passed
//! through the target's id rules and taken by no other id of the request.
//! The spelling is deterministic, so the same history always encodes to the
//! same bytes. Encoders spell calls and results with [`WireIds::of`].

use std::collections::{HashMap, HashSet};

use crate::message::{AssistantContent, CallId, Message, UserContent};

/// The wire spelling of every tool call and result of one request's history.
#[derive(Debug, Default)]
pub struct WireIds {
    by_call: HashMap<CallId, String>,
}

impl WireIds {
    /// The spelling of every call and result in `history` for `model` on
    /// `target`: a provider id as it stands, and an id rig issued as its
    /// `tool-<n>` alias passed through the target's id rules, so every id a
    /// request sends is one the wire accepts. No alias takes an id a hosted
    /// item holds ([`ReplayTarget::hosted_pair`]).
    ///
    /// [`ReplayTarget::hosted_pair`]: crate::completion::ReplayTarget::hosted_pair
    pub fn for_target(
        history: &[Message],
        target: &dyn crate::completion::ReplayTarget,
        model: &str,
    ) -> Self {
        let mut ids: Vec<&CallId> = Vec::new();
        let mut used: HashSet<String> = HashSet::new();
        for message in history {
            match message {
                Message::Assistant(turn) => {
                    for block in &turn.content {
                        match block {
                            AssistantContent::ToolCall(call) => ids.push(&call.id),
                            AssistantContent::Opaque(opaque) => {
                                used.extend(target.hosted_pair(&opaque.item).map(|(_, id)| id));
                            }
                            _ => {}
                        }
                    }
                }
                Message::User { content } => {
                    ids.extend(content.iter().filter_map(|part| match part {
                        UserContent::ToolResult(result) => Some(&result.call),
                        _ => None,
                    }))
                }
                Message::System { .. } => {}
            }
        }
        used.extend(
            ids.iter()
                .filter_map(|id| id.provider().map(|id| id.as_str().to_owned())),
        );
        let mut by_call = HashMap::new();
        let mut next = 0usize;
        for id in ids {
            if by_call.contains_key(id) {
                continue;
            }
            let spelled = match id {
                CallId::Provider(provider) => provider.as_str().to_owned(),
                CallId::Local(_) => loop {
                    let alias = target.normalize_tool_call_id(&format!("tool-{next}"), model, None);
                    next += 1;
                    if used.insert(alias.clone()) {
                        break alias;
                    }
                },
            };
            by_call.insert(id.clone(), spelled);
        }
        Self { by_call }
    }

    /// The wire spelling of `call`, when it is in the history.
    pub fn of(&self, call: &CallId) -> Option<&str> {
        self.by_call.get(call).map(String::as_str)
    }

    /// The wire spelling of `call`: [`Self::of`], or the id as it stands for
    /// one outside the history the spellings were made for.
    pub fn spell(&self, call: &CallId) -> String {
        self.of(call)
            .map_or_else(|| call.wire().into_owned(), str::to_owned)
    }
}

/// `id` in the alphabet most wires take for call ids, `[A-Za-z0-9_-]`,
/// cut to `max` characters: every other character becomes `_`.
pub fn legal_call_id(id: &str, max: usize) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(max)
        .collect()
}

/// pi's `shortHash`: two 32-bit multiplicative hashes of the UTF-16 code
/// units, written in base 36.
pub(crate) fn short_hash(text: &str) -> String {
    fn base36(mut value: u32) -> String {
        let mut digits = Vec::new();
        loop {
            digits.push(char::from_digit(value % 36, 36).unwrap_or('0'));
            value /= 36;
            if value == 0 {
                break;
            }
        }
        digits.iter().rev().collect()
    }
    let (mut h1, mut h2) = (0xdead_beef_u32, 0x41c6_ce57_u32);
    for unit in text.encode_utf16().map(u32::from) {
        h1 = (h1 ^ unit).wrapping_mul(2_654_435_761);
        h2 = (h2 ^ unit).wrapping_mul(1_597_334_677);
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h2 ^ (h2 >> 13)).wrapping_mul(3_266_489_909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h1 ^ (h1 >> 13)).wrapping_mul(3_266_489_909);
    format!("{}{}", base36(h2), base36(h1))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
pub(crate) mod tests;
