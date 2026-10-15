//! Recent messages: which newest messages of a conversation a request
//! sends. A [`Recent`] rule says where the messages it keeps start; [`view`] moves that
//! start past results whose calls it cut off, so every result it keeps
//! still follows its call.
//!
//! ```
//! use std::sync::Arc;
//!
//! use rig_core::completion::Message;
//! use rig_memory::recent::{LastMessages, view};
//!
//! let messages = [Message::user("1"), Message::assistant("2"), Message::user("3")].map(Arc::new);
//! assert_eq!(view(&messages, &LastMessages(2)), &messages[1..]);
//! ```

use std::sync::Arc;

use rig_core::completion::Message;
use rig_core::conversation::pairing::safe_start;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::{HeuristicTokenCounter, TokenCounter};

/// Where the newest messages a request sends start.
pub trait Recent: WasmCompatSend + WasmCompatSync {
    /// The index of the first message of `messages` to keep, at most
    /// `messages.len()`.
    fn start(&self, messages: &[Arc<Message>]) -> usize;
}

/// The newest so many messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LastMessages(pub usize);

impl Recent for LastMessages {
    fn start(&self, messages: &[Arc<Message>]) -> usize {
        messages.len().saturating_sub(self.0)
    }
}

/// The newest messages within `max_tokens`, as `counter` counts them:
/// walking back from the newest, it stops before the first message that
/// does not fit.
#[derive(Clone, Copy, Debug)]
pub struct TokenBudget<C: TokenCounter = HeuristicTokenCounter> {
    /// The budget.
    pub max_tokens: usize,
    /// How a message is counted.
    pub counter: C,
}

impl TokenBudget {
    /// A budget of `max_tokens`, counted by the default
    /// [`HeuristicTokenCounter`].
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            counter: HeuristicTokenCounter::default(),
        }
    }
}

impl<C: TokenCounter> Recent for TokenBudget<C> {
    fn start(&self, messages: &[Arc<Message>]) -> usize {
        let mut budget = self.max_tokens;
        let mut start = messages.len();
        for (index, message) in messages.iter().enumerate().rev() {
            let cost = self.counter.count(message);
            if cost > budget {
                break;
            }
            budget -= cost;
            start = index;
        }
        start
    }
}

/// The messages of `messages` that `recent` keeps, from its start moved
/// forward by [`safe_start`] so no result loses its call.
pub fn view<'m>(messages: &'m [Arc<Message>], recent: &dyn Recent) -> &'m [Arc<Message>] {
    let start = safe_start(messages, recent.start(messages));
    messages.get(start..).unwrap_or_default()
}

#[cfg(test)]
mod tests;
