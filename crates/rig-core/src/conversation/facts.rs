use serde::{Deserialize, Serialize};

use super::CallRef;
use super::pairing::answered_by;
use crate::message::CallId;
use crate::tool::ToolResultContext;

/// What the tool calls an entry of results answers published, by call, as
/// a conversation keeps it ([`ToolResultContext::kept`]): the host facts a
/// view draws a call's row from, such as rig-core's `ExitStatus` and
/// `FilesChanged`, saved with the entry so a resumed session draws what
/// the live one did. The model never sees them. Each call's facts name
/// the call they are for ([`ToolResultContext::with_at`]).
///
/// ```
/// use rig_core::conversation::{CallRef, Conversation, ResultFacts};
/// use rig_core::message::{CallId, Message};
/// use rig_core::tool::{ExitStatus, ToolContext};
///
/// let mut context = ToolContext::new();
/// context.insert_result(ExitStatus::Exited(3))?;
/// let call = CallId::from_wire("call_1");
/// let reply = Conversation::new().push(Message::assistant("Running it."));
/// let at = CallRef { reply, index: 0 };
/// let facts: ResultFacts = [(call.clone(), context.result_context().with_at(at))]
///     .into_iter()
///     .collect();
/// let found = facts.of([(at, &call)]);
/// let status = found.first().copied().flatten().map(|facts| facts.get::<ExitStatus>());
/// assert_eq!(status.transpose()?.flatten(), Some(ExitStatus::Exited(3)));
/// # Ok::<(), rig_core::tool::ToolContextError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResultFacts(Vec<(CallId, ToolResultContext)>);

impl ResultFacts {
    /// What each of `calls`, a reply's tool calls in order with their
    /// ids, published, if anything: the facts kept for it, else, among
    /// facts that name no call, those [`answered_by`] pairs with it.
    pub fn of<'c>(
        &self,
        calls: impl IntoIterator<Item = (CallRef, &'c CallId)>,
    ) -> Vec<Option<&ToolResultContext>> {
        let calls: Vec<(CallRef, &CallId)> = calls.into_iter().collect();
        let unnamed = self.0.iter().filter(|(_, facts)| facts.at().is_none());
        let unnamed = unnamed.map(|(id, facts)| (id, facts));
        let paired = answered_by(calls.iter().map(|(_, id)| *id), unnamed);
        calls
            .iter()
            .zip(paired)
            .map(|((at, _), paired)| {
                let mut named = self.0.iter().map(|(_, facts)| facts);
                named.find(|facts| facts.at() == Some(*at)).or(paired)
            })
            .collect()
    }

    /// Whether no call published anything.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// These and then `other`'s, as entries merged into one keep them.
    #[must_use]
    pub fn and(mut self, other: &Self) -> Self {
        self.0.extend(other.0.iter().cloned());
        self
    }
}

/// The facts of each call, leaving out a call that published nothing.
impl FromIterator<(CallId, ToolResultContext)> for ResultFacts {
    fn from_iter<I: IntoIterator<Item = (CallId, ToolResultContext)>>(calls: I) -> Self {
        let published = calls.into_iter().filter(|(_, facts)| !facts.is_empty());
        Self(published.collect())
    }
}
