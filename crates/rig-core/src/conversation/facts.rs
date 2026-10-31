use serde::{Deserialize, Serialize};

use crate::message::CallId;
use crate::tool::ToolResultContext;

/// What the tool calls an entry of results answers published, by call, as
/// a conversation keeps it ([`ToolResultContext::kept`]): the host facts a
/// view draws a call's row from, such as rig-core's `ExitStatus` and
/// `FilesChanged`, saved with the entry so a resumed session draws what
/// the live one did. The model never sees them.
///
/// ```
/// use rig_core::conversation::ResultFacts;
/// use rig_core::message::CallId;
/// use rig_core::tool::{ExitStatus, ToolContext};
///
/// let mut context = ToolContext::new();
/// context.insert_result(ExitStatus::Exited(3))?;
/// let call = CallId::from_wire("call_1");
/// let facts: ResultFacts = [(call.clone(), context.result_context())].into_iter().collect();
/// let status = facts.get(&call).map(|facts| facts.get::<ExitStatus>());
/// assert_eq!(status.transpose()?.flatten(), Some(ExitStatus::Exited(3)));
/// # Ok::<(), rig_core::tool::ToolContextError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResultFacts(Vec<(CallId, ToolResultContext)>);

impl ResultFacts {
    /// What the call `call` published, if it published anything.
    pub fn get(&self, call: &CallId) -> Option<&ToolResultContext> {
        self.0
            .iter()
            .find_map(|(id, facts)| (id == call).then_some(facts))
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
