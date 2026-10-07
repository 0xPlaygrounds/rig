//! The engine's tool stream items, built only here from committed history.

use rig_core::completion::Message;
use rig_core::message::ToolCall;

use crate::agent::MultiTurnStreamItem;
use crate::run::{CommittedItem, project};

/// Tool stream items whose field is private to this module, so the drive
/// loop cannot announce a tool call or result of its own.
#[derive(Debug)]
pub(crate) struct ProjectedItems(Vec<MultiTurnStreamItem>);

impl IntoIterator for ProjectedItems {
    type Item = MultiTurnStreamItem;
    type IntoIter = std::vec::IntoIter<MultiTurnStreamItem>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Each tool call and result in `committed`. A non-empty `batch` has one
/// slot per committed result of the settled tool batch, in commit order
/// (provider ids may repeat): the i-th result pairs with the i-th slot, and
/// follows a [`ToolExecutionCommitted`](MultiTurnStreamItem::ToolExecutionCommitted)
/// when that slot's body ran. A batch that does not line up is refused with
/// the first unmatched result's id.
pub(crate) fn committed_stream_items(
    committed: &[Message],
    batch: &[Option<ToolCall>],
) -> Result<ProjectedItems, String> {
    let mut slots = batch.iter();
    let mut items = Vec::new();
    for item in project(committed) {
        match item {
            CommittedItem::ToolCall(tool_call) => items.push(MultiTurnStreamItem::ToolCall {
                tool_call: tool_call.clone(),
            }),
            CommittedItem::ToolResult(tool_result) => {
                match slots.next() {
                    Some(Some(ran)) if ran.id == tool_result.call => {
                        items.push(MultiTurnStreamItem::ToolExecutionCommitted {
                            tool_call: ran.clone(),
                        });
                    }
                    Some(None) => {}
                    None if batch.is_empty() => {}
                    _ => return Err(tool_result.call.to_string()),
                }
                items.push(MultiTurnStreamItem::ToolResult {
                    tool_result: tool_result.clone(),
                });
            }
        }
    }
    let unmatched = slots.next().map(|_| "a slot with no result".to_owned());
    unmatched.map_or(Ok(ProjectedItems(items)), Err)
}

#[cfg(test)]
mod tests;
