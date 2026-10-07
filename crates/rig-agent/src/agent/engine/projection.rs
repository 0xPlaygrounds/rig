//! The engine's tool stream items, built only here from committed history.

use rig_core::completion::Message;
use rig_core::message::{CallId, ToolCall};

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

/// Each tool call and result in `committed`; the result answering each call
/// in `executed` (the settled batch's calls whose bodies ran) by id follows a
/// [`ToolExecutionCommitted`](MultiTurnStreamItem::ToolExecutionCommitted).
/// A ran call that no committed result answers is refused with its id.
pub(crate) fn committed_stream_items(
    committed: &[Message],
    executed: &[ToolCall],
) -> Result<ProjectedItems, CallId> {
    let mut ran: Vec<&ToolCall> = executed.iter().collect();
    let mut items = Vec::new();
    for item in project(committed) {
        match item {
            CommittedItem::ToolCall(tool_call) => items.push(MultiTurnStreamItem::ToolCall {
                tool_call: tool_call.clone(),
            }),
            CommittedItem::ToolResult(tool_result) => {
                if let Some(index) = ran.iter().position(|call| call.id == tool_result.call) {
                    items.push(MultiTurnStreamItem::ToolExecutionCommitted {
                        tool_call: ran.remove(index).clone(),
                    });
                }
                items.push(MultiTurnStreamItem::ToolResult {
                    tool_result: tool_result.clone(),
                });
            }
        }
    }
    ran.first()
        .map_or(Ok(ProjectedItems(items)), |call| Err(call.id.clone()))
}

#[cfg(test)]
mod tests;
