//! Early invalid-name publication from the native bus's delivered stream.

use super::*;
use rig_core::{
    message::{ToolCall, ToolCallId, ToolFunction},
    streaming::{BlockAccumulator, Delta, StreamEvent},
};

/// A consumer stops at its first error. Its item position equals the number
/// of preceding successful events, since there are no earlier error items.
pub(super) fn validation_len(stream: &BusStreamed) -> usize {
    stream
        .errors
        .first()
        .map_or(stream.events.len(), |(position, _)| {
            (*position).min(stream.events.len())
        })
}

/// Resolve a delivered name's block to the final identity using core assembly.
/// The offset disambiguates a block identifier reused later in the stream.
pub(super) fn completed_call_id(events: &[StreamEvent], offset: usize) -> Option<ToolCallId> {
    let block_id = match events.get(offset)? {
        StreamEvent::BlockDelta { id, .. } | StreamEvent::BlockEnd { id, .. } => id,
        _ => return None,
    };
    let mut accumulator = BlockAccumulator::new();
    for (index, event) in events.iter().enumerate() {
        let completed = accumulator.apply(event).ok()?;
        if index >= offset
            && let Some((block, AssistantContent::ToolCall(call))) = completed
            && &block == block_id
        {
            return Some(call.id);
        }
    }
    None
}

/// Publish invalid tool names from real delivered prefixes, before EOF.
///
/// A user system in `RigSet::Judge` may resolve the resulting `InvalidCall`.
/// Failure takes effect immediately; other resolutions remain attached until
/// the producer drains, retaining its actual outcome and usage for continuation.
#[allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    reason = "separate native graph accesses"
)]
pub(super) fn discover_streamed_invalid_calls(
    mut commands: Commands,
    effects: Query<(&ChildOf, &BusStreamed), NotRetrieval>,
    mut turns: Query<(&ChildOf, &mut Outputs), (With<Turn>, Without<Materialised>)>,
    runs: Query<&OutputToolName, (With<Run>, With<AwaitingModel>, Without<Failed>)>,
    children: Query<&Children>,
    adverts: Query<(&Advert, &Order)>,
    bound: Query<&Bound>,
    access: Query<&ToolAccess>,
    invalid: Query<(&ChildOf, &InvalidCall, Option<&Resolution>)>,
    mut progress: ResMut<Progress>,
) {
    for (parent, stream) in &effects {
        let turn = parent.parent();
        let Ok((run_of, mut outputs)) = turns.get_mut(turn) else {
            continue;
        };
        let Ok(minted) = runs.get(run_of.parent()) else {
            continue;
        };
        // An unresolved decision pauses validation. Skip/retry abandons the
        // prefix and drains later events without validating further calls.
        let pending: Vec<_> = invalid
            .iter()
            .filter(|(parent, _, _)| parent.parent() == turn)
            .collect();
        if pending.iter().any(|(_, _, resolution)| {
            !matches!(
                resolution,
                Some(Resolution::Repair { .. } | Resolution::Ignore)
            )
        }) {
            continue;
        }
        let mut allowed: Vec<String> = links_in_order(turn, &children, &adverts)
            .into_iter()
            .filter_map(|Advert(entity)| bound.get(*entity).ok())
            .filter_map(|bound| match &bound.descriptor.family {
                FamilyDescriptor::Tool { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        if let Some(names) = access
            .get(turn)
            .ok()
            .and_then(|access| access.allowed.as_ref())
        {
            allowed = names.iter().cloned().collect();
        }
        allowed.extend(minted.0.iter().cloned());
        // Valid streams scan each event once. Reconstruct a prefix only when
        // there is an actual invalid name to expose to policy.
        for (index, event) in stream
            .events
            .iter()
            .enumerate()
            .take(validation_len(stream))
            .skip(outputs.stream_validated)
        {
            outputs.stream_validated = index + 1;
            progress.mark();
            let (id, name, full) = match event {
                StreamEvent::BlockDelta {
                    id,
                    delta: Delta::ToolName { name },
                } => (id, name, false),
                StreamEvent::BlockEnd {
                    id,
                    block: Some(AssistantContent::ToolCall(call)),
                    ..
                } => (id, &call.function.name, true),
                StreamEvent::BlockEnd {
                    id,
                    end: rig_core::streaming::BlockClose::ToolCall(end),
                    ..
                } => {
                    let Some(name) = &end.name else {
                        continue;
                    };
                    (id, name, true)
                }
                _ => continue,
            };
            if allowed.contains(name) {
                continue;
            }
            let call_id = ToolCallId::from_block(id);
            // Ignore applies to the whole open occurrence, not just the
            // first name delta. An ended block may reuse the same identifier.
            if pending.iter().any(|(_, call, resolution)| {
                (matches!(resolution, Some(Resolution::Ignore))
                    || (full && matches!(resolution, Some(Resolution::Repair { .. }))))
                    && call.stream_offset.is_some_and(|offset| {
                        stream.events.get(offset).is_some_and(|event| matches!(event,
                            StreamEvent::BlockDelta { id: opened, .. } | StreamEvent::BlockEnd { id: opened, .. } if opened == id))
                        && !stream.events.iter().take(index).skip(offset).any(|event| {
                            matches!(event, StreamEvent::BlockEnd { id: closed, .. } if closed == id)
                        })
                    })
            }) {
                continue;
            }
            let mut accumulator = BlockAccumulator::new();
            let mut valid_prefix = true;
            for earlier in stream.events.iter().take(index) {
                if accumulator.apply(earlier).is_err() {
                    valid_prefix = false;
                    break;
                }
            }
            if !valid_prefix {
                break;
            }
            let mut prefix = accumulator.snapshot();
            let completed = match accumulator.apply(event) {
                Ok(Some((_, AssistantContent::ToolCall(call)))) => Some(call),
                Ok(_) => None,
                Err(_) => break,
            };
            // Earlier repairs/ignores already took effect in the driver's
            // view, even while native execution waits for the final outcome.
            for (_, call, resolution) in &pending {
                let Some(offset) = call.stream_offset else {
                    continue;
                };
                let completed_id = stream
                    .events
                    .get(..index)
                    .and_then(|events| completed_call_id(events, offset));
                if let Some(completed_id) = completed_id {
                    match resolution {
                        Some(Resolution::Repair { to }) => {
                            for part in &mut prefix {
                                if let AssistantContent::ToolCall(call) = part && call.id == completed_id { call.function.name = to.clone(); }
                            }
                        }
                        Some(Resolution::Ignore) => prefix.retain(|part| !matches!(part, AssistantContent::ToolCall(call) if call.id == completed_id)),
                        _ => {}
                    }
                }
            }
            let mut fragments: Vec<&str> = stream.events.iter().take(index).rev()
                .take_while(|event| !matches!(event, StreamEvent::BlockStart { id: start, .. } if start == id))
                .filter_map(|event| match event {
                    StreamEvent::BlockDelta { id: other, delta: Delta::ToolArguments { arguments } } if other == id => Some(arguments.as_str()),
                    _ => None,
                }).collect();
            fragments.reverse();
            let args = serde_json::from_str(&fragments.concat()).unwrap_or(serde_json::Value::Null);
            let diagnostic = completed.unwrap_or_else(|| {
                ToolCall::new(
                    call_id.clone(),
                    ToolFunction::new(name.clone(), args.clone()),
                )
            });
            let call_id = diagnostic.id.clone();
            let args = diagnostic.function.arguments.clone();
            prefix.push(AssistantContent::ToolCall(diagnostic));
            commands.spawn((
                InvalidCall {
                    id: call_id,
                    name: name.clone(),
                    arguments: args,
                    prefix,
                    stream_offset: Some(index),
                },
                ChildOf(turn),
            ));
            break;
        }
    }
}
