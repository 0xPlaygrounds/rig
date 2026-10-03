use super::*;
use rig_core::streaming::{Item, StreamEvent};

/// Return the successful item count before the first error, or the full length.
/// The first error's item position equals the number of preceding successful items.
pub(super) fn validation_len(stream: &BusStreamed) -> usize {
    stream
        .errors
        .first()
        .map_or(stream.events.len(), |(position, _)| {
            (*position).min(stream.events.len())
        })
}

/// Publish invalid tool calls from real delivered prefixes, before EOF.
///
/// A user system in `RigSet::Judge` may resolve the resulting `InvalidCall`.
/// Failure takes effect immediately; other resolutions remain attached until
/// the producer drains, retaining its actual outcome and usage for continuation.
pub fn discover_streamed_invalid_calls(
    mut commands: Commands,
    effects: Query<(&ChildOf, &BusStreamed), NotRetrieval>,
    mut turns: Query<(&ChildOf, &mut Outputs), Unread>,
    runs: Query<(&OutputToolName, &RunPhase), LiveRun>,
    children: Query<&Children>,
    adverts: Query<&Advert>,
    bound: Query<&Bound>,
    access: Query<&ToolAccess>,
    invalid: Query<(&ChildOf, &InvalidCall, Option<&Resolution>)>,
) {
    for (parent, stream) in &effects {
        let turn = parent.parent();
        let Ok((run_of, mut outputs)) = turns.get_mut(turn) else {
            continue;
        };
        let Ok((minted, &RunPhase::AwaitingModel)) = runs.get(run_of.parent()) else {
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
            .filter_map(|bound| crate::policy::tool_name(&bound.descriptor).map(str::to_owned))
            .collect();
        if let Some(names) = access
            .get(turn)
            .ok()
            .and_then(|access| access.allowed.as_ref())
        {
            allowed = names.iter().cloned().collect();
        }
        allowed.extend(minted.0.iter().cloned());
        // Track the validated offset to avoid rescanning valid stream prefixes.
        let items = stream.events.items();
        for (index, item) in items
            .iter()
            .enumerate()
            .take(validation_len(stream))
            .skip(outputs.stream_validated)
        {
            outputs.stream_validated = index + 1;
            let Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            }) = item
            else {
                continue;
            };
            let name = call.function.name.as_str();
            if allowed.iter().any(|allowed| allowed == name) {
                continue;
            }
            // A call already decided on stays decided.
            if pending.iter().any(|(_, pending, _)| pending.id == call.id) {
                continue;
            }
            // The prefix up to and with this call, folded as the core folds a
            // partial reply.
            let mut prefix =
                rig_core::streaming::delivered(items.get(..=index).unwrap_or_default());
            // Earlier repairs/ignores already took effect in the driver's
            // view, even while native execution waits for the final outcome.
            for (_, pending, resolution) in &pending {
                match resolution {
                    Some(Resolution::Repair { to }) => {
                        for part in &mut prefix {
                            if let AssistantContent::ToolCall(call) = part
                                && call.id == pending.id
                                && let Ok(to) = rig_core::message::ToolName::new(to.clone())
                            {
                                call.function.name = to;
                            }
                        }
                    }
                    Some(Resolution::Ignore) => prefix.retain(|part| {
                        !matches!(part, AssistantContent::ToolCall(call) if call.id == pending.id)
                    }),
                    _ => {}
                }
            }
            commands.spawn((
                InvalidCall {
                    id: call.id.clone(),
                    name: name.to_owned(),
                    arguments: call.function.arguments_value(),
                    prefix,
                    stream_offset: Some(index),
                    origin: stream.origin.clone(),
                },
                ChildOf(turn),
            ));
            break;
        }
    }
}
