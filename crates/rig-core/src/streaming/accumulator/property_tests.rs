use super::*;
use proptest::prelude::*;

fn split_fragments(payload: &str, points: &[usize]) -> Vec<String> {
    let chars: Vec<char> = payload.chars().collect();
    let mut cuts: Vec<usize> = points
        .iter()
        .map(|point| point % (chars.len() + 1))
        .collect();
    cuts.sort_unstable();
    let mut fragments = Vec::new();
    let mut start = 0usize;
    for cut in cuts {
        fragments.push(chars[start..cut.max(start)].iter().collect());
        start = cut.max(start);
    }
    fragments.push(chars[start..].iter().collect());
    fragments
}

fn text_delta(accumulator: &mut BlockAccumulator, key: &BlockId, text: &str) {
    accumulator
        .apply(&StreamEvent::text(key.clone(), text))
        .expect("text deltas never fail");
}

fn reasoning_delta(accumulator: &mut BlockAccumulator, key: &BlockId, text: &str) {
    accumulator
        .apply(&StreamEvent::BlockDelta {
            id: key.clone(),
            delta: Delta::Reasoning {
                text: text.to_owned(),
            },
        })
        .expect("reasoning deltas never fail");
}

/// A wire-sent bare reasoning end; returns the completed part.
fn reasoning_end(accumulator: &mut BlockAccumulator, key: &BlockId) -> Option<AssistantContent> {
    accumulator
        .apply(&StreamEvent::BlockEnd {
            id: key.clone(),
            end: BlockClose::Reasoning {
                reasoning: None,
                signature: None,
                wire_sent: true,
            },
            block: None,
        })
        .expect("reasoning ends never fail")
        .map(|(_, block)| block)
}

fn tool_name_delta(accumulator: &mut BlockAccumulator, key: &BlockId, name: &str) {
    accumulator
        .apply(&StreamEvent::BlockDelta {
            id: key.clone(),
            delta: Delta::ToolName {
                name: name.to_owned(),
            },
        })
        .expect("tool name deltas never fail");
}

fn tool_args_delta(accumulator: &mut BlockAccumulator, key: &BlockId, arguments: &str) {
    accumulator
        .apply(&StreamEvent::BlockDelta {
            id: key.clone(),
            delta: Delta::ToolArguments {
                arguments: arguments.to_owned(),
            },
        })
        .expect("tool argument deltas never fail");
}

fn tool_end(
    accumulator: &mut BlockAccumulator,
    key: &BlockId,
    end: ToolCallEnd,
) -> Result<Option<ToolCall>, CompletionError> {
    Ok(accumulator
        .apply(&StreamEvent::BlockEnd {
            id: key.clone(),
            end: BlockClose::ToolCall(end),
            block: None,
        })?
        .and_then(|(_, block)| match block {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        }))
}

proptest! {
    /// Text aggregation is invariant under fragmentation.
    #[test]
    fn text_aggregate_is_fragmentation_invariant(
        payload in ".{0,60}",
        points in proptest::collection::vec(0usize..1000, 0..6),
    ) {
        let key = BlockId::wire("msg_1");
        let mut whole = BlockAccumulator::new();
        text_delta(&mut whole, &key, &payload);
        let mut split = BlockAccumulator::new();
        for fragment in split_fragments(&payload, &points) {
            text_delta(&mut split, &key, &fragment);
        }
        prop_assert_eq!(whole.finish(), split.finish());
    }

    /// Reasoning delta accumulation + end is invariant under
    /// fragmentation: accumulated delta content equals the completed
    /// part's payload (the langchain stream-lifecycle conservation law).
    #[test]
    fn reasoning_aggregate_is_fragmentation_invariant(
        payload in ".{1,60}",
        points in proptest::collection::vec(0usize..1000, 0..6),
    ) {
        let key = BlockId::wire("rs_1");
        let mut whole = BlockAccumulator::new();
        reasoning_delta(&mut whole, &key, &payload);
        let completed_whole = reasoning_end(&mut whole, &key);
        let mut split = BlockAccumulator::new();
        let mut pushed = false;
        for fragment in split_fragments(&payload, &points) {
            if !fragment.is_empty() {
                reasoning_delta(&mut split, &key, &fragment);
                pushed = true;
            }
        }
        prop_assume!(pushed);
        let completed_split = reasoning_end(&mut split, &key);
        prop_assert_eq!(completed_whole, completed_split);
        prop_assert_eq!(whole.finish(), split.finish());
    }

    /// Tool-argument assembly is invariant under fragmentation.
    #[test]
    fn tool_arguments_are_fragmentation_invariant(
        value in "[a-z]{0,20}",
        points in proptest::collection::vec(0usize..1000, 0..6),
    ) {
        let payload = format!("{{\"q\":\"{value}\"}}");
        let finalize = |fragments: &[String]| {
            let mut accumulator = BlockAccumulator::new();
            let key = BlockId::wire("call_1");
            tool_name_delta(&mut accumulator, &key, "probe");
            for fragment in fragments {
                tool_args_delta(&mut accumulator, &key, fragment);
            }
            tool_end(&mut accumulator, &key, ToolCallEnd::new(UnparseableToolInput::Drop))
                .expect("no error")
                .map(|call| call.function.arguments)
        };
        let whole = finalize(std::slice::from_ref(&payload));
        let split = finalize(&split_fragments(&payload, &points));
        prop_assert_eq!(whole, split);
    }

    /// Stale-end idempotence, on EVERY route: repeated ends — bare or
    /// carrying an authoritative payload — add nothing after the entity
    /// finished.
    #[test]
    fn stale_tool_input_ends_are_idempotent(
        extra_ends in 1usize..5,
        authoritative in proptest::bool::ANY,
    ) {
        let mut accumulator = BlockAccumulator::new();
        let key = BlockId::wire("call_1");
        tool_name_delta(&mut accumulator, &key, "probe");
        tool_args_delta(&mut accumulator, &key, "{}");
        tool_end(&mut accumulator, &key, ToolCallEnd::new(UnparseableToolInput::Drop))
            .expect("no error")
            .expect("finalizes");
        for _ in 0..extra_ends {
            let mut stale = ToolCallEnd::new(UnparseableToolInput::Drop);
            if authoritative {
                stale.name = Some("probe".to_owned());
                stale.arguments = Some(serde_json::json!({}));
            }
            prop_assert!(
                tool_end(&mut accumulator, &key, stale)
                    .expect("no error")
                    .is_none()
            );
        }
        let calls = accumulator
            .finish()
            .into_iter()
            .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
            .count();
        prop_assert_eq!(calls, 1);
    }

    /// Reasoning-end idempotence: bare repeated ends add nothing.
    #[test]
    fn stale_reasoning_ends_are_idempotent(extra_ends in 1usize..5) {
        let mut accumulator = BlockAccumulator::new();
        let key = BlockId::wire("rs_1");
        reasoning_delta(&mut accumulator, &key, "thought");
        reasoning_end(&mut accumulator, &key);
        for _ in 0..extra_ends {
            prop_assert!(reasoning_end(&mut accumulator, &key).is_none());
        }
        prop_assert_eq!(accumulator.finish().len(), 1);
    }

    /// `snapshot` is a pure read: any number of snapshots at any point of
    /// a fold are equal to each other and leave `finish` unchanged.
    #[test]
    fn snapshots_are_pure_reads(
        payload in ".{0,30}",
        snapshots in 0usize..4,
    ) {
        let key = BlockId::wire("msg_1");
        let mut reference = BlockAccumulator::new();
        text_delta(&mut reference, &key, &payload);
        text_delta(&mut reference, &key, "!");
        let reference = reference.finish();

        let mut observed = BlockAccumulator::new();
        text_delta(&mut observed, &key, &payload);
        let first = observed.snapshot();
        for _ in 0..snapshots {
            prop_assert_eq!(observed.snapshot(), first.clone());
        }
        text_delta(&mut observed, &key, "!");
        let last = observed.snapshot();
        prop_assert_eq!(observed.finish(), last);
        prop_assert_eq!(reference, {
            let mut again = BlockAccumulator::new();
            text_delta(&mut again, &key, &payload);
            let _ = again.snapshot();
            text_delta(&mut again, &key, "!");
            again.finish()
        });
    }
}
