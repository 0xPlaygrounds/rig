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

proptest! {
    /// Text aggregation is invariant under fragmentation.
    #[test]
    fn text_aggregate_is_fragmentation_invariant(
        payload in ".{0,60}",
        points in proptest::collection::vec(0usize..1000, 0..6),
    ) {
        let mut whole = PartsAccumulator::new();
        whole.text_delta(&payload);
        let mut split = PartsAccumulator::new();
        for fragment in split_fragments(&payload, &points) {
            split.text_delta(&fragment);
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
        let mut whole = PartsAccumulator::new();
        whole.reasoning_delta(&key, None, &payload);
        let completed_whole = whole.reasoning_end(&key, None, None);
        let mut split = PartsAccumulator::new();
        let mut pushed = false;
        for fragment in split_fragments(&payload, &points) {
            if !fragment.is_empty() {
                split.reasoning_delta(&key, None, &fragment);
                pushed = true;
            }
        }
        prop_assume!(pushed);
        let completed_split = split.reasoning_end(&key, None, None);
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
            let mut accumulator = PartsAccumulator::new();
            let key = BlockId::wire("call_1");
            accumulator.tool_name_delta(&key, "probe");
            for fragment in fragments {
                accumulator.tool_args_delta(&key, fragment);
            }
            accumulator
                .tool_input_end(ToolInputEnd::new(key, UnparseableToolInput::Drop))
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
        let mut accumulator = PartsAccumulator::new();
        let key = BlockId::wire("call_1");
        accumulator.tool_name_delta(&key, "probe");
        accumulator.tool_args_delta(&key, "{}");
        accumulator
            .tool_input_end(ToolInputEnd::new(key.clone(), UnparseableToolInput::Drop))
            .expect("no error")
            .expect("finalizes");
        for _ in 0..extra_ends {
            let mut stale = ToolInputEnd::new(key.clone(), UnparseableToolInput::Drop);
            if authoritative {
                stale.name = Some("probe".to_owned());
                stale.arguments = Some(serde_json::json!({}));
            }
            prop_assert!(
                accumulator
                    .tool_input_end(stale)
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
        let mut accumulator = PartsAccumulator::new();
        let key = BlockId::wire("rs_1");
        accumulator.reasoning_delta(&key, None, "thought");
        accumulator.reasoning_end(&key, None, None);
        for _ in 0..extra_ends {
            prop_assert!(accumulator.reasoning_end(&key, None, None).is_none());
        }
        prop_assert_eq!(accumulator.finish().len(), 1);
    }
}
