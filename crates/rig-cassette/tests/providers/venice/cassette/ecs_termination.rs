//! Native counterparts of the provider turn-termination matrix.

use super::super::support::with_venice_cassette;
use super::turn_termination_matrix::{
    CONCISE_PREAMBLE, MODEL, RETRY_PROMPT, ROOMY_CAP, SHORT_PROMPT, TINY_CAP, TOOL_PREAMBLE,
    TOOL_PROMPT, TRUNCATING_PROMPT, assert_recorded_request_cap, assert_recorded_wire_reason,
    recorded_request_caps, recorded_wire_reasons,
};
use crate::{
    ecs_agent::EcsAgent,
    ecs_termination::{
        self, NativeEscalation as EscalateCapOnTruncation, NativeProbe as TurnTerminationProbe,
    },
    support::Adder,
};
use rig::completion::FinishReason;
use rig_ecs::agent::{MaxTokens, Temperature};

crate::matrix::case_matrix! {
    wrapper: with_venice_cassette, family: ecs_termination_case;
    # [tokio :: test]
    blocking_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap", blocking_truncated_turn_reports_length_and_cap_7, "venice_termination_blocking_truncated_turn_reports_length_and_cap");
    # [tokio :: test]
    streaming_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap", streaming_truncated_turn_reports_length_and_cap_8, "venice_termination_streaming_truncated_turn_reports_length_and_cap");
    # [tokio :: test]
    blocking_completed_turn_reports_stop_and_cap: ("turn_termination_matrix/blocking_completed_turn_reports_stop_and_cap", blocking_completed_turn_reports_stop_and_cap_9, "venice_termination_blocking_completed_turn_reports_stop_and_cap");
    # [tokio :: test]
    streaming_completed_turn_reports_stop_and_cap: ("turn_termination_matrix/streaming_completed_turn_reports_stop_and_cap", streaming_completed_turn_reports_stop_and_cap_10, "venice_termination_streaming_completed_turn_reports_stop_and_cap");
    # [ignore = "Venice mistral-small-3-2-24b-instruct answered without calling add in attempts 1, 2 and 3 (2026-09-13, record-venice-termination-blocking-attempt-{1,2,3}.log); exhausted the reasoning-matrix prompt's three-attempt limit"]
    # [tokio :: test]
    blocking_tool_turn_reports_tool_calls: ("turn_termination_matrix/blocking_tool_turn_reports_tool_calls", blocking_tool_turn_reports_tool_calls_11, "venice_termination_blocking_tool_turn_reports_tool_calls");
    # [ignore = "Venice mistral-small-3-2-24b-instruct answered without calling add in attempts 1, 2 and 3 (2026-09-13, record-venice-termination-streaming-attempt-{1,2,3}.log); exhausted the reasoning-matrix prompt's three-attempt limit"]
    # [tokio :: test]
    streaming_tool_turn_reports_tool_calls: ("turn_termination_matrix/streaming_tool_turn_reports_tool_calls", streaming_tool_turn_reports_tool_calls_12, "venice_termination_streaming_tool_turn_reports_tool_calls");
    # [tokio :: test]
    blocking_escalating_retry_reports_each_attempts_own_cap: ("turn_termination_matrix/blocking_escalating_retry_reports_each_attempts_own_cap", blocking_escalating_retry_reports_each_attempts_own_cap_13, "venice_termination_blocking_escalating_retry_reports_each_attempts_own_cap");
    # [tokio :: test]
    streaming_escalating_retry_reports_each_attempts_own_cap: ("turn_termination_matrix/streaming_escalating_retry_reports_each_attempts_own_cap", streaming_escalating_retry_reports_each_attempts_own_cap_14, "venice_termination_streaming_escalating_retry_reports_each_attempts_own_cap");
}
