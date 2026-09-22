//! Native counterparts of the provider turn-termination matrix.

use super::turn_termination_matrix::{
    CONCISE_PREAMBLE, MODEL, RETRY_PROMPT, ROOMY_CAP, SHORT_PROMPT, TINY_CAP, TOOL_PREAMBLE,
    TOOL_PROMPT, TRUNCATING_PROMPT, assert_recorded_request_cap, assert_recorded_wire_reason,
    recorded_request_caps, recorded_wire_reasons,
};
use crate::deepseek::support::with_deepseek_cassette;
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
    wrapper: with_deepseek_cassette, family: ecs_termination_case;
    # [tokio :: test]
    blocking_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap", blocking_truncated_turn_reports_length_and_cap_7, "deepseek_termination_blocking_truncated_turn_reports_length_and_cap");
    # [tokio :: test]
    streaming_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap", streaming_truncated_turn_reports_length_and_cap_8, "deepseek_termination_streaming_truncated_turn_reports_length_and_cap");
    # [tokio :: test]
    blocking_completed_turn_reports_stop_and_cap: ("turn_termination_matrix/blocking_completed_turn_reports_stop_and_cap", blocking_completed_turn_reports_stop_and_cap_9, "deepseek_termination_blocking_completed_turn_reports_stop_and_cap");
    # [tokio :: test]
    streaming_completed_turn_reports_stop_and_cap: ("turn_termination_matrix/streaming_completed_turn_reports_stop_and_cap", streaming_completed_turn_reports_stop_and_cap_10, "deepseek_termination_streaming_completed_turn_reports_stop_and_cap");
    # [tokio :: test]
    blocking_tool_turn_reports_tool_calls: ("turn_termination_matrix/blocking_tool_turn_reports_tool_calls", blocking_tool_turn_reports_tool_calls_11, "deepseek_termination_blocking_tool_turn_reports_tool_calls");
    # [tokio :: test]
    streaming_tool_turn_reports_tool_calls: ("turn_termination_matrix/streaming_tool_turn_reports_tool_calls", streaming_tool_turn_reports_tool_calls_12, "deepseek_termination_streaming_tool_turn_reports_tool_calls");
    # [tokio :: test]
    blocking_escalating_retry_reports_each_attempts_own_cap: ("turn_termination_matrix/blocking_escalating_retry_reports_each_attempts_own_cap", blocking_escalating_retry_reports_each_attempts_own_cap_13, "deepseek_termination_blocking_escalating_retry_reports_each_attempts_own_cap");
    # [tokio :: test]
    streaming_escalating_retry_reports_each_attempts_own_cap: ("turn_termination_matrix/streaming_escalating_retry_reports_each_attempts_own_cap", streaming_escalating_retry_reports_each_attempts_own_cap_14, "deepseek_termination_streaming_escalating_retry_reports_each_attempts_own_cap");
}
