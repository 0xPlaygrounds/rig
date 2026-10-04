//! Native counterparts of the provider turn-termination matrix.

use super::turn_termination_matrix::{
    CONCISE_PREAMBLE, MODEL, ROOMY_CAP, TINY_CAP, TOOL_PREAMBLE, TOOL_PROMPT, TRUNCATING_PROMPT,
    assert_recorded_request_cap, assert_recorded_wire_reason,
};
use crate::deepseek::support::with_deepseek_cassette;
use crate::{
    ecs_agent::EcsAgent,
    ecs_termination::{self, NativeProbe as TurnTerminationProbe},
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
    blocking_tool_turn_reports_tool_calls: ("turn_termination_matrix/blocking_tool_turn_reports_tool_calls", blocking_tool_turn_reports_tool_calls_11, "deepseek_termination_blocking_tool_turn_reports_tool_calls");
}
