//! Shared scripted and cassette-backed matrix bodies with wire-local adapters.

/// Emit one matrix row with its original attributes and complete assertions.
#[macro_export]
macro_rules! wire_matrix_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, shaping_thinking_second_turn_9) => {
        $(#[$attribute])*
        async fn $name() {
            $wrapper($scenario, |client| async move {
                run_agent(
                    &reasoning_wire(&client),
                    &cells::SHAPING_THINKING_SECOND_TURN,
                    |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
                )
                .await;
            })
            .await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, reasoning_tool_unary_10) => {
        $(#[$attribute])*
        async fn $name() {
            $wrapper($scenario, |client| async move {
                run_agent(
                    &reasoning_wire(&client),
                    &cells::REASONING_TOOL_UNARY,
                    |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
                )
                .await;
            })
            .await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, reasoning_tool_streamed_11) => {
        $(#[$attribute])*
        async fn $name() {
            $wrapper($scenario, |client| async move {
                run_agent(
                    &reasoning_wire(&client),
                    &cells::REASONING_TOOL_STREAMED,
                    |_| panic!("unrecorded reasoning scenario: see this test\'s ignore disposition"),
                )
                .await;
            })
            .await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, reasoning_off_12) => {
        $(#[$attribute])*
        async fn $name() {
            $wrapper($scenario, |client| async move {
                run_agent(&reasoning_wire(&client), &cells::REASONING_OFF, |_| {
                    panic!("unrecorded reasoning scenario: see this test\'s ignore disposition")
                })
                .await;
            })
            .await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, invalid_args_midway_13, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let replies = long_loop::scripted_replies(THINKING, &long_loop::INVALID_ARGS_MIDWAY, None);
            long_loop::run_scripted(&long_loop::INVALID_ARGS_MIDWAY, || {
                scripted_unary(replies.clone())
            }, |log| $crate::goldens::world_golden_effects($golden, log))
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, provider_fault_midway_14, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let replies = long_loop::scripted_replies(
                THINKING,
                &long_loop::PROVIDER_FAULT_MIDWAY,
                Some(fault_reply()),
            );
            long_loop_world::run_world(&scripted_unary(replies), &long_loop::PROVIDER_FAULT_MIDWAY, |log| $crate::goldens::world_golden_effects($golden, log))
                .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, batch_held_call_approved_by_removing_held_15, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            $wrapper($scenario, |client| async move {
                batch_hold(&wire(&client), Approval::RemoveHeld, |log| $crate::goldens::world_golden_effects($golden, log)).await;
            })
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, batch_held_call_approved_by_releasing_the_batch_owner_16, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            $wrapper($scenario, |client| async move {
                batch_hold(&wire(&client), Approval::ReleaseBatchOwner, |log| $crate::goldens::world_golden_effects($golden, log)).await;
            })
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, despawn_run_waits_for_an_in_flight_stream_17, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            $wrapper($scenario, |client| async move {
                despawn_waits_for_the_stream(&legacy(&client), &cells::BREADTH_TEXT_DELTA_STOP, |log| $crate::goldens::world_golden_effects($golden, log)).await;
            })
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, shaping_thinking_second_turn_18, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            $wrapper($scenario, |client| async move {
                run_world(&reasoning_wire(&client), &cells::SHAPING_THINKING_SECOND_TURN, |log| $crate::goldens::world_golden_effects($golden, log)).await;
            })
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, reasoning_tool_unary_19, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            $wrapper($scenario, |client| async move {
                run_world(&reasoning_wire(&client), &cells::REASONING_TOOL_UNARY, |log| $crate::goldens::world_golden_effects($golden, log)).await;
            })
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, reasoning_tool_streamed_20, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            $wrapper($scenario, |client| async move {
                run_world(&reasoning_wire(&client), &cells::REASONING_TOOL_STREAMED, |log| $crate::goldens::world_golden_effects($golden, log)).await;
            })
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, reasoning_off_21, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            $wrapper($scenario, |client| async move {
                run_world(&reasoning_wire(&client), &cells::REASONING_OFF, |log| $crate::goldens::world_golden_effects($golden, log)).await;
            })
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, despawn_run_waits_for_an_in_flight_stream_22, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            $wrapper($scenario, |client| async move {
                despawn_waits_for_the_stream(&wire(&client), &cells::ENDINGS_TEXT_DELTA_STOP, |log| $crate::goldens::world_golden_effects($golden, log)).await;
            })
            .await;
            }).await;
        }
    };
}

pub use wire_matrix_case;
