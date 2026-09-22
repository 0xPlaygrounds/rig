//! Shared scripted and cassette-backed matrix bodies with wire-local adapters.

/// Emit one matrix row with its original attributes and complete assertions.
#[macro_export]
macro_rules! wire_matrix_case {
    ($(#[$attribute:meta])* $name:ident, truncated_after_text_0, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let frames = SHAPE.text_prefix(&recorded(TEXT_STREAM));
            run_scripted(&faults::TRUNCATED_AFTER_TEXT, || scripted_stream(&frames), |log| $crate::goldens::world_golden_effects($golden, log)).await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, truncated_after_tool_call_1, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let frames = SHAPE.tool_prefix(&recorded(TOOL_STREAM));
            run_scripted(&faults::TRUNCATED_AFTER_TOOL_CALL, || {
                scripted_stream(&frames)
            }, |log| $crate::goldens::world_golden_effects($golden, log))
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, error_after_text_2, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let cell = Cell {
                fault: Some(Fault::ErrorAfterText {
                    code: SHAPE.error_code(),
                    message: SHAPE.error_message(),
                    status: SHAPE.error_status(),
                }),
                ..faults::ERROR_AFTER_TEXT
            };
            let frames = SHAPE.error_frames(&recorded(TEXT_STREAM));
            run_scripted(&cell, || scripted_stream(&frames), |log| $crate::goldens::world_golden_effects($golden, log)).await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, filtered_with_text_3, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let frames = SHAPE.filtered(&recorded(TEXT_STREAM), true);
            run_scripted(&faults::FILTERED_WITH_TEXT, || scripted_stream(&frames), |log| $crate::goldens::world_golden_effects($golden, log)).await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, filtered_empty_4, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let frames = SHAPE.filtered(&recorded(TEXT_STREAM), false);
            run_scripted(&faults::FILTERED_EMPTY, || scripted_stream(&frames), |log| $crate::goldens::world_golden_effects($golden, log)).await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, failing_load_5, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            run_scripted(&faults::FAILING_LOAD, || scripted_stream(&[]), |log| $crate::goldens::world_golden_effects($golden, log)).await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, failing_load_streamed_6, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            run_scripted(&faults::FAILING_LOAD_STREAMED, || scripted_stream(&[]), |log| $crate::goldens::world_golden_effects($golden, log)).await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, cancel_at_first_tool_call_delta_7, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let frames = recorded(TOOL_STREAM);
            cancel_at(
                &scripted_stream(&frames),
                &cells::HOOKS_PATCH_TOOL_ARGS_STREAMED,
                Cut::FirstToolCallDelta,
             |log| $crate::goldens::world_golden_effects($golden, log))
            .await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, cancel_after_terminal_8, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            $wrapper($scenario, |client| async move {
                cancel_at(
                    &legacy(&client),
                    &cells::BREADTH_TEXT_DELTA_STOP,
                    Cut::AfterTerminal,
                 |log| $crate::goldens::world_golden_effects($golden, log))
                .await;
            })
            .await;
            }).await;
        }
    };
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
    ($(#[$attribute:meta])* $name:ident, status_429_23, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let cell = Cell {
                fault: Some(Fault::Status {
                    status: 429,
                    code: Some("model_not_found"),
                    retry_after: true,
                }),
                ..faults::STATUS_429
            };
            run_scripted(&cell, || scripted_unary(vec![reply(429, true)]), |log| $crate::goldens::world_golden_effects($golden, log)).await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, status_503_24, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let cell = Cell {
                fault: Some(Fault::Status {
                    status: 503,
                    code: Some("model_not_found"),
                    retry_after: false,
                }),
                ..faults::STATUS_503
            };
            run_scripted(&cell, || scripted_unary(vec![reply(503, false)]), |log| $crate::goldens::world_golden_effects($golden, log)).await;
            }).await;
        }
    };
    ($(#[$attribute:meta])* $name:ident, status_503_retried_25, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
            let cell = Cell {
                fault: Some(Fault::Status {
                    status: 503,
                    code: Some("model_not_found"),
                    retry_after: false,
                }),
                ..faults::STATUS_503_RETRIED
            };
            let replies = || (0..4).map(|_| reply(503, false)).collect();
            run_world(&scripted_unary(replies()), &cell, |log| $crate::goldens::world_golden_effects($golden, log)).await;
            }).await;
        }
    };
}

pub use wire_matrix_case;
