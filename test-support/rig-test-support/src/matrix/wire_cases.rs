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
}

pub use wire_matrix_case;
