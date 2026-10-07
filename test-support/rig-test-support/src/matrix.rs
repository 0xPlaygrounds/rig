//! Literal matrix rows share execution while keeping stable test identities.

/// Emit registered test rows with the shared execution body.
#[macro_export]
macro_rules! golden_matrix {
    (
        wrapper: $wrapper:path, wire: $wire:path, run: $run:path, oracle: $oracle:path;
        $( $(#[$attribute:meta])* $name:ident: ($scenario:literal, $cell:path, $golden:literal); )*
    ) => {
        $(
            $(#[$attribute])*
            async fn $name() {
                $wrapper($scenario, |client| async move {
                    $run(&$wire(&client), &$cell, |log| $oracle($golden, log)).await;
                }).await;
            }
        )*
    };
}

pub use golden_matrix;

/// Emit registered test rows with the shared execution body.
#[macro_export]
macro_rules! case_matrix {
    (
        family: $family:ident;
        $( $(#[$attribute:meta])* $name:ident: $case:ident $(=> $golden:literal)?; )*
    ) => {
        $(
            $crate::matrix::$family!($(#[$attribute])* $name, $case $(, $golden)?);
        )*
    };
    (
        wrapper: $wrapper:path, family: $family:ident;
        $( $(#[$attribute:meta])* $name:ident: ($scenario:literal, $case:ident $(, $cell:expr)?); )*
    ) => {
        $(
            $crate::matrix::$family!($(#[$attribute])* $name, $wrapper, $scenario, $case $(, $cell)?);
        )*
    };
}

pub use case_matrix;

#[path = "matrix/agent_tool_sessions.rs"]
mod agent_tool_sessions;
pub use agent_tool_sessions::agent_tool_sessions_case;
pub use agent_tool_sessions::session_agent;

#[path = "matrix/turn_termination_matrix.rs"]
mod turn_termination_matrix;
pub use turn_termination_matrix::turn_termination_matrix_case;

#[path = "matrix/tool_lifecycle_matrix.rs"]
mod tool_lifecycle_matrix;
pub use tool_lifecycle_matrix::tool_lifecycle_matrix_case;

#[path = "matrix/streaming_logprobs_matrix.rs"]
mod streaming_logprobs_matrix;
pub use streaming_logprobs_matrix::streaming_logprobs_matrix_case;

#[path = "matrix/history_roundtrip_matrix.rs"]
mod history_roundtrip_matrix;
pub use history_roundtrip_matrix::history_roundtrip_matrix_case;

#[path = "matrix/history_survival_matrix.rs"]
mod history_survival_matrix;
pub use history_survival_matrix::history_survival_case;

#[path = "matrix/portability_matrix.rs"]
mod portability_matrix;
pub use portability_matrix::portability_case;

#[path = "matrix/terminal_metadata_matrix.rs"]
mod terminal_metadata_matrix;
pub use terminal_metadata_matrix::terminal_metadata_matrix_case;

#[path = "matrix/wire_cases.rs"]
mod wire_cases;
pub use wire_cases::wire_matrix_case;
