//! Shared ECS fault bodies with explicit scenario and world golden names.

/// Emit a native cancellation cell and its world golden assertion.
#[macro_export]
macro_rules! ecs_faults_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, cancel_after_terminal_6, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $crate::goldens::capture_world_programs(async {
                $wrapper($scenario, |client| async move {
                    cancel_at(
                        &wire(&client),
                        &cells::ENDINGS_TEXT_DELTA_STOP,
                        Cut::AfterTerminal,
                        |log| $crate::goldens::world_golden_effects($golden, log),
                    ).await;
                }).await;
            }).await;
        }
    };
}

pub use ecs_faults_case;
