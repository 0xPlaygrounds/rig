//! Scripted ECS fault rows over a wire's `faults::Scripted` or `long_loop::Scripted` suite.

/// Emit a scripted fault row: the suite's method of the row's own name,
/// asserted against the row's literal world golden.
#[macro_export]
macro_rules! ecs_faults_case {
    ($(#[$attribute:meta])* $name:ident, $suite:ident, $golden:expr) => {
        $(#[$attribute])*
        async fn $name() {
            $suite
                .$name(|log| $crate::goldens::world_golden_effects($golden, log))
                .await;
        }
    };
}

pub use ecs_faults_case;
