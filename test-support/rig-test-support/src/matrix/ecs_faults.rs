//! Shared ecs faults bodies; each wire supplies explicit scenario rows.

/// Emit registered test rows with the shared execution body.
#[macro_export]
macro_rules! ecs_faults_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, cancel_after_terminal_6) => {
        $(#[$attribute])*
        async fn $name()  { $wrapper ($scenario , | client | async move { cancel_at (& wire (& client) , & cells :: ENDINGS_TEXT_DELTA_STOP , Cut :: AfterTerminal ,) . await ; } ,) . await ; }
    };
}

pub use ecs_faults_case;
