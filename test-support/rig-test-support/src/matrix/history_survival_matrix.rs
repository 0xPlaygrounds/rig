//! Shared history-survival cell execution with explicit per-wire cell descriptors.
//!
//! The provider module supplies `model(client, cell)`; the body runs the
//! shared three-prompt task, asserts the task and the normalized history
//! inside the cassette session, then applies the wire-level survival rule to
//! the finalized recording.

/// Run the declared cell through a plain cassette wrapper.
#[macro_export]
macro_rules! history_survival_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, configured, $cell:expr) => {
        $(#[$attribute])*
        async fn $name() {
            const SCENARIO: &str = $scenario;
            let cell: $crate::history_survival::driver::Cell = $cell;
            let observed: $crate::history_survival::driver::Observed = Default::default();
            let capture = ::std::sync::Arc::clone(&observed);
            $wrapper(SCENARIO, |client| async move {
                let observation =
                    $crate::history_survival::driver::run(model(client, cell), cell).await;
                $crate::history_survival::driver::assert_run(cell, &observation);
                *capture.lock().expect("observation slot") = Some(observation);
            })
            .await;
            assert!(
                observed.lock().expect("observation slot").is_some(),
                "{SCENARIO}: the cell body records an observation"
            );
            $crate::history_survival::driver::assert_recorded(cell, SCENARIO);
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, configured_result, $cell:expr) => {
        $(#[$attribute])*
        async fn $name() {
            const SCENARIO: &str = $scenario;
            let cell: $crate::history_survival::driver::Cell = $cell;
            let observed: $crate::history_survival::driver::Observed = Default::default();
            let capture = ::std::sync::Arc::clone(&observed);
            $wrapper(SCENARIO, |client| async move {
                let observation =
                    $crate::history_survival::driver::run(model(client, cell), cell).await;
                $crate::history_survival::driver::assert_run(cell, &observation);
                *capture.lock().expect("observation slot") = Some(observation);
                Ok::<(), ::anyhow::Error>(())
            })
            .await
            .expect("cassette session");
            assert!(
                observed.lock().expect("observation slot").is_some(),
                "{SCENARIO}: the cell body records an observation"
            );
            $crate::history_survival::driver::assert_recorded(cell, SCENARIO);
        }
    };
}

pub use history_survival_case;
