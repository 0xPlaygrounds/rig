//! Shared cross-wire continuation cells with explicit per-target descriptors.
//!
//! The provider module supplies `model(client, cell)`; the body continues a
//! history decoded from another wire's recording and, after the cassette
//! finalizes, checks the recorded request.

/// Run the declared cell through a plain cassette wrapper.
#[macro_export]
macro_rules! portability_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, configured, $cell:expr) => {
        $(#[$attribute])*
        async fn $name() {
            const SCENARIO: &str = $scenario;
            let cell: $crate::history_survival::portability::Cell = $cell;
            let observed: $crate::history_survival::portability::Observed = Default::default();
            let capture = ::std::sync::Arc::clone(&observed);
            $wrapper(SCENARIO, |client| async move {
                let observation =
                    $crate::history_survival::portability::run(model(client, cell), cell).await;
                $crate::history_survival::portability::assert_run(cell, &observation);
                *capture.lock().expect("observation slot") = Some(observation);
            })
            .await;
            assert!(
                observed.lock().expect("observation slot").is_some(),
                "{SCENARIO}: the cell body records an observation"
            );
            $crate::history_survival::portability::assert_recorded(cell, SCENARIO);
        }
    };
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, configured_result, $cell:expr) => {
        $(#[$attribute])*
        async fn $name() {
            const SCENARIO: &str = $scenario;
            let cell: $crate::history_survival::portability::Cell = $cell;
            let observed: $crate::history_survival::portability::Observed = Default::default();
            let capture = ::std::sync::Arc::clone(&observed);
            $wrapper(SCENARIO, |client| async move {
                let observation =
                    $crate::history_survival::portability::run(model(client, cell), cell).await;
                $crate::history_survival::portability::assert_run(cell, &observation);
                *capture.lock().expect("observation slot") = Some(observation);
                Ok::<(), ::anyhow::Error>(())
            })
            .await
            .expect("cassette session");
            assert!(
                observed.lock().expect("observation slot").is_some(),
                "{SCENARIO}: the cell body records an observation"
            );
            $crate::history_survival::portability::assert_recorded(cell, SCENARIO);
        }
    };
}

pub use portability_case;
