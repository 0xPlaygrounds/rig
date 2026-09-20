//! Shared history roundtrip matrix execution with explicit per-wire cell descriptors.

/// Run the declared cell through the wire and retain its complete observation assertions.
#[macro_export]
macro_rules! history_roundtrip_matrix_case {
    ($(#[$($attribute:tt)*])* $name:ident, $wrapper:path, $scenario:literal, configured, $cell:expr) => {
#[$crate::cassette($crate::recording::Scenario::live(format!("{}/{}", env!("CARGO_CRATE_NAME"), $scenario)))]
        $(#[$($attribute)*])*
        async fn $name() -> Result<()> {
            const SCENARIO: &str = $scenario;
            let cell = $cell;
            let observed = SharedObservation::default();
            let capture = Arc::clone(&observed);
            $wrapper($scenario, |client| async move {
                run_cell(client, cell, capture).await
            })
            .await?;
            assert_cell(SCENARIO, cell, observed);
            Ok(())
        }
};
}

pub use history_roundtrip_matrix_case;
