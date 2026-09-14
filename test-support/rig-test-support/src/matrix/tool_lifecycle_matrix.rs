//! Shared tool lifecycle matrix execution with explicit per-wire cell descriptors.

/// Run the declared cell through the wire and retain its complete observation assertions.
#[macro_export]
macro_rules! tool_lifecycle_matrix_case {
    ($(#[$attribute:meta])* $name:ident, $wrapper:path, $scenario:literal, configured, $cell:expr) => {
        $(#[$attribute])*
        async fn $name() -> Result<()> {
            const S: &str = $scenario;
            let c = $cell;
            let o = SharedObservation::default();
            $wrapper($scenario, {
                let o = Arc::clone(&o);
                move |x| run_cell(x, c, o)
            })
            .await?;
            execute(S, c, o).await;
            Ok(())
        }
    };
}

pub use tool_lifecycle_matrix_case;
