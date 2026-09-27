use super::*;
use rig_core::error::{ErrorKind, RigError};

/// A local failure reports as the Candle transport reports it.
#[test]
fn a_candle_error_converts_as_the_transport_reports_it() {
    let error = RigError::from(CandleError::EmptyBuffer {
        artifact: "weights",
    });
    assert_eq!(error.kind, ErrorKind::Provider);
    assert!(!error.retryable);
    assert_eq!(error.message, "ProviderError: the weights buffer is empty");
    assert!(error.source_chain.is_empty());
}
