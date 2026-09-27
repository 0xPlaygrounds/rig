use super::*;
use rig_core::error::{ErrorKind, RigError};

/// A Vectorize failure reports as the store reports it: a datastore failure.
#[test]
fn a_vectorize_error_converts_as_a_datastore_failure() {
    let error = RigError::from(VectorizeError::ApiError {
        code: 1000,
        message: "index not found".to_owned(),
    });
    assert_eq!(error.kind, ErrorKind::Provider);
    assert!(!error.retryable);
    assert_eq!(
        error.message,
        "Datastore error: Vectorize API error (code: 1000): index not found"
    );
    assert_eq!(
        error.source_chain,
        ["Vectorize API error (code: 1000): index not found"]
    );
}
