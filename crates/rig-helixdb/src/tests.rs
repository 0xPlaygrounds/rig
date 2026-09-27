use super::*;
use rig_core::error::{ErrorKind, RigError};

/// A HelixDB failure reports as the store reports it: a datastore failure.
#[test]
fn a_helix_error_converts_as_a_datastore_failure() {
    let error = RigError::from(HelixError::RemoteError {
        details: "index missing".to_owned(),
    });
    assert_eq!(error.kind, ErrorKind::Provider);
    assert!(!error.retryable);
    assert_eq!(
        error.message,
        "Datastore error: got error from server: index missing"
    );
    assert_eq!(error.source_chain, ["got error from server: index missing"]);
}
