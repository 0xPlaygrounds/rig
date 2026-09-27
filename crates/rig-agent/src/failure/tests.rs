//! The runtime's failures report exactly as the provider error of the same
//! kind does.

use rig_core::error::{ProviderError, RigError};

use super::*;

#[test]
fn a_response_failure_reports_as_a_wire_does() {
    assert_eq!(
        response("provider stream ended without a terminal record"),
        RigError::from(ProviderError::Response(
            "provider stream ended without a terminal record".to_owned()
        ))
    );
}

#[test]
fn a_request_failure_reports_as_a_wire_does() {
    assert_eq!(
        request("Failed to get tool definitions: gone"),
        RigError::from(ProviderError::Request(
            "Failed to get tool definitions: gone".to_owned().into()
        ))
    );
}

#[test]
fn a_request_failure_with_a_cause_reports_as_a_wire_does() {
    let cause = || {
        rig_core::vector_store::VectorStoreError::DatastoreError(Box::new(std::io::Error::other(
            "the index is gone",
        )))
    };
    assert_eq!(
        request_caused_by(cause()),
        RigError::from(ProviderError::Request(Box::new(cause())))
    );
}
