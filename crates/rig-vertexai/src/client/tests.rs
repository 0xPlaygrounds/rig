use super::*;
use rig_core::error::{ErrorKind, RigError};

/// A client that cannot be built reports as the Vertex transport reports it.
#[test]
fn a_client_error_converts_as_the_transport_reports_it() {
    let error = RigError::from(VertexAiClientError::MissingProject);
    assert_eq!(error.kind, ErrorKind::Provider);
    assert!(!error.retryable);
    assert_eq!(
        error.message,
        "ProviderError: Google Cloud project is required. Set it via \
         `VertexAiBuilder::with_project()` or `GOOGLE_CLOUD_PROJECT`"
    );
}
