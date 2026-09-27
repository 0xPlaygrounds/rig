use super::*;
use rig_core::error::{ErrorKind, RigError};

/// A model `fastembed` does not know is the caller's setup; one that will
/// not load is some other failure.
#[test]
fn a_fastembed_error_converts_by_what_failed() {
    let unknown = RigError::from(FastembedError::UnknownModel(FastembedModel::AllMiniLML6V2));
    assert_eq!(unknown.kind, ErrorKind::Configuration);
    assert!(!unknown.retryable);
    assert_eq!(
        unknown.message,
        "Failed to resolve FastEmbed model metadata for AllMiniLML6V2"
    );
    let failed = RigError::from(FastembedError::Initialization("no ONNX runtime".to_owned()));
    assert_eq!(failed.kind, ErrorKind::Other);
    assert!(!failed.retryable);
    assert_eq!(
        failed.message,
        "Failed to initialize FastEmbed model: no ONNX runtime"
    );
}
