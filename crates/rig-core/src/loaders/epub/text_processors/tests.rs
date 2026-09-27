use super::*;
use crate::error::{ErrorKind, RigError};

#[test]
fn a_markup_failure_is_other() {
    let utf8 = String::from_utf8(vec![0xff]).expect_err("not UTF-8");
    let error = RigError::from(XmlProcessingError::Utf8(utf8));
    assert_eq!(error.kind, ErrorKind::Other);
    assert!(!error.retryable);
    assert_eq!(
        error.message,
        "Invalid UTF-8 sequence: invalid utf-8 sequence of 1 bytes from index 0"
    );
    assert_eq!(
        error.source_chain,
        ["invalid utf-8 sequence of 1 bytes from index 0"]
    );
}
