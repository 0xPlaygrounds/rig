//! Every operation that can refuse a credential-less request, in one place,
//! because the two defects this pins were both a second classification
//! table disagreeing with the first: `VerifyError` charged the refusal to
//! the provider, and `ModelListingError::RequestError` reported a fault its
//! own `boundary()` called a request fault as `ErrorKind::Http`.
//!
//! All seven are listed. Adding an operation without adding it here is the
//! failure mode, so `refusals()` names them rather than deriving them, and
//! a reader can check the list against `provider_error_enum!`'s callers.
//!
//! | claim | test |
//! |---|---|
//! | all seven operations call a credential-less request a fault in the request | `a_missing_credential_is_a_request_fault_in_every_operation` |
//! | all seven name the variable the same way | `every_operation_names_the_variable_the_same_way` |

use crate::audio_generation::AudioGenerationError;
use crate::client::VerifyError;
use crate::completion::CompletionError;
use crate::embeddings::EmbeddingError;
use crate::error::{ErrorKind, ErrorReport};
use crate::image_generation::ImageGenerationError;
use crate::model::ModelListingError;
use crate::observe::AdapterErrorBoundary;
use crate::rerank::RerankError;
use crate::transcription::TranscriptionError;
use crate::wire::WireError;

type Refusal = (&'static str, AdapterErrorBoundary, ErrorKind, String);

/// Every error that can refuse a credential-less request, built the one way
/// a wire builds it.
fn refusals() -> Vec<Refusal> {
    fn refusal<E: WireError>(name: &'static str) -> Refusal
    where
        for<'a> ErrorReport: From<&'a E>,
    {
        let error = E::missing_credential("PROVIDER_API_KEY");
        let report = ErrorReport::from(&error);
        (name, error.boundary(), report.kind, error.to_string())
    }

    vec![
        refusal::<CompletionError>("completion"),
        refusal::<EmbeddingError>("embedding"),
        refusal::<TranscriptionError>("transcription"),
        refusal::<ImageGenerationError>("image generation"),
        refusal::<RerankError>("rerank"),
        refusal::<AudioGenerationError>("audio generation"),
        refusal::<ModelListingError>("model listing"),
        refusal::<VerifyError>("verify"),
    ]
}

/// The refusal happens inside `encode`: no request was sent, so no provider
/// gave a verdict. Charging it to the provider boundary is what made
/// `VerifyError` disagree with every other operation, and the disagreement
/// survived until a review read the two tables side by side. One test now
/// reads them for us.
#[test]
fn a_missing_credential_is_a_request_fault_in_every_operation() {
    for (operation, boundary, kind, _) in refusals() {
        assert_eq!(
            boundary,
            AdapterErrorBoundary::Request,
            "{operation}: a refusal before a request exists is not the provider's verdict"
        );
        assert_eq!(
            kind,
            ErrorKind::Request,
            "{operation}: the report must agree with the boundary"
        );
    }
}

/// The message exists once, so the variable a user has to set is named the
/// same way whichever operation refused.
#[test]
fn every_operation_names_the_variable_the_same_way() {
    for (operation, _, _, message) in refusals() {
        assert!(
            message.contains("PROVIDER_API_KEY"),
            "{operation}: {message}"
        );
        assert!(
            message.contains("build its configuration with a key"),
            "{operation}: {message}"
        );
    }
}
