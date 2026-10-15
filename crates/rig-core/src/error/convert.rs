//! `From` conversions of every rig-core error, and of the JSON and URL
//! errors rig classifies, into [`RigError`], so an application propagates
//! them with `?`. A type a code path already reports through
//! [`ProviderError`] or [`ToolExecutionError`] converts the same way.

use super::{EncodeError, ErrorKind, ProviderError, RigError};
use crate::catalog::{CatalogError, NotFound, OverrideErrors, UnknownReasoning};
use crate::client::env::EnvError;
use crate::completion::message::{EmptyCallId, EmptyToolName, MessageError};
use crate::completion::provider_options::OptionsError;
use crate::completion::{CheckError, UnsupportedOption};
use crate::conversation::pairing::{AnswerError, PairingError};
use crate::conversation::{ApplyError, BranchError, ParseIdError as ParseConversationIdError};
use crate::embeddings::embed::EmbedError;
use crate::http_client;
use crate::id::ParseIdError;
use crate::loaders::file::FileLoaderError;
use crate::provider_response::ProviderResponseError;
use crate::providers::internal::auth::AuthError;
use crate::providers::openrouter::extension::EmptyFallbacks;
use crate::providers::registry::{ConnectError, RefError, SelectionError};
use crate::serve::SinkClosed;
use crate::streaming::SequenceError;
use crate::tool::ToolExecutionError;
use crate::tool::builtin::think::ThinkError;
use crate::tool::context::ToolContextError;
use crate::vector_store::request::FilterError;

/// `From<$error> for RigError`, classified as `$kind`.
macro_rules! classify {
    ($kind:expr => $($error:ty),+ $(,)?) => {$(
        impl From<$error> for RigError {
            fn from(error: $error) -> Self {
                Self::from_error($kind, &error)
            }
        }
    )+};
}

classify!(ErrorKind::Configuration =>
    EnvError,
    UnknownReasoning,
    SelectionError,
    RefError,
    ConnectError,
    NotFound,
    OverrideErrors,
    CatalogError,
);

classify!(ErrorKind::Request =>
    EmptyFallbacks,
    EmptyToolName,
    EmptyCallId,
    ParseIdError,
    ParseConversationIdError,
    PairingError,
    AnswerError,
    ApplyError,
    BranchError,
    OptionsError,
);

classify!(ErrorKind::Response => SequenceError);

classify!(ErrorKind::Cancelled => SinkClosed);

classify!(ErrorKind::Other => EmbedError, FileLoaderError);

#[cfg(feature = "pdf")]
classify!(ErrorKind::Other => crate::loaders::pdf::PdfLoaderError);

#[cfg(feature = "epub")]
classify!(ErrorKind::Other =>
    crate::loaders::epub::EpubLoaderError,
    crate::loaders::epub::XmlProcessingError,
);

impl From<ProviderError> for RigError {
    fn from(error: ProviderError) -> Self {
        Self::from(&error)
    }
}

/// As [`ErrorKind::Json`], the way a provider's malformed reply reports.
impl From<serde_json::Error> for RigError {
    fn from(error: serde_json::Error) -> Self {
        ProviderError::from(error).into()
    }
}

/// As [`ErrorKind::Url`].
impl From<url::ParseError> for RigError {
    fn from(error: url::ParseError) -> Self {
        ProviderError::from(error).into()
    }
}

/// As the provider's reply a caller reads from a failed call.
impl From<ProviderResponseError> for RigError {
    fn from(response: ProviderResponseError) -> Self {
        ProviderError::ProviderResponse(response).into()
    }
}

/// As the refusal a caller reads from a failed call, with its detail.
impl From<UnsupportedOption> for RigError {
    fn from(option: UnsupportedOption) -> Self {
        ProviderError::UnsupportedOption(option).into()
    }
}

impl From<EncodeError> for RigError {
    fn from(error: EncodeError) -> Self {
        ProviderError::from(error).into()
    }
}

impl From<MessageError> for RigError {
    fn from(error: MessageError) -> Self {
        ProviderError::from(error).into()
    }
}

impl From<http_client::Error> for RigError {
    fn from(error: http_client::Error) -> Self {
        ProviderError::from(error).into()
    }
}

impl From<FilterError> for RigError {
    fn from(error: FilterError) -> Self {
        crate::vector_store::VectorStoreError::from(error).into()
    }
}

impl From<ToolContextError> for RigError {
    fn from(error: ToolContextError) -> Self {
        ToolExecutionError::from(error).into()
    }
}

/// As the tool runner reports a failed tool.
impl From<ThinkError> for RigError {
    fn from(error: ThinkError) -> Self {
        ToolExecutionError::from_error(error).into()
    }
}

/// A transport or HTTP failure reports as the provider's would; a malformed
/// reply as [`ErrorKind::Json`].
impl From<AuthError> for RigError {
    fn from(error: AuthError) -> Self {
        match error {
            AuthError::Http(error) => error.into(),
            AuthError::Json(_) => Self::from_error(ErrorKind::Json, &error),
            AuthError::Message(_) | AuthError::Io(_) => Self::from_error(ErrorKind::Other, &error),
        }
    }
}

/// Every refusal is a request the caller must change; any other failure is
/// the one the dry run met.
impl From<CheckError> for RigError {
    fn from(error: CheckError) -> Self {
        match error {
            CheckError::Invalid(error) => error,
            CheckError::Unsupported(_) => Self::from_error(ErrorKind::Request, &error),
        }
    }
}
