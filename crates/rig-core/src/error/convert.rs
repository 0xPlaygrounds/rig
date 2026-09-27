//! Every rig-core error a caller can receive converts into [`RigError`], so
//! application code propagates it with `?`. A type rig already reports
//! through another error keeps that report.

use super::{ErrorKind, ProviderError, RigError};
use crate::client::env::EnvError;
use crate::completion::message::MessageError;
use crate::embeddings::EmbedError;
use crate::error::EncodeError;
use crate::http_client;
use crate::id::ParseIdError;
use crate::loaders::file::FileLoaderError;
use crate::providers::internal::auth::AuthError;
use crate::providers::registry::{RefError, SelectionError};
use crate::tool::ToolExecutionError;
use crate::tool::builtin::think::ThinkError;
use crate::tool::context::ToolContextError;
use crate::transcript::TranscriptError;
use crate::vector_store::VectorStoreError;
use crate::vector_store::request::FilterError;

/// A variable is missing or unusable: the caller's setup.
impl From<EnvError> for RigError {
    fn from(error: EnvError) -> Self {
        Self::classified(ErrorKind::Configuration, &error)
    }
}

/// A transport failure, or a reply the sign-in endpoint refused, reports as
/// the transport's error does. Invalid JSON in a reply is
/// [`ErrorKind::Json`]; a sign-in the provider declined, and a token cache
/// that could not be read or written, are [`ErrorKind::Other`].
impl From<AuthError> for RigError {
    fn from(error: AuthError) -> Self {
        match error {
            AuthError::Http(error) => Self::from(error),
            AuthError::Json(_) => Self::classified(ErrorKind::Json, &error),
            AuthError::Message(_) | AuthError::Io(_) => Self::classified(ErrorKind::Other, &error),
        }
    }
}

/// The reference names no provider or model this build serves.
impl From<RefError> for RigError {
    fn from(error: RefError) -> Self {
        match error {
            RefError::Selection(error) => Self::from(error),
            RefError::EmptyModel | RefError::NoModel { .. } => {
                Self::classified(ErrorKind::Configuration, &error)
            }
        }
    }
}

/// The selection names no provider this build serves.
impl From<SelectionError> for RigError {
    fn from(error: SelectionError) -> Self {
        Self::classified(ErrorKind::Configuration, &error)
    }
}

/// The text is not an id.
impl From<ParseIdError> for RigError {
    fn from(error: ParseIdError) -> Self {
        Self::classified(ErrorKind::Configuration, &error)
    }
}

/// As a wire reports it: a request that could not be built.
impl From<MessageError> for RigError {
    fn from(error: MessageError) -> Self {
        Self::from(ProviderError::from(error))
    }
}

/// As a wire reports it: a request that could not be built.
impl From<EncodeError> for RigError {
    fn from(error: EncodeError) -> Self {
        Self::from(ProviderError::from(error))
    }
}

/// As a transport failure reports: a reply the server made is
/// [`ErrorKind::ProviderResponse`], anything else [`ErrorKind::Http`].
impl From<http_client::Error> for RigError {
    fn from(error: http_client::Error) -> Self {
        Self::from(ProviderError::from(error))
    }
}

/// As a tool reports it: the tool failed.
impl From<ToolContextError> for RigError {
    fn from(error: ToolContextError) -> Self {
        Self::from(ToolExecutionError::from(error))
    }
}

/// As the think tool's failure reports.
impl From<ThinkError> for RigError {
    fn from(error: ThinkError) -> Self {
        Self::from(ToolExecutionError::from_error(error))
    }
}

/// The history cannot be sent as a conversation.
impl From<TranscriptError> for RigError {
    fn from(error: TranscriptError) -> Self {
        Self::classified(ErrorKind::Request, &error)
    }
}

/// A document's text could not be extracted for embedding.
impl From<EmbedError> for RigError {
    fn from(error: EmbedError) -> Self {
        Self::classified(ErrorKind::Other, &error)
    }
}

/// As a vector store reports it: a query that could not be built.
impl From<FilterError> for RigError {
    fn from(error: FilterError) -> Self {
        Self::from(VectorStoreError::from(error))
    }
}

/// Files could not be listed or read.
impl From<FileLoaderError> for RigError {
    fn from(error: FileLoaderError) -> Self {
        Self::classified(ErrorKind::Other, &error)
    }
}

/// A document could not be read or parsed.
#[cfg(feature = "pdf")]
impl From<crate::loaders::pdf::PdfLoaderError> for RigError {
    fn from(error: crate::loaders::pdf::PdfLoaderError) -> Self {
        match error {
            crate::loaders::pdf::PdfLoaderError::FileLoaderError(error) => Self::from(error),
            other => Self::classified(ErrorKind::Other, &other),
        }
    }
}

/// A book could not be read or its text extracted.
#[cfg(feature = "epub")]
impl From<crate::loaders::epub::EpubLoaderError> for RigError {
    fn from(error: crate::loaders::epub::EpubLoaderError) -> Self {
        match error {
            crate::loaders::epub::EpubLoaderError::FileLoaderError(error) => Self::from(error),
            other => Self::classified(ErrorKind::Other, &other),
        }
    }
}

#[cfg(test)]
mod tests;
