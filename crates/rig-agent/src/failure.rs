//! Failures the agent runtime detects itself rather than reads from a
//! provider. Each reads as a wire's failure of the same kind does, so one
//! failure has one message on every path.

use rig_core::error::{ErrorKind, RigError};

/// A reply rig cannot use.
pub(crate) fn response(message: impl std::fmt::Display) -> RigError {
    RigError::new(ErrorKind::Response, format!("ResponseError: {message}"))
}

/// A request rig will not send, with the message as its source.
pub(crate) fn request(message: impl std::fmt::Display) -> RigError {
    let message = message.to_string();
    RigError {
        source_chain: vec![message.clone()],
        ..RigError::new(ErrorKind::Request, format!("RequestError: {message}"))
    }
}

/// A request rig will not send because of `cause`, which leads the source
/// chain.
pub(crate) fn request_caused_by(cause: impl std::error::Error) -> RigError {
    let cause = RigError::other(cause);
    RigError {
        source_chain: std::iter::once(cause.message.clone())
            .chain(cause.source_chain)
            .collect(),
        ..RigError::new(
            ErrorKind::Request,
            format!("RequestError: {}", cause.message),
        )
    }
}

#[cfg(test)]
mod tests;
