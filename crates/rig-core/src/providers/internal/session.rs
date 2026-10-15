//! The conversation a gateway routes and caches a completion by.

use crate::completion::CompletionRequest;

/// The session `request` names in a gateway's session header: its
/// [`cache_key`](crate::completion::GenerationOptions::cache_key), else a
/// fresh id.
pub(crate) fn session_id(request: &CompletionRequest) -> String {
    request
        .options
        .cache_key
        .clone()
        .unwrap_or_else(crate::id::generate)
}
