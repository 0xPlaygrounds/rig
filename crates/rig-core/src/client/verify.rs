//! Credential verification reads a 401 or 403 reply as a verdict on the
//! credential rather than as a provider failure.

use crate::error::ProviderError;

/// Reclassifies a 401 or 403 reply as [`ProviderError::InvalidAuthentication`],
/// keeping the reply. Other failures are unchanged.
pub(crate) fn authentication(error: ProviderError) -> ProviderError {
    match error {
        ProviderError::ProviderResponse(response)
            if matches!(
                response.status,
                Some(http::StatusCode::UNAUTHORIZED | http::StatusCode::FORBIDDEN)
            ) =>
        {
            ProviderError::InvalidAuthentication(response)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests;
