//! Credential verification reads a 401 or 403 reply as a verdict on the
//! credential rather than as a provider failure.

use crate::driver::{Model, Transport};
use crate::error::ProviderError;
use crate::wire::Wire;

impl<W, T> Model<W, T>
where
    W: Wire<Op = crate::operation::Verify> + Clone,
    T: Transport<W>,
{
    /// Check that the provider accepts the configured credentials. A 401 or
    /// 403 reply is [`ProviderError::InvalidAuthentication`].
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.call((), None).await.map_err(authentication)
    }
}

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
