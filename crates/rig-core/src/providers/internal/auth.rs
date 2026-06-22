//! Shared helpers for provider-native authentication flows.

use crate::http_client::{self, HttpClientExt};
use http::{HeaderValue, Method, header::HeaderName};

pub(crate) type AuthHeader = (HeaderName, HeaderValue);

#[derive(Debug, thiserror::Error)]
pub(crate) enum JsonAuthRequestError {
    #[error(transparent)]
    Transport(#[from] http_client::Error),
    #[error("{provider} auth response could not be parsed: {source}; body: {body}")]
    Parse {
        provider: &'static str,
        source: serde_json::Error,
        body: String,
    },
}

pub(crate) async fn send_json_request<H, T>(
    http_client: &H,
    provider: &'static str,
    method: Method,
    uri: &str,
    headers: impl IntoIterator<Item = AuthHeader>,
    body: Vec<u8>,
    default_json_content_type: bool,
) -> Result<T, JsonAuthRequestError>
where
    H: HttpClientExt,
    T: serde::de::DeserializeOwned,
{
    let mut req = http::Request::builder().method(method).uri(uri);
    let mut has_content_type = false;
    for (name, value) in headers {
        if name == http::header::CONTENT_TYPE {
            has_content_type = true;
        }
        req = req.header(name, value);
    }
    if default_json_content_type && !has_content_type {
        req = req.header(http::header::CONTENT_TYPE, "application/json");
    }

    let req = req.body(body).map_err(http_client::Error::Protocol)?;
    let response = http_client.send::<_, Vec<u8>>(req).await?;
    let body = response.into_body().await?;
    serde_json::from_slice(&body).map_err(|source| JsonAuthRequestError::Parse {
        provider,
        source,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}
