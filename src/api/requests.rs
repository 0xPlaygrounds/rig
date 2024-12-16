use crate::error::Result;
use lazy_static::lazy_static;
use reqwest::multipart::Form;
use reqwest::{header::HeaderMap, Method};
use serde::de::DeserializeOwned;
use std::sync::Arc;

lazy_static! {
    static ref SHARED_CLIENT: Arc<reqwest::Client> = Arc::new(
        reqwest::Client::builder()
            .cookie_store(true)
            .build()
            .expect("Failed to create HTTP client")
    );
}

pub async fn request_api<T>(
    url: &str,
    headers: HeaderMap,
    method: Method,
    body: Option<serde_json::Value>,
) -> Result<(T, HeaderMap)>
where
    T: DeserializeOwned,
{
    let mut request = SHARED_CLIENT.request(method, url).headers(headers);

    if let Some(json_body) = body {
        request = request.json(&json_body);
    }

    let response = request.send().await?;

    if response.status().is_success() {
        let headers = response.headers().clone();
        let text = response.text().await?;
        let parsed: T = serde_json::from_str(&text)?;
        Ok((parsed, headers))
    } else {
        Err(crate::error::TwitterError::Api(format!(
            "Request failed with status: {}",
            response.status()
        )))
    }
}

pub async fn get_guest_token(bearer_token: &str) -> Result<String> {
    let mut headers = HeaderMap::new();
    headers.insert(
        "Authorization",
        format!("Bearer {}", bearer_token).parse().unwrap(),
    );

    let (response, _) = request_api::<serde_json::Value>(
        "https://api.twitter.com/1.1/guest/activate.json",
        headers,
        Method::POST,
        None,
    )
    .await?;

    response
        .get("guest_token")
        .and_then(|token| token.as_str())
        .map(String::from)
        .ok_or_else(|| crate::error::TwitterError::Auth("Failed to get guest token".into()))
}

pub async fn request_multipart_api<T>(
    url: &str,
    headers: HeaderMap,
    form: Form,
) -> Result<(T, HeaderMap)>
where
    T: DeserializeOwned,
{
    let request = SHARED_CLIENT
        .request(Method::POST, url)
        .headers(headers)
        .multipart(form);

    let response = request.send().await?;

    if response.status().is_success() {
        let headers = response.headers().clone();
        let text = response.text().await?;
        let parsed: T = serde_json::from_str(&text)?;
        Ok((parsed, headers))
    } else {
        Err(crate::error::TwitterError::Api(format!(
            "Request failed with status: {}",
            response.status()
        )))
    }
}

pub async fn request_form_api<T>(
    url: &str,
    headers: HeaderMap,
    form_data: Vec<(String, String)>,
) -> Result<(T, HeaderMap)>
where
    T: DeserializeOwned,
{
    let request = SHARED_CLIENT
        .request(Method::POST, url)
        .headers(headers)
        .form(&form_data);

    let response = request.send().await?;

    if response.status().is_success() {
        let headers = response.headers().clone();
        let text = response.text().await?;
        let parsed: T = serde_json::from_str(&text)?;
        Ok((parsed, headers))
    } else {
        Err(crate::error::TwitterError::Api(format!(
            "Request failed with status: {}",
            response.status()
        )))
    }
}
