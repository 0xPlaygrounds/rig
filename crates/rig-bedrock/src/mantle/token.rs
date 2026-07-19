//! Bedrock Mantle short-term IAM bearer token generation.
//!
//! Token algorithm matches the official AWS Bedrock token generator:
//! <https://github.com/aws/aws-bedrock-token-generator-python>
//!
//! 1. Presign `POST https://bedrock.amazonaws.com/?Action=CallWithBearerToken`
//!    with SigV4 query params, service=`bedrock`, expires=43200s
//! 2. Strip the URL scheme (`https://` / `http://`)
//! 3. Append `&Version=1` (not part of the signed payload)
//! 4. Base64-encode the UTF-8 string
//! 5. Prefix `bedrock-api-key-`
//! 6. Send as `Authorization: Bearer <token>` against Mantle

use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use aws_config::{BehaviorVersion, Region};
use aws_credential_types::provider::ProvideCredentials;
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    sign, SignableBody, SignableRequest, SignatureLocation, SigningSettings,
};
use aws_sigv4::sign::v4;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;

use super::MantleError;

/// Prefix applied to every Mantle short-term API key.
const AUTH_PREFIX: &str = "bedrock-api-key-";
/// Appended after SigV4 signing; not included in the signed payload.
const TOKEN_VERSION_SUFFIX: &str = "&Version=1";
/// Token lifetime: 12 hours.
const TOKEN_DURATION_SECS: u64 = 43_200;
/// Refresh when less than this remains on the 12h token.
const REFRESH_BUFFER: Duration = Duration::from_secs(3_600);
const TOKEN_HOST: &str = "bedrock.amazonaws.com";
const TOKEN_URL: &str = "https://bedrock.amazonaws.com/?Action=CallWithBearerToken";
const SERVICE_NAME: &str = "bedrock";

struct CachedToken {
    region: String,
    token: String,
    /// Wall-clock instant after which we must mint a new token.
    refresh_after: Instant,
}

static TOKEN_CACHE: Mutex<Option<CachedToken>> = Mutex::new(None);

/// Format a signed (scheme-stripped) presigned URL into a Bedrock API key token.
///
/// `presigned_without_scheme` is e.g. `bedrock.amazonaws.com/?Action=...&X-Amz-...`.
/// Appends `&Version=1` (not part of the SigV4 payload), base64-encodes, and prefixes
/// with `bedrock-api-key-`.
pub fn format_api_key_token(presigned_without_scheme: &str) -> String {
    let with_version = format!("{presigned_without_scheme}{TOKEN_VERSION_SUFFIX}");
    let encoded = BASE64_STANDARD.encode(with_version.as_bytes());
    format!("{AUTH_PREFIX}{encoded}")
}

/// Generate a short-term Bedrock bearer token from explicit credentials.
///
/// This is the pure signing path and is suitable for unit tests with static keys.
pub fn generate_token_from_credentials(
    credentials: &Credentials,
    region: &str,
) -> Result<String, MantleError> {
    let identity = credentials.clone().into();

    let mut settings = SigningSettings::default();
    settings.signature_location = SignatureLocation::QueryParams;
    settings.expires_in = Some(Duration::from_secs(TOKEN_DURATION_SECS));

    let signing_params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(SERVICE_NAME)
        .time(SystemTime::now())
        .settings(settings)
        .build()
        .map_err(|e| MantleError::Token(format!("signing params: {e}")))?
        .into();

    let signable = SignableRequest::new(
        "POST",
        TOKEN_URL,
        std::iter::once(("host", TOKEN_HOST)),
        SignableBody::Bytes(&[]),
    )
    .map_err(|e| MantleError::Token(format!("signable request: {e}")))?;

    let (instructions, _signature) = sign(signable, &signing_params)
        .map_err(|e| MantleError::Token(format!("sign: {e}")))?
        .into_parts();

    let mut req = http::Request::builder()
        .method("POST")
        .uri(TOKEN_URL)
        .header("host", TOKEN_HOST)
        .body(())
        .map_err(|e| MantleError::Token(format!("http request: {e}")))?;

    instructions.apply_to_request_http1x(&mut req);

    let uri = req.uri().to_string();
    let without_scheme = uri
        .strip_prefix("https://")
        .or_else(|| uri.strip_prefix("http://"))
        .unwrap_or(uri.as_str());

    Ok(format_api_key_token(without_scheme))
}

/// Generate (or return a cached) short-term Bedrock API key from the default AWS
/// credential chain.
///
/// Tokens are cached process-wide for the caller's region and refreshed one hour
/// before the 12-hour expiry. If the cache lock is poisoned, a new token is minted.
pub async fn generate_short_term_token(region: &str) -> Result<String, MantleError> {
    if let Ok(guard) = TOKEN_CACHE.lock()
        && let Some(cached) = guard.as_ref()
        && cached.region == region
        && Instant::now() < cached.refresh_after
    {
        return Ok(cached.token.clone());
    }

    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(region.to_string()))
        .load()
        .await;

    let provider = config.credentials_provider().ok_or_else(|| {
        MantleError::Credentials(
            "no AWS credentials provider available for Bedrock Mantle token".into(),
        )
    })?;

    let credentials = provider.provide_credentials().await.map_err(|e| {
        MantleError::Credentials(format!(
            "failed to resolve AWS credentials for Bedrock Mantle token: {e}"
        ))
    })?;

    let token = generate_token_from_credentials(&credentials, region)?;

    // Cache for (token TTL - refresh buffer) so we mint before the 12h expiry.
    let refresh_after =
        Instant::now() + Duration::from_secs(TOKEN_DURATION_SECS).saturating_sub(REFRESH_BUFFER);

    if let Ok(mut guard) = TOKEN_CACHE.lock() {
        *guard = Some(CachedToken {
            region: region.to_string(),
            token: token.clone(),
            refresh_after,
        });
    }

    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_api_key_token_prefix_and_version() {
        let token = format_api_key_token(
            "bedrock.amazonaws.com/?Action=CallWithBearerToken&X-Amz-Algorithm=AWS4-HMAC-SHA256",
        );
        assert!(token.starts_with(AUTH_PREFIX));
        let b64 = &token[AUTH_PREFIX.len()..];
        let decoded = BASE64_STANDARD.decode(b64).unwrap();
        let s = String::from_utf8(decoded).unwrap();
        assert!(s.starts_with("bedrock.amazonaws.com/?Action=CallWithBearerToken"));
        assert!(s.ends_with("&Version=1"));
    }

    #[test]
    fn generate_token_from_static_credentials() {
        let creds = Credentials::new(
            "AKIATESTACCESSKEYID",
            "testsecretaccesskey",
            None,
            None,
            "unit-test",
        );
        let token = generate_token_from_credentials(&creds, "us-east-1").expect("token");
        assert!(
            token.starts_with(AUTH_PREFIX),
            "token must start with {AUTH_PREFIX}, got {}",
            &token[..token.len().min(32)]
        );

        let b64 = &token[AUTH_PREFIX.len()..];
        let decoded = BASE64_STANDARD.decode(b64).expect("base64");
        let s = String::from_utf8(decoded).expect("utf8");
        assert!(
            s.starts_with("bedrock.amazonaws.com/?Action=CallWithBearerToken"),
            "decoded={s}"
        );
        assert!(s.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"), "decoded={s}");
        assert!(s.contains("X-Amz-Signature="), "decoded={s}");
        assert!(s.contains("X-Amz-Expires=43200"), "decoded={s}");
        assert!(s.ends_with("&Version=1"), "decoded={s}");
    }
}
