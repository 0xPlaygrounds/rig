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
/// Token lifetime: 12 hours (matches AWS short-term Bedrock API keys).
pub const TOKEN_TTL: Duration = Duration::from_secs(43_200);
/// Token lifetime in seconds (same as [`TOKEN_TTL`]).
pub const TOKEN_TTL_SECS: u64 = 43_200;
/// Refresh when less than this remains on the 12h token.
const REFRESH_BUFFER: Duration = Duration::from_secs(3_600);
const TOKEN_HOST: &str = "bedrock.amazonaws.com";
const TOKEN_URL: &str = "https://bedrock.amazonaws.com/?Action=CallWithBearerToken";
const SERVICE_NAME: &str = "bedrock";

struct CachedToken {
    region: String,
    /// Fingerprint from `credentials.access_key_id()` so distinct IAM principals
    /// never share a minted bearer token in the process-wide cache.
    access_key_id: String,
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
    settings.expires_in = Some(TOKEN_TTL);

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
/// Tokens are cached process-wide keyed by `(region, access_key_id)` and refreshed
/// one hour before the 12-hour expiry. If the cache lock is poisoned, a new token
/// is minted.
///
/// The minted token is snapshotted into the OpenAI-compatible client at build time.
/// Long-lived processes should rebuild the client before the 12h TTL elapses.
pub async fn generate_short_term_token(region: &str) -> Result<String, MantleError> {
    generate_short_term_token_with_profile(region, None).await
}

/// Like [`generate_short_term_token`], but optionally pins an AWS shared-config profile.
pub async fn generate_short_term_token_with_profile(
    region: &str,
    profile_name: Option<&str>,
) -> Result<String, MantleError> {
    let credentials = resolve_credentials(region, profile_name).await?;
    let access_key_id = credentials.access_key_id().to_string();

    if let Ok(guard) = TOKEN_CACHE.lock()
        && let Some(cached) = guard.as_ref()
        && cache_matches(cached, region, &access_key_id)
    {
        return Ok(cached.token.clone());
    }

    let token = generate_token_from_credentials(&credentials, region)?;

    // Cache for (token TTL - refresh buffer) so we mint before the 12h expiry.
    let refresh_after = Instant::now() + TOKEN_TTL.saturating_sub(REFRESH_BUFFER);

    if let Ok(mut guard) = TOKEN_CACHE.lock() {
        *guard = Some(CachedToken {
            region: region.to_string(),
            access_key_id,
            token: token.clone(),
            refresh_after,
        });
    }

    Ok(token)
}

async fn resolve_credentials(
    region: &str,
    profile_name: Option<&str>,
) -> Result<Credentials, MantleError> {
    let mut loader = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(region.to_string()));
    if let Some(profile) = profile_name {
        loader = loader.profile_name(profile);
    }
    let config = loader.load().await;

    let provider = config.credentials_provider().ok_or_else(|| {
        MantleError::Credentials(
            "no AWS credentials provider available for Bedrock Mantle token".into(),
        )
    })?;

    provider.provide_credentials().await.map_err(|e| {
        MantleError::Credentials(format!(
            "failed to resolve AWS credentials for Bedrock Mantle token: {e}"
        ))
    })
}

fn cache_matches(cached: &CachedToken, region: &str, access_key_id: &str) -> bool {
    cached.region == region
        && cached.access_key_id == access_key_id
        && Instant::now() < cached.refresh_after
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

    #[test]
    fn token_ttl_is_twelve_hours() {
        assert_eq!(TOKEN_TTL, Duration::from_secs(43_200));
        assert_eq!(TOKEN_TTL_SECS, 43_200);
    }

    #[test]
    fn cache_matches_requires_region_and_access_key() {
        let cached = CachedToken {
            region: "us-east-1".into(),
            access_key_id: "AKIA_A".into(),
            token: "token".into(),
            refresh_after: Instant::now() + Duration::from_secs(60),
        };
        assert!(cache_matches(&cached, "us-east-1", "AKIA_A"));
        assert!(!cache_matches(&cached, "us-west-2", "AKIA_A"));
        assert!(!cache_matches(&cached, "us-east-1", "AKIA_B"));
    }

    #[test]
    fn cache_matches_rejects_expired_entry() {
        let cached = CachedToken {
            region: "us-east-1".into(),
            access_key_id: "AKIA_A".into(),
            token: "token".into(),
            refresh_after: Instant::now() - Duration::from_secs(1),
        };
        assert!(!cache_matches(&cached, "us-east-1", "AKIA_A"));
    }

    #[test]
    fn different_access_keys_mint_different_tokens() {
        let a = Credentials::new("AKIA_A", "secret-a", None, None, "unit-test");
        let b = Credentials::new("AKIA_B", "secret-b", None, None, "unit-test");
        let token_a = generate_token_from_credentials(&a, "us-east-1").expect("token a");
        let token_b = generate_token_from_credentials(&b, "us-east-1").expect("token b");
        assert_ne!(token_a, token_b);
    }
}
