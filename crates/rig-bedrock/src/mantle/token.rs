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

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, SystemTime};

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
/// Refresh when less than this remains on the 12h token.
const REFRESH_BUFFER: Duration = Duration::from_secs(3_600);
const TOKEN_HOST: &str = "bedrock.amazonaws.com";
const TOKEN_URL: &str = "https://bedrock.amazonaws.com/?Action=CallWithBearerToken";
const SERVICE_NAME: &str = "bedrock";

struct CachedToken {
    token: String,
    /// Wall-clock time after which we must mint a new token.
    ///
    /// Uses [`SystemTime`] (not [`std::time::Instant`]) because the token's real
    /// expiry is wall-clock (`X-Amz-Date` + `X-Amz-Expires`). Monotonic clocks
    /// pause across host suspend on some platforms and would serve expired tokens.
    refresh_after: SystemTime,
}

type CacheKey = (String, String);

static TOKEN_CACHE: LazyLock<Mutex<HashMap<CacheKey, CachedToken>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Format a signed (scheme-stripped) presigned URL into a Bedrock API key token.
///
/// `presigned_without_scheme` is e.g. `bedrock.amazonaws.com/?Action=...&X-Amz-...`.
/// Appends `&Version=1` (not part of the SigV4 payload), base64-encodes, and prefixes
/// with `bedrock-api-key-`.
pub(crate) fn format_api_key_token(presigned_without_scheme: &str) -> String {
    let with_version = format!("{presigned_without_scheme}{TOKEN_VERSION_SUFFIX}");
    let encoded = BASE64_STANDARD.encode(with_version.as_bytes());
    format!("{AUTH_PREFIX}{encoded}")
}

/// Effective short-term token lifetime for these credentials.
///
/// AWS short-term Bedrock API keys last at most 12 hours ([`TOKEN_TTL`]), but
/// never longer than the source AWS credential session (SSO, AssumeRole, ECS,
/// EC2 instance profile, etc.). Returns `Duration::ZERO` when the session has
/// already expired.
pub fn effective_token_ttl(credentials: &Credentials) -> Duration {
    match credentials.expiry() {
        Some(exp) => exp
            .duration_since(SystemTime::now())
            .unwrap_or(Duration::ZERO)
            .min(TOKEN_TTL),
        None => TOKEN_TTL,
    }
}

/// Wall-clock time when a cached mint for `ttl` should be refreshed.
///
/// Uses a 1h buffer for long sessions; for short sessions (e.g. 1h SSO) remints
/// when ~20% of the TTL remains (minimum 30s buffer when TTL allows).
pub(crate) fn refresh_after_from_ttl(ttl: Duration) -> SystemTime {
    let buffer = if ttl > REFRESH_BUFFER.saturating_mul(2) {
        REFRESH_BUFFER
    } else if ttl > Duration::from_secs(60) {
        (ttl / 5).max(Duration::from_secs(30))
    } else {
        Duration::from_secs(0)
    };
    SystemTime::now() + ttl.saturating_sub(buffer)
}

fn token_err(source: impl std::error::Error + Send + Sync + 'static) -> MantleError {
    MantleError::Token {
        source: Box::new(source),
    }
}

/// Generate a short-term Bedrock bearer token from explicit credentials.
///
/// This is the pure signing path and is suitable for unit tests with static keys.
/// `X-Amz-Expires` is capped by [`effective_token_ttl`].
pub fn generate_token_from_credentials(
    credentials: &Credentials,
    region: &str,
) -> Result<String, MantleError> {
    let ttl = effective_token_ttl(credentials);
    if ttl.is_zero() {
        return Err(MantleError::CredentialsExpired);
    }

    let identity = credentials.clone().into();

    let mut settings = SigningSettings::default();
    settings.signature_location = SignatureLocation::QueryParams;
    settings.expires_in = Some(ttl);

    let signing_params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(SERVICE_NAME)
        .time(SystemTime::now())
        .settings(settings)
        .build()
        .map_err(token_err)?
        .into();

    let signable = SignableRequest::new(
        "POST",
        TOKEN_URL,
        std::iter::once(("host", TOKEN_HOST)),
        SignableBody::Bytes(&[]),
    )
    .map_err(token_err)?;

    let (instructions, _signature) = sign(signable, &signing_params)
        .map_err(token_err)?
        .into_parts();

    let mut req = http::Request::builder()
        .method("POST")
        .uri(TOKEN_URL)
        .header("host", TOKEN_HOST)
        .body(())
        .map_err(token_err)?;

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
/// Tokens are cached process-wide in a map keyed by `(region, access_key_id)`.
/// Refresh timing uses [`effective_token_ttl`] (min of 12h and source credential
/// expiry). If the cache lock is poisoned, a new token is minted.
///
/// The minted token is **snapshotted** into the Mantle HTTP client at build time.
/// Long-lived processes must rebuild the client before the effective TTL elapses
/// (often much sooner than 12h when using SSO / AssumeRole sessions).
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
    let key = (region.to_string(), access_key_id);

    if let Ok(guard) = TOKEN_CACHE.lock()
        && let Some(cached) = guard.get(&key)
        && cache_fresh(cached)
    {
        return Ok(cached.token.clone());
    }

    let ttl = effective_token_ttl(&credentials);
    let token = generate_token_from_credentials(&credentials, region)?;
    let refresh_after = refresh_after_from_ttl(ttl);

    if let Ok(mut guard) = TOKEN_CACHE.lock() {
        guard.insert(
            key,
            CachedToken {
                token: token.clone(),
                refresh_after,
            },
        );
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

    let provider = config
        .credentials_provider()
        .ok_or(MantleError::NoCredentialsProvider)?;

    provider
        .provide_credentials()
        .await
        .map_err(|source| MantleError::Credentials {
            source: Box::new(source),
        })
}

fn cache_fresh(cached: &CachedToken) -> bool {
    SystemTime::now() < cached.refresh_after
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
        // Static long-lived keys use full TOKEN_TTL
        assert!(s.contains("X-Amz-Expires=43200"), "decoded={s}");
        assert!(s.ends_with("&Version=1"), "decoded={s}");
    }

    #[test]
    fn generate_token_respects_short_credential_expiry() {
        let exp = SystemTime::now() + Duration::from_secs(1_800);
        let creds = Credentials::new(
            "AKIATESTACCESSKEYID",
            "testsecretaccesskey",
            None,
            Some(exp),
            "unit-test",
        );
        let token = generate_token_from_credentials(&creds, "us-east-1").expect("token");
        let b64 = &token[AUTH_PREFIX.len()..];
        let decoded = BASE64_STANDARD.decode(b64).expect("base64");
        let s = String::from_utf8(decoded).expect("utf8");
        // Expires should be ~1800s, not 43200
        assert!(
            !s.contains("X-Amz-Expires=43200"),
            "should not use full 12h when session is shorter: {s}"
        );
        assert!(s.contains("X-Amz-Expires="), "decoded={s}");
    }

    #[test]
    fn token_ttl_is_twelve_hours() {
        assert_eq!(TOKEN_TTL, Duration::from_secs(43_200));
        assert_eq!(TOKEN_TTL.as_secs(), 43_200);
    }

    #[test]
    fn effective_token_ttl_uses_credential_expiry_when_shorter() {
        let exp = SystemTime::now() + Duration::from_secs(3_600);
        let creds = Credentials::new(
            "AKIATEST",
            "secret",
            None,
            Some(exp),
            "unit-test",
        );
        let ttl = effective_token_ttl(&creds);
        assert!(ttl <= Duration::from_secs(3_600));
        assert!(ttl > Duration::from_secs(3_500));
        assert!(ttl < TOKEN_TTL);
    }

    #[test]
    fn effective_token_ttl_caps_at_twelve_hours() {
        let exp = SystemTime::now() + Duration::from_secs(86_400);
        let creds = Credentials::new(
            "AKIATEST",
            "secret",
            None,
            Some(exp),
            "unit-test",
        );
        assert_eq!(effective_token_ttl(&creds), TOKEN_TTL);
    }

    #[test]
    fn effective_token_ttl_without_expiry_is_twelve_hours() {
        let creds = Credentials::new("AKIATEST", "secret", None, None, "unit-test");
        assert_eq!(effective_token_ttl(&creds), TOKEN_TTL);
    }

    #[test]
    fn refresh_after_short_session_before_expiry() {
        let ttl = Duration::from_secs(3_600);
        let after = refresh_after_from_ttl(ttl);
        let remaining = after
            .duration_since(SystemTime::now())
            .expect("refresh_after is in the future");
        // 20% buffer => remint after ~2880s; allow clock skew
        assert!(remaining < ttl);
        assert!(remaining > Duration::from_secs(2_000));
    }

    #[test]
    fn cache_fresh_rejects_expired_entry() {
        let expired = CachedToken {
            token: "token".into(),
            refresh_after: SystemTime::now() - Duration::from_secs(1),
        };
        assert!(!cache_fresh(&expired));
        let fresh = CachedToken {
            token: "token".into(),
            refresh_after: SystemTime::now() + Duration::from_secs(60),
        };
        assert!(cache_fresh(&fresh));
    }

    #[test]
    fn cache_map_holds_multiple_keys() {
        let mut map: HashMap<CacheKey, CachedToken> = HashMap::new();
        map.insert(
            ("us-east-1".to_string(), "AKIA_A".to_string()),
            CachedToken {
                token: "token-a".into(),
                refresh_after: SystemTime::now() + Duration::from_secs(60),
            },
        );
        map.insert(
            ("us-west-2".to_string(), "AKIA_A".to_string()),
            CachedToken {
                token: "token-b".into(),
                refresh_after: SystemTime::now() + Duration::from_secs(60),
            },
        );
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get(&("us-east-1".to_string(), "AKIA_A".to_string()))
                .map(|c| c.token.as_str()),
            Some("token-a")
        );
        assert_eq!(
            map.get(&("us-west-2".to_string(), "AKIA_A".to_string()))
                .map(|c| c.token.as_str()),
            Some("token-b")
        );
    }

    #[test]
    fn different_access_keys_mint_different_tokens() {
        let a = Credentials::new("AKIA_A", "secret-a", None, None, "unit-test");
        let b = Credentials::new("AKIA_B", "secret-b", None, None, "unit-test");
        let token_a = generate_token_from_credentials(&a, "us-east-1").expect("token a");
        let token_b = generate_token_from_credentials(&b, "us-east-1").expect("token b");
        assert_ne!(token_a, token_b);
    }

    #[test]
    fn expired_credentials_error_variant() {
        let exp = SystemTime::now() - Duration::from_secs(1);
        let creds = Credentials::new("AKIATEST", "secret", None, Some(exp), "unit-test");
        let err = generate_token_from_credentials(&creds, "us-east-1").unwrap_err();
        assert!(matches!(err, MantleError::CredentialsExpired));
    }
}
