//! Small, bounded diagnostics; credential-bearing transport fields are never copied.

const LIMIT: usize = 512;

pub(super) fn request_secrets<B>(request: &http::Request<B>) -> Vec<String> {
    let mut secrets = Vec::new();
    for name in ["authorization", "x-api-key", "api-key", "x-goog-api-key"] {
        for value in request
            .headers()
            .get_all(name)
            .iter()
            .filter_map(|v| v.to_str().ok())
        {
            secrets.push(value.to_owned());
            if let Some((_, token)) = value.split_once(' ') {
                secrets.push(token.to_owned());
            }
        }
    }
    secrets.extend(url_secrets(&request.uri().to_string()));
    secrets.retain(|s| !s.is_empty());
    secrets
}

pub(super) fn url_secrets(value: &str) -> Vec<String> {
    let mut secrets = Vec::new();
    let parsed = url::Url::parse(value).ok();
    if let Some(url) = &parsed {
        secrets.extend([url.username().to_owned(), normalized(url.username())]);
        if let Some(password) = url.password() {
            secrets.extend([password.to_owned(), normalized(password)]);
        }
    }
    // Origin-form HTTP request URIs have a query but no absolute URL. Keep raw
    // and form-decoded values: '+' is literal in userinfo, but a space in queries.
    let query = parsed.as_ref().and_then(url::Url::query).or_else(|| {
        value
            .split_once('?')
            .map(|(_, query)| query.split('#').next().unwrap_or_default())
    });
    for pair in query.unwrap_or_default().split('&') {
        let Some((name, decoded)) = url::form_urlencoded::parse(pair.as_bytes()).next() else {
            continue;
        };
        if matches!(
            name.to_ascii_lowercase().as_str(),
            "key" | "api_key" | "api-key" | "token" | "access_token"
        ) {
            if let Some((_, raw)) = pair.split_once('=') {
                secrets.push(raw.to_owned());
            }
            secrets.push(decoded.into_owned());
        }
    }
    secrets.retain(|secret| !secret.is_empty());
    secrets.sort();
    secrets.dedup();
    secrets
}

// Decode percent escapes only for comparison; preserve the original scrubbed
// diagnostic when safe. Unlike form decoding, '+' remains a literal character.
fn normalized(value: &str) -> String {
    let value: String = value.chars().filter(|c| !c.is_control()).collect();
    let bytes = value.as_bytes();
    let mut decoded = Vec::new();
    let mut index = 0;
    while let Some(&byte) = bytes.get(index) {
        if byte == b'%'
            && let (Some(high), Some(low)) = (
                bytes
                    .get(index + 1)
                    .and_then(|byte| char::from(*byte).to_digit(16)),
                bytes
                    .get(index + 2)
                    .and_then(|byte| char::from(*byte).to_digit(16)),
            )
        {
            decoded.push((high * 16 + low) as u8);
            index += 3;
        } else {
            decoded.push(byte);
            index += 1;
        }
    }
    String::from_utf8_lossy(&decoded)
        .chars()
        .filter(|c| !c.is_control())
        .collect()
}

pub(super) fn text(value: &str, secrets: &[String]) -> String {
    // Do not truncate a credential and accidentally persist a useful prefix.
    if value.len() > LIMIT {
        return "[truncated]".into();
    }
    let value: String = value.chars().filter(|c| !c.is_control()).collect();
    let comparable = normalized(&value);
    let lower = comparable.to_ascii_lowercase();
    if secrets.iter().any(|secret| {
        let raw: String = secret.chars().filter(|c| !c.is_control()).collect();
        let decoded = normalized(secret);
        [raw, decoded].iter().any(|secret| {
            !secret.is_empty() && (value.contains(secret) || comparable.contains(secret))
        })
    }) || [
        "bearer ",
        "api_key",
        "api-key",
        "access_token",
        "key=",
        "token=",
        "authorization",
        "sk-",
        "aiza",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return "[redacted]".into();
    }
    value
}

pub(super) fn headers(
    headers: &http::HeaderMap,
    secrets: &[String],
) -> std::collections::BTreeMap<String, String> {
    // Fixed names bound both cardinality and exposure. Never persist arbitrary
    // x-* headers, cookies, authentication headers, or the changing Date header.
    [
        "retry-after",
        "request-id",
        "x-request-id",
        "x-goog-request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ]
    .into_iter()
    .filter_map(|name| {
        let value = headers.get(name)?.to_str().ok()?;
        Some((name.to_owned(), text(value, secrets)))
    })
    .collect()
}
