//! The one scrubber of recorded provider traffic, shared by the cassette
//! engine and the traffic log. It removes credentials (sensitive headers and
//! query parameters, and every secret [`rig_core::redact`] finds), AWS account
//! numbers and home-directory paths, and keeps everything else the provider
//! sent verbatim. A [`Purpose`] selects the few behaviours a replayable
//! fixture needs and a log of live traffic must not have.

use rig_core::redact;
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) const REDACTED: &str = redact::REDACTED;

pub(crate) const SENSITIVE_HEADER_NAMES: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
    // A session cookie, and the ChatGPT account a sign-in belongs to.
    "cookie",
    "chatgpt-account-id",
    "x-goog-api-key",
    "ocp-apim-subscription-key",
    "set-cookie",
    "openai-organization",
    "openai-project",
    "anthropic-organization-id",
    "x-amz-security-token",
    "x-amz-content-sha256",
    "x-amz-date",
    "key",
];

pub(crate) const SENSITIVE_QUERY_PARAMS: &[&str] = &[
    "key",
    "api_key",
    "apikey",
    "access_token",
    "x-amz-credential",
    "x-amz-signature",
    "x-amz-security-token",
];

pub(crate) const VOLATILE_JSON_KEYS: &[&str] = &[
    "completed_at",
    "created",
    "created_at",
    "updated",
    "updated_at",
];

/// Credential material a token exchange can return in a body.
const OAUTH_TOKEN_KEYS: &[&str] = &["access_token", "id_token", "refresh_token"];

/// What a scrubbed exchange is for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Purpose {
    /// A replayable fixture: volatile timestamps are normalized and an OAuth
    /// token becomes a numbered placeholder, so a replay can tell two apart.
    #[default]
    Cassette,
    /// A log of live traffic: timestamps stay as sent and an OAuth token is
    /// [`REDACTED`] like every other secret.
    Traffic,
}

/// Scrubs the text, JSON and SSE bodies of one recording. Placeholders are
/// numbered per scrubber, so one scrubber serves one recording.
#[derive(Default)]
pub(crate) struct Scrubber {
    purpose: Purpose,
    placeholders: BTreeMap<String, String>,
    counters: BTreeMap<&'static str, usize>,
    /// Credentials the recording carried, scrubbed wherever they show up
    /// again, such as an error body that echoes the key; longest first.
    learned: Vec<String>,
}

impl Scrubber {
    /// A scrubber for `purpose`.
    pub(crate) fn new(purpose: Purpose) -> Self {
        Self {
            purpose,
            ..Self::default()
        }
    }

    /// Remembers the credential a header carries when its name is a
    /// credential header's: its value, and an authorization's after its
    /// scheme.
    pub(crate) fn learn_header(&mut self, name: &str, value: &str) {
        if !is_secret_header(name) {
            return;
        }
        self.learn(value);
        if let Some((_, credential)) = value.split_once(' ') {
            self.learn(credential.trim());
        }
    }

    /// Remembers the credentials of `url`: its secret query parameters'
    /// values.
    pub(crate) fn learn_url(&mut self, url: &str) {
        let Some((_, query)) = url.split_once('?') else {
            return;
        };
        for pair in query.split('&') {
            if let Some((name, value)) = pair.split_once('=')
                && contains_case_insensitive(SENSITIVE_QUERY_PARAMS, name)
            {
                self.learn(value);
            }
        }
    }

    /// Remembers `secret`, unless it is too short to tell from ordinary
    /// text or is a redaction already.
    fn learn(&mut self, secret: &str) {
        const SHORTEST: usize = 8;
        if secret.len() < SHORTEST
            || secret == REDACTED
            || is_redacted_placeholder(secret)
            || self.learned.iter().any(|known| known == secret)
        {
            return;
        }
        self.learned.push(secret.to_owned());
        self.learned
            .sort_by_key(|known| std::cmp::Reverse(known.len()));
    }

    /// `url` with its credentials scrubbed.
    pub(crate) fn scrub_url(&mut self, url: &str) -> String {
        self.scrub_text(url)
    }

    /// `body` scrubbed as JSON (written canonically), SSE, or text.
    #[cfg(feature = "http")]
    pub(crate) fn scrub_body(&mut self, body: &str) -> String {
        if let Some(mut json) = canonical_json(body) {
            self.scrub_json_value(None, &mut json);
            return json.to_string();
        }

        if body
            .lines()
            .any(|line| line.trim_start().starts_with("data:"))
        {
            return self.scrub_sse_body(body);
        }

        self.scrub_text(body)
    }

    #[cfg(feature = "http")]
    pub(crate) fn scrub_sse_body(&mut self, body: &str) -> String {
        let mut output = String::with_capacity(body.len());

        for line in body.split_inclusive('\n') {
            let (line_without_newline, newline) = line
                .strip_suffix('\n')
                .map_or((line, ""), |line| (line, "\n"));
            let (line_without_cr, cr) = line_without_newline
                .strip_suffix('\r')
                .map_or((line_without_newline, ""), |line| (line, "\r"));
            let trimmed = line_without_cr.trim_start();
            let indentation = line_without_cr
                .get(..line_without_cr.len() - trimmed.len())
                .unwrap_or_default();

            if let Some(payload) = trimmed.strip_prefix("data:") {
                let payload = payload.trim_start();
                if payload == "[DONE]" {
                    output.push_str(line_without_cr);
                } else if let Some(mut json) = canonical_json(payload) {
                    self.scrub_json_value(None, &mut json);
                    output.push_str(indentation);
                    output.push_str("data: ");
                    output.push_str(&json.to_string());
                } else {
                    output.push_str(&self.scrub_text(line_without_cr));
                }
            } else {
                output.push_str(&self.scrub_text(line_without_cr));
            }

            output.push_str(cr);
            output.push_str(newline);
        }

        output
    }

    pub(crate) fn scrub_json_value(&mut self, key: Option<&str>, value: &mut Value) {
        let key_lower = key.map(str::to_ascii_lowercase);
        let cassette = self.purpose == Purpose::Cassette;

        match value {
            Value::Object(map) => {
                for (key, value) in map {
                    self.scrub_json_value(Some(key), value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    self.scrub_json_value(key, value);
                }
            }
            Value::String(text) => {
                if is_redacted_placeholder(text) {
                    return;
                }

                if let Some(key) = key_lower.as_deref() {
                    if cassette && VOLATILE_JSON_KEYS.contains(&key) {
                        *text = "1970-01-01T00:00:00Z".to_string();
                        return;
                    }

                    if OAUTH_TOKEN_KEYS.contains(&key) {
                        *text = match self.purpose {
                            Purpose::Cassette => self.placeholder(text, "id_"),
                            Purpose::Traffic => REDACTED.to_owned(),
                        };
                        return;
                    }
                }

                *text = self.scrub_text(text);
            }
            Value::Number(_) => {
                if cassette
                    && key_lower
                        .as_deref()
                        .is_some_and(|key| VOLATILE_JSON_KEYS.contains(&key))
                {
                    *value = Value::Number(0.into());
                }
            }
            Value::Bool(_) | Value::Null => {}
        }
    }

    pub(crate) fn scrub_text(&mut self, text: &str) -> String {
        let mut scrubbed = redact::text(text).into_owned();
        for secret in &self.learned {
            if scrubbed.contains(secret.as_str()) {
                scrubbed = scrubbed.replace(secret.as_str(), REDACTED);
            }
        }
        for key in SENSITIVE_QUERY_PARAMS {
            scrubbed = scrub_query_param(&scrubbed, key, REDACTED);
        }
        let scrubbed = self.scrub_aws_account_ids(&scrubbed);
        scrub_local_filesystem_paths(&scrubbed)
    }

    /// Replace 12-digit account-ID segments in ARNs, preserving partition,
    /// service, region, and resource fields.
    fn scrub_aws_account_ids(&mut self, text: &str) -> String {
        const PREFIX: &str = "arn:";
        const ACCOUNT_FIELD: usize = 4;
        const ACCOUNT_LEN: usize = 12;

        let mut output = String::with_capacity(text.len());
        let mut rest = text;

        while let Some(start) = rest.find(PREFIX) {
            let (before, arn) = rest.split_at(start);
            output.push_str(before);
            // An ARN ends at the first character that cannot appear in one;
            // the surrounding JSON quote or comma is the usual terminator.
            let end = arn
                .find(['"', ',', ' ', '\n', '}', ']'])
                .unwrap_or(arn.len());
            let (arn, tail) = arn.split_at(end);

            let mut fields = arn.split(':').map(str::to_string).collect::<Vec<_>>();
            match fields.get(ACCOUNT_FIELD).cloned() {
                Some(account)
                    if account.len() == ACCOUNT_LEN
                        && account.chars().all(|ch| ch.is_ascii_digit()) =>
                {
                    let placeholder = self.placeholder(&account, "account_");
                    if let Some(field) = fields.get_mut(ACCOUNT_FIELD) {
                        *field = placeholder;
                    }
                    output.push_str(&fields.join(":"));
                }
                _ => output.push_str(arn),
            }

            rest = tail;
        }

        output.push_str(rest);
        output
    }

    pub(crate) fn placeholder(&mut self, original: &str, kind: &'static str) -> String {
        // Empty wire values must stay empty rather than acquire invented identifiers.
        if original.is_empty() {
            return String::new();
        }

        if let Some(existing) = self.placeholders.get(original) {
            return existing.clone();
        }

        let counter = self.counters.entry(kind).or_insert(0);
        *counter += 1;
        let placeholder = format!("{kind}REDACTED_{counter}");
        self.placeholders
            .insert(original.to_string(), placeholder.clone());
        placeholder
    }
}

/// Whether a header named `name` carries a credential, or names the
/// account one belongs to.
pub(crate) fn is_secret_header(name: &str) -> bool {
    contains_case_insensitive(SENSITIVE_HEADER_NAMES, name)
}

pub(crate) fn contains_case_insensitive(values: &[&str], needle: &str) -> bool {
    values
        .iter()
        .any(|value| value.eq_ignore_ascii_case(needle))
}

#[cfg(feature = "http")]
pub(crate) fn canonical_json(body: &str) -> Option<Value> {
    serde_json::from_str::<Value>(body)
        .ok()
        .map(rig_core::json_utils::canonical)
}

pub(crate) fn is_redacted_placeholder(value: &str) -> bool {
    let Some((kind, counter)) = value.split_once("REDACTED_") else {
        return false;
    };

    !kind.is_empty()
        && !counter.is_empty()
        && kind
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
        && counter.chars().all(|ch| ch.is_ascii_digit())
}

fn scrub_query_param(input: &str, key: &str, replacement: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut remainder = input;
    let needle = format!("{key}=");

    while let Some(index) = find_query_param(remainder, &needle) {
        let (prefix, after_prefix) = remainder.split_at(index);
        output.push_str(prefix);

        let (name, after_value_start) = after_prefix.split_at(needle.len());
        output.push_str(name);
        let value_end = after_value_start
            .find(['&', '"', '\'', ' ', '\n', '\r', '<'])
            .unwrap_or(after_value_start.len());
        output.push_str(replacement);
        remainder = after_value_start.get(value_end..).unwrap_or_default();
    }

    output.push_str(remainder);
    output
}

fn find_query_param(input: &str, needle: &str) -> Option<usize> {
    let mut search_start = 0;

    while let Some(relative_index) = find_ascii_case_insensitive(input.get(search_start..)?, needle)
    {
        let index = search_start + relative_index;
        let starts_param = index == 0
            || input
                .get(..index)
                .and_then(|before| before.chars().next_back())
                .is_some_and(|ch| matches!(ch, '?' | '&' | '"' | '\'' | ' '));

        if starts_param {
            return Some(index);
        }

        search_start = index + needle.len();
    }

    None
}

fn find_ascii_case_insensitive(input: &str, needle: &str) -> Option<usize> {
    let input = input.as_bytes();
    let needle = needle.as_bytes();

    input
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle))
}

/// Redact directories of home paths beginning at the input start or after a
/// double quote, preserving the final component. Embedded path segments are
/// unchanged to avoid corrupting URLs.
pub(crate) fn scrub_local_filesystem_paths(text: &str) -> String {
    const HOME_PREFIXES: &[&str] = &["/Users/", "/home/", "/root/"];

    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    let mut consumed = 0usize;

    while let Some((prefix, at)) = HOME_PREFIXES
        .iter()
        .filter_map(|p| rest.find(*p).map(|i| (*p, i)))
        .min_by_key(|(_, i)| *i)
    {
        // Anchored: the value must begin here. Anything else is a path segment
        // inside a larger string (a URL, prose) and is not ours to rewrite.
        let absolute = consumed + at;
        let starts_value = absolute == 0 || text.as_bytes().get(absolute - 1) == Some(&b'"');
        let (before, path) = rest.split_at(at);
        if !starts_value {
            let (segment, after) = path.split_at(prefix.len());
            output.push_str(before);
            output.push_str(segment);
            rest = after;
            consumed = absolute + prefix.len();
            continue;
        }

        output.push_str(before);
        // Spaces may belong to account names; treating them as terminators would
        // leak the remainder of the home path.
        let end = path.find(['"', '\\', '\n', '\r']).unwrap_or(path.len());
        let (path, tail) = path.split_at(end);

        match path.rsplit_once('/') {
            Some((_, base)) if !base.is_empty() => {
                output.push_str("/REDACTED_PATH/");
                output.push_str(base);
            }
            _ => output.push_str("/REDACTED_PATH"),
        }
        consumed = absolute + path.len();
        rest = tail;
        if rest.is_empty() {
            break;
        }
    }

    output.push_str(rest);
    output
}
