//! The one scrubber of recorded provider traffic, shared by the cassette
//! engine and the traffic log. It removes credentials (sensitive headers and
//! query parameters, and every secret [`rig_core::redact`] finds), AWS account
//! numbers and home-directory paths, and keeps everything else the provider
//! sent verbatim. A [`Purpose`] selects the few behaviours a replayable
//! fixture needs and a log of live traffic must not have.

use rig_core::redact;
use serde_json::{Map, Value};
use std::ops::Range;

#[cfg(test)]
mod tests;
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

    /// Remembers credentials in a header, including individual cookies.
    pub(crate) fn learn_header(&mut self, name: &str, value: &str) {
        if is_secret_header(name) {
            self.learn(value);
            if name.to_ascii_lowercase().contains("auth")
                && let Some((_, credential)) = value.split_once(' ')
            {
                self.learn(credential.trim());
            }
            if name.eq_ignore_ascii_case("cookie") || name.eq_ignore_ascii_case("set-cookie") {
                for cookie in value.split(';') {
                    if let Some((_, value)) = cookie.split_once('=') {
                        self.learn(value.trim());
                    }
                    if name.eq_ignore_ascii_case("set-cookie") {
                        break;
                    }
                }
            }
        } else {
            self.learn_text(value);
        }
    }

    /// Remembers raw and decoded URL credentials.
    pub(crate) fn learn_url(&mut self, uri: &str) {
        if let Some((_, query)) = uri.split_once('?') {
            self.learn_form(query.split('#').next().unwrap_or_default());
        }
        if let Ok(url) = url::Url::parse(uri) {
            for username in [
                url.username().to_owned(),
                decode_form_component(url.username()),
            ] {
                if username.len() >= 8 {
                    self.learn(&username);
                }
            }
            if let Some(password) = url.password() {
                self.learn(password);
                self.learn(&decode_form_component(password));
            }
        }
    }

    /// Remembers raw and decoded values of a sensitive query or form pair.
    pub(crate) fn learn_param(&mut self, name: &str, value: &str) {
        if is_secret_param(&decode_form_component(name)) {
            self.learn(value);
            self.learn(&decode_form_component(value));
        }
    }

    fn learn_form(&mut self, form: &str) {
        for pair in form.split('&') {
            if let Some((name, value)) = pair.split_once('=') {
                self.learn_param(name, value);
            }
        }
    }

    /// Remembers credentials in JSON, form, SSE, or incomplete text bodies.
    pub(crate) fn learn_body(&mut self, body: &str) {
        if let Ok(json) = serde_json::from_str::<Value>(body) {
            self.learn_json(&json);
            return;
        }
        if is_form(body) {
            self.learn_form(body);
        }
        if body
            .lines()
            .any(|line| line.trim_start().starts_with("data:"))
        {
            let normalized = body.replace("\r\n", "\n");
            for event in normalized.split("\n\n") {
                let data: Vec<_> = event
                    .lines()
                    .filter_map(|line| line.trim_start().strip_prefix("data:"))
                    .collect();
                if let Ok(json) = serde_json::from_str::<Value>(&data.join("\n")) {
                    self.learn_json(&json);
                } else {
                    self.learn_text(event);
                }
            }
            return;
        }
        self.learn_text(body);
    }

    fn learn_text(&mut self, text: &str) {
        for range in url_spans(text) {
            self.learn_url(text.get(range).unwrap_or_default());
        }
        for range in secret_values(text, self.purpose) {
            if let Some(raw) = text.get(range) {
                if let Ok(value) = serde_json::from_str::<Value>(raw) {
                    self.learn_secret_value(&value);
                } else {
                    self.learn(raw.trim_matches('"'));
                }
            }
        }
    }

    fn learn_secret_value(&mut self, value: &Value) {
        match value {
            Value::String(text) => self.learn(text),
            Value::Number(number) => self.learn(&number.to_string()),
            Value::Array(items) => items.iter().for_each(|item| self.learn_secret_value(item)),
            Value::Object(fields) if !is_schema(fields) => {
                fields
                    .values()
                    .for_each(|field| self.learn_secret_value(field));
            }
            _ => {}
        }
    }

    fn learn_json(&mut self, value: &Value) {
        match value {
            Value::String(text) => match canonical_json(text) {
                Some(json) if json.is_object() || json.is_array() => self.learn_json(&json),
                _ => self.learn_text(text),
            },
            Value::Array(items) => items.iter().for_each(|item| self.learn_json(item)),
            Value::Object(fields) => {
                for (name, field) in fields {
                    if is_secret_field(name, self.purpose)
                        && !(name.eq_ignore_ascii_case("token") && is_logprobs(fields))
                    {
                        self.learn_secret_value(field);
                    } else {
                        self.learn_json(field);
                    }
                }
            }
            _ => {}
        }
    }

    /// Remembers `secret`, unless it is too short to tell from ordinary
    /// text or is a redaction already.
    fn learn(&mut self, secret: &str) {
        const SHORTEST: usize = 3;
        if secret.len() < SHORTEST
            || secret == REDACTED
            || decode_form_component(secret)
                .split_whitespace()
                .any(|part| part == REDACTED || is_redacted_placeholder(part))
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
        self.scrub_text(&mask_url(url))
    }

    /// `body` scrubbed as JSON (written canonically), SSE, or text.
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

        if is_form(body) {
            return self.scrub_text(&mask_form(body));
        }
        self.scrub_text(body)
    }

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
        if key.is_none() {
            self.learn_json(value);
        }
        let key_lower = key.map(str::to_ascii_lowercase);
        let cassette = self.purpose == Purpose::Cassette;

        match value {
            Value::Object(map) => {
                let logprobs = is_logprobs(map);
                for (name, mut field) in std::mem::take(map) {
                    let key = (!(logprobs && name.eq_ignore_ascii_case("token")))
                        .then_some(name.as_str());
                    self.scrub_json_value(key, &mut field);
                    let name = self.scrub_text(&name);
                    let mut unique = name.clone();
                    let mut n = 1;
                    while map.contains_key(&unique) {
                        n += 1;
                        unique = format!("{name}#{n}");
                    }
                    map.insert(unique, field);
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

                if key.is_some_and(|key| is_secret_field(key, self.purpose)) {
                    *text = REDACTED.to_owned();
                } else if let Some(mut json) =
                    canonical_json(text).filter(|value| value.is_object() || value.is_array())
                {
                    let original = json.clone();
                    self.scrub_json_value(None, &mut json);
                    if json != original {
                        *text = json.to_string();
                    }
                } else {
                    *text = self.scrub_text(text);
                }
            }
            Value::Number(number) => {
                if key.is_some_and(|key| is_secret_field(key, self.purpose))
                    || self.learned.contains(&number.to_string())
                {
                    *value = Value::String(REDACTED.to_owned());
                } else if cassette
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
        let mut scrubbed = mask_urls(text);
        for secret in &self.learned {
            if scrubbed.contains(secret.as_str()) {
                scrubbed = replace_secret(&scrubbed, secret);
            }
        }
        for key in SENSITIVE_QUERY_PARAMS {
            scrubbed = scrub_query_param(&scrubbed, key, REDACTED);
        }
        let mut kept = 0;
        let mut masked = String::new();
        for range in secret_values(&scrubbed, self.purpose) {
            masked.push_str(scrubbed.get(kept..range.start).unwrap_or_default());
            masked.push('"');
            masked.push_str(REDACTED);
            masked.push('"');
            kept = range.end;
        }
        if kept != 0 {
            masked.push_str(scrubbed.get(kept..).unwrap_or_default());
            scrubbed = masked;
        }
        let scrubbed = redact::text(&scrubbed);
        match self.purpose {
            Purpose::Cassette => {
                let scrubbed = self.scrub_aws_account_ids(&scrubbed);
                scrub_local_filesystem_paths(&scrubbed)
            }
            Purpose::Traffic => scrubbed.into_owned(),
        }
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
    let lower = name.to_ascii_lowercase();
    contains_case_insensitive(SENSITIVE_HEADER_NAMES, name)
        || matches!(lower.as_str(), "token" | "apikey" | "secret" | "password")
        || (lower.contains("auth") && !lower.ends_with("-authenticate"))
        || ["-key", "-token", "-secret", "-signature"]
            .iter()
            .any(|suffix| lower.ends_with(suffix))
}

pub(crate) fn contains_case_insensitive(values: &[&str], needle: &str) -> bool {
    values
        .iter()
        .any(|value| value.eq_ignore_ascii_case(needle))
}

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

fn replace_secret(text: &str, secret: &str) -> String {
    if secret.len() >= 8 || text == secret {
        return text.replace(secret, REDACTED);
    }
    let mut out = String::with_capacity(text.len());
    let mut kept = 0;
    for (at, _) in text.match_indices(secret) {
        let before = text.get(..at).and_then(|text| text.chars().next_back());
        let after = text
            .get(at + secret.len()..)
            .and_then(|text| text.chars().next());
        if before.is_some_and(char::is_alphanumeric) || after.is_some_and(char::is_alphanumeric) {
            continue;
        }
        out.push_str(text.get(kept..at).unwrap_or_default());
        out.push_str(REDACTED);
        kept = at + secret.len();
    }
    out.push_str(text.get(kept..).unwrap_or_default());
    out
}

fn string_end(text: &str, open: usize) -> usize {
    let mut escaped = false;
    text.get(open + 1..)
        .and_then(|rest| {
            rest.char_indices().find(|&(_, c)| {
                let end = c == '"' && !escaped;
                escaped = c == '\\' && !escaped;
                end
            })
        })
        .map_or(text.len(), |(at, _)| open + 1 + at + 1)
}

fn value_end(text: &str, start: usize) -> usize {
    match text.get(start..).and_then(|rest| rest.chars().next()) {
        Some('"') => string_end(text, start),
        Some('{' | '[') => {
            let mut depth = 0usize;
            let mut at = start;
            while let Some(c) = text.get(at..).and_then(|rest| rest.chars().next()) {
                match c {
                    '"' => {
                        at = string_end(text, at);
                        continue;
                    }
                    '{' | '[' => depth += 1,
                    '}' | ']' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            return at + 1;
                        }
                    }
                    _ => {}
                }
                at += c.len_utf8();
            }
            text.len()
        }
        _ => text
            .get(start..)
            .and_then(|rest| rest.find([',', '}', ']']))
            .map_or(text.len(), |len| start + len),
    }
}

fn secret_values(text: &str, purpose: Purpose) -> Vec<Range<usize>> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(open) = text.get(at..).and_then(|rest| rest.find('"')) {
        let open = at + open;
        let end = string_end(text, open);
        at = end;
        let raw = text.get(open..end).unwrap_or_default();
        let key = serde_json::from_str::<String>(raw)
            .unwrap_or_else(|_| raw.trim_matches('"').to_owned());
        let after = text.get(end..).unwrap_or_default();
        let Some(value) = after
            .trim_start()
            .strip_prefix(':')
            .map(str::trim_start)
            .filter(|_| is_secret_field(&key, purpose))
        else {
            continue;
        };
        let start = text.len() - value.len();
        let stop = value_end(text, start);
        if stop > start {
            found.push(start..stop);
        }
        at = stop.max(end);
    }
    found
}

fn is_secret_field(name: &str, purpose: Purpose) -> bool {
    let normalized: String = name
        .chars()
        .filter(|c| !matches!(c, '_' | '-'))
        .map(|c| c.to_ascii_lowercase())
        .collect();
    match normalized.as_str() {
        "token" => purpose == Purpose::Traffic,
        "apikey"
        | "apitoken"
        | "accesstoken"
        | "authtoken"
        | "authorizationcode"
        | "refreshtoken"
        | "idtoken"
        | "sessiontoken"
        | "clientsecret"
        | "secretkey"
        | "privatekey"
        | "subscriptionkey"
        | "password"
        | "devicecode"
        | "deviceauthid"
        | "usercode"
        | "codeverifier"
        | "authorizationtoken"
        | "fallbackcredittoken" => true,
        _ => false,
    }
}

pub(crate) fn is_secret_param(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    is_secret_field(name, Purpose::Traffic)
        || matches!(
            lower.as_str(),
            "key" | "secret" | "signature" | "sig" | "code" | "assertion"
        )
        || lower.starts_with("x-amz-")
        || ["key", "token", "secret", "signature"]
            .iter()
            .any(|suffix| {
                ["_", "-"]
                    .iter()
                    .any(|sep| lower.ends_with(&format!("{sep}{suffix}")))
            })
}

fn is_schema(fields: &Map<String, Value>) -> bool {
    let json_type = |value: &Value| {
        matches!(
            value.as_str(),
            Some("string" | "object" | "array" | "number" | "integer" | "boolean" | "null")
        )
    };
    fields.get("type").is_some_and(|value| match value {
        Value::Array(types) => !types.is_empty() && types.iter().all(json_type),
        value => json_type(value),
    }) || [
        "anyOf",
        "oneOf",
        "allOf",
        "$ref",
        "properties",
        "items",
        "enum",
        "description",
        "title",
        "required",
    ]
    .iter()
    .any(|key| fields.contains_key(*key))
}

fn is_logprobs(fields: &Map<String, Value>) -> bool {
    fields.iter().any(|(name, value)| {
        value.is_number()
            && matches!(
                name.to_ascii_lowercase().as_str(),
                "logprob" | "logprobability"
            )
    })
}

fn decode_form_component(text: &str) -> String {
    url::form_urlencoded::parse(format!("value={text}").as_bytes())
        .next()
        .map(|(_, value)| value.into_owned())
        .unwrap_or_default()
}

fn is_form(text: &str) -> bool {
    text.contains('=') && !text.trim_end().contains(char::is_whitespace)
}

fn mask_form(form: &str) -> String {
    form.split('&')
        .map(|pair| match pair.split_once('=') {
            Some((name, _)) if is_secret_param(&decode_form_component(name)) => {
                format!("{name}={REDACTED}")
            }
            _ => pair.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("&")
}

fn mask_url(uri: &str) -> String {
    let mut uri = uri.to_owned();
    if let Some((scheme, rest)) = uri.split_once("://") {
        let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        if let Some((_, host)) = rest
            .get(..end)
            .and_then(|authority| authority.rsplit_once('@'))
        {
            uri = format!(
                "{scheme}://{REDACTED}@{host}{}",
                rest.get(end..).unwrap_or_default()
            );
        }
    }
    if let Some((path, query)) = uri.split_once('?') {
        return format!("{path}?{}", mask_form(query));
    }
    uri
}

fn url_spans(text: &str) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut after = 0;
    for (at, _) in text.match_indices("://") {
        let Some(scheme) = ["https", "http", "wss", "ws"].into_iter().find(|scheme| {
            text.get(..at)
                .is_some_and(|before| before.ends_with(scheme))
        }) else {
            continue;
        };
        let start = at - scheme.len();
        if start < after {
            continue;
        }
        let rest = text.get(start..).unwrap_or_default();
        let len = rest
            .find(|c: char| c.is_whitespace() || "\"'<>()[]{}`".contains(c))
            .unwrap_or(rest.len());
        spans.push(start..start + len);
        after = start + len;
    }
    spans
}

fn mask_urls(text: &str) -> String {
    let mut out = String::new();
    let mut kept = 0;
    for range in url_spans(text) {
        out.push_str(text.get(kept..range.start).unwrap_or_default());
        out.push_str(&mask_url(text.get(range.clone()).unwrap_or_default()));
        kept = range.end;
    }
    out.push_str(text.get(kept..).unwrap_or_default());
    out
}
