//! Finding and redacting secrets in text a host logs, shows or shares:
//! API keys and access tokens by their well-known shapes, private key
//! blocks, and values assigned to secret-named variables. Bevy or not, every host that writes model traffic, a prompt
//! history or a shared transcript somewhere needs the same rules.
//!
//! The rules match a secret by its shape or the name it is assigned to,
//! so a credential with neither (a password in prose, a bare hex key) is
//! not found. A shape
//! matches only where a token starts: the same letters inside a word or a
//! base64 blob are not a key.
//!
//! ```
//! use rig_core::redact;
//!
//! let line = "export ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz";
//! assert!(redact::has_secret(line));
//! assert_eq!(redact::text(line), "export ANTHROPIC_API_KEY=[REDACTED]");
//! assert_eq!(redact::text("nothing to hide"), "nothing to hide");
//! ```

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;

use serde_json::Value;

#[cfg(test)]
mod tests;

/// What [`text`] and [`json`] put in place of a secret.
pub const REDACTED: &str = "[REDACTED]";

/// What kind of secret a [`SecretMatch`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Kind {
    /// An OpenAI API key: `sk-`, `sk-proj-` or `sk-svcacct-`.
    OpenAiApiKey,
    /// An Anthropic API key: `sk-ant-`.
    AnthropicApiKey,
    /// A Google API key: `AIza`.
    GoogleApiKey,
    /// An AWS access key id: `AKIA` or `ASIA`.
    AwsAccessKey,
    /// A GitHub token: `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_` or
    /// `github_pat_`.
    GitHubToken,
    /// A GitLab personal access token: `glpat-`.
    GitLabToken,
    /// A Slack token: `xoxa-`, `xoxb-`, `xoxp-`, `xoxr-` or `xoxs-`.
    SlackToken,
    /// A Stripe secret or restricted key: `sk_live_`, `sk_test_`,
    /// `rk_live_` or `rk_test_`.
    StripeKey,
    /// A Hugging Face access token: `hf_`.
    HuggingFaceToken,
    /// A PEM private key block, `-----BEGIN … PRIVATE KEY-----` through its
    /// end line, or through the end of the text when it has none.
    PrivateKey,
    /// A value assigned to a name that says it is secret, `NAME=value` or
    /// `NAME: value`: the name in upper snake case and, or ending in,
    /// `SECRET`, `TOKEN`, `PASSWORD`, `PASSWD`, `API_KEY`, `ACCESS_KEY`,
    /// `SECRET_KEY` or `PRIVATE_KEY` (so `AWS_SECRET_ACCESS_KEY` and
    /// `GITHUB_TOKEN`, not `MAX_TOKENS` or `TOKEN_COUNT`). The value runs
    /// to its closing quote when quoted, else to whitespace, a quote, `,`,
    /// `;` or `&` (the next assignment of a query string); one of at most
    /// 7 characters, all digits, starting with `$` (a reference such as
    /// `$GITHUB_TOKEN`), with characters other than printable ASCII (a
    /// placeholder such as `‹SECRET_1›`), or [`REDACTED`] is not a secret.
    /// Only the value is the secret.
    Assigned,
}

impl Kind {
    /// The kind's name, such as `OpenAI API key`.
    pub fn name(self) -> &'static str {
        match self {
            Self::OpenAiApiKey => "OpenAI API key",
            Self::AnthropicApiKey => "Anthropic API key",
            Self::GoogleApiKey => "Google API key",
            Self::AwsAccessKey => "AWS access key",
            Self::GitHubToken => "GitHub token",
            Self::GitLabToken => "GitLab token",
            Self::SlackToken => "Slack token",
            Self::StripeKey => "Stripe key",
            Self::HuggingFaceToken => "Hugging Face token",
            Self::PrivateKey => "private key",
            Self::Assigned => "assigned secret",
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A secret [`find`] found: its kind, and where it is in the text, in
/// bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecretMatch {
    /// What kind of secret it is.
    pub kind: Kind,
    /// Where it is in the text searched.
    pub range: Range<usize>,
}

/// Every secret in `input`, in order and not overlapping.
///
/// ```
/// use rig_core::redact::{self, Kind};
///
/// let found: Vec<_> = redact::find("id AKIA0123456789ABCDEF here").collect();
/// assert_eq!(found.len(), 1);
/// assert_eq!(found[0].kind, Kind::AwsAccessKey);
/// assert_eq!(found[0].range, 3..23);
/// ```
pub fn find(input: &str) -> impl Iterator<Item = SecretMatch> + '_ {
    let mut at = 0;
    std::iter::from_fn(move || {
        while let Some(rest) = input.get(at..) {
            let next = rest.chars().next()?;
            if starts_token(input, at) {
                let found = secret_at(rest).map(|(kind, len)| (kind, 0..len));
                let found = found.or_else(|| Some((Kind::Assigned, assigned(rest)?)));
                if let Some((kind, range)) = found {
                    let range = at + range.start..at + range.end;
                    at = range.end;
                    return Some(SecretMatch { kind, range });
                }
            }
            // A token's tail starts no key, so skip to the token's end.
            at += if is_token_char(next) {
                token_len(rest)
            } else {
                next.len_utf8()
            };
        }
        None
    })
}

/// Whether `input` holds a secret.
pub fn has_secret(input: &str) -> bool {
    find(input).next().is_some()
}

/// `input` with every secret replaced by [`REDACTED`]; borrowed when it has
/// none.
pub fn text(input: &str) -> Cow<'_, str> {
    let mut found = find(input).peekable();
    if found.peek().is_none() {
        return Cow::Borrowed(input);
    }
    let mut out = String::with_capacity(input.len());
    let mut from = 0;
    for secret in found {
        out.push_str(input.get(from..secret.range.start).unwrap_or_default());
        out.push_str(REDACTED);
        from = secret.range.end;
    }
    out.push_str(input.get(from..).unwrap_or_default());
    Cow::Owned(out)
}

/// Every string in `value` redacted as [`text`] redacts it. Object keys
/// are left as they are.
pub fn json(value: &mut Value) {
    match value {
        Value::String(string) => {
            if let Cow::Owned(redacted) = text(string) {
                *string = redacted;
            }
        }
        Value::Array(items) => items.iter_mut().for_each(json),
        Value::Object(fields) => fields.values_mut().for_each(json),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

/// The secret `rest` starts with, and its length in bytes.
fn secret_at(rest: &str) -> Option<(Kind, usize)> {
    if rest.starts_with("-----BEGIN ") {
        return private_key(rest).map(|len| (Kind::PrivateKey, len));
    }
    let token = rest.get(..token_len(rest))?;
    if let Some(len) = aws_access_key(token) {
        return Some((Kind::AwsAccessKey, len));
    }
    let kind = TOKENS.iter().find_map(|&(prefix, min, kind)| {
        let tail = token.strip_prefix(prefix)?;
        is_plausible(tail, min).then_some(kind)
    })?;
    Some((kind, token.len()))
}

/// Token shapes: a prefix, the fewest characters after it, and the kind. A
/// longer prefix comes before a shorter one it starts with.
const TOKENS: &[(&str, usize, Kind)] = &[
    ("sk-ant-", 16, Kind::AnthropicApiKey),
    ("sk-proj-", 16, Kind::OpenAiApiKey),
    ("sk-svcacct-", 16, Kind::OpenAiApiKey),
    ("sk-", 32, Kind::OpenAiApiKey),
    ("AIza", 20, Kind::GoogleApiKey),
    ("github_pat_", 22, Kind::GitHubToken),
    ("ghp_", 30, Kind::GitHubToken),
    ("gho_", 30, Kind::GitHubToken),
    ("ghu_", 30, Kind::GitHubToken),
    ("ghs_", 30, Kind::GitHubToken),
    ("ghr_", 30, Kind::GitHubToken),
    ("glpat-", 20, Kind::GitLabToken),
    ("xoxa-", 10, Kind::SlackToken),
    ("xoxb-", 10, Kind::SlackToken),
    ("xoxp-", 10, Kind::SlackToken),
    ("xoxr-", 10, Kind::SlackToken),
    ("xoxs-", 10, Kind::SlackToken),
    ("sk_live_", 16, Kind::StripeKey),
    ("sk_test_", 16, Kind::StripeKey),
    ("rk_live_", 16, Kind::StripeKey),
    ("rk_test_", 16, Kind::StripeKey),
    ("hf_", 30, Kind::HuggingFaceToken),
];

/// The names, or ends of names after a `_`, that say a variable holds a
/// secret.
const SECRET_NAMES: &[&str] = &[
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "ACCESS_KEY",
    "SECRET_KEY",
    "PRIVATE_KEY",
];

/// Where the secret value assigned to the secret-named variable `rest`
/// starts with is, in `rest`: see [`Kind::Assigned`]. `None` when the value
/// is a key of a known shape, which the scan finds as its own kind.
fn assigned(rest: &str) -> Option<Range<usize>> {
    let name = rest.get(..token_len(rest))?;
    let upper_snake = name.starts_with(|ch: char| ch.is_ascii_uppercase())
        && name
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_');
    let secret_named = SECRET_NAMES.iter().any(|end| {
        name.strip_suffix(end)
            .is_some_and(|head| head.is_empty() || head.ends_with('_'))
    });
    if !upper_snake || !secret_named {
        return None;
    }
    // A quoted name's closing quote, then `=` or `:`, spaces around it.
    let after = rest.get(name.len()..)?;
    let after = after.strip_prefix(['"', '\'']).unwrap_or(after);
    let after = after.trim_start_matches([' ', '\t']);
    let after = after.strip_prefix(['=', ':'])?;
    let after = after.trim_start_matches([' ', '\t']);
    let start = rest.len() - after.len();
    let (from, value) = match after.chars().next()? {
        quote @ ('"' | '\'') => {
            let inner = after.get(1..)?;
            (start + 1, inner.get(..inner.find(quote)?)?)
        }
        _ => (
            start,
            after.get(..after.find(ends_value).unwrap_or(after.len()))?,
        ),
    };
    let plausible = value.len() >= 8
        && value.chars().all(|ch| ch.is_ascii_graphic() || ch == ' ')
        && !value.starts_with('$')
        && !value.chars().all(|ch| ch.is_ascii_digit())
        && value != REDACTED
        && secret_at(value).is_none();
    plausible.then(|| from..from + value.len())
}

/// Whether `ch` ends an unquoted assigned value: whitespace, a quote, or a
/// separator between assignments (`,`, `;`, `&`).
fn ends_value(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '"' | '\'' | '`' | ',' | ';' | '&')
}

/// Whether `tail`, the rest of a token after a key's prefix, is long
/// enough and has a letter, so `sk-` followed by a number or a short word
/// is not a key.
fn is_plausible(tail: &str, min: usize) -> bool {
    tail.len() >= min && tail.chars().any(|ch| ch.is_ascii_alphabetic())
}

/// The length of the AWS access key id `token` starts with: `AKIA` or
/// `ASIA` and 16 upper-case letters or digits, no more.
fn aws_access_key(token: &str) -> Option<usize> {
    let tail = token
        .strip_prefix("AKIA")
        .or_else(|| token.strip_prefix("ASIA"))?;
    let len = tail
        .chars()
        .take_while(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit())
        .count();
    (len == 16).then_some(4 + len)
}

/// The length of the private key block `rest` starts with, through its end
/// line, else through the end of `rest`.
fn private_key(rest: &str) -> Option<usize> {
    let header = rest.get("-----BEGIN ".len()..)?;
    let label = header.split_once("-----")?.0;
    let is_key = label.ends_with("PRIVATE KEY")
        && label.chars().all(|ch| ch.is_ascii_uppercase() || ch == ' ');
    if !is_key {
        return None;
    }
    let end = format!("-----END {label}-----");
    Some(
        rest.find(end.as_str())
            .map_or(rest.len(), |at| at + end.len()),
    )
}

/// The length of the run of token characters `rest` starts with.
fn token_len(rest: &str) -> usize {
    rest.find(|ch| !is_token_char(ch)).unwrap_or(rest.len())
}

/// Whether `ch` may be part of a key.
fn is_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')
}

/// Whether a token starts at `at` in `input`: the character before is
/// neither a token character nor base64's `+` or `/`, so the same letters
/// inside a word or a base64 blob (a signature, ciphertext, an image) are
/// not read as the start of a key.
fn starts_token(input: &str, at: usize) -> bool {
    input
        .get(..at)
        .and_then(|before| before.chars().next_back())
        .is_none_or(|ch| !is_token_char(ch) && !matches!(ch, '+' | '/'))
}
