//! Finding and redacting secrets in text a host logs, shows or shares: API keys and tokens by
//! their known shapes, private key blocks, and values assigned to secret-named variables.
//!
//! A shape matches only where a token starts, so the same letters inside a word or a base64
//! blob are not a key. A secret with neither a known shape nor a telling name, such as a
//! password in prose, is not found.
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
    /// A JSON Web Token, such as an OAuth access or id token: a base64url
    /// header and payload, each a JSON object (`eyJ…`), and a signature,
    /// joined by `.`; the signature may be empty.
    JsonWebToken,
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
    /// placeholder such as `‹SECRET_1›`), or one holding [`REDACTED`] is
    /// not a secret, so redacting text again changes nothing.
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
            Self::JsonWebToken => "JSON Web Token",
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
    // Bytes, not chars: every token character is ASCII, so a token starts
    // on a char boundary, and the bytes of any other character are skipped.
    let bytes = input.as_bytes();
    let mut at = 0;
    // Where the last secret ended: what follows it starts a token, as it
    // does after the secret is redacted, so redacting twice finds nothing new.
    let mut ended = 0;
    std::iter::from_fn(move || {
        while let Some(&byte) = bytes.get(at) {
            if !is_token_byte(byte) {
                at += 1;
                continue;
            }
            let rest = bytes.get(at..).unwrap_or_default();
            let len = rest
                .iter()
                .position(|&byte| !is_token_byte(byte))
                .unwrap_or(rest.len());
            let token = rest.get(..len).unwrap_or_default();
            if may_be_secret(token)
                && (starts_token(input, at) || at == ended)
                && let Some(rest) = input.get(at..)
            {
                let found = secret_at(rest).map(|(kind, len)| (kind, 0..len));
                let found = found.or_else(|| Some((Kind::Assigned, assigned(rest)?)));
                if let Some((kind, range)) = found {
                    let range = at + range.start..at + range.end;
                    (at, ended) = (range.end, range.end);
                    return Some(SecretMatch { kind, range });
                }
            }
            // A token's tail starts no key, so skip to the token's end.
            at += len;
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

/// The most a [`Streamed`] holds back, in bytes: a private key block, the
/// longest secret, is a few kilobytes.
const HELD_MOST: usize = 16 * 1024;

/// A text that arrives in pieces, such as a streamed reply, redacted as
/// [`text`] redacts it whole: what [`Self::push`] and [`Self::finish`]
/// return, joined, is [`text`] of the pieces joined. It holds back what a
/// later piece may still make part of a secret: the last run of characters
/// a key is made of, a line that names a secret (its value may follow), and
/// a private key block not yet ended. It holds at most 16 KiB,
/// far more than a key, so a line that names a secret, a private key block
/// or a run of key characters longer than that may show some of itself.
///
/// ```
/// use rig_core::redact::Streamed;
///
/// let mut streamed = Streamed::default();
/// let mut shown = streamed.push("export GITHUB_TOKEN=gh");
/// assert_eq!(shown, "");
/// shown += &streamed.push("p_abcdefghijklmnopqrstuvwxyz0123456789\nnext ");
/// shown += &streamed.finish();
/// assert_eq!(shown, "export GITHUB_TOKEN=[REDACTED]\nnext ");
/// ```
#[derive(Clone, Debug, Default)]
pub struct Streamed {
    /// What came and was not returned yet.
    held: String,
}

impl Streamed {
    /// Takes `piece` and returns, redacted, what no later piece can change.
    pub fn push(&mut self, piece: &str) -> String {
        self.held.push_str(piece);
        let rest = self.held.split_off(settled(&self.held));
        let shown = text(&self.held).into_owned();
        self.held = rest;
        shown
    }

    /// Returns, redacted, what it still holds: the text ended.
    pub fn finish(&mut self) -> String {
        let shown = text(&self.held).into_owned();
        self.held.clear();
        shown
    }
}

/// How much of `input`, a text that may still grow, ends where no secret
/// [`find`] finds in the grown text can start before it and end after it:
/// before the last run of characters a key or a JSON Web Token is made of,
/// before a line that names a secret, and before the line of a secret that
/// runs past the cut, such as a private key block. Each such place follows
/// a character no token continues, so a token starts after it as in the
/// whole text. At most [`HELD_MOST`] bytes stay held.
fn settled(input: &str) -> usize {
    let word = input
        .char_indices()
        .rfind(|&(_, ch)| !is_token_char(ch) && !matches!(ch, '.' | '+' | '/'))
        .map_or(0, |(at, ch)| at + ch.len_utf8());
    let line_of = |at: usize| {
        input
            .get(..at)
            .and_then(|before| before.rfind('\n'))
            .map_or(0, |at| at + 1)
    };
    let line = line_of(input.len());
    let last = input.get(line..).unwrap_or_default();
    // A run of key characters that ends as a secret name does, wherever in
    // the run [`find`] may start a token.
    let named = SECRET_NAMES.iter().any(|name| {
        last.match_indices(name).any(|(at, _)| {
            let after = last.get(at + name.len()..).unwrap_or_default();
            !after.starts_with(is_token_char)
        })
    });
    let mut cut = word;
    if named || last.contains("-----BEGIN ") {
        cut = cut.min(line);
    }
    let found: Vec<_> = find(input).collect();
    // A secret that reaches the end may still grow, as a block not yet ended.
    for found in found.iter().rev() {
        let past = cut < found.range.end || found.range.end == input.len();
        if found.range.start < cut && past {
            cut = line_of(found.range.start);
        }
    }
    let least = input.len().saturating_sub(HELD_MOST);
    let cut = (cut.max(least)..=input.len())
        .find(|&at| input.is_char_boundary(at))
        .unwrap_or(input.len());
    // A whole secret the bound cuts through goes out whole, so it is
    // redacted as in the whole text: an assigned value with its name before
    // it and its closing quote after it.
    let whole = |found: &SecretMatch| {
        let (start, end) = (found.range.start, found.range.end);
        if found.kind != Kind::Assigned {
            return start..end;
        }
        let before = input.get(..start).unwrap_or_default();
        let named = before.trim_end_matches(['=', ':', '"', '\'', ' ', '\t']);
        let quoted = input
            .get(end..)
            .is_some_and(|after| after.starts_with(['"', '\'']));
        named.trim_end_matches(is_token_char).len()..end + usize::from(quoted)
    };
    let through = found
        .iter()
        .map(whole)
        .find(|whole| whole.start < cut && cut < whole.end && whole.end < input.len());
    through.map_or(cut, |whole| whole.end)
}

/// Every string in `value` redacted as [`text`] redacts it; whether any
/// was. Object keys are left as they are.
pub fn json(value: &mut Value) -> bool {
    match value {
        Value::String(string) => match text(string) {
            Cow::Owned(redacted) => {
                *string = redacted;
                true
            }
            Cow::Borrowed(_) => false,
        },
        Value::Array(items) => items.iter_mut().fold(false, |any, item| json(item) | any),
        Value::Object(fields) => fields
            .values_mut()
            .fold(false, |any, field| json(field) | any),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

/// The secret `rest` starts with, and its length in bytes.
fn secret_at(rest: &str) -> Option<(Kind, usize)> {
    if rest.starts_with("-----BEGIN ") {
        return private_key(rest).map(|len| (Kind::PrivateKey, len));
    }
    if let Some(len) = json_web_token(rest) {
        return Some((Kind::JsonWebToken, len));
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
        && !value.contains(REDACTED)
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

/// The length of the JSON Web Token `rest` starts with: a header and a
/// payload of at least 10 base64url characters each, both starting `eyJ`
/// (`{"`), each followed by `.`, then the signature's base64url characters,
/// if any.
fn json_web_token(rest: &str) -> Option<usize> {
    let mut len = 0;
    for _ in 0..2 {
        let part = rest.get(len..)?;
        let segment = token_len(part);
        if !part.starts_with("eyJ") || segment < 10 || !part.get(segment..)?.starts_with('.') {
            return None;
        }
        len += segment + 1;
    }
    Some(len + token_len(rest.get(len..)?))
}

/// Whether a secret may start at `token`, a run of token characters: it
/// starts as a key's prefix, a private key's header or a JSON Web Token
/// does, or it is upper case and ends as a secret name does. A cheap test
/// before the full one.
fn may_be_secret(token: &[u8]) -> bool {
    let first = token.first().copied().unwrap_or_default();
    let second = token.get(1).copied().unwrap_or_default();
    let shaped = matches!(
        (first, second),
        (b's' | b'r', b'k')
            | (b'A', b'I' | b'K' | b'S')
            | (b'g', b'h' | b'i' | b'l')
            | (b'x', b'o')
            | (b'h', b'f')
            | (b'e', b'y')
            | (b'-', b'-')
    );
    let named =
        first.is_ascii_uppercase() && matches!(token.last(), Some(b'T' | b'N' | b'D' | b'Y'));
    shaped || named
}

/// Whether `byte` may be part of a key: [`is_token_char`] for bytes.
fn is_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
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
