//! `cargo xtask cassette scan`: look for credentials and account data in
//! new or changed fixtures.
//!
//! Key shapes are matched against whole tokens (maximal runs of
//! `[A-Za-z0-9_.-]`), case-sensitively, so base64 image data does not
//! produce false hits and a key embedded in a longer token is not mistaken
//! for one. Every exported `*_API_KEY`, `*_TOKEN` and `*_SECRET` value of at
//! least 12 characters is also searched for literally, and never printed.

use std::path::Path;
use std::process::Command;

/// One suspicious match: the kind and a redacted excerpt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Hit {
    pub(crate) kind: &'static str,
    pub(crate) excerpt: String,
}

fn is_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')
}

fn tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|ch: char| !is_token_char(ch))
        .map(|token| token.trim_matches('.'))
        .filter(|token| !token.is_empty())
}

fn all(text: &str, allowed: impl Fn(char) -> bool) -> bool {
    !text.is_empty() && text.chars().all(allowed)
}

fn key_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')
}

/// The credential shape `token` has, if any.
pub(crate) fn key_shape(token: &str) -> Option<&'static str> {
    if let Some(rest) = token.strip_prefix("AIza") {
        return (rest.len() == 35 && all(rest, key_char)).then_some("google_key");
    }
    if let Some(rest) = token.strip_prefix("sk-ant-") {
        let (_, key) = rest.split_once('-')?;
        return (key.len() >= 40 && all(key, key_char)).then_some("anthropic_key");
    }
    if let Some(rest) = token.strip_prefix("sk-or-v1-") {
        return (rest.len() == 64 && all(rest, |ch| ch.is_ascii_hexdigit()))
            .then_some("openrouter_key");
    }
    if let Some(rest) = token.strip_prefix("sk-") {
        let rest = ["proj-", "svcacct-", "admin-"]
            .iter()
            .find_map(|prefix| rest.strip_prefix(prefix))
            .unwrap_or(rest);
        return (rest.len() >= 32 && all(rest, key_char)).then_some("openai_key");
    }
    if let Some(rest) = token.strip_prefix("xai-") {
        return (rest.len() >= 40 && all(rest, |ch| ch.is_ascii_alphanumeric()))
            .then_some("xai_key");
    }
    if let Some(rest) = token.strip_prefix("gsk_") {
        return (rest.len() >= 40 && all(rest, |ch| ch.is_ascii_alphanumeric()))
            .then_some("groq_key");
    }
    if let Some(rest) = token
        .strip_prefix("AKIA")
        .or_else(|| token.strip_prefix("ASIA"))
    {
        return (rest.len() == 16
            && all(rest, |ch| ch.is_ascii_uppercase() || ch.is_ascii_digit()))
        .then_some("aws_key");
    }
    if let Some(rest) = token.strip_prefix("org_") {
        return (rest.len() == 26
            && all(rest, |ch| ch.is_ascii_lowercase() || ch.is_ascii_digit()))
        .then_some("groq_org");
    }
    if let Some(rest) = token.strip_prefix("org-") {
        return (rest.len() == 24 && all(rest, |ch| ch.is_ascii_alphanumeric()))
            .then_some("openai_org");
    }
    None
}

fn is_uuid(token: &str) -> bool {
    let parts: Vec<&str> = token.split('-').collect();
    parts.len() == 5
        && parts
            .iter()
            .zip([8, 4, 4, 4, 12])
            .all(|(part, len)| part.len() == len && all(part, |ch| ch.is_ascii_hexdigit()))
}

fn redact(found: &str) -> String {
    let head: String = found.chars().take(6).collect();
    let tail: String = found
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if found.chars().count() > 12 {
        format!("{head}…{tail}")
    } else {
        format!("{head}…")
    }
}

/// Every hit in `text`, with `secrets` searched for literally.
pub(crate) fn scan(text: &str, secrets: &[String]) -> Vec<Hit> {
    let mut hits = Vec::new();
    let mut hit = |kind: &'static str, found: &str| {
        hits.push(Hit {
            kind,
            excerpt: redact(found),
        });
    };
    for token in tokens(text) {
        if let Some(kind) = key_shape(token) {
            hit(kind, token);
        }
    }
    // `Bearer <credential>` with a long credential.
    for (index, _) in text.match_indices("Bearer ") {
        let credential: String = text[index + 7..]
            .chars()
            .take_while(|ch| is_token_char(*ch))
            .collect();
        if credential.len() >= 20 {
            hit("bearer", &credential);
        }
    }
    let lower = text.to_ascii_lowercase();
    for name in [
        "set-cookie",
        "cookie",
        "openai-organization",
        "openai-project",
        "anthropic-organization-id",
        "x-organization-id",
    ] {
        if lower.contains(&format!("name: {name}\n")) {
            hit(
                if name.contains("cookie") {
                    "cookie_header"
                } else {
                    "org_header"
                },
                name,
            );
        }
    }
    // `"user_id": "<value>"`.
    for (index, _) in text.match_indices("\"user_id\"") {
        let rest = text[index + 9..].trim_start();
        if let Some(value) = rest
            .strip_prefix(':')
            .map(str::trim_start)
            .and_then(|rest| rest.strip_prefix('"'))
            && !value.starts_with('"')
        {
            hit("user_id", "user_id");
        }
    }
    // A UUID named as a team id.
    for token in tokens(text) {
        if is_uuid(token)
            && let Some(position) = text.find(token)
        {
            let mut start = position.saturating_sub(16);
            while !lower.is_char_boundary(start) {
                start -= 1;
            }
            if lower[start..position].contains("team") {
                hit("xai_team", token);
            }
        }
    }
    // Email addresses: a local part, `@`, and a dotted domain with a TLD.
    for (index, _) in text.match_indices('@') {
        let local: String = text[..index]
            .chars()
            .rev()
            .take_while(|ch| {
                ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '%' | '+' | '-')
            })
            .collect();
        let domain: String = text[index + 1..]
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-'))
            .collect();
        let domain = domain.trim_end_matches('.');
        let tld_ok = domain.rsplit_once('.').is_some_and(|(host, tld)| {
            !host.is_empty() && tld.len() >= 2 && all(tld, |ch| ch.is_ascii_alphabetic())
        });
        if !local.is_empty() && tld_ok {
            hit(
                "email",
                &format!("{}@{domain}", local.chars().rev().collect::<String>()),
            );
        }
    }
    for home in ["/Users/", "/home/", "C:\\\\Users"] {
        for (index, _) in text.match_indices(home) {
            if text[index + home.len()..]
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic())
            {
                hit("home_path", home);
            }
        }
    }
    for secret in secrets {
        if text.contains(secret.as_str()) {
            hits.push(Hit {
                kind: "exported_secret",
                excerpt: "<exported credential>".into(),
            });
        }
    }
    hits
}

/// Exported credential values worth searching for.
pub(crate) fn exported_secrets() -> Vec<String> {
    std::env::vars()
        .filter(|(name, value)| {
            (name.ends_with("_API_KEY") || name.ends_with("_TOKEN") || name.ends_with("_SECRET"))
                && value.len() >= 12
        })
        .map(|(_, value)| value)
        .collect()
}

/// Fixtures changed against `base`, plus untracked ones.
fn changed_fixtures(root: &Path, base: &str) -> Result<Vec<String>, String> {
    let git = |args: &[&str]| -> Result<Vec<String>, String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .map_err(|error| format!("git: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "git {}: {}",
                args.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::to_owned)
            .collect())
    };
    let mut files = git(&[
        "diff",
        "--name-only",
        base,
        "--",
        "crates/rig-cassette/fixtures",
    ])?;
    files.extend(git(&[
        "ls-files",
        "--others",
        "--exclude-standard",
        "crates/rig-cassette/fixtures",
    ])?);
    files.sort();
    files.dedup();
    Ok(files
        .into_iter()
        .filter(|file| root.join(file).is_file())
        .collect())
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let mut base = "origin/main".to_owned();
    let mut paths = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base" => base = args.next().cloned().ok_or("--base needs a ref")?,
            path => paths.push(path.to_owned()),
        }
    }
    if paths.is_empty() {
        paths = changed_fixtures(root, &base)?;
    }
    let secrets = exported_secrets();
    let mut total = 0;
    for path in &paths {
        let contents =
            std::fs::read_to_string(root.join(path)).map_err(|error| format!("{path}: {error}"))?;
        for hit in scan(&contents, &secrets) {
            total += 1;
            println!("{path}\t{}\t{}", hit.kind, hit.excerpt);
        }
    }
    println!("scanned {} file(s), {total} hit(s)", paths.len());
    if total == 0 {
        Ok(())
    } else {
        Err(format!("{total} hit(s) to review"))
    }
}

#[cfg(test)]
mod tests;
