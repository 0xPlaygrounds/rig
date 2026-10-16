//! Finding issue and pull request references in what the user typed:
//! `#142`, `!87`, `PR#87` and tracker URLs, outside code spans and fenced
//! blocks.

use std::fmt;

/// What a reference points at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefKind {
    /// An issue: `#142`.
    Issue,
    /// A pull or merge request: `!87` or `PR#87`.
    Pull,
}

/// One issue or pull request of the tracker.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Reference {
    /// The host a URL named; none for `#142`, which is the tracker's.
    pub host: Option<String>,
    /// The repository a URL named (`owner/repo`); none for the current one.
    pub repo: Option<String>,
    /// Issue or pull request.
    pub kind: RefKind,
    /// Its number.
    pub number: u64,
}

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mark = match self.kind {
            RefKind::Issue => '#',
            RefKind::Pull => '!',
        };
        match &self.repo {
            Some(repo) => write!(f, "{repo}{mark}{}", self.number),
            None => write!(f, "{mark}{}", self.number),
        }
    }
}

/// The references in `text`, each once, in the order they first appear;
/// none from code spans or fenced blocks.
pub fn references(text: &str) -> Vec<Reference> {
    let mut found = Vec::new();
    for token in prose(text).split_whitespace() {
        let url = token.trim_matches(|c: char| matches!(c, '(' | ')' | '<' | '>' | ',' | '.'));
        if url.starts_with("http://") || url.starts_with("https://") {
            found.extend(from_url(url));
        } else {
            scan(token, &mut found);
        }
    }
    let mut unique: Vec<Reference> = Vec::with_capacity(found.len());
    for reference in found {
        if !unique.contains(&reference) {
            unique.push(reference);
        }
    }
    unique
}

/// The references in one word without spaces.
fn scan(token: &str, found: &mut Vec<Reference>) {
    let mut before: Option<char> = None;
    for (at, c) in token.char_indices() {
        let boundary = before.is_none_or(|b| !b.is_alphanumeric() && !"/&#!_`".contains(b));
        let rest = token.get(at..).unwrap_or_default();
        let marked = rest
            .strip_prefix("PR#")
            .map(|digits| (RefKind::Pull, digits))
            .or_else(|| {
                rest.strip_prefix('#')
                    .map(|digits| (RefKind::Issue, digits))
            })
            .or_else(|| rest.strip_prefix('!').map(|digits| (RefKind::Pull, digits)));
        if boundary && let Some((kind, digits)) = marked {
            let end = digits
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(digits.len());
            let after = digits.get(end..).and_then(|after| after.chars().next());
            let number = digits.get(..end).and_then(|n| n.parse().ok());
            if let (Some(number), false) = (number, after.is_some_and(char::is_alphanumeric)) {
                found.push(Reference {
                    host: None,
                    repo: None,
                    kind,
                    number,
                });
            }
        }
        before = Some(c);
    }
}

/// A GitHub (`/owner/repo/issues/1`, `/pull/1`) or GitLab
/// (`/group/project/-/issues/1`, `/-/merge_requests/1`) URL.
fn from_url(url: &str) -> Option<Reference> {
    let rest = url.split_once("://")?.1;
    let (host, path) = rest.split_once('/')?;
    let path = path.split(['?', '#']).next()?.trim_end_matches('/');
    let segments: Vec<&str> = path.split('/').collect();
    let (number, segments) = segments.split_last()?;
    let number = number.parse().ok()?;
    let (what, segments) = segments.split_last()?;
    let kind = match *what {
        "issues" => RefKind::Issue,
        "pull" | "pulls" | "merge_requests" => RefKind::Pull,
        _ => return None,
    };
    let segments = segments.strip_suffix(&["-"]).unwrap_or(segments);
    (segments.len() >= 2).then(|| Reference {
        host: Some(host.to_owned()),
        repo: Some(segments.join("/")),
        kind,
        number,
    })
}

/// `text` with every fenced block and code span blanked out.
fn prose(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut fence: Option<String> = None;
    for line in text.split_inclusive('\n') {
        let marker = fence_marker(line.trim_start());
        match (&fence, marker) {
            (Some(open), Some(marker)) if marker.starts_with(open.as_str()) => fence = None,
            (Some(_), _) => {}
            (None, Some(marker)) => fence = Some(marker.to_owned()),
            (None, None) => out.push_str(&without_spans(line)),
        }
        out.push('\n');
    }
    out
}

/// The run of three or more backticks or tildes `line` starts with.
fn fence_marker(line: &str) -> Option<&str> {
    let mark = line.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let run = line.len() - line.trim_start_matches(mark).len();
    (run >= 3).then(|| line.get(..run)).flatten()
}

/// `line` with each code span (`` `x` ``, ```` ``x`` ````) a space.
fn without_spans(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find('`') {
        out.push_str(rest.get(..open).unwrap_or_default());
        let from = rest.get(open..).unwrap_or_default();
        let ticks = from.len() - from.trim_start_matches('`').len();
        let body = from.get(ticks..).unwrap_or_default();
        match closing(body, ticks) {
            Some(close) => {
                out.push(' ');
                rest = body.get(close + ticks..).unwrap_or_default();
            }
            None => {
                out.push_str(from.get(..ticks).unwrap_or_default());
                rest = body;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Where the next run of exactly `ticks` backticks starts in `body`.
fn closing(body: &str, ticks: usize) -> Option<usize> {
    let mut from = 0;
    while let Some(at) = body.get(from..)?.find('`') {
        let start = from + at;
        let run = body.get(start..)?;
        let length = run.len() - run.trim_start_matches('`').len();
        if length == ticks {
            return Some(start);
        }
        from = start + length;
    }
    None
}
