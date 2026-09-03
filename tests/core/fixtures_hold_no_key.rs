//! No committed fixture holds a key. The cassette recorder scrubs
//! sensitive headers, Gemini's `key` query parameter and key-bearing body
//! fields before writing (`tests/common/cassettes.rs`); this guard is the
//! check on the tree itself — every cassette and every effect-log golden —
//! that the scrubbing held, so a recording session can never commit what
//! the recorder was supposed to remove.
//!
//! Three rules. The exported keys themselves (every `*_API_KEY`, `*_TOKEN`
//! and `*_SECRET` in the environment) are compared in memory and reported
//! by env **name** and `path:line`, never by value. Key-shaped tokens of the
//! providers' documented formats are reported wherever they appear. A
//! sensitive header or query parameter whose recorded value is not the
//! scrubber's placeholder is reported by name.

use std::path::{Path, PathBuf};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Every `.yaml` under `tests/cassettes` and every `.effects.json` under
/// `crates/rig-verify/fixtures`.
fn fixtures() -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![
        root().join("tests/cassettes"),
        root().join("crates/rig-verify/fixtures"),
    ];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("a fixture directory") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path
                .extension()
                .is_some_and(|ext| ext == "yaml" || ext == "json")
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// The exported secrets, by env name: long enough that a substring match
/// is a match, never a coincidence.
fn exported_secrets() -> Vec<(String, String)> {
    std::env::vars()
        .filter(|(name, value)| {
            (name.ends_with("_API_KEY") || name.ends_with("_TOKEN") || name.ends_with("_SECRET"))
                && value.len() >= 12
        })
        .collect()
}

/// The prefix of a documented key format and the alphabet and minimum
/// length of what follows it.
struct KeyShape {
    prefix: &'static str,
    alphabet: fn(char) -> bool,
    min_len: usize,
}

fn base64url(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'
}

fn upper_digit(ch: char) -> bool {
    ch.is_ascii_uppercase() || ch.is_ascii_digit()
}

fn token(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')
}

const KEY_SHAPES: &[KeyShape] = &[
    KeyShape {
        prefix: "sk-proj-",
        alphabet: base64url,
        min_len: 20,
    },
    KeyShape {
        prefix: "sk-ant-api",
        alphabet: base64url,
        min_len: 20,
    },
    KeyShape {
        prefix: "AIzaSy",
        alphabet: base64url,
        min_len: 30,
    },
    KeyShape {
        prefix: "AKIA",
        alphabet: upper_digit,
        min_len: 16,
    },
    KeyShape {
        prefix: "Bearer ",
        alphabet: token,
        min_len: 20,
    },
];

/// Whether `line` holds a token of `shape`: the prefix followed by at least
/// `min_len` characters of its alphabet, where the token then ends.
fn holds_shape(line: &str, shape: &KeyShape) -> bool {
    line.match_indices(shape.prefix).any(|(at, _)| {
        let rest = &line[at + shape.prefix.len()..];
        let run = rest.chars().take_while(|ch| (shape.alphabet)(*ch)).count();
        run >= shape.min_len && (shape.prefix != "AKIA" || run == 16)
    })
}

/// Header and query-parameter names whose value is a credential; the
/// recorder writes them only as its placeholder.
const SENSITIVE_NAMES: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "ocp-apim-subscription-key",
    "key",
    "x-amz-security-token",
];

fn is_placeholder(value: &str) -> bool {
    value.contains("REDACTED")
}

#[test]
fn fixtures_hold_no_key() {
    let secrets = exported_secrets();
    let mut findings = Vec::new();
    for path in fixtures() {
        let text = std::fs::read_to_string(&path).expect("a fixture reads");
        let relative = path
            .strip_prefix(root())
            .expect("under the root")
            .display()
            .to_string();
        let lines: Vec<&str> = text.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            let number = index + 1;
            for (name, value) in &secrets {
                if line.contains(value.as_str()) {
                    findings.push(format!("{relative}:{number}: holds the value of ${name}"));
                }
            }
            for shape in KEY_SHAPES {
                if holds_shape(line, shape) {
                    findings.push(format!(
                        "{relative}:{number}: holds a `{}…` shaped token",
                        shape.prefix.trim_end()
                    ));
                }
            }
            if let Some(name) = line.trim_start().strip_prefix("- name: ")
                && SENSITIVE_NAMES.contains(&name.trim().to_ascii_lowercase().as_str())
                && let Some(next) = lines.get(index + 1)
                && let Some(value) = next.trim_start().strip_prefix("value:")
                && !is_placeholder(value)
            {
                findings.push(format!(
                    "{relative}:{}: `{}` carries a recorded value, not the placeholder",
                    number + 1,
                    name.trim()
                ));
            }
        }
    }
    assert!(
        findings.is_empty(),
        "committed fixtures hold a key:\n{}",
        findings.join("\n")
    );
}
