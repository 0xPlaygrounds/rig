//! `cargo xtask cassette acceptance`: the acceptance index. For each request
//! fact Rig sends, per provider and encoder, it names one cassette
//! interaction recorded live whose request holds that fact, in
//! `crates/rig-cassette/fixtures/acceptance.toml`.
//!
//! A fact is one object of a request body, as the coverage gate reads it
//! (`coverage/shapes.rs`): its path, its keys, and each key's type, with
//! discriminator values kept. What Rig sends is read offline from the corpus:
//! each recorded request with its request snapshot applied, which every
//! workspace replay checks against what Rig actually sends. A recording whose
//! request body the proxy recorder dropped holds no facts of its own; the
//! index pins the ones its snapshot showed when the fixture was first
//! indexed, and drops the pins when the fixture changes.
//!
//! `--check` fails when a fact Rig sends has no live recording, naming the
//! smallest cassette that sends it to re-record, and when the committed index
//! differs from the one a rewrite would write.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer};
use serde_json::Value;

use crate::coverage::shapes::{self, DISCRIMINATORS, Exchange, Fact, Facts, Fixture, facts};

const CASSETTES: &str = "crates/rig-cassette/fixtures/cassettes";
/// The index, relative to the workspace root.
pub(crate) const INDEX: &str = "crates/rig-cassette/fixtures/acceptance.toml";

const PREAMBLE: &str = "\
# The acceptance index, written by `cargo xtask cassette acceptance`.
# Every request fact Rig sends, per provider and encoder, with one cassette
# interaction recorded live whose request holds it. A fact is one object of
# the body: its path with array indices collapsed, its keys, and each key's
# type, with the values of discriminators such as `role` and `type` kept
# (see tests/README.md).
";

const UNRECORDED: &str = "\
# Recordings whose request body the proxy recorder dropped, each with the
# fixture hash it was indexed at and the facts its snapshot then showed.
";

/// Where a fact is sent or recorded: a provider, an encoder and the fact.
type Cell = (String, String, Fact);

/// One index entry: a fact and the interaction that recorded it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) provider: String,
    pub(crate) encoder: String,
    pub(crate) fact: Fact,
    pub(crate) recording: String,
}

/// A fact of a recording the proxy recorder kept no request body for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Pin {
    pub(crate) recording: String,
    pub(crate) fixture: String,
    pub(crate) fact: Fact,
}

/// The whole index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Index {
    pub(crate) facts: Vec<Entry>,
    pub(crate) unrecorded: Vec<Pin>,
}

/// One interaction of the corpus: what was recorded, and what Rig sends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Observed {
    pub(crate) provider: String,
    pub(crate) encoder: String,
    pub(crate) recording: String,
    /// The fixture's hash.
    pub(crate) fixture: String,
    /// The fixture's length in bytes.
    pub(crate) size: usize,
    /// The recorded request's facts; `None` when the body was not recorded.
    pub(crate) recorded: Option<Facts>,
    /// The facts of the request Rig sends.
    pub(crate) sent: Facts,
}

/// A fact Rig sends with no live recording, and the smallest cassette
/// interaction that sends it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Missing {
    pub(crate) provider: String,
    pub(crate) encoder: String,
    pub(crate) fact: Fact,
    pub(crate) rerecord: String,
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let check = match args {
        [] => false,
        [flag] if flag == "--check" => true,
        _ => return Err("usage: cassette acceptance [--check]".into()),
    };
    let cassettes = root.join(CASSETTES);
    let observed = observe(&cassettes)?;
    let path = root.join(INDEX);
    let committed = match std::fs::read_to_string(&path) {
        Ok(text) => Some(parse(&text)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let (index, missing) = build(&observed, committed.as_ref().unwrap_or(&Index::default()));
    println!(
        "acceptance: {} fact(s) over {} interaction(s), {} unrecorded body pin(s)",
        index.facts.len(),
        observed.len(),
        index.unrecorded.len()
    );
    let mut failures: Vec<String> = missing
        .iter()
        .map(|missing| {
            format!(
                "{} `{}` at {}: {} has no live recording; re-record {}, \
                 the smallest cassette that sends it",
                missing.provider,
                missing.encoder,
                missing.fact.path,
                missing.fact.object,
                missing.rerecord
            )
        })
        .collect();
    let text = render(&index);
    if check {
        if committed.as_ref() != Some(&index) {
            failures.push(format!(
                "{INDEX} is not the index the corpus gives; rewrite it with \
                 `cargo xtask cassette acceptance` and review its diff"
            ));
        }
    } else {
        std::fs::write(&path, &text).map_err(|error| format!("{}: {error}", path.display()))?;
        println!("wrote {}", path.display());
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} acceptance failure(s):\n  {}",
            failures.len(),
            failures.join("\n  ")
        ))
    }
}

/// Every interaction under `cassettes`, with its snapshot applied.
pub(crate) fn observe(cassettes: &Path) -> Result<Vec<Observed>, String> {
    let mut observed = Vec::new();
    for fixture in shapes::corpus(cassettes).map_err(|error| error.to_string())? {
        let snapshot = read_snapshot(&snapshot_path(&cassettes.join(&fixture.relative)))?;
        observed.extend(observe_fixture(&fixture, &snapshot)?);
    }
    Ok(observed)
}

/// The snapshot beside a fixture, as rig-cassette names it.
fn snapshot_path(fixture: &Path) -> PathBuf {
    fixture.with_extension("requests.json")
}

#[derive(Debug, Default, Deserialize)]
struct SnapshotFile {
    interactions: Vec<InteractionChanges>,
}

#[derive(Debug, Deserialize)]
struct InteractionChanges {
    index: usize,
    changes: Vec<Change>,
}

/// A snapshot change, as rig-cassette writes it.
#[derive(Debug, Deserialize)]
pub(crate) struct Change {
    path: String,
    #[serde(default)]
    splice: Option<usize>,
    #[serde(default, deserialize_with = "present")]
    recorded: Option<Value>,
    #[serde(default, deserialize_with = "present")]
    sent: Option<Value>,
}

/// A present field is `Some`, even when it holds `null`.
fn present<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
}

/// Each interaction's snapshot changes; none when the file is absent.
fn read_snapshot(path: &Path) -> Result<BTreeMap<usize, Vec<Change>>, String> {
    let file: SnapshotFile = match std::fs::read_to_string(path) {
        Ok(text) => {
            serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => SnapshotFile::default(),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    Ok(file
        .interactions
        .into_iter()
        .map(|entry| (entry.index, entry.changes))
        .collect())
}

fn observe_fixture(
    fixture: &Fixture,
    snapshot: &BTreeMap<usize, Vec<Change>>,
) -> Result<Vec<Observed>, String> {
    let mut observed = Vec::new();
    for (index, interaction) in fixture.interactions.iter().enumerate() {
        let recording = format!("{}#{index}", fixture.relative);
        let request = &interaction.when;
        let unrecorded =
            request.body.is_none() && request.content_type().starts_with("multipart/form-data");
        let recorded = shapes::request_facts(request);
        let sent = match snapshot.get(&index) {
            Some(changes) => {
                let mut view = recorded_view(request);
                for change in changes {
                    apply(&mut view, change).map_err(|reason| {
                        format!("{recording}: snapshot {}: {reason}", change.path)
                    })?;
                }
                view_facts(&view)
            }
            None => recorded.clone(),
        };
        observed.push(Observed {
            provider: fixture.provider.clone(),
            encoder: request.encoder(),
            recording,
            fixture: fixture.hash.clone(),
            size: fixture.size,
            recorded: (!unrecorded).then_some(recorded),
            sent,
        });
    }
    Ok(observed)
}

/// The recorded body in the form snapshot paths point into: JSON as
/// parsed, multipart as its parts' headers, anything else whole.
fn recorded_view(request: &Exchange) -> Value {
    let Some(body) = &request.body else {
        return Value::Null;
    };
    if request.is_base64() {
        return serde_json::json!({ "bytes": body.len(), "fnv1a64": "" });
    }
    if let Some(boundary) = multipart_boundary(request.content_type()) {
        let marker = format!("--{boundary}");
        let parts: Vec<Value> = body
            .split(marker.as_str())
            .skip(1)
            .filter_map(|part| {
                let (headers, _) = part.trim_start_matches("\r\n").split_once("\r\n\r\n")?;
                let headers: serde_json::Map<String, Value> = headers
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .map(|(name, value)| {
                        (
                            name.trim().to_ascii_lowercase(),
                            Value::from(value.trim().to_owned()),
                        )
                    })
                    .collect();
                Some(serde_json::json!({ "headers": headers, "body": null }))
            })
            .collect();
        return serde_json::json!({ "multipart": parts });
    }
    serde_json::from_str(body).unwrap_or_else(|_| Value::from(body.clone()))
}

fn multipart_boundary(content_type: &str) -> Option<&str> {
    content_type
        .strip_prefix("multipart/form-data")?
        .split(';')
        .find_map(|part| part.trim().strip_prefix("boundary="))
        .map(|boundary| boundary.trim_matches('"'))
        .filter(|boundary| !boundary.is_empty())
}

/// Apply one snapshot change. Replay already checked that every change
/// finds what it says was recorded, so this only follows the pointers.
pub(crate) fn apply(view: &mut Value, change: &Change) -> Result<(), String> {
    match (change.splice, &change.recorded, &change.sent) {
        (Some(at), Some(Value::Array(removed)), Some(Value::Array(inserted))) => {
            let Some(Value::Array(items)) = view.pointer_mut(&change.path) else {
                return Err("no array here".into());
            };
            let end = at.saturating_add(removed.len()).min(items.len());
            if at > end {
                return Err("the splice starts past the array".into());
            }
            items.splice(at..end, inserted.iter().cloned());
            Ok(())
        }
        (Some(_), _, _) => Err("a splice needs recorded and sent arrays".into()),
        (None, Some(_), Some(sent)) => {
            let slot = view.pointer_mut(&change.path).ok_or("no value here")?;
            *slot = sent.clone();
            Ok(())
        }
        (None, recorded, sent) => {
            let (parent, token) = change
                .path
                .rsplit_once('/')
                .ok_or("only an object key is added or removed")?;
            let Some(Value::Object(map)) = view.pointer_mut(parent) else {
                return Err("no object here".into());
            };
            let key = token.replace("~1", "/").replace("~0", "~");
            match (recorded, sent) {
                (None, Some(sent)) => {
                    map.insert(key, sent.clone());
                }
                (Some(_), None) => {
                    map.shift_remove(&key);
                }
                _ => return Err("a change needs a recorded or a sent value".into()),
            }
            Ok(())
        }
    }
}

/// The facts of a body in its snapshot form, as
/// [`shapes::request_facts`] reads the recorded one.
pub(crate) fn view_facts(view: &Value) -> Facts {
    let root = |object: String| {
        Facts::from([Fact {
            path: "$".to_owned(),
            object,
        }])
    };
    match view {
        Value::Null => root("empty".to_owned()),
        Value::String(_) => root("text".to_owned()),
        Value::Object(map) if map.len() == 2 && map.contains_key("bytes") => {
            root("binary".to_owned())
        }
        Value::Object(map) if map.len() == 1 && map.contains_key("multipart") => {
            let names: BTreeSet<&str> = map
                .get("multipart")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|part| part.pointer("/headers/content-disposition")?.as_str())
                .filter_map(|disposition| disposition.split("name=\"").nth(1)?.split('"').next())
                .collect();
            root(format!(
                "multipart[{}]",
                names.into_iter().collect::<Vec<_>>().join(",")
            ))
        }
        value => facts(value, DISCRIMINATORS),
    }
}

/// The index the corpus gives, and every sent fact no live recording holds.
/// `committed` contributes only its pins: a bodyless recording keeps the
/// facts it was indexed with while its fixture is unchanged.
pub(crate) fn build(observed: &[Observed], committed: &Index) -> (Index, Vec<Missing>) {
    let mut old_pins: BTreeMap<&str, (&str, Facts)> = BTreeMap::new();
    for pin in &committed.unrecorded {
        old_pins
            .entry(pin.recording.as_str())
            .or_insert_with(|| (pin.fixture.as_str(), Facts::new()))
            .1
            .insert(pin.fact.clone());
    }
    let mut pins = Vec::new();
    // Per cell: the recordings that hold it, with those whose request Rig
    // still sends with the fact first.
    let mut live: BTreeMap<Cell, BTreeSet<(bool, &str)>> = BTreeMap::new();
    for item in observed {
        let recorded = match &item.recorded {
            Some(recorded) => recorded.clone(),
            None => {
                let facts = match old_pins.get(item.recording.as_str()) {
                    Some((fixture, facts)) if *fixture == item.fixture => facts.clone(),
                    _ => item.sent.clone(),
                };
                pins.extend(facts.iter().map(|fact| Pin {
                    recording: item.recording.clone(),
                    fixture: item.fixture.clone(),
                    fact: fact.clone(),
                }));
                facts
            }
        };
        for fact in recorded {
            let dropped = !item.sent.contains(&fact);
            live.entry((item.provider.clone(), item.encoder.clone(), fact))
                .or_default()
                .insert((dropped, item.recording.as_str()));
        }
    }
    // Per cell Rig sends: the smallest fixture that sends it.
    let mut sent: BTreeMap<Cell, (usize, &str)> = BTreeMap::new();
    for item in observed {
        for fact in &item.sent {
            let candidate = (item.size, item.recording.as_str());
            sent.entry((item.provider.clone(), item.encoder.clone(), fact.clone()))
                .and_modify(|best| *best = (*best).min(candidate))
                .or_insert(candidate);
        }
    }
    let mut index = Index {
        facts: Vec::new(),
        unrecorded: pins,
    };
    let mut missing = Vec::new();
    for (cell, (_, rerecord)) in sent {
        let recording = live.get(&cell).and_then(|recordings| recordings.first());
        let (provider, encoder, fact) = cell;
        match recording {
            Some((_, recording)) => index.facts.push(Entry {
                provider,
                encoder,
                fact,
                recording: (*recording).to_owned(),
            }),
            None => missing.push(Missing {
                provider,
                encoder,
                fact,
                rerecord: rerecord.to_owned(),
            }),
        }
    }
    (index, missing)
}

/// The index as TOML: one inline table per line.
pub(crate) fn render(index: &Index) -> String {
    let mut out = format!("{PREAMBLE}facts = [\n");
    for entry in &index.facts {
        let _ = writeln!(
            out,
            "  {{ provider = {}, encoder = {}, path = {}, fact = {}, recording = {} }},",
            quote(&entry.provider),
            quote(&entry.encoder),
            quote(&entry.fact.path),
            quote(&entry.fact.object),
            quote(&entry.recording)
        );
    }
    out.push_str("]\n\n");
    out.push_str(UNRECORDED);
    out.push_str("unrecorded = [\n");
    for pin in &index.unrecorded {
        let _ = writeln!(
            out,
            "  {{ recording = {}, fixture = {}, path = {}, fact = {} }},",
            quote(&pin.recording),
            quote(&pin.fixture),
            quote(&pin.fact.path),
            quote(&pin.fact.object)
        );
    }
    out.push_str("]\n");
    out
}

fn quote(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Parse an index written by [`render`].
pub(crate) fn parse(text: &str) -> Result<Index, String> {
    let mut index = Index::default();
    let mut section = None;
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line == "]" {
            continue;
        }
        if let Some(name) = line.strip_suffix(" = [") {
            section = Some(name.to_owned());
            continue;
        }
        let at = |reason: &str| format!("{INDEX} line {}: {reason}: {line}", number + 1);
        let fields = inline_table(line).ok_or_else(|| at("not an inline table"))?;
        let field = |name: &str| {
            fields
                .get(name)
                .cloned()
                .ok_or_else(|| at(&format!("no `{name}`")))
        };
        let fact = || -> Result<Fact, String> {
            Ok(Fact {
                path: field("path")?,
                object: field("fact")?,
            })
        };
        match section.as_deref() {
            Some("facts") => index.facts.push(Entry {
                provider: field("provider")?,
                encoder: field("encoder")?,
                fact: fact()?,
                recording: field("recording")?,
            }),
            Some("unrecorded") => index.unrecorded.push(Pin {
                recording: field("recording")?,
                fixture: field("fixture")?,
                fact: fact()?,
            }),
            _ => return Err(at("outside `facts` and `unrecorded`")),
        }
    }
    Ok(index)
}

/// The string fields of `{ key = "value", ... },`.
fn inline_table(line: &str) -> Option<BTreeMap<String, String>> {
    let body = line
        .strip_suffix(',')
        .unwrap_or(line)
        .strip_prefix('{')?
        .strip_suffix('}')?;
    let mut fields = BTreeMap::new();
    let mut rest = body.trim_start();
    while !rest.is_empty() {
        let (key, after) = rest.split_once('=')?;
        let after = after.trim_start().strip_prefix('"')?;
        let mut value = String::new();
        let mut chars = after.char_indices();
        let end = loop {
            match chars.next()? {
                (_, '\\') => value.push(chars.next()?.1),
                (offset, '"') => break offset,
                (_, ch) => value.push(ch),
            }
        };
        fields.insert(key.trim().to_owned(), value);
        rest = after.get(end + 1..)?.trim_start();
        rest = rest.strip_prefix(',').unwrap_or(rest).trim_start();
    }
    Some(fields)
}

#[cfg(test)]
mod tests;
