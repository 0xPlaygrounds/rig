//! `cargo xtask cassette bank`: the reply bank. For every completion reply
//! shape the corpus records, per provider and encoder, it keeps one real
//! recorded reply under `crates/rig-cassette/fixtures/bank/<provider>.yaml`,
//! so the runtime scenarios can run once against real replies instead of
//! once per provider's cassette.
//!
//! A reply shape is the coverage gate's (`coverage/shapes.rs`): the status,
//! the framing and the body skeleton with discriminators kept. The bank adds
//! the names of the tools a reply calls to its key, because the runtime
//! dispatches by name: two replies of one shape that call different tools
//! drive different runs. Of the replies with one key the bank keeps the
//! smallest, then the first in path order, and records where it came from.
//! An entry whose source fixture no longer exists stays a candidate, so a
//! pruned cassette does not take its reply out of the bank. A fixture whose
//! owning test marks it hand-derived (`skip_when_recording`) holds bytes no
//! provider returned and gives the bank nothing.
//!
//! Beside the replies, `scripts.tsv` keeps every fixture's reply sequence as
//! bank keys, so a scenario can be served the replies of the shapes it
//! recorded after its own cassette is gone. A script whose fixture no longer
//! exists is kept too.
//!
//! A scenario whose assertions read what a reply says, not only its shape
//! (an answer that must end in a marker, reasoning that must be non-empty),
//! is listed in `pinned.txt` with its reason, and the bank keeps that
//! fixture's replies verbatim in `pinned.yaml`.
//!
//! `--check` fails when the committed bank differs from the one a rewrite
//! would write.
//!
//! The coverage gate counts a bank entry as a recording of its reply shape
//! when the `runtime` target's `decode` tests decode it ([`held_shapes`]):
//! the entry is a verbatim provider reply with its source, and a decoded
//! reply pins its provider's decoder as a cassette would.

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::coverage::shapes::{self, Exchange, Kind, NameValue, ShapeKey, hash};

const CASSETTES: &str = "crates/rig-cassette/fixtures/cassettes";
/// The bank's directory, relative to the workspace root.
pub(crate) const BANK: &str = "crates/rig-cassette/fixtures/bank";
/// The target that decodes every bank reply of the providers its
/// `sweeps!` invocation names, relative to the workspace root.
pub(crate) const DECODE: &str = "crates/rig-cassette/tests/runtime/decode.rs";

const PREAMBLE: &str = "\
# The reply bank, written by `cargo xtask cassette bank`. One real recorded
# reply per provider, completion encoder, reply shape and called tools, the
# smallest the corpus holds, with the interaction it came from. Never edit
# an entry by hand (see tests/README.md).
";

/// Path suffixes of the encoders that answer a completion.
const COMPLETION_PATHS: &[&str] = &[
    "/chat/completions",
    "/converse",
    "/converse-stream",
    "/interactions",
    "/messages",
    "/responses",
    ":generateContent",
    ":streamGenerateContent",
];

/// The list of fixtures whose replies the bank keeps verbatim, each with
/// its reason; written by hand.
pub(crate) const PINNED_LIST: &str = "pinned.txt";
/// The replies of the pinned fixtures.
pub(crate) const PINNED: &str = "pinned.yaml";

/// The scripts file inside the bank directory.
pub(crate) const SCRIPTS: &str = "scripts.tsv";

const SCRIPTS_HEADER: &str =
    "fixture\treplies (encoder;shape;calls, `-` for a reply the bank does not hold)";

/// Keys whose string value says how a turn ended.
const ENDINGS: &[&str] = &["finishReason", "finish_reason", "stopReason", "stop_reason"];

/// Keys whose value is a tool call's arguments: an object with a `name` and
/// one of these is a tool call.
const ARGUMENTS: &[&str] = &["args", "arguments", "input"];

/// Keys a tool's definition has and a call never does.
const DEFINITION: &[&str] = &["description", "input_schema", "parameters"];

/// Call types the provider runs itself: no client tool is dispatched.
const SERVER_CALLS: &[&str] = &["mcp_tool_use", "server_tool_use"];

/// One recorded reply, as the cassette holds it.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct Reply {
    pub(crate) status: u16,
    #[serde(default)]
    pub(crate) header: Vec<NameValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) body: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) body_encoding: Option<String>,
}

impl Reply {
    fn from_exchange(exchange: &Exchange) -> Self {
        Self {
            status: exchange.status,
            header: exchange.header.clone(),
            body: exchange.body.clone(),
            body_encoding: exchange.body_encoding.clone(),
        }
    }

    fn exchange(&self) -> Exchange {
        Exchange {
            path: String::new(),
            method: String::new(),
            status: self.status,
            header: self.header.clone(),
            body: self.body.clone(),
            body_encoding: self.body_encoding.clone(),
        }
    }

    fn size(&self) -> usize {
        self.body.as_ref().map_or(0, String::len)
    }
}

/// One bank entry: a reply and what the runtime reads off it.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct Entry {
    pub(crate) provider: String,
    pub(crate) encoder: String,
    /// The coverage gate's reply shape, hashed.
    pub(crate) shape: String,
    /// The distinct names of the tools the reply calls, sorted.
    pub(crate) calls: Vec<String>,
    /// The distinct values of the reply's ending keys, sorted.
    pub(crate) ends: Vec<String>,
    /// The interaction it was taken from, `<provider>/<scenario>.yaml#<n>`.
    pub(crate) source: String,
    pub(crate) then: Reply,
}

impl Entry {
    fn key(&self) -> (String, String, String, Vec<String>) {
        (
            self.provider.clone(),
            self.encoder.clone(),
            self.shape.clone(),
            self.calls.clone(),
        )
    }

    /// The entry's key inside its provider's file, as a script names it.
    pub(crate) fn script_key(&self) -> String {
        format!("{};{};{}", self.encoder, self.shape, self.calls.join(","))
    }
}

/// Every fixture's replies as bank keys, by fixture path.
pub(crate) type Scripts = BTreeMap<String, Vec<Option<String>>>;

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let check = match args {
        [] => false,
        [flag] if flag == "--check" => true,
        _ => return Err("usage: cassette bank [--check]".into()),
    };
    let cassettes = root.join(CASSETTES);
    let dir = root.join(BANK);
    let committed = read(&dir)?;
    let committed_scripts = match std::fs::read_to_string(dir.join(SCRIPTS)) {
        Ok(text) => parse_scripts(&text)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Scripts::new(),
        Err(error) => return Err(format!("{BANK}/{SCRIPTS}: {error}")),
    };
    let excluded = hand_derived(root)?;
    let (mut candidates, mut scripts) = observe(&cassettes, &excluded)?;
    let pinned = pin(&dir, &cassettes, &candidates)?;
    // An entry whose fixture is gone is still a real recording; what the
    // bank reads off it is read again.
    candidates.extend(
        committed
            .iter()
            .filter(|kept| !source_exists(&cassettes, &kept.source))
            .map(|kept| {
                entry(
                    &kept.provider,
                    &kept.encoder,
                    &kept.then.exchange(),
                    kept.source.clone(),
                )
            }),
    );
    for (fixture, script) in committed_scripts {
        if !cassettes.join(&fixture).is_file() {
            scripts.insert(fixture, script);
        }
    }
    let bank = build(candidates);
    let mut rendered = render(&bank)?;
    rendered.insert(SCRIPTS.to_owned(), render_scripts(&scripts));
    rendered.insert(PINNED.to_owned(), render_entries(&pinned)?);
    let per_provider: BTreeMap<&str, usize> =
        bank.iter().fold(BTreeMap::new(), |mut counts, entry| {
            *counts.entry(entry.provider.as_str()).or_default() += 1;
            counts
        });
    for (provider, count) in &per_provider {
        println!("bank {provider}: {count} replies");
    }
    let size: usize = rendered.values().map(String::len).sum();
    println!(
        "bank: {} replies, {} pinned, {} scripts, {size} bytes; {} hand-derived fixture(s) left out",
        bank.len(),
        pinned.len(),
        scripts.len(),
        excluded.len()
    );
    if check {
        let committed_text = read_texts(&dir)?;
        if committed_text != rendered {
            return Err(format!(
                "{BANK} is not the bank the corpus gives; rewrite it with \
                 `cargo xtask cassette bank` and review its diff"
            ));
        }
        return Ok(());
    }
    std::fs::create_dir_all(&dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for (file, _) in read_texts(&dir)? {
        if !rendered.contains_key(&file) {
            let path = dir.join(&file);
            std::fs::remove_file(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        }
    }
    for (file, text) in &rendered {
        let path = dir.join(file);
        std::fs::write(&path, text).map_err(|error| format!("{}: {error}", path.display()))?;
    }
    println!("wrote {}", dir.display());
    Ok(())
}

/// The fixtures (`<provider>/<scenario>.yaml`) whose owning test marks them
/// hand-derived.
fn hand_derived(root: &Path) -> Result<BTreeSet<String>, String> {
    let providers = root.join("crates/rig-cassette/tests/providers");
    let cassettes = root.join(CASSETTES);
    let mut excluded = BTreeSet::new();
    for file in crate::support::files_under(&providers, Some("rs"))? {
        let source = std::fs::read_to_string(&file)
            .map_err(|error| format!("{}: {error}", file.display()))?;
        if !source.contains("skip_when_recording(") {
            continue;
        }
        let Some(provider) = file
            .strip_prefix(&providers)
            .ok()
            .and_then(|relative| relative.iter().next())
            .map(|name| name.to_string_lossy().into_owned())
        else {
            continue;
        };
        for scenario in string_literals(&source) {
            let fixture = format!("{provider}/{scenario}.yaml");
            if !cassettes.join(&fixture).is_file() {
                continue;
            }
            let owners = super::owner::owners(root, &provider, &scenario)?;
            if owners
                .iter()
                .any(|owner| matches!(owner, super::owner::Owner::HandDerived(_)))
            {
                excluded.insert(fixture);
            }
        }
    }
    Ok(excluded)
}

/// The contents of every plain string literal in `source` that looks like a
/// scenario (`dir/name`).
fn string_literals(source: &str) -> BTreeSet<String> {
    source
        .split('"')
        .skip(1)
        .step_by(2)
        .filter(|text| {
            text.contains('/')
                && text
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '/' | '.'))
        })
        .map(str::to_owned)
        .collect()
}

/// Whether the encoder answers a completion.
pub(crate) fn is_completion(encoder: &str) -> bool {
    encoder.starts_with("POST ") && COMPLETION_PATHS.iter().any(|path| encoder.ends_with(path))
}

/// Every completion reply of the corpus outside `excluded`, as a candidate
/// entry, and every fixture's script.
fn observe(cassettes: &Path, excluded: &BTreeSet<String>) -> Result<(Vec<Entry>, Scripts), String> {
    let mut entries = Vec::new();
    let mut scripts = Scripts::new();
    for fixture in shapes::corpus(cassettes).map_err(|error| error.to_string())? {
        if excluded.contains(&fixture.relative) {
            continue;
        }
        let mut script = Vec::new();
        for (index, interaction) in fixture.interactions.iter().enumerate() {
            let encoder = interaction.when.encoder();
            if !is_completion(&encoder) {
                script.push(None);
                continue;
            }
            let entry = entry(
                &fixture.provider,
                &encoder,
                &interaction.then,
                format!("{}#{index}", fixture.relative),
            );
            script.push(Some(entry.script_key()));
            entries.push(entry);
        }
        if script.iter().any(Option::is_some) {
            scripts.insert(fixture.relative, script);
        }
    }
    Ok((entries, scripts))
}

/// The scripts file's text.
pub(crate) fn render_scripts(scripts: &Scripts) -> String {
    let mut out = format!("{SCRIPTS_HEADER}\n");
    for (fixture, script) in scripts {
        out.push_str(fixture);
        for key in script {
            out.push('\t');
            out.push_str(key.as_deref().unwrap_or("-"));
        }
        out.push('\n');
    }
    out
}

/// Parse a scripts file written by [`render_scripts`].
pub(crate) fn parse_scripts(text: &str) -> Result<Scripts, String> {
    let mut scripts = Scripts::new();
    for (number, line) in text.lines().enumerate().skip(1) {
        let mut columns = line.split('\t');
        let fixture = columns
            .next()
            .filter(|fixture| !fixture.is_empty())
            .ok_or_else(|| format!("{SCRIPTS} line {}: no fixture", number + 1))?;
        let script = columns
            .map(|key| (key != "-").then(|| key.to_owned()))
            .collect();
        scripts.insert(fixture.to_owned(), script);
    }
    Ok(scripts)
}

/// A candidate entry for one recorded reply.
pub(crate) fn entry(provider: &str, encoder: &str, reply: &Exchange, source: String) -> Entry {
    let documents = shapes::reply_documents(reply);
    let mut calls = BTreeSet::new();
    let mut ends = BTreeSet::new();
    for document in &documents {
        walk(document, None, &mut calls, &mut ends);
    }
    Entry {
        provider: provider.to_owned(),
        encoder: encoder.to_owned(),
        shape: hash(&shapes::reply_shape(reply)),
        calls: calls.into_iter().collect(),
        ends: ends.into_iter().collect(),
        source,
        then: Reply::from_exchange(reply),
    }
}

/// Collect the tool names `value` calls and the values of its ending keys.
/// A Responses object's `status` and its `incomplete_details.reason` end it
/// too.
fn walk(
    value: &Value,
    parent: Option<&str>,
    calls: &mut BTreeSet<String>,
    ends: &mut BTreeSet<String>,
) {
    match value {
        Value::Array(items) => {
            for item in items {
                walk(item, parent, calls, ends);
            }
        }
        Value::Object(map) => {
            if let Some(Value::String(name)) = map.get("name")
                && is_call(map, parent)
            {
                calls.insert(name.clone());
            }
            if let Some(Value::String(status)) = map.get("status")
                && map.contains_key("output")
            {
                ends.insert(status.clone());
            }
            for (key, child) in map {
                match child {
                    Value::String(text)
                        if ENDINGS.contains(&key.as_str())
                            || (key == "reason" && parent == Some("incomplete_details")) =>
                    {
                        ends.insert(text.clone());
                    }
                    _ => walk(child, Some(key), calls, ends),
                }
            }
        }
        _ => {}
    }
}

/// Whether an object with a `name` is a call of a client tool: it carries
/// arguments, or it is a function under `function` (a streamed call's first
/// delta names the function before any argument), and it is neither a
/// definition nor a call the provider runs itself.
fn is_call(map: &serde_json::Map<String, Value>, parent: Option<&str>) -> bool {
    let server = map
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| SERVER_CALLS.contains(&kind));
    let definition = DEFINITION.iter().any(|key| map.contains_key(*key));
    let arguments = ARGUMENTS.iter().any(|key| map.contains_key(*key));
    !server && !definition && (arguments || matches!(parent, Some("function" | "functionCall")))
}

/// One entry per key: the smallest reply, then the first source in order.
pub(crate) fn build(candidates: Vec<Entry>) -> Vec<Entry> {
    let mut chosen: BTreeMap<(String, String, String, Vec<String>), Entry> = BTreeMap::new();
    for candidate in candidates {
        let key = candidate.key();
        let better = chosen.get(&key).is_none_or(|current| {
            (candidate.then.size(), &candidate.source) < (current.then.size(), &current.source)
        });
        if better {
            chosen.insert(key, candidate);
        }
    }
    chosen.into_values().collect()
}

/// The bank's files, one per provider, as text.
pub(crate) fn render(bank: &[Entry]) -> Result<BTreeMap<String, String>, String> {
    let mut by_provider: BTreeMap<String, Vec<Entry>> = BTreeMap::new();
    for entry in bank {
        by_provider
            .entry(format!("{}.yaml", entry.provider))
            .or_default()
            .push(entry.clone());
    }
    by_provider
        .into_iter()
        .map(|(file, entries)| Ok((file, render_entries(&entries)?)))
        .collect()
}

/// One bank file's text: the preamble, then every entry as a document.
fn render_entries(entries: &[Entry]) -> Result<String, String> {
    let mut text = PREAMBLE.to_owned();
    for (index, entry) in entries.iter().enumerate() {
        if index > 0 {
            text.push_str("---\n");
        }
        text.push_str(&serde_yaml::to_string(entry).map_err(|error| error.to_string())?);
    }
    Ok(text)
}

/// The replies of every fixture `pinned.txt` lists, in fixture and
/// interaction order: from the corpus while the fixture exists, else as the
/// committed bank kept them.
fn pin(dir: &Path, cassettes: &Path, candidates: &[Entry]) -> Result<Vec<Entry>, String> {
    let listed = match std::fs::read_to_string(dir.join(PINNED_LIST)) {
        Ok(text) => parse_pinned(&text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => BTreeSet::new(),
        Err(error) => return Err(format!("{BANK}/{PINNED_LIST}: {error}")),
    };
    let committed = match std::fs::read_to_string(dir.join(PINNED)) {
        Ok(text) => parse(&text).map_err(|error| format!("{BANK}/{PINNED}: {error}"))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(format!("{BANK}/{PINNED}: {error}")),
    };
    let mut pinned = Vec::new();
    for fixture in &listed {
        let source = if cassettes.join(fixture).is_file() {
            candidates
        } else {
            committed.as_slice()
        };
        let mut replies: Vec<Entry> = source
            .iter()
            .filter(|entry| fixture_of(&entry.source) == fixture)
            .cloned()
            .collect();
        if replies.is_empty() {
            return Err(format!(
                "{BANK}/{PINNED_LIST} pins {fixture}, which holds no completion reply"
            ));
        }
        replies.sort_by_key(|entry| interaction_of(&entry.source));
        pinned.extend(replies);
    }
    Ok(pinned)
}

/// The fixtures a pinned list names: the first word of every line that is
/// not blank or a comment.
pub(crate) fn parse_pinned(text: &str) -> BTreeSet<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

fn fixture_of(source: &str) -> &str {
    source
        .rsplit_once('#')
        .map_or(source, |(fixture, _)| fixture)
}

fn interaction_of(source: &str) -> usize {
    source
        .rsplit_once('#')
        .and_then(|(_, index)| index.parse().ok())
        .unwrap_or(0)
}

/// Every committed entry.
pub(crate) fn read(dir: &Path) -> Result<Vec<Entry>, String> {
    let mut entries = Vec::new();
    for (file, text) in read_texts(dir)? {
        if file == SCRIPTS || file == PINNED {
            continue;
        }
        entries.extend(parse(&text).map_err(|error| format!("{BANK}/{file}: {error}"))?);
    }
    Ok(entries)
}

/// The entries of one bank file.
pub(crate) fn parse(text: &str) -> Result<Vec<Entry>, serde_yaml::Error> {
    serde_yaml::Deserializer::from_str(text)
        .map(Entry::deserialize)
        .collect()
}

/// The committed bank files (the replies and the scripts) by name; none
/// when the directory is missing.
fn read_texts(dir: &Path) -> Result<BTreeMap<String, String>, String> {
    let mut texts = BTreeMap::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(texts),
        Err(error) => return Err(format!("{}: {error}", dir.display())),
    };
    for entry in entries {
        let path = entry
            .map_err(|error| format!("{}: {error}", dir.display()))?
            .path();
        let banked = path.extension().is_some_and(|ext| ext == "yaml")
            || path.file_name().is_some_and(|name| name == SCRIPTS);
        if banked && let Some(name) = path.file_name() {
            let text = std::fs::read_to_string(&path)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            texts.insert(name.to_string_lossy().into_owned(), text);
        }
    }
    Ok(texts)
}

/// Whether `source` (`<fixture>#<n>`) names a fixture under `cassettes`.
fn source_exists(cassettes: &Path, source: &str) -> bool {
    cassettes.join(fixture_of(source)).is_file()
}

/// The reply shapes the bank holds for the coverage gate, each with the
/// source of its first entry: every entry of a provider the `decode`
/// target sweeps. That target fails on an entry it cannot decode, so each
/// of these replies passes through its provider's decoder in every replay.
pub(crate) fn held_shapes(root: &Path) -> Result<BTreeMap<ShapeKey, String>, String> {
    let source =
        std::fs::read_to_string(root.join(DECODE)).map_err(|error| format!("{DECODE}: {error}"))?;
    let swept = swept_providers(&source)
        .ok_or_else(|| format!("{DECODE}: no `sweeps!(...)` invocation names a provider"))?;
    let mut held = BTreeMap::new();
    for entry in read(&root.join(BANK))? {
        if swept.contains(entry.provider.as_str()) {
            held.entry(ShapeKey {
                provider: entry.provider,
                encoder: entry.encoder,
                kind: Kind::Reply,
                shape: entry.shape,
            })
            .or_insert(entry.source);
        }
    }
    Ok(held)
}

/// The providers the `sweeps!(...)` invocation of the decode target names;
/// `None` when there is no such invocation or it names none.
pub(crate) fn swept_providers(source: &str) -> Option<BTreeSet<&str>> {
    let (_, rest) = source.split_once("\nsweeps!(")?;
    let (list, _) = rest.split_once(')')?;
    let providers: BTreeSet<&str> = list
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect();
    (!providers.is_empty()).then_some(providers)
}
