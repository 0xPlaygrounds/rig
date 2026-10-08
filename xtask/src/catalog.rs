//! `cargo xtask catalog sync`: generate rig-core's model catalog,
//! `crates/rig-core/src/catalog/models.json`, from models.dev and the facts
//! reviewed by hand in `xtask/src/catalog/review.json`.
//!
//! The steps, in order:
//!
//! 1. Keep the models.dev providers rig serves, renamed to rig's vendor
//!    names ([`KEYS`]), and in each model row the fields the catalog reads
//!    ([`trim`]).
//! 2. Lay each reviewed row over its models.dev row, or add it: objects merge
//!    key by key, anything else replaces. A reviewed row that names
//!    `"from": "<vendor>/<id>"` starts from that generated row instead. The
//!    review's own keys (`from`, `source`, `note`) are dropped.
//! 3. Give each gateway row whose models.dev `canonical_model_id` names an
//!    Anthropic model that model's reviewed facts ([`ANTHROPIC_SERVERS`]),
//!    then lay the reviewed rows over them again.
//!
//! `--from <file>` reads a saved models.dev `api.json`; without it the
//! command fetches <https://models.dev/api.json> with `curl`. `--check`
//! writes nothing and fails when the checked-in file differs from what sync
//! would write. `catalog check` runs offline: it fails when the checked-in
//! file is not in the form sync writes, or a reviewed fact is missing from
//! it.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Map, Value};

pub(crate) const USAGE: &str = "\
  catalog sync [--from FILE] [--check]
                              regenerate crates/rig-core/src/catalog/models.json
                              from models.dev (fetched, or a saved api.json) and
                              xtask/src/catalog/review.json; with --check fail
                              when the checked-in file differs
  catalog check               fail when models.json is not in sync's form or
                              lacks a reviewed fact (offline)
";

/// The generated catalog, relative to the workspace root.
const OUTPUT: &str = "crates/rig-core/src/catalog/models.json";
/// The reviewed facts, relative to the workspace root.
const REVIEW: &str = "xtask/src/catalog/review.json";
/// Where models.dev publishes its catalog.
const MODELS_DEV: &str = "https://models.dev/api.json";

/// The models.dev provider keys rig serves, and the rig vendor each names.
const KEYS: [(&str, &str); 22] = [
    ("openai", "openai"),
    ("azure", "azure.openai"),
    ("anthropic", "anthropic"),
    ("google", "gcp.gemini"),
    ("google-vertex", "vertexai"),
    ("amazon-bedrock", "aws_bedrock"),
    ("openrouter", "openrouter"),
    ("deepseek", "deepseek"),
    ("groq", "groq"),
    ("mistral", "mistral"),
    ("xai", "xai"),
    ("venice", "venice"),
    ("zai", "zai"),
    ("minimax", "minimax"),
    ("cohere", "cohere"),
    ("perplexity", "perplexity"),
    ("huggingface", "huggingface"),
    ("togetherai", "together"),
    ("moonshotai", "moonshot"),
    ("xiaomi", "xiaomimimo"),
    ("ollama-cloud", "ollama"),
    ("github-copilot", "copilot"),
];

/// The keys of a model row the catalog reads, in the order they are written.
const ROW_KEYS: [&str; 12] = [
    "name",
    "canonical_model_id",
    "reasoning",
    "reasoning_options",
    "tool_call",
    "structured_output",
    "temperature",
    "modalities",
    "limit",
    "cost",
    "status",
    "rig",
];

/// The keys a reviewed row may carry for the review alone.
const REVIEW_KEYS: [&str; 3] = ["from", "source", "note"];

/// The facts a reviewed row may set under `rig`, as rig-core reads them.
const RIG_KEYS: [&str; 13] = [
    "format",
    "reasoning_default",
    "reasoning_control",
    "cache",
    "sampling",
    "reasoning_field",
    "adaptive_thinking",
    "thinking_off",
    "mid_conversation_system",
    "rejects_forced_tool_choice",
    "binds_context",
    "prompt_cache_options",
    "chat_tools_need_reasoning_off",
];

/// The vendors that serve Anthropic's models, and the keys of the Anthropic
/// row a row of theirs takes when its `canonical_model_id` names that row.
/// Bedrock and Vertex AI take Anthropic's own request fields, so they take
/// its reasoning options too; OpenRouter maps reasoning its own way and
/// keeps the options models.dev lists for it.
const ANTHROPIC_SERVERS: [(&str, &[&str]); 3] = [
    ("aws_bedrock", &["reasoning_options", "rig"]),
    ("openrouter", &["rig"]),
    ("vertexai", &["reasoning_options", "rig"]),
];

/// The catalog: vendor, then model id, then the row.
type Rows = BTreeMap<String, BTreeMap<String, Map<String, Value>>>;

/// Run `catalog <args>` from `root`.
pub(crate) fn run(root: &Path, args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    match args.next().as_deref() {
        Some("sync") => {
            let mut from = None;
            let mut check = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--check" => check = true,
                    "--from" => from = Some(args.next().ok_or("--from needs a file")?),
                    other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
                }
            }
            sync(root, from.as_deref(), check)
        }
        Some("check") => check(root),
        _ => Err(format!("usage:\n{USAGE}")),
    }
}

/// Regenerate the catalog, or with `check` compare it.
fn sync(root: &Path, from: Option<&str>, check: bool) -> Result<(), String> {
    let source = match from {
        Some(file) => std::fs::read_to_string(file).map_err(|error| format!("{file}: {error}"))?,
        None => crate::support::output(root, "curl", &["-sSfL", MODELS_DEV])?,
    };
    let models_dev: Value =
        serde_json::from_str(&source).map_err(|error| format!("models.dev: {error}"))?;
    let review = read_json(&root.join(REVIEW))?;
    let written = render(&generate(&models_dev, &review)?);
    let path = root.join(OUTPUT);
    if check {
        let current = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        return match current == written {
            true => Ok(()),
            false => Err(format!(
                "{OUTPUT} differs from what `cargo xtask catalog sync` writes; run it"
            )),
        };
    }
    std::fs::write(&path, written).map_err(|error| format!("{}: {error}", path.display()))
}

/// Check the checked-in catalog offline.
fn check(root: &Path) -> Result<(), String> {
    let path = root.join(OUTPUT);
    let current =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let catalog =
        read_rows(&serde_json::from_str(&current).map_err(|e| format!("{OUTPUT}: {e}"))?)?;
    if render(&catalog) != current {
        return Err(format!(
            "{OUTPUT} is not in the form `cargo xtask catalog sync` writes"
        ));
    }
    let review = read_json(&root.join(REVIEW))?;
    let missing = missing_facts(&catalog, &review)?;
    match missing.is_empty() {
        true => Ok(()),
        false => Err(format!(
            "{OUTPUT} lacks reviewed facts; run `cargo xtask catalog sync`:\n{}",
            missing.join("\n")
        )),
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// The catalog models.dev and the review give.
pub(crate) fn generate(models_dev: &Value, review: &Value) -> Result<Rows, String> {
    let mut rows = Rows::new();
    let providers = models_dev.as_object().ok_or("models.dev: not an object")?;
    for (key, vendor) in KEYS {
        let Some(models) = providers
            .get(key)
            .and_then(|provider| provider.get("models"))
            .and_then(Value::as_object)
        else {
            continue;
        };
        let section = rows.entry(vendor.to_owned()).or_default();
        for (id, row) in models {
            section.insert(id.clone(), trim(row));
        }
    }
    let review = read_rows(review)?;
    validate_review(&review)?;
    apply(&mut rows, &review)?;
    serve_anthropic(&mut rows);
    apply(&mut rows, &review)?;
    rows.retain(|_, models| !models.is_empty());
    Ok(rows)
}

/// The rows of a models.dev-shaped document: provider keys, each with a
/// `models` object.
fn read_rows(document: &Value) -> Result<Rows, String> {
    let mut rows = Rows::new();
    for (vendor, provider) in document.as_object().ok_or("not an object")? {
        let models = provider
            .get("models")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{vendor}: no `models` object"))?;
        let section = rows.entry(vendor.clone()).or_default();
        for (id, row) in models {
            let row = row
                .as_object()
                .ok_or_else(|| format!("{vendor}/{id}: not an object"))?;
            section.insert(id.clone(), row.clone());
        }
    }
    Ok(rows)
}

/// Fail on a reviewed row with a key the catalog does not read.
fn validate_review(review: &Rows) -> Result<(), String> {
    for (vendor, models) in review {
        for (id, row) in models {
            for key in row.keys() {
                if !ROW_KEYS.contains(&key.as_str()) && !REVIEW_KEYS.contains(&key.as_str()) {
                    return Err(format!("{REVIEW}: {vendor}/{id}: unknown key `{key}`"));
                }
            }
            for key in row
                .get("rig")
                .and_then(Value::as_object)
                .into_iter()
                .flatten()
            {
                if !RIG_KEYS.contains(&key.0.as_str()) {
                    return Err(format!(
                        "{REVIEW}: {vendor}/{id}: unknown rig fact `{}`",
                        key.0
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Lay every reviewed row over `rows`: first the rows that start from their
/// own models.dev row, then those that name another row in `from`, so a
/// `from` row starts from a reviewed one.
fn apply(rows: &mut Rows, review: &Rows) -> Result<(), String> {
    let reviewed_rows = |copies: bool| {
        review.iter().flat_map(move |(vendor, models)| {
            models
                .iter()
                .filter(move |(_, row)| row.contains_key("from") == copies)
                .map(move |(id, row)| (vendor, id, row))
        })
    };
    for (vendor, id, reviewed) in reviewed_rows(false).chain(reviewed_rows(true)) {
        {
            let base = match reviewed.get("from").and_then(Value::as_str) {
                Some(from) => {
                    let (from_vendor, from_id) = from
                        .split_once('/')
                        .ok_or_else(|| format!("{vendor}/{id}: `from` is not vendor/id"))?;
                    rows.get(from_vendor)
                        .and_then(|models| models.get(from_id))
                        .cloned()
                        .ok_or_else(|| format!("{vendor}/{id}: no row {from} to start from"))?
                }
                None => rows
                    .get(vendor)
                    .and_then(|models| models.get(id))
                    .cloned()
                    .unwrap_or_default(),
            };
            let mut row = Value::Object(base);
            let mut over = reviewed.clone();
            for key in REVIEW_KEYS {
                over.shift_remove(key);
            }
            merge(&mut row, Value::Object(over));
            if let Value::Object(row) = row {
                rows.entry(vendor.clone())
                    .or_default()
                    .insert(id.clone(), row);
            }
        }
    }
    Ok(())
}

/// Give each row of an [`ANTHROPIC_SERVERS`] vendor whose
/// `canonical_model_id` is `anthropic/<model>` the keys its vendor takes from
/// that Anthropic row. `<model>` is found as rig-core finds a model: the id
/// as listed, or else the longest listed id it extends with `-20` and a
/// year.
fn serve_anthropic(rows: &mut Rows) {
    let Some(anthropic) = rows.get("anthropic").cloned() else {
        return;
    };
    for (vendor, keys) in ANTHROPIC_SERVERS {
        let Some(models) = rows.get_mut(vendor) else {
            continue;
        };
        for row in models.values_mut() {
            let Some(source) = row
                .get("canonical_model_id")
                .and_then(Value::as_str)
                .and_then(|canonical| canonical.strip_prefix("anthropic/"))
                .and_then(|model| anthropic_row(&anthropic, model))
            else {
                continue;
            };
            for key in keys {
                if let Some(value) = source.get(*key) {
                    row.insert((*key).to_owned(), value.clone());
                }
            }
        }
    }
}

/// The Anthropic row of `model`, or of the model it is a dated snapshot of.
fn anthropic_row<'a>(
    anthropic: &'a BTreeMap<String, Map<String, Value>>,
    model: &str,
) -> Option<&'a Map<String, Value>> {
    anthropic.get(model).or_else(|| {
        model
            .match_indices("-20")
            .filter(|(at, _)| {
                model
                    .get(at + 3..at + 5)
                    .is_some_and(|year| year.bytes().all(|byte| byte.is_ascii_digit()))
            })
            .filter_map(|(at, _)| anthropic.get(model.get(..at)?))
            .last()
    })
}

/// `models.dev` row reduced to the keys the catalog reads, each trimmed to
/// the parts it reads.
fn trim(row: &Value) -> Map<String, Value> {
    let mut trimmed = Map::new();
    for key in ROW_KEYS {
        let Some(value) = row.get(key) else {
            continue;
        };
        let value = match key {
            "modalities" => keep(value, &["input"]),
            "limit" => keep(value, &["context", "output"]),
            "cost" => keep(value, &["input", "output", "cache_read", "cache_write"]),
            "reasoning_options" => Value::Array(
                value
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|option| keep(option, &["type", "values", "min", "max"]))
                    .collect(),
            ),
            _ => value.clone(),
        };
        trimmed.insert(key.to_owned(), value);
    }
    trimmed
}

/// `value`'s object keys in `keys`, in that order.
fn keep(value: &Value, keys: &[&str]) -> Value {
    let mut kept = Map::new();
    for key in keys {
        if let Some(field) = value.get(*key) {
            kept.insert((*key).to_owned(), field.clone());
        }
    }
    Value::Object(kept)
}

/// Deep-merge `over` into `base`: objects key by key, anything else
/// replaces.
fn merge(base: &mut Value, over: Value) {
    match (base, over) {
        (Value::Object(base), Value::Object(over)) => {
            for (key, value) in over {
                match base.get_mut(&key) {
                    Some(slot) => merge(slot, value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, over) => *base = over,
    }
}

/// The reviewed facts `catalog` lacks, as `vendor/id: key`.
fn missing_facts(catalog: &Rows, review: &Value) -> Result<Vec<String>, String> {
    let review = read_rows(review)?;
    let mut missing = Vec::new();
    for (vendor, models) in &review {
        for (id, reviewed) in models {
            let row = catalog.get(vendor).and_then(|models| models.get(id));
            for (key, value) in reviewed {
                if REVIEW_KEYS.contains(&key.as_str()) {
                    continue;
                }
                let held = row.and_then(|row| row.get(key));
                if !held.is_some_and(|held| contains(held, value)) {
                    missing.push(format!("{vendor}/{id}: {key}"));
                }
            }
        }
    }
    Ok(missing)
}

/// Whether `held` carries everything `wanted` sets: objects key by key,
/// anything else equal.
fn contains(held: &Value, wanted: &Value) -> bool {
    match (held, wanted) {
        (Value::Object(held), Value::Object(wanted)) => wanted.iter().all(|(key, value)| {
            held.get(key)
                .is_some_and(|held_value| contains(held_value, value))
        }),
        _ => held == wanted,
    }
}

/// The catalog as sync writes it: one provider per section, one model per
/// line, every key in a fixed order.
pub(crate) fn render(rows: &Rows) -> String {
    let mut out = String::from("{\n");
    let mut vendors = rows.iter().peekable();
    while let Some((vendor, models)) = vendors.next() {
        out.push_str(&format!("{}: {{\"models\": {{\n", quote(vendor)));
        let mut entries = models.iter().peekable();
        while let Some((id, row)) = entries.next() {
            out.push_str(&format!("  {}: {}", quote(id), canonical(row)));
            out.push_str(if entries.peek().is_some() {
                ",\n"
            } else {
                "\n"
            });
        }
        out.push_str("}}");
        out.push_str(if vendors.peek().is_some() {
            ",\n"
        } else {
            "\n"
        });
    }
    out.push_str("}\n");
    out
}

fn quote(text: &str) -> String {
    Value::String(text.to_owned()).to_string()
}

/// `row` as compact JSON, trimmed and ordered as [`trim`] orders it, with
/// its `rig` facts sorted by name.
fn canonical(row: &Map<String, Value>) -> String {
    let mut ordered = trim(&Value::Object(row.clone()));
    if let Some(Value::Object(facts)) = ordered.get_mut("rig") {
        facts.sort_keys();
    }
    Value::Object(ordered).to_string()
}

#[cfg(test)]
mod tests;
