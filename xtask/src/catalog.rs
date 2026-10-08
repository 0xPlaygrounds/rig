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
//! 3. Give each row of a vendor that serves another's models, whose
//!    models.dev `canonical_model_id` names that vendor's model, that
//!    model's facts it takes ([`SERVERS`]),
//!    give each reasoning row of a gateway that translates reasoning
//!    controls the controls it translates ([`REASONING_GATEWAYS`]), then lay
//!    the reviewed rows over them again.
//! 4. Pin under `rig.pinned` every key steps 2 and 3 set ([`pin`]), so that
//!    rig-core's `Catalog::with_models_dev` keeps them when an application
//!    lays a newer copy of models.dev over the built-in catalog.
//!
//! `--from <file>` reads a saved models.dev `api.json`; without it the
//! command fetches <https://models.dev/api.json> with `curl`. Sync also
//! writes when that data was read (the file's modification time, or now)
//! to `crates/rig-core/src/catalog/generated_at`. `--check`
//! writes nothing and fails when the checked-in file differs from what sync
//! would write. `catalog check` runs offline: it fails when the checked-in
//! file is not in the form sync writes, or a reviewed fact is missing from
//! it.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

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
/// When the models.dev data behind [`OUTPUT`] was read, relative to the
/// workspace root: one RFC 3339 UTC time, which rig-core reads at compile
/// time as `Catalog::generated_at`.
const GENERATED_AT: &str = "crates/rig-core/src/catalog/generated_at";
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

/// The row keys a row may pin, in the order `rig.pinned` lists them: the
/// keys rig-core reads, with `limit` and `cost` by part, as a refresh
/// replaces those part by part.
const PINNABLE: [&str; 14] = [
    "name",
    "reasoning",
    "reasoning_options",
    "tool_call",
    "structured_output",
    "temperature",
    "modalities",
    "limit.context",
    "limit.output",
    "cost.input",
    "cost.output",
    "cost.cache_read",
    "cost.cache_write",
    "status",
];

/// The vendors that serve another vendor's models, and the facts a row of
/// theirs takes from the origin's row when its `canonical_model_id` names
/// that row: `(origin, server, keys)`, where a key is a row key or
/// `rig.<fact>` for one fact under `rig`. Bedrock and Vertex AI take
/// Anthropic's own request fields, so they take its reasoning options too;
/// OpenRouter maps reasoning its own way and keeps the options models.dev
/// lists for it. Azure OpenAI and Copilot reach OpenAI's models through
/// OpenAI's wire, which applies OpenAI's sampling rule, so they take it and
/// the default effort it depends on.
const SERVERS: [(&str, &str, &[&str]); 5] = [
    ("anthropic", "aws_bedrock", &["reasoning_options", "rig"]),
    ("anthropic", "openrouter", &["rig"]),
    ("anthropic", "vertexai", &["reasoning_options", "rig"]),
    (
        "openai",
        "azure.openai",
        &["rig.sampling", "rig.reasoning_default"],
    ),
    (
        "openai",
        "copilot",
        &["rig.sampling", "rig.reasoning_default"],
    ),
];

/// The gateways that translate reasoning controls for every model they
/// route: each takes every effort level in [`GATEWAY_EFFORTS`] and a token
/// budget for any reasoning model, whatever the upstream lists, except a
/// budget for the upstreams by these id prefixes, which take only an
/// effort.
const REASONING_GATEWAYS: [(&str, &[&str]); 1] = [("openrouter", &["openai/", "x-ai/"])];

/// The effort levels a [`REASONING_GATEWAYS`] gateway takes for every
/// reasoning model.
const GATEWAY_EFFORTS: [&str; 5] = ["minimal", "low", "medium", "high", "xhigh"];

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
    // The data is as new as the file it was saved to, or as now when fetched.
    let (source, read_at) = match from {
        Some(file) => (
            std::fs::read_to_string(file).map_err(|error| format!("{file}: {error}"))?,
            std::fs::metadata(file)
                .and_then(|metadata| metadata.modified())
                .map_err(|error| format!("{file}: {error}"))?,
        ),
        None => (
            crate::support::output(root, "curl", &["-sSfL", MODELS_DEV])?,
            SystemTime::now(),
        ),
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
    std::fs::write(&path, written).map_err(|error| format!("{}: {error}", path.display()))?;
    let stamp = root.join(GENERATED_AT);
    std::fs::write(&stamp, format!("{}\n", rfc3339(read_at)?))
        .map_err(|error| format!("{}: {error}", stamp.display()))
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
    let stamp = root.join(GENERATED_AT);
    let stamp =
        std::fs::read_to_string(&stamp).map_err(|error| format!("{}: {error}", stamp.display()))?;
    if !is_rfc3339_utc(stamp.trim_end()) {
        return Err(format!(
            "{GENERATED_AT} is not a `YYYY-MM-DDTHH:MM:SSZ` time; run `cargo xtask catalog sync`"
        ));
    }
    let review = read_json(&root.join(REVIEW))?;
    let mut pinned = catalog.clone();
    pin(&mut pinned, &read_rows(&review)?);
    if pinned != catalog {
        return Err(format!(
            "{OUTPUT} pins other keys than the review and joins set; run `cargo xtask catalog sync`"
        ));
    }
    let missing = missing_facts(&catalog, &review)?;
    match missing.is_empty() {
        true => Ok(()),
        false => Err(format!(
            "{OUTPUT} lacks reviewed facts; run `cargo xtask catalog sync`:\n{}",
            missing.join("\n")
        )),
    }
}

/// `time` as `YYYY-MM-DDTHH:MM:SSZ`.
pub(crate) fn rfc3339(time: SystemTime) -> Result<String, String> {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "the models.dev data predates 1970".to_owned())?
        .as_secs();
    let (days, of_day) = (seconds / 86_400, seconds % 86_400);
    // The civil date of a day count, by Howard Hinnant's algorithm, with
    // the year starting in March so the leap day falls last.
    let days = days + 719_468;
    let era = days / 146_097;
    let of_era = days - era * 146_097;
    let year_of_era = (of_era - of_era / 1460 + of_era / 36_524 - of_era / 146_096) / 365;
    let of_year = of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * of_year + 2) / 153;
    let day = of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        of_day / 3600,
        of_day % 3600 / 60,
        of_day % 60
    ))
}

/// Whether `text` is written `YYYY-MM-DDTHH:MM:SSZ`.
pub(crate) fn is_rfc3339_utc(text: &str) -> bool {
    text.len() == 20
        && text.char_indices().all(|(at, char)| match at {
            4 | 7 => char == '-',
            10 => char == 'T',
            13 | 16 => char == ':',
            19 => char == 'Z',
            _ => char.is_ascii_digit(),
        })
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
    serve(&mut rows);
    translate_reasoning(&mut rows);
    apply(&mut rows, &review)?;
    rows.retain(|_, models| !models.is_empty());
    pin(&mut rows, &review);
    Ok(rows)
}

/// Write under each row's `rig.pinned` the [`PINNABLE`] keys the review or
/// a join set: every key a reviewed row names, every key of a row copied
/// with `from`, the non-`rig` keys a [`SERVERS`] row takes from its origin,
/// and the reasoning options of a [`REASONING_GATEWAYS`] row that lists
/// them. It reads only the finished rows and the review, so `catalog check`
/// can tell offline whether the pins are current.
fn pin(rows: &mut Rows, review: &Rows) {
    let mut pins: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    let mut add = |vendor: &str, id: &str, key: String| {
        pins.entry((vendor.to_owned(), id.to_owned()))
            .or_default()
            .push(key);
    };
    let paths = |row: &Map<String, Value>| -> Vec<String> {
        row.iter()
            .flat_map(|(key, value)| match (key.as_str(), value) {
                ("limit" | "cost", Value::Object(parts)) => {
                    parts.keys().map(|part| format!("{key}.{part}")).collect()
                }
                _ => vec![key.clone()],
            })
            .collect()
    };
    for (vendor, models) in review {
        for (id, reviewed) in models {
            let named = match reviewed.contains_key("from") {
                true => rows.get(vendor).and_then(|models| models.get(id)),
                false => Some(reviewed),
            };
            for key in named.map(paths).into_iter().flatten() {
                add(vendor, id, key);
            }
        }
    }
    for (origin, vendor, keys) in SERVERS {
        let (Some(origin_rows), Some(models)) = (rows.get(origin), rows.get(vendor)) else {
            continue;
        };
        let prefix = format!("{origin}/");
        for (id, row) in models {
            let Some(source) = row
                .get("canonical_model_id")
                .and_then(Value::as_str)
                .and_then(|canonical| canonical.strip_prefix(prefix.as_str()))
                .and_then(|model| origin_row(origin_rows, model))
            else {
                continue;
            };
            for key in keys.iter().filter(|key| source.contains_key(**key)) {
                add(vendor, id, (*key).to_owned());
            }
        }
    }
    for (vendor, _) in REASONING_GATEWAYS {
        for (id, row) in rows.get(vendor).into_iter().flatten() {
            let lists_options = row
                .get("reasoning_options")
                .and_then(Value::as_array)
                .is_some_and(|options| !options.is_empty());
            if row.get("reasoning") == Some(&Value::Bool(true)) && lists_options {
                add(vendor, id, "reasoning_options".to_owned());
            }
        }
    }
    for (vendor, models) in rows.iter_mut() {
        for (id, row) in models.iter_mut() {
            let keys = pins
                .remove(&(vendor.clone(), id.clone()))
                .unwrap_or_default();
            let pinned: Vec<Value> = PINNABLE
                .iter()
                .filter(|key| keys.iter().any(|pinned| pinned == *key))
                .map(|&key| Value::from(key))
                .collect();
            let facts = row.get_mut("rig").and_then(Value::as_object_mut);
            match (facts, pinned.is_empty()) {
                (Some(facts), true) => {
                    facts.shift_remove("pinned");
                    if facts.is_empty() {
                        row.shift_remove("rig");
                    }
                }
                (None, true) => {}
                (Some(facts), false) => {
                    facts.insert("pinned".to_owned(), Value::Array(pinned));
                }
                (None, false) => {
                    row.insert("rig".to_owned(), serde_json::json!({ "pinned": pinned }));
                }
            }
        }
    }
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

/// Give each row of a [`SERVERS`] vendor whose `canonical_model_id` is
/// `<origin>/<model>` the keys its vendor takes from that origin row.
/// `<model>` is found as rig-core finds a model: the id as listed, or else
/// the longest listed id it extends with `-20` and a year.
fn serve(rows: &mut Rows) {
    for (origin, vendor, keys) in SERVERS {
        let Some(origin_rows) = rows.get(origin).cloned() else {
            continue;
        };
        let Some(models) = rows.get_mut(vendor) else {
            continue;
        };
        let prefix = format!("{origin}/");
        for row in models.values_mut() {
            let Some(source) = row
                .get("canonical_model_id")
                .and_then(Value::as_str)
                .and_then(|canonical| canonical.strip_prefix(prefix.as_str()))
                .and_then(|model| origin_row(&origin_rows, model))
            else {
                continue;
            };
            for key in keys {
                match key.split_once('.') {
                    Some((outer, fact)) => {
                        let Some(value) = source.get(outer).and_then(|facts| facts.get(fact))
                        else {
                            continue;
                        };
                        if let Value::Object(facts) = row
                            .entry(outer.to_owned())
                            .or_insert_with(|| Value::Object(Map::new()))
                        {
                            facts.insert(fact.to_owned(), value.clone());
                        }
                    }
                    None => {
                        if let Some(value) = source.get(*key) {
                            row.insert((*key).to_owned(), value.clone());
                        }
                    }
                }
            }
        }
    }
}

/// Give each reasoning row of a [`REASONING_GATEWAYS`] gateway that lists
/// reasoning options the controls the gateway translates: the
/// [`GATEWAY_EFFORTS`] beside the efforts it lists, and a token budget
/// unless it lists one or its upstream takes none. Whether reasoning turns
/// off stays as listed. A row that lists no options stays unknown.
fn translate_reasoning(rows: &mut Rows) {
    for (vendor, effort_only) in REASONING_GATEWAYS {
        let Some(models) = rows.get_mut(vendor) else {
            continue;
        };
        for (id, row) in models.iter_mut() {
            if row.get("reasoning") != Some(&Value::Bool(true)) {
                continue;
            }
            let Some(Value::Array(options)) = row.get_mut("reasoning_options") else {
                continue;
            };
            if options.is_empty() {
                continue;
            }
            let kind = |option: &Value| {
                option
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            };
            let mut values: Vec<Value> = GATEWAY_EFFORTS
                .iter()
                .map(|&level| Value::from(level))
                .collect();
            for option in options
                .iter()
                .filter(|option| kind(option).as_deref() == Some("effort"))
            {
                for value in option
                    .get("values")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if !values.contains(value) {
                        values.push(value.clone());
                    }
                }
            }
            let budget = options
                .iter()
                .find(|option| kind(option).as_deref() == Some("budget_tokens"))
                .cloned()
                .or_else(|| {
                    (!effort_only.iter().any(|prefix| id.starts_with(prefix)))
                        .then(|| serde_json::json!({"type": "budget_tokens"}))
                });
            let toggle = options
                .iter()
                .find(|option| kind(option).as_deref() == Some("toggle"))
                .cloned();
            let mut translated = vec![serde_json::json!({"type": "effort", "values": values})];
            translated.extend(budget);
            translated.extend(toggle);
            *options = translated;
        }
    }
}

/// The row of `model` among `origin`'s rows, or of the model it is a dated
/// snapshot of.
fn origin_row<'a>(
    origin: &'a BTreeMap<String, Map<String, Value>>,
    model: &str,
) -> Option<&'a Map<String, Value>> {
    origin.get(model).or_else(|| {
        model
            .match_indices("-20")
            .filter(|(at, _)| {
                model
                    .get(at + 3..at + 5)
                    .is_some_and(|year| year.bytes().all(|byte| byte.is_ascii_digit()))
            })
            .filter_map(|(at, _)| origin.get(model.get(..at)?))
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
