//! The guard `catalog sync` runs before it writes. It reads each row of the
//! checked-in catalog and of the new one the way rig-core reads it
//! ([`reading`]), refuses a capability the new row loses unless the review
//! answers it ([`Comparison::refusal`]), and accounts for every other change
//! in a markdown report ([`Comparison::markdown`]).
//!
//! It also holds the review's provenance rule ([`provenance`]): a review row
//! a change adds or edits says where its facts came from.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use super::{REVIEW, REVIEW_KEYS, Rows, UNSOURCED};

/// The effort words rig-core reads as an `Effort`, lowest first.
const EFFORTS: [&str; 6] = ["minimal", "low", "medium", "high", "xhigh", "max"];

/// The input modalities rig-core reads, in its order.
const INPUTS: [&str; 5] = ["text", "image", "audio", "video", "pdf"];

/// The review keys that set a fact a `source` must back: the capabilities
/// rig reads, the limits, and every `rig` fact.
const SOURCED: [&str; 8] = [
    "reasoning",
    "reasoning_options",
    "tool_call",
    "temperature",
    "structured_output",
    "modalities",
    "limit",
    "rig",
];

/// What rig-core reads from a finished row, by the rules of `Row::spec` in
/// `crates/rig-core/src/catalog/row.rs`. Keep the two in step;
/// `the_reading_is_rig_cores` checks them over the checked-in catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Reading {
    pub(super) reasoning: Reasoning,
    /// The input modalities, in [`INPUTS`] order.
    pub(super) input: Vec<&'static str>,
    pub(super) tools: bool,
    pub(super) structured_output: bool,
    /// When the model takes sampling parameters: `any`, `reasoning_off` or
    /// `never`, or `None` when unknown.
    pub(super) sampling: Option<String>,
}

/// rig-core's `ReasoningSupport`, without the default effort, which only
/// the review sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Reasoning {
    None,
    Unknown,
    Listed {
        /// The effort levels, in [`EFFORTS`] order.
        levels: Vec<&'static str>,
        off: bool,
        budget: Option<(u32, u32)>,
    },
}

/// What rig-core reads from `row`.
pub(super) fn reading(row: &Map<String, Value>) -> Reading {
    let flag = |key: &str| row.get(key).and_then(Value::as_bool).unwrap_or(false);
    let listed = |value: Option<&Value>| -> Vec<String> {
        value
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()
    };
    let input = listed(row.get("modalities").and_then(|m| m.get("input")));
    let facts = row.get("rig");
    let fact = |name: &str| {
        facts
            .and_then(|facts| facts.get(name))
            .and_then(Value::as_str)
    };
    Reading {
        reasoning: reasoning(row, fact("reasoning_control") == Some("none")),
        input: INPUTS
            .into_iter()
            .filter(|word| input.iter().any(|listed| listed == word))
            .collect(),
        tools: flag("tool_call"),
        structured_output: flag("structured_output"),
        sampling: fact("sampling")
            .map(str::to_owned)
            .or_else(|| flag("temperature").then(|| "any".to_owned())),
    }
}

/// `Row::reasoning`: effort words but `none`, which turns reasoning off, as
/// a `toggle` does; a `budget_tokens` range whose absent `max` is the output
/// limit; unknown when a reasoning row lists none of these and its facts do
/// not say it takes no control.
fn reasoning(row: &Map<String, Value>, no_control: bool) -> Reasoning {
    if row.get("reasoning").and_then(Value::as_bool) != Some(true) {
        return Reasoning::None;
    }
    let number = |value: Option<&Value>| {
        value
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
    };
    let output = number(row.get("limit").and_then(|limit| limit.get("output"))).filter(|n| *n > 0);
    let mut words = Vec::new();
    let mut off = false;
    let mut budget = None;
    let options = row.get("reasoning_options").and_then(Value::as_array);
    for option in options.into_iter().flatten() {
        match option.get("type").and_then(Value::as_str) {
            Some("effort") => words.extend(
                option
                    .get("values")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str),
            ),
            Some("toggle") => off = true,
            Some("budget_tokens") => {
                let min = number(option.get("min")).unwrap_or(0);
                let max = number(option.get("max")).or(output).unwrap_or(u32::MAX);
                budget = Some((min, max));
            }
            _ => {}
        }
    }
    off |= words.contains(&"none");
    let levels: Vec<&'static str> = EFFORTS
        .into_iter()
        .filter(|level| words.contains(level))
        .collect();
    if levels.is_empty() && budget.is_none() && !off && !no_control {
        return Reasoning::Unknown;
    }
    Reasoning::Listed {
        levels,
        off,
        budget,
    }
}

/// A capability a row loses, and the review keys that answer it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Loss {
    pub(super) what: String,
    pub(super) answers: &'static [&'static str],
}

const REASONING_OPTIONS: &[&str] = &["reasoning_options"];

/// What `new` loses and gains against `old`. `sampling_fact` says whether
/// the new row's `rig` facts set its sampling rule, which then decides it
/// over `temperature`.
fn changes(old: &Reading, new: &Reading, sampling_fact: bool) -> (Vec<Loss>, Vec<String>) {
    let mut losses = Vec::new();
    let mut gains = Vec::new();
    let mut lose =
        |what: String, answers: &'static [&'static str]| losses.push(Loss { what, answers });
    match (&old.reasoning, &new.reasoning) {
        (Reasoning::None, Reasoning::None) | (Reasoning::Unknown, Reasoning::Unknown) => {}
        (_, Reasoning::None) => lose("stops reasoning".to_owned(), &["reasoning"]),
        (Reasoning::None, now) => gains.push(format!("reasons ({})", controls(now))),
        (was @ Reasoning::Listed { .. }, Reasoning::Unknown) => lose(
            format!("reasoning controls become unknown (were {})", controls(was)),
            REASONING_OPTIONS,
        ),
        (Reasoning::Unknown, now) => {
            gains.push(format!("reasoning controls listed ({})", controls(now)))
        }
        (
            Reasoning::Listed {
                levels: was,
                off: was_off,
                budget: was_budget,
            },
            Reasoning::Listed {
                levels: now,
                off: now_off,
                budget: now_budget,
            },
        ) => {
            let lost = missing(was, now);
            if !lost.is_empty() {
                lose(
                    format!("effort lost: {}", lost.join(", ")),
                    REASONING_OPTIONS,
                );
            }
            let gained = missing(now, was);
            if !gained.is_empty() {
                gains.push(format!("effort: {}", gained.join(", ")));
            }
            match (was_off, now_off) {
                (true, false) => lose(
                    "can no longer turn reasoning off".to_owned(),
                    REASONING_OPTIONS,
                ),
                (false, true) => gains.push("can turn reasoning off".to_owned()),
                _ => {}
            }
            match (was_budget, now_budget) {
                (Some(_), None) => lose("reasoning budget lost".to_owned(), REASONING_OPTIONS),
                (None, Some(_)) => gains.push("reasoning budget".to_owned()),
                _ => {}
            }
        }
    }
    let lost = missing(&old.input, &new.input);
    if !lost.is_empty() {
        lose(
            format!("input lost: {}", lost.join(", ")),
            &["modalities.input"],
        );
    }
    let gained = missing(&new.input, &old.input);
    if !gained.is_empty() {
        gains.push(format!("input: {}", gained.join(", ")));
    }
    let flags: [(&str, bool, bool, &'static [&'static str]); 2] = [
        ("tool calls", old.tools, new.tools, &["tool_call"]),
        (
            "structured output",
            old.structured_output,
            new.structured_output,
            &["structured_output"],
        ),
    ];
    for (name, was, now, answers) in flags {
        match (was, now) {
            (true, false) => lose(format!("{name} lost"), answers),
            (false, true) => gains.push(name.to_owned()),
            _ => {}
        }
    }
    let (was, now) = (old.sampling.as_deref(), new.sampling.as_deref());
    let sampling = format!(
        "sampling {} -> {}",
        was.unwrap_or("unknown"),
        now.unwrap_or("unknown")
    );
    if samples_less(was, now) {
        let answers: &'static [&'static str] = match sampling_fact {
            true => &["rig.sampling"],
            false => &["temperature", "rig.sampling"],
        };
        lose(sampling, answers);
    } else if samples_less(now, was) {
        gains.push(sampling);
    }
    (losses, gains)
}

/// Whether a model whose sampling rule was `was` takes sampling parameters
/// in fewer cases under `now`: a documented `any` that becomes anything
/// else, no rule (which refuses nothing) that becomes a restriction, or
/// `reasoning_off` that becomes `never`.
fn samples_less(was: Option<&str>, now: Option<&str>) -> bool {
    match (was, now) {
        (Some("any"), now) => now != Some("any"),
        (None, Some("reasoning_off" | "never")) | (Some("reasoning_off"), Some("never")) => true,
        _ => false,
    }
}

/// The items of `from` that `of` lacks.
fn missing<'a>(from: &[&'a str], of: &[&str]) -> Vec<&'a str> {
    from.iter()
        .filter(|item| !of.contains(item))
        .copied()
        .collect()
}

/// A listed reading's controls, spelled for the report.
fn controls(reasoning: &Reasoning) -> String {
    let Reasoning::Listed {
        levels,
        off,
        budget,
    } = reasoning
    else {
        return "unknown".to_owned();
    };
    let mut parts = Vec::new();
    if !levels.is_empty() {
        parts.push(format!("effort {}", levels.join(" ")));
    }
    if *off {
        parts.push("off".to_owned());
    }
    if budget.is_some() {
        parts.push("budget".to_owned());
    }
    match parts.is_empty() {
        true => "no control".to_owned(),
        false => parts.join(", "),
    }
}

/// The `source` that answers a loss of `vendor/id` with one of `paths`:
/// the row's own review when it sets one, else the review of the row it
/// copies with `from` when that one does. A key a join sets is no answer.
fn answer(review: &Rows, vendor: &str, id: &str, paths: &[&str]) -> Option<String> {
    let row = review.get(vendor)?.get(id)?;
    let origin = origin(review, row);
    let answering = match paths.iter().any(|path| sets(row, path)) {
        true => source(row).or_else(|| origin.and_then(source)),
        false => origin
            .filter(|origin| paths.iter().any(|path| sets(origin, path)))
            .and_then(source),
    };
    answering.map(str::to_owned)
}

/// The review row a review row copies with `from`, when it is reviewed.
fn origin<'a>(review: &'a Rows, row: &Map<String, Value>) -> Option<&'a Map<String, Value>> {
    let (vendor, id) = row.get("from")?.as_str()?.split_once('/')?;
    review.get(vendor)?.get(id)
}

/// Whether a review row sets the value at the dotted `path`: the value
/// itself, or a parent it replaces with something that is not an object,
/// as the review's merge does.
fn sets(row: &Map<String, Value>, path: &str) -> bool {
    let mut fields = row;
    for part in path.split('.') {
        match fields.get(part) {
            None => return false,
            Some(Value::Object(inner)) => fields = inner,
            Some(_) => return true,
        }
    }
    true
}

/// The value at a dotted `path` of `row`.
fn at_path<'a>(row: &'a Map<String, Value>, path: &str) -> Option<&'a Value> {
    let mut parts = path.split('.');
    let first = row.get(parts.next()?)?;
    parts.try_fold(first, |value, part| value.get(part))
}

/// A value as the report writes it: a string bare, anything else as JSON,
/// and an absent one as `none`.
fn spell(value: Option<&Value>) -> String {
    match value {
        None => "none".to_owned(),
        Some(Value::String(text)) => text.clone(),
        Some(value) => value.to_string(),
    }
}

/// A review row's `source`, when it is a non-empty string.
fn source(row: &Map<String, Value>) -> Option<&str> {
    row.get("source")
        .and_then(Value::as_str)
        .filter(|source| !source.trim().is_empty())
}

/// The changes between two catalogs, sorted by row.
#[derive(Debug, Default)]
pub(super) struct Comparison {
    /// Losses no review answers.
    pub(super) refused: Vec<(String, Vec<Loss>)>,
    /// Losses a review answers, each with its source.
    pub(super) answered: Vec<(String, Loss, String)>,
    pub(super) removed: Vec<String>,
    /// Each new row, and whether the review names it.
    pub(super) added: Vec<(String, bool)>,
    pub(super) gains: Vec<(String, Vec<String>)>,
    pub(super) limits: Vec<(String, Vec<String>)>,
    /// Rows whose `rig` facts changed, and how.
    pub(super) facts: Vec<(String, Vec<String>)>,
    pub(super) agrees: Vec<String>,
    pub(super) disagrees: Vec<String>,
    /// How many rows changed each counted kind of field.
    pub(super) counted: BTreeMap<String, usize>,
}

/// Compare the finished rows `old` and `new`; `review` answers losses, and
/// `models_dev` (its rows before the review) is what the reviewed facts are
/// checked against.
pub(super) fn compare(old: &Rows, new: &Rows, review: &Rows, models_dev: &Rows) -> Comparison {
    let mut comparison = Comparison::default();
    let ids = |rows: &Rows| -> BTreeSet<(String, String)> {
        rows.iter()
            .flat_map(|(vendor, models)| models.keys().map(|id| (vendor.clone(), id.clone())))
            .collect()
    };
    let (before, after) = (ids(old), ids(new));
    for (vendor, id) in before.union(&after) {
        let key = format!("{vendor}/{id}");
        let row = |rows: &'_ Rows| rows.get(vendor).and_then(|models| models.get(id)).cloned();
        let (was, now) = match (row(old), row(new)) {
            (Some(was), Some(now)) => (was, now),
            (Some(_), None) => {
                comparison.removed.push(key);
                continue;
            }
            (None, Some(_)) => {
                let reviewed = review
                    .get(vendor)
                    .is_some_and(|models| models.contains_key(id));
                comparison.added.push((key, reviewed));
                continue;
            }
            (None, None) => continue,
        };
        let (old_reading, new_reading) = (reading(&was), reading(&now));
        let sampling_fact = now
            .get("rig")
            .and_then(|facts| facts.get("sampling"))
            .and_then(Value::as_str)
            .is_some();
        let (losses, gains) = changes(&old_reading, &new_reading, sampling_fact);
        let mut refused = Vec::new();
        for loss in losses {
            match answer(review, vendor, id, loss.answers) {
                Some(source) => comparison.answered.push((key.clone(), loss, source)),
                None => refused.push(loss),
            }
        }
        if !refused.is_empty() {
            comparison.refused.push((key.clone(), refused));
        }
        if !gains.is_empty() {
            comparison.gains.push((key.clone(), gains));
        }
        let limits = limit_changes(&was, &now, &old_reading, &new_reading);
        if !limits.is_empty() {
            comparison.limits.push((key.clone(), limits));
        }
        let facts = fact_changes(&was, &now);
        if !facts.is_empty() {
            comparison.facts.push((key.clone(), facts));
        }
        count(
            &mut comparison.counted,
            &was,
            &now,
            old_reading == new_reading,
        );
    }
    reviewed_facts(&mut comparison, review, models_dev);
    comparison
}

/// The limit and status changes of a row, and a change to its reasoning
/// budget range, whose ceiling follows `limit.output` unless the row sets
/// its own. A narrower range is called out.
fn limit_changes(
    was: &Map<String, Value>,
    now: &Map<String, Value>,
    old: &Reading,
    new: &Reading,
) -> Vec<String> {
    let mut changes = Vec::new();
    for part in ["context", "input", "output"] {
        let path = format!("limit.{part}");
        let (before, after) = (at_path(was, &path), at_path(now, &path));
        if before != after {
            changes.push(format!("{part} {} -> {}", spell(before), spell(after)));
        }
    }
    if let (
        Reasoning::Listed {
            budget: Some((was_min, was_max)),
            ..
        },
        Reasoning::Listed {
            budget: Some((min, max)),
            ..
        },
    ) = (&old.reasoning, &new.reasoning)
        && (was_min, was_max) != (min, max)
    {
        let change = format!("reasoning budget {was_min}..{was_max} -> {min}..{max}");
        changes.push(match min > was_min || max < was_max {
            true => format!("**{change}, narrower**"),
            false => change,
        });
    }
    if was.get("status") != now.get("status") {
        changes.push(format!(
            "status {} -> {}",
            spell(was.get("status")),
            spell(now.get("status"))
        ));
    }
    changes
}

/// The `rig` facts of a row that changed, but its pins, as
/// `name old -> new`.
fn fact_changes(was: &Map<String, Value>, now: &Map<String, Value>) -> Vec<String> {
    let facts = |row: &Map<String, Value>| -> Map<String, Value> {
        let mut facts = row
            .get("rig")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        facts.retain(|name, _| name != "pinned");
        facts
    };
    let (before, after) = (facts(was), facts(now));
    let names: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    names
        .into_iter()
        .filter(|name| before.get(*name) != after.get(*name))
        .map(|name| {
            format!(
                "{name} {} -> {}",
                spell(before.get(name)),
                spell(after.get(name))
            )
        })
        .collect()
}

/// The row keys whose changes the report counts rather than lists, by the
/// name it counts them under.
const COUNTED: [(&str, &str); 6] = [
    ("cost", "prices"),
    ("description", "descriptions"),
    ("release_date", "dates"),
    ("last_updated", "dates"),
    ("knowledge", "dates"),
    ("name", "names"),
];

/// The row keys [`reading`] reads, and the part of `modalities` it does.
const READ: [&str; 6] = [
    "reasoning",
    "reasoning_options",
    "tool_call",
    "structured_output",
    "temperature",
    "modalities.input",
];

/// Count the fields of a row that changed and that no list above names:
/// the [`COUNTED`] kinds, a field rig reads whose change leaves its reading
/// as it was (`same reading`), and any other field by its name.
fn count(
    counted: &mut BTreeMap<String, usize>,
    was: &Map<String, Value>,
    now: &Map<String, Value>,
    same_reading: bool,
) {
    let mut kinds = BTreeSet::new();
    let keys: BTreeSet<&String> = was.keys().chain(now.keys()).collect();
    for key in keys {
        let paths: Vec<String> = match key.as_str() {
            "modalities" => vec!["modalities.input".into(), "modalities.output".into()],
            "limit" | "status" | "rig" => continue,
            _ => vec![key.clone()],
        };
        for path in paths {
            if at_path(was, &path) == at_path(now, &path) {
                continue;
            }
            let kind = match COUNTED.iter().find(|(counted, _)| *counted == path) {
                Some((_, kind)) => (*kind).to_owned(),
                None if READ.contains(&path.as_str()) => match same_reading {
                    true => format!("same reading: {path}"),
                    false => continue,
                },
                None => format!("other: {path}"),
            };
            kinds.insert(kind);
        }
    }
    for kind in kinds {
        *counted.entry(kind).or_default() += 1;
    }
}

/// List each reviewed value of a row models.dev lists as agreeing or not
/// with models.dev's own value: an object's values part by part, and a
/// `from` row's own values.
fn reviewed_facts(comparison: &mut Comparison, review: &Rows, models_dev: &Rows) {
    for (vendor, models) in review {
        for (id, reviewed) in models {
            let Some(upstream) = models_dev.get(vendor).and_then(|models| models.get(id)) else {
                continue;
            };
            for (key, value) in reviewed {
                if REVIEW_KEYS.contains(&key.as_str()) || key == "rig" {
                    continue;
                }
                let parts: Vec<(String, &Value)> = match value {
                    Value::Object(parts) => parts
                        .iter()
                        .map(|(part, value)| (format!("{key}.{part}"), value))
                        .collect(),
                    _ => vec![(key.clone(), value)],
                };
                for (path, value) in parts {
                    let theirs = at_path(upstream, &path);
                    let line = format!("`{vendor}/{id}` `{path}`");
                    match theirs == Some(value) {
                        true => comparison.agrees.push(line),
                        false => comparison.disagrees.push(format!(
                            "{line}: review {}, models.dev {}",
                            spell(Some(value)),
                            spell(theirs)
                        )),
                    }
                }
            }
        }
    }
}

impl Comparison {
    /// Why sync must not write, when a loss is unanswered.
    pub(super) fn refusal(&self) -> Option<String> {
        if self.refused.is_empty() {
            return None;
        }
        let mut out = format!(
            "catalog sync: {} rows lose a capability; nothing written.\n",
            self.refused.len()
        );
        for (key, losses) in &self.refused {
            let losses: Vec<String> = losses
                .iter()
                .map(|loss| {
                    let answers: Vec<String> = loss
                        .answers
                        .iter()
                        .map(|path| format!("`{path}`"))
                        .collect();
                    format!("{} (answer with {})", loss.what, answers.join(" or "))
                })
                .collect();
            out.push_str(&format!("  {key}: {}\n", losses.join("; ")));
        }
        out.push_str(&format!(
            "Answer each in {REVIEW}: set that key on the row, with a \"source\", to the old \
             value if models.dev is wrong or to the new one if it is right. Then run sync again \
             with the same `--from` file."
        ));
        Some(out)
    }

    /// The report, for a refresh PR's "Catalog changes".
    pub(super) fn markdown(&self, read_at: &str) -> String {
        let mut out = format!("## Catalog changes\n\nmodels.dev data read {read_at}.\n");
        let section = |out: &mut String, title: String, intro: &str, lines: Vec<String>| {
            if lines.is_empty() {
                return;
            }
            out.push_str(&format!("\n### {title}\n\n"));
            if !intro.is_empty() {
                out.push_str(intro);
                out.push_str("\n\n");
            }
            for line in lines {
                out.push_str(&format!("- {line}\n"));
            }
        };
        let rows = |list: &[(String, Vec<String>)]| -> Vec<String> {
            list.iter()
                .map(|(key, changes)| format!("`{key}`: {}", changes.join("; ")))
                .collect()
        };
        section(
            &mut out,
            format!("Losses answered by the review ({})", self.answered.len()),
            "",
            self.answered
                .iter()
                .map(|(key, loss, source)| format!("`{key}`: {} (source: {source})", loss.what))
                .collect(),
        );
        section(
            &mut out,
            format!(
                "Removed ({}): each needs a decision in this PR",
                self.removed.len()
            ),
            "Retired (say how you know), or kept with a review row that has a `name`. A public \
             model constant naming it is deleted or kept to match.",
            self.removed.iter().map(|key| format!("`{key}`")).collect(),
        );
        let reviewed = self.added.iter().filter(|(_, reviewed)| *reviewed).count();
        section(
            &mut out,
            format!(
                "Added ({}: {reviewed} reviewed, {} not)",
                self.added.len(),
                self.added.len() - reviewed
            ),
            "",
            self.added
                .iter()
                .map(|(key, reviewed)| match reviewed {
                    true => format!("`{key}` (reviewed)"),
                    false => format!("`{key}`"),
                })
                .collect(),
        );
        section(
            &mut out,
            format!("Gains ({})", self.gains.len()),
            "",
            rows(&self.gains),
        );
        section(
            &mut out,
            format!("Limits and status ({})", self.limits.len()),
            "",
            rows(&self.limits),
        );
        section(
            &mut out,
            format!("rig facts ({})", self.facts.len()),
            "",
            rows(&self.facts),
        );
        section(
            &mut out,
            format!(
                "Reviewed values models.dev disagrees with ({})",
                self.disagrees.len()
            ),
            "Every such value today, not only this refresh's. Information only: models.dev \
             values flap, so agreement is no reason to drop one.",
            self.disagrees.clone(),
        );
        if !self.agrees.is_empty() {
            out.push_str(&format!(
                "\nReviewed values models.dev agrees with ({}): {}.\n",
                self.agrees.len(),
                self.agrees.join(", ")
            ));
        }
        let counted: Vec<String> = self
            .counted
            .iter()
            .map(|(kind, rows)| format!("{kind} {rows}"))
            .collect();
        if !counted.is_empty() {
            out.push_str(&format!("\nRows changed: {}.\n", counted.join(", ")));
        }
        out
    }
}

/// Fail on a review row that sets a [`SOURCED`] key, or copies a row's
/// facts with `from`, with no `source` of its own or of the review of the
/// row it copies, unless `unsourced` (the
/// text of [`UNSOURCED`]) lists it as it was when sources became required:
/// a line `<fingerprint> <vendor>/<id>`. A listed row that changed, gained a
/// source or left the review fails too, so the list only shrinks.
pub(super) fn provenance(review: &Rows, unsourced: &str) -> Result<(), String> {
    let mut listed = BTreeMap::new();
    for line in unsourced
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let (print, key) = line
            .split_once(' ')
            .ok_or_else(|| format!("{UNSOURCED}: `{line}` is not `<fingerprint> <vendor>/<id>`"))?;
        listed.insert(key.trim().to_owned(), print.to_owned());
    }
    let mut errors = Vec::new();
    for (vendor, models) in review {
        for (id, row) in models {
            let key = format!("{vendor}/{id}");
            let sets: Vec<&str> = SOURCED
                .into_iter()
                .chain(["from"])
                .filter(|field| row.contains_key(*field))
                .collect();
            let needs = !sets.is_empty()
                && source(row)
                    .or_else(|| origin(review, row).and_then(source))
                    .is_none();
            let sets = sets.join(", ");
            match (needs, listed.remove(&key)) {
                (false, None) => {}
                (false, Some(_)) => {
                    errors.push(format!("  {key}: has a source now; drop its line from {UNSOURCED}"))
                }
                (true, Some(print)) if print == fingerprint(row) => {}
                (true, Some(_)) => errors.push(format!(
                    "  {key}: changed, and sets {sets}; give it a \"source\" and drop its line from \
                     {UNSOURCED}"
                )),
                (true, None) => errors.push(format!("  {key}: sets {sets}")),
            }
        }
    }
    for key in listed.keys() {
        errors.push(format!(
            "  {key}: no such review row; drop its line from {UNSOURCED}"
        ));
    }
    match errors.is_empty() {
        true => Ok(()),
        false => Err(format!(
            "{REVIEW}: {} rows set a fact with no \"source\":\n{}\n\
             Say what proved each fact (a doc URL, a cassette, a live check), or write \
             \"unverified: <where it came from>\". Never claim a check that was not made.",
            errors.len(),
            errors.join("\n")
        )),
    }
}

/// A review row's fingerprint: FNV-1a (64 bits) of its JSON, keys in the
/// order the review writes them, as 16 hex digits.
pub(super) fn fingerprint(row: &Map<String, Value>) -> String {
    let hash = Value::Object(row.clone())
        .to_string()
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    format!("{hash:016x}")
}
