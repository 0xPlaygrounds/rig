//! A notebook's JSON rendered as cells for the model, and one cell changed
//! in the file's text, leaving every other byte as it was.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::hash::{BuildHasher, RandomState};
use std::ops::Range;

use rig_core::json_utils::remove_preserving_order;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::ser::PrettyFormatter;
use serde_json::value::RawValue;
use serde_json::{Map, Value, json};

/// Lines of one text output shown; the rest are counted.
const MAX_OUTPUT_LINES: usize = 40;

/// What `notebook_edit` is called with.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CellEdit {
    /// The notebook (`.ipynb`) to edit.
    pub path: String,
    /// The cell: its 0-based index, or its id. With `insert_after`, -1
    /// inserts at the start, and creates the notebook if the file is
    /// missing.
    pub cell: CellRef,
    /// What to do with the cell.
    pub action: Action,
    /// The cell's new source; needed for every action but `delete`.
    #[serde(default)]
    pub source: Option<String>,
    /// The new cell's type (default `code`); a replaced cell keeps its own
    /// unless this is given.
    #[serde(default)]
    pub cell_type: Option<CellType>,
}

/// A cell, by index or by id.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum CellRef {
    /// Its 0-based index.
    Index(i64),
    /// Its id.
    Id(String),
}

/// What `notebook_edit` does with the cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Replace its source (a code cell's outputs are cleared).
    Replace,
    /// Insert a new cell before it.
    InsertBefore,
    /// Insert a new cell after it.
    InsertAfter,
    /// Delete it.
    Delete,
}

/// A cell's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CellType {
    /// Code.
    Code,
    /// Markdown.
    Markdown,
}

impl CellType {
    /// Its name in nbformat.
    fn name(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::Markdown => "markdown",
        }
    }
}

/// The notebook `text` as its cells, each a header (`[3] code (exec 7)`),
/// its source, then its outputs: text cut at [`MAX_OUTPUT_LINES`] lines,
/// rich data as its type and size, an error as its name and message. Why
/// not, when `text` is no notebook.
pub fn render(text: &str) -> Result<String, String> {
    let notebook: Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
    let cells = notebook
        .get("cells")
        .and_then(Value::as_array)
        .ok_or("it has no `cells` list")?;
    let mut out = format!("[notebook: {} cells]\n", cells.len());
    for (index, cell) in cells.iter().enumerate() {
        let kind = cell
            .get("cell_type")
            .and_then(Value::as_str)
            .unwrap_or("cell");
        write!(out, "[{index}] {kind}").ok();
        if let Some(count) = cell.get("execution_count").and_then(Value::as_u64) {
            write!(out, " (exec {count})").ok();
        }
        if let Some(id) = cell.get("id").and_then(Value::as_str) {
            write!(out, " id={id}").ok();
        }
        out.push('\n');
        push_text(&mut out, &joined(cell.get("source")));
        let outputs = cell.get("outputs").and_then(Value::as_array);
        for output in outputs.into_iter().flatten() {
            render_output(&mut out, output);
        }
    }
    Ok(out)
}

/// One output of a code cell, after a line naming its kind.
fn render_output(out: &mut String, output: &Value) {
    let text = |key| output.get(key).and_then(Value::as_str).unwrap_or_default();
    match text("output_type") {
        "stream" => {
            writeln!(out, "--- {}", text("name")).ok();
            push_text(out, &capped(&joined(output.get("text"))));
        }
        "error" => {
            writeln!(out, "--- error\n{}: {}", text("ename"), text("evalue")).ok();
        }
        kind => {
            let kind = if kind == "execute_result" {
                "result"
            } else {
                "display"
            };
            writeln!(out, "--- {kind}").ok();
            let data = output.get("data").and_then(Value::as_object);
            for (mime, value) in data.into_iter().flatten() {
                if mime == "text/plain" {
                    push_text(out, &capped(&joined(Some(value))));
                } else {
                    writeln!(out, "[{mime}, {}]", size(mime, &joined(Some(value)))).ok();
                }
            }
        }
    }
}

/// `text`, ending in a newline.
fn push_text(out: &mut String, text: &str) {
    out.push_str(text);
    if !text.is_empty() && !text.ends_with('\n') {
        out.push('\n');
    }
}

/// A multiline string of nbformat: a string, or a list of its lines.
fn joined(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(lines)) => lines.iter().filter_map(Value::as_str).collect(),
        _ => String::new(),
    }
}

/// `text`'s first [`MAX_OUTPUT_LINES`] lines, and how many more there are.
fn capped(text: &str) -> String {
    let total = text.lines().count();
    if total <= MAX_OUTPUT_LINES {
        return text.to_owned();
    }
    let mut out = String::new();
    for line in text.lines().take(MAX_OUTPUT_LINES) {
        out.push_str(line);
        out.push('\n');
    }
    writeln!(out, "[{} more lines]", total - MAX_OUTPUT_LINES).ok();
    out
}

/// How large the data `text` of type `mime` is: an image's bytes, decoded
/// from its base64, else the text's.
fn size(mime: &str, text: &str) -> String {
    let base64 = mime.starts_with("image/") && !mime.contains("svg");
    let bytes = if base64 {
        text.bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .count()
            * 3
            / 4
    } else {
        text.len()
    };
    if bytes < 1024 {
        format!("{bytes} B")
    } else {
        format!("{} KB", bytes.div_ceil(1024))
    }
}

/// The notebook's `cells` list, as written in its text.
#[derive(Deserialize)]
struct Cells<'a> {
    #[serde(borrow)]
    cells: &'a RawValue,
}

/// How the file lays out its JSON: the indent of one level (`None` when it
/// is all on one line) and the whitespace before each cell.
struct Layout {
    unit: Option<String>,
    prefix: String,
}

impl Layout {
    /// The layout of `text`, whose first cell, if any, is at `first`.
    fn of(text: &str, first: Option<&Range<usize>>) -> Self {
        let unit = text.split_once('\n').map(|(_, rest)| {
            rest.chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect::<String>()
        });
        let before = first.and_then(|span| text.get(..span.start));
        let line = before
            .and_then(|before| before.rsplit_once('\n'))
            .map(|(_, line)| line);
        let prefix = line
            .filter(|line| line.chars().all(char::is_whitespace))
            .map(str::to_owned)
            .unwrap_or_else(|| unit.as_deref().unwrap_or_default().repeat(2));
        Self { unit, prefix }
    }

    /// `cell` written as the file writes its cells.
    fn write(&self, cell: &Value) -> Result<String, String> {
        let Some(unit) = &self.unit else {
            return serde_json::to_string(cell).map_err(|error| error.to_string());
        };
        let mut bytes = Vec::new();
        let formatter = PrettyFormatter::with_indent(unit.as_bytes());
        let mut writer = serde_json::Serializer::with_formatter(&mut bytes, formatter);
        cell.serialize(&mut writer)
            .map_err(|error| error.to_string())?;
        let text = String::from_utf8(bytes).map_err(|error| error.to_string())?;
        Ok(text.replace('\n', &format!("\n{}", self.prefix)))
    }

    /// What goes between two cells, when the file has fewer than two.
    fn separator(&self) -> String {
        match &self.unit {
            Some(_) => format!(",\n{}", self.prefix),
            None => ",".to_owned(),
        }
    }

    /// The `cells` list holding only `cell`, written.
    fn only(&self, cell: &str) -> String {
        match &self.unit {
            Some(unit) => format!("[\n{}{cell}\n{unit}]", self.prefix),
            None => format!("[{cell}]"),
        }
    }
}

/// Where `part`, a slice of `text`, lies in it.
fn span(text: &str, part: &str) -> Option<Range<usize>> {
    let start = part.as_ptr().addr().checked_sub(text.as_ptr().addr())?;
    let end = start.checked_add(part.len())?;
    (end <= text.len()).then_some(start..end)
}

/// The notebook `text` with `edit` made, and what was done. Only the bytes
/// of the cell changed (or the cell added, or removed, with its separator)
/// differ.
pub fn edit(text: &str, edit: &CellEdit) -> Result<(String, String), String> {
    let not_notebook = |error: serde_json::Error| format!("not a notebook: {error}");
    let list: Cells<'_> = serde_json::from_str(text).map_err(not_notebook)?;
    let list = list.cells.get();
    let raws: Vec<&RawValue> = serde_json::from_str(list).map_err(not_notebook)?;
    let cells = raws
        .iter()
        .map(|raw| serde_json::from_str::<Value>(raw.get()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(not_notebook)?;
    let lost = || "the notebook's cells could not be found in its text".to_owned();
    let spans = raws
        .iter()
        .map(|raw| span(text, raw.get()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(lost)?;
    let list_span = span(text, list).ok_or_else(lost)?;
    let layout = Layout::of(text, spans.first());
    let index = || index_of(&edit.cell, &cells);
    let source = || {
        edit.source
            .as_deref()
            .ok_or_else(|| format!("`source` is needed to {:?} a cell", edit.action))
    };
    let (range, written, done) = match edit.action {
        Action::Replace => {
            let index = index()?;
            let (span, old) = spans.get(index).zip(cells.get(index)).ok_or_else(lost)?;
            let cell = replaced(old, source()?, edit.cell_type);
            (
                span.clone(),
                layout.write(&cell)?,
                format!("Replaced cell {index}"),
            )
        }
        Action::InsertBefore | Action::InsertAfter => {
            let after = edit.action == Action::InsertAfter;
            let position = if after && is_start(&edit.cell) {
                0
            } else {
                index()? + usize::from(after)
            };
            let ids = cells
                .iter()
                .filter_map(|cell| cell.get("id")?.as_str())
                .collect();
            let kind = edit.cell_type.unwrap_or(CellType::Code);
            let cell = layout.write(&new_cell(kind, source()?, fresh_id(&ids)))?;
            let done = format!("Inserted {} cell {position}", kind.name());
            let at = |at: usize| at..at;
            let separator = match (spans.first(), spans.get(1)) {
                (Some(first), Some(second)) => text
                    .get(first.end..second.start)
                    .ok_or_else(lost)?
                    .to_owned(),
                _ => layout.separator(),
            };
            match (
                spans.get(position),
                position.checked_sub(1).and_then(|last| spans.get(last)),
            ) {
                (Some(next), _) => (at(next.start), cell + &separator, done),
                (None, Some(last)) => (at(last.end), separator + &cell, done),
                (None, None) => (list_span, layout.only(&cell), done),
            }
        }
        Action::Delete => {
            let index = index()?;
            let span = spans.get(index).ok_or_else(lost)?;
            let range = match (
                spans.get(index + 1),
                index.checked_sub(1).and_then(|i| spans.get(i)),
            ) {
                (Some(next), _) => span.start..next.start,
                (None, Some(previous)) => previous.end..span.end,
                (None, None) => list_span.clone(),
            };
            let empty = if range == list_span { "[]" } else { "" };
            (range, empty.to_owned(), format!("Deleted cell {index}"))
        }
    };
    let (head, tail) = text
        .get(..range.start)
        .zip(text.get(range.end..))
        .ok_or_else(lost)?;
    let out = format!("{head}{written}{tail}");
    serde_json::from_str::<Value>(&out)
        .map_err(|error| format!("the edit broke the JSON: {error}"))?;
    let count = cells.len()
        + usize::from(matches!(
            edit.action,
            Action::InsertBefore | Action::InsertAfter
        ));
    let count = count - usize::from(edit.action == Action::Delete);
    Ok((out, format!("{done}; the notebook has {count} cells")))
}

/// A new nbformat 4 notebook holding the one cell `edit` inserts, which
/// must be an `insert_after` of cell -1.
pub fn create(edit: &CellEdit) -> Result<(String, String), String> {
    if edit.action != Action::InsertAfter || !is_start(&edit.cell) {
        return Err("no such notebook; `insert_after` cell -1 creates one".to_owned());
    }
    let source = edit.source.as_deref().ok_or("`source` is needed")?;
    let kind = edit.cell_type.unwrap_or(CellType::Code);
    let notebook = json!({
        "cells": [new_cell(kind, source, fresh_id(&HashSet::new()))],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    });
    let layout = Layout {
        unit: Some(" ".to_owned()),
        prefix: String::new(),
    };
    let text = layout.write(&notebook)? + "\n";
    Ok((
        text,
        format!("Created the notebook with one {} cell", kind.name()),
    ))
}

/// Whether `cell` is -1, the place before the first cell.
fn is_start(cell: &CellRef) -> bool {
    match cell {
        CellRef::Index(index) => *index == -1,
        CellRef::Id(id) => id == "-1",
    }
}

/// The index in `cells` of the cell `cell` names: by its id, else by its
/// index.
fn index_of(cell: &CellRef, cells: &[Value]) -> Result<usize, String> {
    let index = match cell {
        CellRef::Index(index) => *index,
        CellRef::Id(id) => {
            let named = |cell: &Value| cell.get("id").and_then(Value::as_str) == Some(id.as_str());
            if let Some(index) = cells.iter().position(named) {
                return Ok(index);
            }
            id.parse()
                .map_err(|_| format!("no cell has the id {id:?}"))?
        }
    };
    usize::try_from(index)
        .ok()
        .filter(|index| *index < cells.len())
        .ok_or_else(|| format!("no cell {index}: the notebook has {} cells", cells.len()))
}

/// `old` with `source`, as `kind` if given, else its own type; a code
/// cell's outputs cleared and execution count nulled.
fn replaced(old: &Value, source: &str, kind: Option<CellType>) -> Value {
    let mut cell = old.as_object().cloned().unwrap_or_default();
    let own = old
        .get("cell_type")
        .and_then(Value::as_str)
        .unwrap_or("code");
    let kind = kind.map_or(own, |kind| kind.name()).to_owned();
    cell.insert("cell_type".to_owned(), Value::String(kind.clone()));
    if kind == "code" {
        cell.insert("execution_count".to_owned(), Value::Null);
        cell.insert("outputs".to_owned(), json!([]));
    } else {
        remove_preserving_order(cell, "execution_count");
        remove_preserving_order(cell, "outputs");
    }
    cell.insert("source".to_owned(), lines(source));
    Value::Object(cell)
}

/// A new cell of `kind` holding `source`, with the id `id`, its keys in
/// the order nbformat writes them.
fn new_cell(kind: CellType, source: &str, id: String) -> Value {
    let mut cell = Map::new();
    cell.insert("cell_type".to_owned(), kind.name().into());
    if kind == CellType::Code {
        cell.insert("execution_count".to_owned(), Value::Null);
    }
    cell.insert("id".to_owned(), id.into());
    cell.insert("metadata".to_owned(), json!({}));
    if kind == CellType::Code {
        cell.insert("outputs".to_owned(), json!([]));
    }
    cell.insert("source".to_owned(), lines(source));
    Value::Object(cell)
}

/// `source` as nbformat writes it: a list of its lines, each with its
/// newline.
fn lines(source: &str) -> Value {
    source.split_inclusive('\n').map(Value::from).collect()
}

/// A cell id of 8 hex digits that none of `taken` is.
fn fresh_id(taken: &HashSet<&str>) -> String {
    (0_u64..)
        .map(|n| format!("{:016x}", RandomState::new().hash_one(n)))
        .filter_map(|id| id.get(..8).map(str::to_owned))
        .find(|id| !taken.contains(id.as_str()))
        .unwrap_or_default()
}
