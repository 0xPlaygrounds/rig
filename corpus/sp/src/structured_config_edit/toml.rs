//! TOML files, through `toml_edit`, which keeps every byte it is not asked
//! to change: comments, order, spacing and quoting.

use serde_json::{Map, Number, Value};
use toml_edit::{Array, DocumentMut, InlineTable, Item, Table};

use crate::structured_config_edit::key::{Seg, line_column};

/// Reads `text` as TOML.
///
/// # Errors
///
/// Where it stops parsing, by line and column.
pub fn parse(text: &str) -> Result<DocumentMut, String> {
    text.parse::<DocumentMut>().map_err(|error| {
        let (line, column) = line_column(text, error.span().map_or(0, |span| span.start));
        format!(
            "does not parse as TOML at line {line}, column {column}: {}",
            error.message()
        )
    })
}

/// The document as JSON, to follow a key path through.
pub fn to_json(doc: &DocumentMut) -> Value {
    item_json(doc.as_item())
}

fn item_json(item: &Item) -> Value {
    match item {
        Item::None => Value::Null,
        Item::Value(value) => value_json(value),
        Item::Table(table) => table_json(table.iter()),
        Item::ArrayOfTables(tables) => Value::Array(
            tables
                .iter()
                .map(|table| table_json(table.iter()))
                .collect(),
        ),
    }
}

fn table_json<'a>(entries: impl Iterator<Item = (&'a str, &'a Item)>) -> Value {
    Value::Object(
        entries
            .map(|(key, item)| (key.to_owned(), item_json(item)))
            .collect::<Map<_, _>>(),
    )
}

fn value_json(value: &toml_edit::Value) -> Value {
    match value {
        toml_edit::Value::String(text) => Value::String(text.value().clone()),
        toml_edit::Value::Integer(number) => Value::from(*number.value()),
        toml_edit::Value::Float(number) => {
            Number::from_f64(*number.value()).map_or(Value::Null, Value::Number)
        }
        toml_edit::Value::Boolean(flag) => Value::Bool(*flag.value()),
        toml_edit::Value::Datetime(when) => Value::String(when.value().to_string()),
        toml_edit::Value::Array(items) => Value::Array(items.iter().map(value_json).collect()),
        toml_edit::Value::InlineTable(table) => Value::Object(
            table
                .iter()
                .map(|(key, value)| (key.to_owned(), value_json(value)))
                .collect(),
        ),
    }
}

/// `value` as a TOML value.
///
/// # Errors
///
/// `null`, which TOML has no way to write, and integers past `i64`.
pub fn from_json(value: &Value) -> Result<toml_edit::Value, String> {
    Ok(match value {
        Value::Null => return Err("TOML has no null; use config_delete to remove a key".into()),
        Value::Bool(flag) => (*flag).into(),
        Value::Number(number) => match (number.as_i64(), number.as_f64()) {
            (Some(integer), _) => integer.into(),
            (None, Some(float)) if !number.is_u64() => float.into(),
            _ => return Err(format!("{number} does not fit a TOML integer")),
        },
        Value::String(text) => text.as_str().into(),
        Value::Array(items) => {
            let mut array = Array::new();
            for item in items {
                array.push(from_json(item)?);
            }
            array.into()
        }
        Value::Object(map) => {
            let mut table = InlineTable::new();
            for (key, item) in map {
                table.insert(key.as_str(), from_json(item)?);
            }
            table.into()
        }
    })
}

/// The item `path` names, which [`crate::structured_config_edit::key::reach`] found.
fn item_mut<'d>(doc: &'d mut DocumentMut, path: &[Seg]) -> Result<&'d mut Item, String> {
    path.iter()
        .try_fold(doc.as_item_mut(), |here, seg| match seg {
            Seg::Key(key) => here.get_mut(key.as_str()),
            Seg::Index(index) => here.get_mut(*index),
        })
        .ok_or_else(|| "the key was not found in the document".to_owned())
}

/// Sets the item `path` names, which exists, to `value`, keeping the
/// spacing around the old value.
pub fn replace(doc: &mut DocumentMut, path: &[Seg], value: &Value) -> Result<(), String> {
    let mut new = from_json(value)?;
    let item = item_mut(doc, path)?;
    if let Item::Value(old) = item {
        *new.decor_mut() = old.decor().clone();
    }
    *item = Item::Value(new);
    Ok(())
}

/// Adds `key` = `value` to the table `parent` names: a `[table]` of its
/// own for a table at the top level, an inline value anywhere else.
pub fn insert(
    doc: &mut DocumentMut,
    parent: &[Seg],
    key: &str,
    value: &Value,
) -> Result<(), String> {
    let top = parent.is_empty();
    match item_mut(doc, parent)? {
        Item::Table(table) => {
            let item = match value {
                Value::Object(map) if top && !map.is_empty() => {
                    let mut section = Table::new();
                    for (name, inner) in map {
                        section.insert(name, Item::Value(from_json(inner)?));
                    }
                    Item::Table(section)
                }
                other => Item::Value(from_json(other)?),
            };
            table.insert(key, item);
        }
        Item::Value(toml_edit::Value::InlineTable(table)) => {
            let mut new = from_json(value)?;
            // The space before `}` stays at the end.
            if let Some((_, last)) = table.iter_mut().last() {
                let suffix = last.decor().suffix().cloned();
                if let Some(suffix) = suffix {
                    new.decor_mut().set_suffix(suffix);
                    last.decor_mut().set_suffix("");
                }
                new.decor_mut().set_prefix(" ");
            }
            table.insert(key, new);
        }
        _ => return Err("the table to add to was not found in the document".to_owned()),
    }
    Ok(())
}

/// Removes the item `path` names, which exists.
pub fn remove(doc: &mut DocumentMut, path: &[Seg]) -> Result<(), String> {
    let missing = || "the key was not found in the document".to_owned();
    let (last, parent) = path.split_last().ok_or("the key is empty")?;
    let parent = item_mut(doc, parent)?;
    match last {
        Seg::Key(key) => parent
            .as_table_like_mut()
            .and_then(|table| table.remove(key))
            .map(drop)
            .ok_or_else(missing),
        Seg::Index(index) => match parent {
            Item::ArrayOfTables(tables) if *index < tables.len() => {
                tables.remove(*index);
                Ok(())
            }
            Item::Value(toml_edit::Value::Array(items)) if *index < items.len() => {
                items.remove(*index);
                Ok(())
            }
            _ => Err(missing()),
        },
    }
}
