//! The two readers of catalog data: a strict one for a user's override file,
//! which reports every mistake with its path, and a lenient one for
//! models.dev's `api.json`, which skips what it cannot use and says so.

use std::fmt;

use serde::Deserialize;
use serde_json::{Map, Value};

use super::lookup::{self, edit_distance};
use super::row::{Row, known_fields};
use super::{Catalog, CatalogError, KEYS, vendor_of};
use crate::providers::registry::ProviderId;

/// The row keys a model the base catalog does not list must set, since an
/// absent one reads as "the model does not": reason, call tools, or read
/// any input.
const NEW_MODEL_NEEDS: [&str; 3] = ["reasoning", "tool_call", "modalities"];

/// Every mistake [`Catalog::from_overrides`] found, in file order.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub struct OverrideErrors(pub Vec<OverrideError>);

impl fmt::Display for OverrideErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the catalog override file has ")?;
        match self.0.as_slice() {
            [one] => write!(f, "an error: {one}"),
            errors => {
                write!(f, "{} errors:", errors.len())?;
                errors.iter().try_for_each(|error| write!(f, "\n  {error}"))
            }
        }
    }
}

/// One mistake in an override file.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverrideError {
    /// Where it is: the keys leading to it, joined by `.`
    /// (`antropic`, `anthropic.models.claude-opus-5-5.limits`). Empty for
    /// the file as a whole.
    pub path: String,
    /// What is wrong there.
    pub kind: OverrideErrorKind,
}

/// What is wrong at an [`OverrideError`]'s path.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OverrideErrorKind {
    /// A provider key that is neither a rig vendor nor a models.dev key for
    /// one, with the closest that is.
    UnknownVendor {
        /// The closest known key, if any is close.
        suggestion: Option<&'static str>,
    },
    /// A key rig does not read at that place, with the closest that it does.
    UnknownField {
        /// The closest known key, if any is close.
        suggestion: Option<&'static str>,
    },
    /// A model the catalog the file is laid over does not list, whose row
    /// leaves out keys that would otherwise read as "the model does not".
    /// Most often the id is misspelt.
    NewModel {
        /// The keys the row must set to add the model.
        missing: Vec<&'static str>,
        /// The closest id the catalog lists for that vendor, if any is
        /// close: for a dated snapshot, the model it is a snapshot of.
        suggestion: Option<String>,
    },
    /// A value of the wrong type or form, or text that is not JSON.
    InvalidValue(String),
}

impl fmt::Display for OverrideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let at = if self.path.is_empty() {
            String::new()
        } else {
            format!("`{}`: ", self.path)
        };
        let hint = |suggestion: Option<&str>| {
            suggestion.map_or_else(String::new, |close| format!("; did you mean `{close}`?"))
        };
        match &self.kind {
            OverrideErrorKind::UnknownVendor { suggestion } => {
                write!(f, "{at}unknown vendor{}", hint(*suggestion))
            }
            OverrideErrorKind::UnknownField { suggestion } => {
                write!(f, "{at}unknown field{}", hint(*suggestion))
            }
            OverrideErrorKind::NewModel {
                missing,
                suggestion,
            } => write!(
                f,
                "{at}not a listed model{}; to add it, set {}",
                hint(suggestion.as_deref()),
                missing
                    .iter()
                    .map(|key| format!("`{key}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            OverrideErrorKind::InvalidValue(message) => write!(f, "{at}{message}"),
        }
    }
}

/// A part of a models.dev file [`Catalog::from_models_dev`] did not read.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skipped {
    /// The provider key, or `vendor.models.id` for one row.
    pub path: String,
    /// Why it was skipped.
    pub reason: SkipReason,
}

/// Why [`Catalog::from_models_dev`] skipped part of its input.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkipReason {
    /// The provider is not one rig knows.
    UnknownVendor,
    /// The provider's section, or one model's row, does not read.
    Invalid(String),
}

impl Catalog {
    /// Read a user's override file, laid over `base` afterwards with
    /// [`Self::with_overrides`]. The file is in the shape of
    /// [`Self::from_models_dev`]'s, with each section holding only
    /// `models`, and each row only the keys rig reads plus its hand-entered
    /// facts under `rig`.
    ///
    /// `base` is the catalog the file will be laid over. A row for a model
    /// `base` does not list exactly adds that model, so it must set
    /// `reasoning`, `tool_call` and `modalities`, which an absent key would
    /// otherwise turn into "the model does not". That catches a misspelt
    /// model id, and a dated snapshot id whose row would hide its base
    /// model's facts.
    ///
    /// # Errors
    ///
    /// [`OverrideErrors`] holding every mistake found, each with its path:
    /// an unknown vendor key or field, with the closest known one; a new
    /// model that leaves out a needed key, with the closest listed id; and
    /// a value of the wrong type. Text that is not a JSON object is one
    /// error with an empty path.
    pub fn from_overrides(json: &str, base: &Catalog) -> Result<Catalog, OverrideErrors> {
        let whole = |message: String| {
            OverrideErrors(vec![OverrideError {
                path: String::new(),
                kind: OverrideErrorKind::InvalidValue(message),
            }])
        };
        let sections: Map<String, Value> =
            serde_json::from_str(json).map_err(|error| whole(error.to_string()))?;
        let mut errors = Vec::new();
        let mut error = |path: String, kind| errors.push(OverrideError { path, kind });
        let mut catalog = Catalog::default();
        for (key, section) in &sections {
            let Some(provider) = ProviderId::catalog(vendor_of(key)) else {
                let suggestion = closest(key, vendor_keys());
                error(key.clone(), OverrideErrorKind::UnknownVendor { suggestion });
                continue;
            };
            let Some(section) = section.as_object() else {
                error(key.clone(), invalid("expected an object with `models`"));
                continue;
            };
            for field in section.keys().filter(|field| *field != "models") {
                let suggestion = closest(field, ["models"]);
                error(
                    format!("{key}.{field}"),
                    OverrideErrorKind::UnknownField { suggestion },
                );
            }
            let Some(models) = section.get("models") else {
                continue;
            };
            let Some(models) = models.as_object() else {
                error(
                    format!("{key}.models"),
                    invalid("expected an object of model ids"),
                );
                continue;
            };
            for (id, value) in models {
                let path = format!("{key}.models.{id}");
                let Some(fields) = value.as_object() else {
                    error(path, invalid("expected an object"));
                    continue;
                };
                let mut problems = Vec::new();
                unknown_fields(fields, &[], &mut |at, suggestion| {
                    problems.push((
                        format!("{path}.{at}"),
                        OverrideErrorKind::UnknownField { suggestion },
                    ));
                });
                let row = match serde_path_to_error::deserialize::<_, Row>(value) {
                    Ok(row) => Some(row),
                    Err(failed) => {
                        let at = failed.path().to_string();
                        let at = if at == "." {
                            path.clone()
                        } else {
                            format!("{path}.{at}")
                        };
                        problems.push((at, invalid(&failed.into_inner().to_string())));
                        None
                    }
                };
                if let Some(format) = row.as_ref().and_then(Row::format)
                    && ProviderId::new(provider.vendor(), format).is_none()
                {
                    problems.push((
                        format!("{path}.rig.format"),
                        invalid(&format!(
                            "`{}` speaks no {format} endpoint in this build",
                            provider.vendor()
                        )),
                    ));
                }
                if base.exact(provider.vendor(), id).is_none() {
                    let missing: Vec<&'static str> = NEW_MODEL_NEEDS
                        .into_iter()
                        .filter(|needed| !fields.contains_key(*needed))
                        .collect();
                    if !missing.is_empty() {
                        let suggestion = closest_model(base, provider, id);
                        problems.push((
                            path.clone(),
                            OverrideErrorKind::NewModel {
                                missing,
                                suggestion,
                            },
                        ));
                    }
                }
                match (row, problems.is_empty()) {
                    (Some(row), true) => catalog.put(provider, id, row),
                    _ => problems
                        .into_iter()
                        .for_each(|(path, kind)| error(path, kind)),
                }
            }
        }
        match errors.is_empty() {
            true => Ok(catalog),
            false => Err(OverrideErrors(errors)),
        }
    }

    /// Read models.dev's `api.json`, or a file in its shape: an object of
    /// provider keys, each with a `models` object keyed by model id. A
    /// provider key is a models.dev key (`google`, `amazon-bedrock`) or a
    /// rig vendor name (`gcp.gemini`). Keys rig does not read are ignored,
    /// and a provider rig does not know, or a section or row that does not
    /// read, is skipped and listed, so the rest of the file still loads.
    ///
    /// A row may carry rig's hand-entered facts under `rig`, among them
    /// `format`, the protocol family the model is reached by when that is
    /// not its vendor's own (see [`ProviderId::catalog`]).
    ///
    /// # Errors
    ///
    /// [`CatalogError::Json`] when the text is not a JSON object.
    pub fn from_models_dev(json: &str) -> Result<(Catalog, Vec<Skipped>), CatalogError> {
        let sections: Map<String, Value> = serde_json::from_str(json)?;
        let mut skipped = Vec::new();
        let mut skip = |path: String, reason| skipped.push(Skipped { path, reason });
        let mut catalog = Catalog::default();
        for (key, section) in &sections {
            let Some(provider) = ProviderId::catalog(vendor_of(key)) else {
                skip(key.clone(), SkipReason::UnknownVendor);
                continue;
            };
            let Some(models) = section.get("models").and_then(Value::as_object) else {
                skip(
                    key.clone(),
                    SkipReason::Invalid("no `models` object".to_owned()),
                );
                continue;
            };
            for (id, value) in models {
                let path = format!("{key}.models.{id}");
                let row = match Row::deserialize(value) {
                    Ok(row) => row,
                    Err(error) => {
                        skip(path, SkipReason::Invalid(error.to_string()));
                        continue;
                    }
                };
                if let Some(format) = row.format()
                    && ProviderId::new(provider.vendor(), format).is_none()
                {
                    let reason = format!(
                        "`rig.format`: `{}` speaks no {format} endpoint in this build",
                        provider.vendor()
                    );
                    skip(path, SkipReason::Invalid(reason));
                    continue;
                }
                catalog.put(provider, id, row);
            }
        }
        Ok((catalog, skipped))
    }

    /// [`Self::from_models_dev`] without the list of skipped entries.
    ///
    /// # Errors
    ///
    /// [`CatalogError::Json`] when the text is not a JSON object.
    #[deprecated(
        since = "0.45.0",
        note = "use `from_models_dev` for models.dev data, or `from_overrides` for an override file"
    )]
    pub fn from_json(json: &str) -> Result<Catalog, CatalogError> {
        Catalog::from_models_dev(json).map(|(catalog, _)| catalog)
    }
}

fn invalid(message: &str) -> OverrideErrorKind {
    OverrideErrorKind::InvalidValue(message.to_owned())
}

/// Report each key of `object`, the object at `path` inside a row, that rig
/// does not read there, then the same for each object below it.
fn unknown_fields(
    object: &Map<String, Value>,
    path: &[String],
    report: &mut dyn FnMut(String, Option<&'static str>),
) {
    let keys: Vec<&str> = path.iter().map(String::as_str).collect();
    let Some(known) = known_fields(&keys) else {
        return;
    };
    for (key, value) in object {
        let mut at = path.to_vec();
        at.push(key.clone());
        if !known.contains(&key.as_str()) {
            report(at.join("."), closest(key, known.iter().copied()));
            continue;
        }
        match value {
            Value::Object(inner) => unknown_fields(inner, &at, report),
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    if let Value::Object(inner) = item {
                        let mut at = at.clone();
                        at.push(index.to_string());
                        unknown_fields(inner, &at, report);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Every provider key a catalog file may use: rig's vendor names and the
/// models.dev keys that name one.
fn vendor_keys() -> impl Iterator<Item = &'static str> {
    ProviderId::catalog_vendors().chain(KEYS.iter().map(|(models_dev, _)| *models_dev))
}

/// The candidate closest to `word` by edit distance, if within a quarter of
/// its length (at least one edit), so a typo finds its key and a different
/// word (`output` for `input`) finds none.
fn closest(word: &str, candidates: impl IntoIterator<Item = &'static str>) -> Option<&'static str> {
    let limit = (word.chars().count() / 4).max(1);
    candidates
        .into_iter()
        .map(|candidate| (edit_distance(word, candidate), candidate))
        .filter(|(distance, _)| *distance <= limit)
        .min()
        .map(|(_, candidate)| candidate)
}

/// The id `base` lists for `provider`'s vendor that `id` most likely means:
/// the model a dated snapshot id extends, else the closest listed id.
fn closest_model(base: &Catalog, provider: ProviderId, id: &str) -> Option<String> {
    if let Some(found) = base.get_vendor(provider.vendor(), id) {
        return Some(found.spec.id.clone());
    }
    let vendor = provider.vendor();
    let reference = format!("{vendor}/{id}");
    lookup::suggestions(
        &reference,
        base.iter().filter(|spec| spec.provider.vendor() == vendor),
    )
    .into_iter()
    .next()
    .and_then(|close| close.strip_prefix(&format!("{vendor}/")).map(str::to_owned))
}

#[cfg(test)]
mod tests;
