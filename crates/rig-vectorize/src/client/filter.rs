//! Filter implementation for Cloudflare Vectorize.

use rig_core::vector_store::request::SearchFilter;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::VectorizeError;

/// Metadata filter for Vectorize queries, supporting the `$eq`, `$ne`, `$gt`,
/// `$lt`, `$gte`, `$lte`, `$in`, and `$nin` operators over indexed fields.
///
/// Vectorize has no disjunction, so `or` produces a filter that
/// [`VectorizeFilter::validate`] rejects.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorizeFilter(Value);

impl VectorizeFilter {
    /// Creates an empty filter.
    pub fn new() -> Self {
        Self(json!({}))
    }

    /// Returns the inner JSON value.
    pub fn into_inner(self) -> Value {
        self.0
    }

    /// Returns a reference to the inner JSON value.
    pub fn as_value(&self) -> &Value {
        &self.0
    }

    /// Whether the filter constrains nothing.
    pub fn is_empty(&self) -> bool {
        self.0.as_object().is_none_or(serde_json::Map::is_empty)
    }
}

impl SearchFilter for VectorizeFilter {
    type Value = Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(json!({ key.as_ref(): { "$eq": value } }))
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(json!({ key.as_ref(): { "$gt": value } }))
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self(json!({ key.as_ref(): { "$lt": value } }))
    }

    /// Merges both filters into one object, which Vectorize evaluates as a
    /// conjunction. Keys present in both are taken from `rhs`.
    fn and(self, rhs: Self) -> Self {
        let mut merged = match self.0 {
            Value::Object(obj) => obj,
            _ => serde_json::Map::new(),
        };

        if let Value::Object(rhs_obj) = rhs.0 {
            for (k, v) in rhs_obj {
                merged.insert(k, v);
            }
        }

        Self(Value::Object(merged))
    }

    /// Discards both operands and returns a sentinel filter, since Vectorize has
    /// no disjunction and this method cannot fail. Searching with the result
    /// errors during validation.
    fn or(self, _rhs: Self) -> Self {
        tracing::error!("Vectorize does not support OR filters. This filter will fail.");
        Self(json!({ "$unsupported_or": "Vectorize does not support OR filters" }))
    }
}

impl VectorizeFilter {
    /// Matches records whose value at `key` differs from `value`.
    pub fn ne(key: impl AsRef<str>, value: &Value) -> Self {
        Self(json!({ key.as_ref(): { "$ne": value } }))
    }

    /// Matches records whose value at `key` is at least `value`.
    pub fn gte(key: impl AsRef<str>, value: &Value) -> Self {
        Self(json!({ key.as_ref(): { "$gte": value } }))
    }

    /// Matches records whose value at `key` is at most `value`.
    pub fn lte(key: impl AsRef<str>, value: &Value) -> Self {
        Self(json!({ key.as_ref(): { "$lte": value } }))
    }

    /// Matches records whose value at `key` is one of `values`.
    pub fn in_values(key: impl AsRef<str>, values: &[Value]) -> Self {
        Self(json!({ key.as_ref(): { "$in": values } }))
    }

    /// Matches records whose value at `key` is none of `values`.
    pub fn nin(key: impl AsRef<str>, values: &[Value]) -> Self {
        Self(json!({ key.as_ref(): { "$nin": values } }))
    }

    /// Rejects filters built with a top-level disjunction. Nested disjunctions
    /// are not detected.
    pub fn validate(&self) -> Result<(), VectorizeError> {
        if let Some(obj) = self.0.as_object()
            && obj.contains_key("$unsupported_or")
        {
            return Err(VectorizeError::UnsupportedFilterOperation(
                "OR filters are not supported by Vectorize".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
