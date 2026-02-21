//! Filter implementation for Cloudflare Vectorize.

use rig::vector_store::request::SearchFilter;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::VectorizeError;

/// Filter for Vectorize vector search queries.
///
/// Vectorize supports filtering on indexed metadata fields using operators like
/// `$eq`, `$ne`, `$gt`, `$lt`, `$gte`, `$lte`, `$in`, and `$nin`.
///
/// Note: Vectorize does NOT support OR filters. Calling `or()` will return an error.
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

    /// Returns true if the filter is empty.
    pub fn is_empty(&self) -> bool {
        self.0.as_object().is_none_or(|obj| obj.is_empty())
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

    fn and(self, rhs: Self) -> Self {
        // Vectorize uses implicit AND by merging filter objects
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

    fn or(self, _rhs: Self) -> Self {
        // Vectorize does not support OR filters.
        // We cannot return an error from this trait method, so we log a warning
        // and return a filter that will cause an API error.
        // Users should check for OR support before using it.
        tracing::error!("Vectorize does not support OR filters. This filter will fail.");
        Self(json!({ "$unsupported_or": "Vectorize does not support OR filters" }))
    }
}

impl VectorizeFilter {
    /// Creates a "not equal" filter.
    pub fn ne(key: impl AsRef<str>, value: Value) -> Self {
        Self(json!({ key.as_ref(): { "$ne": value } }))
    }

    /// Creates a "greater than or equal" filter.
    pub fn gte(key: impl AsRef<str>, value: Value) -> Self {
        Self(json!({ key.as_ref(): { "$gte": value } }))
    }

    /// Creates a "less than or equal" filter.
    pub fn lte(key: impl AsRef<str>, value: Value) -> Self {
        Self(json!({ key.as_ref(): { "$lte": value } }))
    }

    /// Creates an "in" filter (value must be one of the provided values).
    pub fn in_values(key: impl AsRef<str>, values: Vec<Value>) -> Self {
        Self(json!({ key.as_ref(): { "$in": values } }))
    }

    /// Creates a "not in" filter (value must not be any of the provided values).
    pub fn nin(key: impl AsRef<str>, values: Vec<Value>) -> Self {
        Self(json!({ key.as_ref(): { "$nin": values } }))
    }

    /// Validates that the filter doesn't contain unsupported operations.
    /// Returns an error if the filter contains OR operations.
    pub fn validate(&self) -> Result<(), VectorizeError> {
        if let Some(obj) = self.0.as_object() {
            if obj.contains_key("$unsupported_or") {
                return Err(VectorizeError::UnsupportedFilterOperation(
                    "OR filters are not supported by Vectorize".to_string(),
                ));
            }
        }
        Ok(())
    }
}
