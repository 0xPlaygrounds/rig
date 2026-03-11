//! Filter implementation for Cloudflare Vectorize.

use rig::vector_store::request::SearchFilter;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::VectorizeError;

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
mod tests {
    use super::*;
    use rig::vector_store::request::SearchFilter;
    use serde_json::json;

    #[test]
    fn test_eq_filter() {
        let filter = VectorizeFilter::eq("category", json!("programming"));
        assert_eq!(
            filter.into_inner(),
            json!({ "category": { "$eq": "programming" } })
        );
    }

    #[test]
    fn test_gt_filter() {
        let filter = VectorizeFilter::gt("score", json!(0.5));
        assert_eq!(filter.into_inner(), json!({ "score": { "$gt": 0.5 } }));
    }

    #[test]
    fn test_lt_filter() {
        let filter = VectorizeFilter::lt("price", json!(100));
        assert_eq!(filter.into_inner(), json!({ "price": { "$lt": 100 } }));
    }

    #[test]
    fn test_ne_filter() {
        let filter = VectorizeFilter::ne("status", json!("deleted"));
        assert_eq!(
            filter.into_inner(),
            json!({ "status": { "$ne": "deleted" } })
        );
    }

    #[test]
    fn test_gte_filter() {
        let filter = VectorizeFilter::gte("count", json!(10));
        assert_eq!(filter.into_inner(), json!({ "count": { "$gte": 10 } }));
    }

    #[test]
    fn test_lte_filter() {
        let filter = VectorizeFilter::lte("age", json!(65));
        assert_eq!(filter.into_inner(), json!({ "age": { "$lte": 65 } }));
    }

    #[test]
    fn test_in_filter() {
        let filter =
            VectorizeFilter::in_values("category", vec![json!("a"), json!("b"), json!("c")]);
        assert_eq!(
            filter.into_inner(),
            json!({ "category": { "$in": ["a", "b", "c"] } })
        );
    }

    #[test]
    fn test_nin_filter() {
        let filter = VectorizeFilter::nin("status", vec![json!("deleted"), json!("archived")]);
        assert_eq!(
            filter.into_inner(),
            json!({ "status": { "$nin": ["deleted", "archived"] } })
        );
    }

    #[test]
    fn test_and_filter() {
        let filter1 = VectorizeFilter::eq("category", json!("programming"));
        let filter2 = VectorizeFilter::gt("score", json!(0.5));
        let combined = filter1.and(filter2);

        let result = combined.into_inner();
        let obj = result.as_object().unwrap();

        // Both keys should be present (implicit AND)
        assert!(obj.contains_key("category"));
        assert!(obj.contains_key("score"));
        assert_eq!(
            obj.get("category").unwrap(),
            &json!({ "$eq": "programming" })
        );
        assert_eq!(obj.get("score").unwrap(), &json!({ "$gt": 0.5 }));
    }

    #[test]
    fn test_or_filter_validation_fails() {
        let filter1 = VectorizeFilter::eq("a", json!(1));
        let filter2 = VectorizeFilter::eq("b", json!(2));
        let combined = filter1.or(filter2);

        // OR should create an invalid filter
        let result = combined.validate();
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            VectorizeError::UnsupportedFilterOperation(msg) => {
                assert!(msg.contains("OR"));
            }
            _ => panic!("Expected UnsupportedFilterOperation error"),
        }
    }

    #[test]
    fn test_empty_filter() {
        let filter = VectorizeFilter::new();
        assert!(filter.is_empty());
        assert_eq!(filter.into_inner(), json!({}));
    }

    #[test]
    fn test_non_empty_filter() {
        let filter = VectorizeFilter::eq("key", json!("value"));
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_multiple_and_filters() {
        let filter = VectorizeFilter::eq("category", json!("tech"))
            .and(VectorizeFilter::gt("score", json!(0.5)))
            .and(VectorizeFilter::lt("price", json!(100)));

        let result = filter.into_inner();
        let obj = result.as_object().unwrap();

        assert_eq!(obj.len(), 3);
        assert!(obj.contains_key("category"));
        assert!(obj.contains_key("score"));
        assert!(obj.contains_key("price"));
    }
}
