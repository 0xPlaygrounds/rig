//! Types for constructing vector search queries.
//!
//! - [`VectorSearchRequest`]: Query parameters (pre-embedded query, result count, threshold, filters).
//! - [`Filter`]: The canonical, serializable filter expression. Each backend
//!   translates it with its own `from_filter` constructor.

use serde::{Deserialize, Serialize};

use crate::{OneOrMany, embeddings::Embedding};

/// A pre-embedded vector search request.
///
/// The query arrives already embedded (as one or more [`Embedding`]s); stores never
/// embed text themselves. The type parameter `F` specifies the filter type and
/// defaults to the portable [`Filter`]; backends substitute their own native
/// filter records for operators the portable language does not model.
///
/// Construct with [`VectorSearchRequest::new`] (the two required fields) and refine
/// with the `with_*` methods:
///
/// ```
/// use rig_core::{OneOrMany, embeddings::Embedding, vector_store::request::{Filter, VectorSearchRequest}};
///
/// # fn example(embedding: Embedding) {
/// let request = VectorSearchRequest::new(OneOrMany::one(embedding), 10)
///     .with_threshold(0.7)
///     .with_filter(Filter::eq("category", serde_json::json!("fruit")));
/// # }
/// ```
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequest<F = Filter> {
    /// The pre-embedded query to search with.
    query: OneOrMany<Embedding>,
    /// Maximum number of results to return.
    samples: u64,
    /// Minimum similarity score for results.
    threshold: Option<f64>,
    /// Backend-specific parameters as a JSON object.
    additional_params: Option<serde_json::Value>,
    /// Filter expression to narrow results by metadata.
    filter: Option<F>,
}

impl<F> VectorSearchRequest<F> {
    /// Creates a request from the two required fields: the pre-embedded query and the
    /// maximum number of results.
    pub fn new(query: OneOrMany<Embedding>, samples: u64) -> Self {
        Self {
            query,
            samples,
            threshold: None,
            additional_params: None,
            filter: None,
        }
    }

    /// Replaces the query with a single pre-embedded query embedding.
    pub fn with_query(mut self, query: Embedding) -> Self {
        self.query = OneOrMany::one(query);
        self
    }

    /// Replaces the query with one or more pre-embedded query embeddings.
    pub fn with_queries(mut self, query: OneOrMany<Embedding>) -> Self {
        self.query = query;
        self
    }

    /// Sets the maximum number of results to return.
    pub fn with_samples(mut self, samples: u64) -> Self {
        self.samples = samples;
        self
    }

    /// Sets the minimum similarity threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Sets backend-specific parameters.
    pub fn with_additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Sets a filter expression to narrow results by metadata.
    pub fn with_filter(mut self, filter: F) -> Self {
        self.filter = Some(filter);
        self
    }

    /// The pre-embedded query used in similarity search.
    pub fn query(&self) -> &OneOrMany<Embedding> {
        &self.query
    }

    /// Returns the maximum number of results to return.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Returns the optional similarity threshold.
    pub fn threshold(&self) -> Option<f64> {
        self.threshold
    }

    /// Returns a reference to the optional filter expression.
    pub fn filter(&self) -> &Option<F> {
        &self.filter
    }

    /// Transforms the filter type using the provided function.
    ///
    /// This is useful for converting between filter representations, such as
    /// translating the canonical [`super::request::Filter`] to a backend-specific filter type.
    pub fn map_filter<T, Func>(self, f: Func) -> VectorSearchRequest<T>
    where
        Func: Fn(F) -> T,
    {
        VectorSearchRequest {
            query: self.query,
            samples: self.samples,
            threshold: self.threshold,
            additional_params: self.additional_params,
            filter: self.filter.map(f),
        }
    }

    /// Transforms the filter type using a provided function which can additionally return a result.
    ///
    /// Useful for converting between filter representations where the conversion can potentially fail (eg, unrepresentable or invalid values).
    pub fn try_map_filter<T, Func>(self, f: Func) -> Result<VectorSearchRequest<T>, FilterError>
    where
        Func: Fn(F) -> Result<T, FilterError>,
    {
        let filter = self.filter.map(f).transpose()?;

        Ok(VectorSearchRequest {
            query: self.query,
            samples: self.samples,
            threshold: self.threshold,
            additional_params: self.additional_params,
            filter,
        })
    }
}

/// Errors from constructing or converting filter expressions.
#[derive(Debug, Clone, thiserror::Error)]
pub enum FilterError {
    #[error("Expected: {expected}, got: {got}")]
    Expected { expected: String, got: String },

    #[error("Cannot compile '{0}' to the backend's filter type")]
    TypeError(String),

    #[error("Missing field '{0}'")]
    MissingField(String),

    #[error("'{0}' must {1}")]
    Must(String, String),

    // NOTE: Uses String because `serde_json::Error` is not `Clone`.
    #[error("Filter serialization failed: {0}")]
    Serialization(String),
}

/// Canonical, serializable filter representation.
///
/// This is the portable filter language every backend understands: build one with
/// [`Filter::eq`] and friends, and the backend translates it with its own
/// `from_filter` constructor. Operands are [`serde_json::Value`], the one shape
/// every backend can translate.
///
/// The portable language is deliberately small. Backends additionally expose
/// native filter types with richer operators (array membership, negation,
/// backend-native expressions), which [`VectorSearchRequest`] still accepts via
/// its `F` parameter — reach for those directly when the portable operators are
/// not enough.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Filter {
    Eq(String, serde_json::Value),
    Gt(String, serde_json::Value),
    Lt(String, serde_json::Value),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

impl Filter {
    /// Select values where the entry at `key` is equal to `value`
    #[allow(clippy::should_implement_trait)]
    pub fn eq(key: impl AsRef<str>, value: impl Into<serde_json::Value>) -> Self {
        Self::Eq(key.as_ref().to_owned(), value.into())
    }

    /// Select values where the entry at `key` is greater than `value`
    pub fn gt(key: impl AsRef<str>, value: impl Into<serde_json::Value>) -> Self {
        Self::Gt(key.as_ref().to_owned(), value.into())
    }

    /// Select values where the entry at `key` is less than `value`
    pub fn lt(key: impl AsRef<str>, value: impl Into<serde_json::Value>) -> Self {
        Self::Lt(key.as_ref().to_owned(), value.into())
    }

    /// Select values where the entry satisfies `self` *and* `rhs`
    pub fn and(self, rhs: Self) -> Self {
        Self::And(self.into(), rhs.into())
    }

    /// Select values where the entry satisfies `self` *or* `rhs`
    pub fn or(self, rhs: Self) -> Self {
        Self::Or(self.into(), rhs.into())
    }

    /// Tests whether a JSON document satisfies this filter.
    ///
    /// Leaf filters (`Eq`/`Gt`/`Lt`) look their key up in `value` (expected to be
    /// a JSON object) and compare the resulting field against the filter operand.
    /// A missing field, or an operand that is not order-comparable with the field,
    /// never satisfies the leaf. `And`/`Or` combine leaf results.
    pub fn satisfies(&self, value: &serde_json::Value) -> bool {
        use Filter::*;
        use serde_json::{Value, Value::*};
        use std::cmp::Ordering;

        fn compare_pair(l: &Value, r: &Value) -> Option<Ordering> {
            match (l, r) {
                // Compare integers exactly; fall back to f64 only for floats or
                // mixed int/float operands. Trying `as_f64` first (as the old
                // code did) would lose precision for integers beyond 2^53.
                (Number(l), Number(r)) => {
                    if let (Some(l), Some(r)) = (l.as_i64(), r.as_i64()) {
                        Some(l.cmp(&r))
                    } else if let (Some(l), Some(r)) = (l.as_u64(), r.as_u64()) {
                        Some(l.cmp(&r))
                    } else {
                        l.as_f64()
                            .zip(r.as_f64())
                            .and_then(|(l, r)| l.partial_cmp(&r))
                    }
                }
                (String(l), String(r)) => Some(l.cmp(r)),
                (Null, Null) => Some(Ordering::Equal),
                (Bool(l), Bool(r)) => Some(l.cmp(r)),
                _ => None,
            }
        }

        match self {
            // Numbers compare numerically so `5` matches `5.0`, consistent with
            // `Gt`/`Lt`; other JSON types fall back to structural equality so
            // strings/bools/arrays/objects still match exactly.
            Eq(k, v) => value
                .get(k)
                .is_some_and(|field| compare_pair(field, v) == Some(Ordering::Equal) || field == v),
            Gt(k, v) => value
                .get(k)
                .and_then(|field| compare_pair(field, v))
                .is_some_and(|ord| ord == Ordering::Greater),
            Lt(k, v) => value
                .get(k)
                .and_then(|field| compare_pair(field, v))
                .is_some_and(|ord| ord == Ordering::Less),
            And(l, r) => l.satisfies(value) && r.satisfies(value),
            Or(l, r) => l.satisfies(value) || r.satisfies(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Filter;
    use serde_json::json;

    type F = Filter;

    #[test]
    fn eq_matches_field_within_multi_field_document() {
        let doc = json!({ "category": "fruit", "text": "banana" });
        assert!(F::eq("category", json!("fruit")).satisfies(&doc));
        assert!(!F::eq("category", json!("veg")).satisfies(&doc));
        // A field that does not exist never matches.
        assert!(!F::eq("missing", json!("fruit")).satisfies(&doc));
    }

    #[test]
    fn gt_and_lt_compare_the_named_field() {
        let doc = json!({ "price": 10, "text": "banana" });
        assert!(F::gt("price", json!(5)).satisfies(&doc));
        assert!(!F::gt("price", json!(10)).satisfies(&doc));
        assert!(F::lt("price", json!(20)).satisfies(&doc));
        assert!(!F::lt("price", json!(10)).satisfies(&doc));
        // Missing / non-comparable fields never satisfy an ordering filter.
        assert!(!F::gt("missing", json!(1)).satisfies(&doc));
        assert!(!F::gt("text", json!(1)).satisfies(&doc));
    }

    #[test]
    fn eq_matches_integer_and_float_representations() {
        // A field stored as a float still matches an integer operand and vice
        // versa, consistent with Gt/Lt numeric coercion.
        assert!(F::eq("score", json!(5)).satisfies(&json!({ "score": 5.0 })));
        assert!(F::eq("score", json!(5.0)).satisfies(&json!({ "score": 5 })));
        assert!(!F::eq("score", json!(6)).satisfies(&json!({ "score": 5.0 })));
        // Non-numeric fields still use structural equality.
        assert!(F::eq("tag", json!("a")).satisfies(&json!({ "tag": "a" })));
        assert!(F::eq("tags", json!(["a", "b"])).satisfies(&json!({ "tags": ["a", "b"] })));
        assert!(!F::eq("tags", json!(["a"])).satisfies(&json!({ "tags": ["a", "b"] })));
    }

    #[test]
    fn ordering_compares_large_integers_exactly() {
        // Integers beyond 2^53 must not collapse to the same f64.
        let doc = json!({ "id": 9007199254740993_u64 }); // 2^53 + 1
        assert!(F::gt("id", json!(9007199254740992_u64)).satisfies(&doc)); // > 2^53
        assert!(!F::gt("id", json!(9007199254740993_u64)).satisfies(&doc));
        assert!(F::lt("id", json!(9007199254740994_u64)).satisfies(&doc));
    }

    #[test]
    fn serde_representation_is_stable() {
        // `Filter` is serializable and may be persisted, so dropping the operand
        // type parameter must not move the wire format. Variants stay lowercase
        // and newtype-shaped, with the operand as raw JSON.
        let filter = F::eq("category", json!("fruit")).and(F::gt("price", json!(5)));
        let encoded = serde_json::to_value(&filter).expect("filter serializes");
        assert_eq!(
            encoded,
            json!({
                "and": [
                    { "eq": ["category", "fruit"] },
                    { "gt": ["price", 5] },
                ]
            })
        );

        let decoded: Filter = serde_json::from_value(encoded).expect("filter round-trips");
        let doc = json!({ "category": "fruit", "price": 10 });
        assert!(decoded.satisfies(&doc));
    }

    #[test]
    fn operands_accept_any_json_convertible_value() {
        // Operands are `impl Into<serde_json::Value>`, so plain Rust values work
        // without a `json!` wrapper. `Value` itself still converts.
        let doc = json!({ "category": "fruit", "price": 10, "fresh": true });
        assert!(F::eq("category", "fruit").satisfies(&doc));
        assert!(F::gt("price", 5).satisfies(&doc));
        assert!(F::eq("fresh", true).satisfies(&doc));
        assert!(F::eq("category", json!("fruit")).satisfies(&doc));
    }

    #[test]
    fn and_or_combine_leaf_filters() {
        let doc = json!({ "category": "fruit", "price": 10 });
        let both = F::eq("category", json!("fruit")).and(F::gt("price", json!(5)));
        assert!(both.satisfies(&doc));

        let missing_branch = F::eq("category", json!("fruit")).and(F::gt("price", json!(50)));
        assert!(!missing_branch.satisfies(&doc));

        let either = F::eq("category", json!("veg")).or(F::lt("price", json!(50)));
        assert!(either.satisfies(&doc));
    }
}
