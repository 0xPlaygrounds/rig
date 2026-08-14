//! Types for constructing vector search queries.
//!
//! - [`VectorSearchRequest`]: Query parameters (text, result count, threshold, filters).
//! - [`SearchFilter`]: Trait for backend-agnostic filter expressions.
//! - [`Filter`]: Canonical, serializable filter representation.

use serde::{Deserialize, Serialize};

use super::VectorStoreError;
use crate::markers::{Missing, Provided};

/// A vector search request for querying a [`super::VectorStoreIndex`].
///
/// The type parameter `F` specifies the filter type (defaults to [`Filter<serde_json::Value>`]).
/// Use [`VectorSearchRequest::builder()`] to construct instances.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequest<F = Filter<serde_json::Value>> {
    /// The query text to embed and search with.
    query: String,
    /// Maximum number of results to return.
    samples: u64,
    /// Minimum similarity score for results.
    threshold: Option<f64>,
    /// Backend-specific parameters as a JSON object.
    additional_params: Option<serde_json::Value>,
    /// Filter expression to narrow results by metadata.
    filter: Option<F>,
}

impl<Filter> VectorSearchRequest<Filter> {
    /// Creates a [`VectorSearchRequestBuilder`] which you can use to instantiate this struct.
    pub fn builder() -> VectorSearchRequestBuilder<Filter> {
        VectorSearchRequestBuilder::<Filter>::default()
    }

    /// The query to be embedded and used in similarity search.
    pub fn query(&self) -> &str {
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
    pub fn filter(&self) -> &Option<Filter> {
        &self.filter
    }

    /// Transforms the filter type using the provided function.
    ///
    /// This is useful for converting between filter representations, such as
    /// translating the canonical [`super::request::Filter`] to a backend-specific filter type.
    pub fn map_filter<T, F>(self, f: F) -> VectorSearchRequest<T>
    where
        F: Fn(Filter) -> T,
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
    pub fn try_map_filter<T, F>(self, f: F) -> Result<VectorSearchRequest<T>, FilterError>
    where
        F: Fn(Filter) -> Result<T, FilterError>,
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

/// Trait for constructing filter expressions in vector search queries.
///
/// Uses [tagless final](https://nrinaudo.github.io/articles/tagless_final.html) encoding
/// for backend-agnostic filters. Use `SearchFilter::eq(...)` etc. directly and let
/// type inference resolve the concrete filter type.
pub trait SearchFilter {
    type Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self;
    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self;
    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self;
    fn and(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
}

/// A rendered SQL-style condition together with its positional bind parameters.
///
/// Shared by the SQL-flavoured vector stores, whose filter algebra differs only
/// in the parameter type `P` and in the placeholder token their driver expects
/// (`$` for Postgres, `?` for CQL). The placeholder is therefore supplied by the
/// caller on every leaf constructor rather than baked into this type.
///
/// Parameters are collected left to right in the order their placeholders appear
/// in [`SqlCondition::condition`], which is the order drivers bind them in.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SqlCondition<P> {
    condition: String,
    params: Vec<P>,
}

/// Hand-written so that `P` needs no [`Default`] of its own: a parameterless
/// empty condition is meaningful for every parameter type.
impl<P> Default for SqlCondition<P> {
    fn default() -> Self {
        Self {
            condition: String::new(),
            params: Vec::new(),
        }
    }
}

impl<P> SqlCondition<P> {
    /// Renders `<key> <op> <placeholder>` bound to a single parameter, e.g.
    /// `price >= $`.
    pub fn binary(key: impl AsRef<str>, op: &str, placeholder: &str, value: P) -> Self {
        Self {
            condition: format!("{} {op} {placeholder}", key.as_ref()),
            params: vec![value],
        }
    }

    /// Renders `<key> <op> (<placeholder>, ...)` with one placeholder per value,
    /// e.g. `id IN (?, ?)`.
    pub fn list(key: impl AsRef<str>, op: &str, placeholder: &str, values: Vec<P>) -> Self {
        let placeholders = vec![placeholder; values.len()].join(", ");

        Self {
            condition: format!("{} {op} ({placeholders})", key.as_ref()),
            params: values,
        }
    }

    /// Wraps an already-rendered, parameterless condition such as `id is null`.
    pub fn raw(condition: impl Into<String>) -> Self {
        Self {
            condition: condition.into(),
            params: Vec::new(),
        }
    }

    /// Conjoins two conditions as `(lhs) AND (rhs)`, concatenating their parameters.
    pub fn and(self, rhs: Self) -> Self {
        self.combine("AND", rhs)
    }

    /// Disjoins two conditions as `(lhs) OR (rhs)`, concatenating their parameters.
    pub fn or(self, rhs: Self) -> Self {
        self.combine("OR", rhs)
    }

    /// Negates the condition as `NOT (condition)`, keeping its parameters.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self {
            condition: format!("NOT ({})", self.condition),
            ..self
        }
    }

    fn combine(self, joiner: &str, rhs: Self) -> Self {
        Self {
            condition: format!("({}) {joiner} ({})", self.condition, rhs.condition),
            params: self.params.into_iter().chain(rhs.params).collect(),
        }
    }

    /// The rendered condition, with placeholders as the caller supplied them.
    pub fn condition(&self) -> &str {
        &self.condition
    }

    /// The bind parameters, in placeholder order.
    pub fn params(&self) -> &[P] {
        &self.params
    }

    /// Consumes the condition, returning the rendered text and its parameters.
    pub fn into_parts(self) -> (String, Vec<P>) {
        (self.condition, self.params)
    }
}

/// Converts the canonical JSON-valued [`Filter`] used by type-erased vector
/// searches into a backend's native filter representation.
///
/// JSON-valued [`SearchFilter`] implementations receive this automatically.
/// Backends with native value types implement the conversion once here rather
/// than hand-writing both [`VectorStoreIndexDyn`](super::VectorStoreIndexDyn)
/// methods.
pub trait DynamicSearchFilter: SearchFilter + Sized {
    /// Converts a canonical dynamic filter into this backend's filter type.
    fn from_dynamic_filter(filter: Filter<serde_json::Value>) -> Result<Self, FilterError>;

    /// Normalizes a document returned through the type-erased search surface.
    ///
    /// Native backend filters retain their documents exactly. JSON-valued
    /// filters override this to preserve the dynamic surface's historical
    /// payload pruning.
    fn normalize_dynamic_document(document: serde_json::Value) -> serde_json::Value {
        document
    }
}

impl<F> DynamicSearchFilter for F
where
    F: SearchFilter<Value = serde_json::Value>,
{
    fn from_dynamic_filter(filter: Filter<serde_json::Value>) -> Result<Self, FilterError> {
        Ok(filter.interpret())
    }

    fn normalize_dynamic_document(document: serde_json::Value) -> serde_json::Value {
        prune_document(document).unwrap_or_default()
    }
}

fn prune_document(document: serde_json::Value) -> Option<serde_json::Value> {
    match document {
        serde_json::Value::Object(mut map) => {
            let new_map = map
                .iter_mut()
                .filter_map(|(key, value)| {
                    prune_document(value.take()).map(|value| (key.clone(), value))
                })
                .collect::<serde_json::Map<_, _>>();

            Some(serde_json::Value::Object(new_map))
        }
        serde_json::Value::Array(vec) if vec.len() > 400 => None,
        serde_json::Value::Array(vec) => Some(serde_json::Value::Array(
            vec.into_iter().filter_map(prune_document).collect(),
        )),
        value => Some(value),
    }
}

/// Canonical, serializable filter representation.
///
/// Use for serialization, runtime inspection, or translating between backends via
/// [`Filter::interpret`]. Prefer [`SearchFilter`] trait methods for writing queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Filter<V>
where
    V: std::fmt::Debug + Clone,
{
    Eq(String, V),
    Gt(String, V),
    Lt(String, V),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

impl<V> SearchFilter for Filter<V>
where
    V: std::fmt::Debug + Clone + Serialize + for<'de> Deserialize<'de>,
{
    type Value = V;

    /// Select values where the entry at `key` is equal to `value`
    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self::Eq(key.as_ref().to_owned(), value)
    }

    /// Select values where the entry at `key` is greater than `value`
    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self::Gt(key.as_ref().to_owned(), value)
    }

    /// Select values where the entry at `key` is less than `value`
    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self::Lt(key.as_ref().to_owned(), value)
    }

    /// Select values where the entry satisfies `self` *and* `rhs`
    fn and(self, rhs: Self) -> Self {
        Self::And(self.into(), rhs.into())
    }

    /// Select values where the entry satisfies `self` *or* `rhs`
    fn or(self, rhs: Self) -> Self {
        Self::Or(self.into(), rhs.into())
    }
}

impl<V> Filter<V>
where
    V: std::fmt::Debug + Clone,
{
    /// Converts this filter into a backend-specific filter type.
    pub fn interpret<F>(self) -> F
    where
        F: SearchFilter<Value = V>,
    {
        self.interpret_with(|v| v)
    }

    /// Converts this filter into a backend-specific filter type, converting
    /// each leaf value with `conv`.
    pub fn interpret_with<F, W>(self, conv: impl Fn(V) -> W + Copy) -> F
    where
        F: SearchFilter<Value = W>,
    {
        match self.try_interpret(|v| Ok::<W, std::convert::Infallible>(conv(v))) {
            Ok(filter) => filter,
            Err(never) => match never {},
        }
    }

    /// Converts this filter into a backend-specific filter type, converting
    /// each leaf value with `conv`. Fails on the first value that fails to convert.
    pub fn try_interpret<F, W, E>(self, conv: impl Fn(V) -> Result<W, E> + Copy) -> Result<F, E>
    where
        F: SearchFilter<Value = W>,
    {
        Ok(match self {
            Self::Eq(key, val) => F::eq(key, conv(val)?),
            Self::Gt(key, val) => F::gt(key, conv(val)?),
            Self::Lt(key, val) => F::lt(key, conv(val)?),
            Self::And(lhs, rhs) => F::and(lhs.try_interpret(conv)?, rhs.try_interpret(conv)?),
            Self::Or(lhs, rhs) => F::or(lhs.try_interpret(conv)?, rhs.try_interpret(conv)?),
        })
    }
}

impl Filter<serde_json::Value> {
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

/// Builder for [`VectorSearchRequest`]. Requires `query` and `samples`.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequestBuilder<F = Filter<serde_json::Value>, Q = Missing, S = Missing> {
    query: Q,
    samples: S,
    threshold: Option<f64>,
    additional_params: Option<serde_json::Value>,
    filter: Option<F>,
}

impl<F> Default for VectorSearchRequestBuilder<F, Missing, Missing> {
    fn default() -> Self {
        Self {
            query: Missing,
            samples: Missing,
            threshold: None,
            additional_params: None,
            filter: None,
        }
    }
}

impl<F, Q, S> VectorSearchRequestBuilder<F, Q, S>
where
    F: SearchFilter,
{
    /// Sets the query text. Required.
    pub fn query<T>(self, query: T) -> VectorSearchRequestBuilder<F, Provided<String>, S>
    where
        T: Into<String>,
    {
        VectorSearchRequestBuilder {
            query: Provided(query.into()),
            samples: self.samples,
            threshold: self.threshold,
            additional_params: self.additional_params,
            filter: self.filter,
        }
    }

    /// Sets the maximum number of results. Required.
    pub fn samples(self, samples: u64) -> VectorSearchRequestBuilder<F, Q, Provided<u64>> {
        VectorSearchRequestBuilder {
            query: self.query,
            samples: Provided(samples),
            threshold: self.threshold,
            additional_params: self.additional_params,
            filter: self.filter,
        }
    }

    /// Sets the minimum similarity threshold.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Sets backend-specific parameters.
    pub fn additional_params(
        mut self,
        params: serde_json::Value,
    ) -> Result<Self, VectorStoreError> {
        self.additional_params = Some(params);
        Ok(self)
    }

    /// Sets a filter expression.
    pub fn filter(mut self, filter: F) -> Self {
        self.filter = Some(filter);
        self
    }
}

/// Only implement `build()` when both `query` and `samples` have been provided.
impl<F> VectorSearchRequestBuilder<F, Provided<String>, Provided<u64>> {
    /// Builds the request
    pub fn build(self) -> VectorSearchRequest<F> {
        VectorSearchRequest {
            query: self.query.0,
            samples: self.samples.0,
            threshold: self.threshold,
            additional_params: self.additional_params,
            filter: self.filter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Filter, SearchFilter};
    use serde_json::json;

    type F = Filter<serde_json::Value>;

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
    fn and_or_combine_leaf_filters() {
        let doc = json!({ "category": "fruit", "price": 10 });
        let both = F::eq("category", json!("fruit")).and(F::gt("price", json!(5)));
        assert!(both.satisfies(&doc));

        let missing_branch = F::eq("category", json!("fruit")).and(F::gt("price", json!(50)));
        assert!(!missing_branch.satisfies(&doc));

        let either = F::eq("category", json!("veg")).or(F::lt("price", json!(50)));
        assert!(either.satisfies(&doc));
    }

    #[test]
    fn try_interpret_converts_nested_leaf_values() {
        let f: Filter<i64> =
            Filter::Eq("a".into(), 1).and(Filter::Gt("b".into(), 2).or(Filter::Lt("c".into(), 3)));
        let out: Filter<String> = f
            .try_interpret(|v| Ok::<_, std::convert::Infallible>(v.to_string()))
            .unwrap();
        match out {
            Filter::And(lhs, rhs) => {
                assert!(matches!(*lhs, Filter::Eq(ref k, ref v) if k == "a" && v == "1"));
                match *rhs {
                    Filter::Or(l, r) => {
                        assert!(matches!(*l, Filter::Gt(ref k, ref v) if k == "b" && v == "2"));
                        assert!(matches!(*r, Filter::Lt(ref k, ref v) if k == "c" && v == "3"));
                    }
                    other => panic!("expected Or, got {other:?}"),
                }
            }
            other => panic!("expected And, got {other:?}"),
        }
    }

    #[test]
    fn try_interpret_propagates_conversion_errors() {
        let f: Filter<i64> = Filter::Eq("a".into(), 1).and(Filter::Gt("b".into(), -2));
        let out: Result<Filter<u64>, String> =
            f.try_interpret(|v| u64::try_from(v).map_err(|e| e.to_string()));
        assert!(out.is_err());
    }
}
