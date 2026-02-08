//! Types for constructing vector search queries.
//!
//! - [`VectorSearchRequest`]: Query parameters (text, result count, threshold, filters).
//! - [`SearchFilter`]: Trait for backend-agnostic filter expressions.
//! - [`Filter`]: Canonical, serializable filter representation.

use serde::{Deserialize, Serialize};

use super::VectorStoreError;

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
        match self {
            Self::Eq(key, val) => F::eq(key, val),
            Self::Gt(key, val) => F::gt(key, val),
            Self::Lt(key, val) => F::lt(key, val),
            Self::And(lhs, rhs) => F::and(lhs.interpret(), rhs.interpret()),
            Self::Or(lhs, rhs) => F::or(lhs.interpret(), rhs.interpret()),
        }
    }
}

impl Filter<serde_json::Value> {
    /// Tests whether a JSON value satisfies this filter.
    pub fn satisfies(&self, value: &serde_json::Value) -> bool {
        use Filter::*;
        use serde_json::{Value, Value::*, json};
        use std::cmp::Ordering;

        fn compare_pair(l: &Value, r: &Value) -> Option<std::cmp::Ordering> {
            match (l, r) {
                (Number(l), Number(r)) => l
                    .as_f64()
                    .zip(r.as_f64())
                    .and_then(|(l, r)| l.partial_cmp(&r))
                    .or(l.as_i64().zip(r.as_i64()).map(|(l, r)| l.cmp(&r)))
                    .or(l.as_u64().zip(r.as_u64()).map(|(l, r)| l.cmp(&r))),
                (String(l), String(r)) => Some(l.cmp(r)),
                (Null, Null) => Some(std::cmp::Ordering::Equal),
                (Bool(l), Bool(r)) => Some(l.cmp(r)),
                _ => None,
            }
        }

        match self {
            Eq(k, v) => &json!({ k: v }) == value,
            Gt(k, v) => {
                compare_pair(&json!({k: v}), value).is_some_and(|ord| ord == Ordering::Greater)
            }
            Lt(k, v) => {
                compare_pair(&json!({k: v}), value).is_some_and(|ord| ord == Ordering::Less)
            }
            And(l, r) => l.satisfies(value) && r.satisfies(value),
            Or(l, r) => l.satisfies(value) || r.satisfies(value),
        }
    }
}

/// Builder for [`VectorSearchRequest`]. Requires `query` and `samples`.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequestBuilder<F = Filter<serde_json::Value>> {
    query: Option<String>,
    samples: Option<u64>,
    threshold: Option<f64>,
    additional_params: Option<serde_json::Value>,
    filter: Option<F>,
}

impl<F> Default for VectorSearchRequestBuilder<F> {
    fn default() -> Self {
        Self {
            query: None,
            samples: None,
            threshold: None,
            additional_params: None,
            filter: None,
        }
    }
}

impl<F> VectorSearchRequestBuilder<F>
where
    F: SearchFilter,
{
    /// Sets the query text. Required.
    pub fn query<T>(mut self, query: T) -> Self
    where
        T: Into<String>,
    {
        self.query = Some(query.into());
        self
    }

    /// Sets the maximum number of results. Required.
    pub fn samples(mut self, samples: u64) -> Self {
        self.samples = Some(samples);
        self
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

    /// Builds the request, returning an error if required fields are missing.
    pub fn build(self) -> Result<VectorSearchRequest<F>, VectorStoreError> {
        let Some(query) = self.query else {
            return Err(VectorStoreError::BuilderError(
                "`query` is a required variable for building a vector search request".into(),
            ));
        };

        let Some(samples) = self.samples else {
            return Err(VectorStoreError::BuilderError(
                "`samples` is a required variable for building a vector search request".into(),
            ));
        };

        let additional_params = if let Some(params) = self.additional_params {
            if !params.is_object() {
                return Err(VectorStoreError::BuilderError(
                    "Expected JSON object for additional params, got something else".into(),
                ));
            }
            Some(params)
        } else {
            None
        };

        Ok(VectorSearchRequest {
            query,
            samples,
            threshold: self.threshold,
            additional_params,
            filter: self.filter,
        })
    }
}
