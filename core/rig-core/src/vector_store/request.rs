use serde::{Deserialize, Serialize};

use super::VectorStoreError;

/// A vector search request - used in the [`super::VectorStoreIndex`] trait.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequest<F = Filter<serde_json::Value>> {
    /// The query to be embedded and used in similarity search.
    query: String,
    /// The maximum number of samples that may be returned. If adding a similarity search threshold, you may receive less than the inputted number if there aren't enough results that satisfy the threshold.
    samples: u64,
    /// Similarity search threshold. If present, any result with a distance less than this may be omitted from the final result.
    threshold: Option<f64>,
    /// Any additional parameters that are required by the vector store.
    additional_params: Option<serde_json::Value>,
    /// An expression used to filter samples
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

    /// The maximum number of samples that may be returned. If adding a similarity search threshold, you may receive less than the inputted number if there aren't enough results that satisfy the threshold.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    pub fn threshold(&self) -> Option<f64> {
        self.threshold
    }

    pub fn filter(&self) -> &Option<Filter> {
        &self.filter
    }

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
}

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
    // NOTE: @FayCarsons - string because `serde_json::Error` is not `Clone`
    // and we need this to be `Clone`
    #[error("Filter serialization failed: {0}")]
    Serialization(String),
}

pub trait SearchFilter {
    type Value;

    fn eq(key: String, value: Self::Value) -> Self;
    fn gt(key: String, value: Self::Value) -> Self;
    fn lt(key: String, value: Self::Value) -> Self;
    fn and(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
}

/// A canonical, serializable retpresentation of filter expressions.
/// This serves as an intermediary form whenever you need to inspect,
/// store, or translate between specific vector store backends
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

    fn eq(key: String, value: Self::Value) -> Self {
        Self::Eq(key, value)
    }

    fn gt(key: String, value: Self::Value) -> Self {
        Self::Gt(key, value)
    }

    fn lt(key: String, value: Self::Value) -> Self {
        Self::Lt(key, value)
    }

    fn and(self, rhs: Self) -> Self {
        Self::And(self.into(), rhs.into())
    }

    fn or(self, rhs: Self) -> Self {
        Self::Or(self.into(), rhs.into())
    }
}

impl<V> Filter<V>
where
    V: std::fmt::Debug + Clone,
{
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

/// The builder struct to instantiate [`VectorSearchRequest`].
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
    /// Set the query (that will then be embedded )
    pub fn query<T>(mut self, query: T) -> Self
    where
        T: Into<String>,
    {
        self.query = Some(query.into());
        self
    }

    pub fn samples(mut self, samples: u64) -> Self {
        self.samples = Some(samples);
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn additional_params(
        mut self,
        params: serde_json::Value,
    ) -> Result<Self, VectorStoreError> {
        self.additional_params = Some(params);
        Ok(self)
    }

    pub fn filter(mut self, filter: F) -> Self {
        self.filter = Some(filter);
        self
    }

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
