use serde::{Deserialize, Serialize};

use super::VectorStoreError;

/// A vector search request - used in the [`super::VectorStoreIndex`] trait.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequest {
    /// The query to be embedded and used in similarity search.
    query: String,
    /// The maximum number of samples that may be returned. If adding a similarity search threshold, you may receive less than the inputted number if there aren't enough results that satisfy the threshold.
    samples: u64,
    /// Similarity search threshold. If present, any result with a distance less than this may be omitted from the final result.
    threshold: Option<f64>,
    /// Any additional parameters that are required by the vector store.
    additional_params: Option<serde_json::Value>,
}

impl VectorSearchRequest {
    /// Creates a [`VectorSearchRequestBuilder`] which you can use to instantiate this struct.
    pub fn builder() -> VectorSearchRequestBuilder {
        VectorSearchRequestBuilder::default()
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
}

/// The builder struct to instantiate [`VectorSearchRequest`].
#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct VectorSearchRequestBuilder {
    query: Option<String>,
    samples: Option<u64>,
    threshold: Option<f64>,
    additional_params: Option<serde_json::Value>,
}

impl VectorSearchRequestBuilder {
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

    pub fn build(self) -> Result<VectorSearchRequest, VectorStoreError> {
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
        })
    }
}
