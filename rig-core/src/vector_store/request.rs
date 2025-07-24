use serde::{Deserialize, Serialize};

use super::VectorStoreError;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VectorSearchRequest {
    query: String,
    samples: u64,
    additional_params: Option<serde_json::Value>,
}

impl VectorSearchRequest {
    pub fn builder() -> VectorSearchRequestBuilder {
        VectorSearchRequestBuilder::default()
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn samples(&self) -> &u64 {
        &self.samples
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct VectorSearchRequestBuilder {
    query: Option<String>,
    samples: Option<u64>,
    additional_params: Option<serde_json::Value>,
}

impl VectorSearchRequestBuilder {
    pub fn query(mut self, query: &str) -> Self {
        self.query = Some(query.to_string());
        self
    }

    pub fn samples(mut self, samples: u64) -> Self {
        self.samples = Some(samples);
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
            additional_params,
        })
    }
}
