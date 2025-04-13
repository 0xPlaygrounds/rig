use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{embeddings::EmbeddingModel, vector_store::VectorStoreIndex};

pub struct SemanticRouter<V> {
    store: V,
    threshold: f64,
}

impl<V> SemanticRouter<V> {
    pub fn builder() -> SemanticRouterBuilder<V> {
        SemanticRouterBuilder::new()
    }
}

impl<V> SemanticRouter<V>
where
    V: VectorStoreIndex,
{
    pub async fn select_route(&self, query: &str) -> Option<String> {
        let res = self.store.top_n(query, 1).await.ok()?;
        let (score, _, SemanticRoute { tag }) = res.first()?;

        if *score < self.threshold {
            return None;
        }

        Some(tag.to_owned())
    }
}

#[derive(Serialize, Deserialize)]
pub struct SemanticRoute {
    tag: String,
}

pub trait Router: VectorStoreIndex {
    fn retrieve_route() -> impl std::future::Future<Output = Option<String>> + Send;
}

pub struct SemanticRouterBuilder<V> {
    store: Option<V>,
    threshold: Option<f64>,
}

impl<V> SemanticRouterBuilder<V> {
    pub fn new() -> Self {
        Self {
            store: None,
            threshold: None,
        }
    }

    pub fn store(mut self, router: V) -> Self {
        self.store = Some(router);

        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);

        self
    }

    pub fn build(self) -> Result<SemanticRouter<V>, Box<dyn std::error::Error>> {
        let Some(store) = self.store else {
            return Err("Vector store not present".into());
        };

        let threshold = self.threshold.unwrap_or(0.9);

        Ok(SemanticRouter { store, threshold })
    }
}
