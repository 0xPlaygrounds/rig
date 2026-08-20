//! Model-listing helpers for deterministic tests.

use crate::{
    client::ModelLister,
    model::{Model, ModelList, ModelListingError},
    wasm_compat::WasmCompatSend,
};

/// A [`ModelLister`] that returns a preconfigured list of models.
pub struct MockModelLister {
    models: Vec<Model>,
}

impl MockModelLister {
    /// Create a model lister from a fixed model list.
    pub fn new(models: Vec<Model>) -> Self {
        Self { models }
    }
}

impl ModelLister for MockModelLister {
    fn list_all(
        &self,
    ) -> impl std::future::Future<Output = Result<ModelList, ModelListingError>> + WasmCompatSend
    {
        let models = self.models.clone();
        async move { Ok(ModelList::new(models)) }
    }
}

impl crate::client::ConstructModelLister<Vec<Model>> for MockModelLister {
    fn construct(client: &Vec<Model>) -> Self {
        let client = client.clone();
        Self { models: client }
    }
}
