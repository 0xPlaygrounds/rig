use crate::{
    completion::{self, CompletionModel},
    extractor::{ExtractionError, Extractor},
    vector_store,
};

use super::Op;

pub struct Lookup<I, In, T> {
    index: I,
    n: usize,
    _in: std::marker::PhantomData<In>,
    _t: std::marker::PhantomData<T>,
}

impl<I, In, T> Lookup<I, In, T>
where
    I: vector_store::VectorStoreIndex,
{
    pub(crate) fn new(index: I, n: usize) -> Self {
        Self {
            index,
            n,
            _in: std::marker::PhantomData,
            _t: std::marker::PhantomData,
        }
    }
}

impl<I, In, T> Op for Lookup<I, In, T>
where
    I: vector_store::VectorStoreIndex,
    In: Into<String> + Send + Sync,
    T: Send + Sync + for<'a> serde::Deserialize<'a>,
{
    type Input = In;
    type Output = Result<Vec<(f64, String, T)>, vector_store::VectorStoreError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let query: String = input.into();

        let docs = self
            .index
            .top_n::<T>(&query, self.n)
            .await?
            .into_iter()
            .collect();

        Ok(docs)
    }
}

/// Create a new lookup operation.
///
/// The op will perform semantic search on the provided index and return the top `n`
/// results closest results to the input.
pub fn lookup<I, In, T>(index: I, n: usize) -> Lookup<I, In, T>
where
    I: vector_store::VectorStoreIndex,
    In: Into<String> + Send + Sync,
    T: Send + Sync + for<'a> serde::Deserialize<'a>,
{
    Lookup::new(index, n)
}

pub struct Prompt<P, In> {
    prompt: P,
    _in: std::marker::PhantomData<In>,
}

impl<P, In> Prompt<P, In> {
    pub(crate) fn new(prompt: P) -> Self {
        Self {
            prompt,
            _in: std::marker::PhantomData,
        }
    }
}

impl<P, In> Op for Prompt<P, In>
where
    P: completion::Prompt,
    In: Into<String> + Send + Sync,
{
    type Input = In;
    type Output = Result<String, completion::PromptError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let prompt: String = input.into();
        self.prompt.prompt(&prompt).await
    }
}

/// Create a new prompt operation.
///
/// The op will prompt the `model` with the input and return the response.
pub fn prompt<P, In>(model: P) -> Prompt<P, In>
where
    P: completion::Prompt,
    In: Into<String> + Send + Sync,
{
    Prompt::new(model)
}

pub struct Extract<M, Input, Output>
where
    M: CompletionModel,
    Output: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + Send + Sync,
{
    extractor: Extractor<M, Output>,
    _in: std::marker::PhantomData<Input>,
}

impl<M, Input, Output> Extract<M, Input, Output>
where
    M: CompletionModel,
    Output: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + Send + Sync,
{
    pub(crate) fn new(extractor: Extractor<M, Output>) -> Self {
        Self {
            extractor,
            _in: std::marker::PhantomData,
        }
    }
}

impl<M, Input, Output> Op for Extract<M, Input, Output>
where
    M: CompletionModel,
    Output: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + Send + Sync,
    Input: Into<String> + Send + Sync,
{
    type Input = Input;
    type Output = Result<Output, ExtractionError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        self.extractor.extract(&input.into()).await
    }
}

/// Create a new extract operation.
///
/// The op will extract the structured data from the input using the provided `extractor`.
pub fn extract<M, Input, Output>(extractor: Extractor<M, Output>) -> Extract<M, Input, Output>
where
    M: CompletionModel,
    Output: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + Send + Sync,
    Input: Into<String> + Send + Sync,
{
    Extract::new(extractor)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use completion::{Prompt, PromptError};
    use vector_store::{VectorStoreError, VectorStoreIndex};

    pub struct MockModel;

    impl Prompt for MockModel {
        async fn prompt(&self, prompt: &str) -> Result<String, PromptError> {
            Ok(format!("Mock response: {}", prompt))
        }
    }

    pub struct MockIndex;

    impl VectorStoreIndex for MockIndex {
        async fn top_n<T: for<'a> serde::Deserialize<'a> + std::marker::Send>(
            &self,
            _query: &str,
            _n: usize,
        ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
            let doc = serde_json::from_value(serde_json::json!({
                "foo": "bar",
            }))
            .unwrap();

            Ok(vec![(1.0, "doc1".to_string(), doc)])
        }

        async fn top_n_ids(
            &self,
            _query: &str,
            _n: usize,
        ) -> Result<Vec<(f64, String)>, VectorStoreError> {
            Ok(vec![(1.0, "doc1".to_string())])
        }
    }

    #[derive(Debug, serde::Deserialize, PartialEq)]
    pub struct Foo {
        pub foo: String,
    }

    #[tokio::test]
    async fn test_lookup() {
        let index = MockIndex;
        let lookup = lookup::<MockIndex, String, Foo>(index, 1);

        let result = lookup.call("query".to_string()).await.unwrap();
        assert_eq!(
            result,
            vec![(
                1.0,
                "doc1".to_string(),
                Foo {
                    foo: "bar".to_string()
                }
            )]
        );
    }

    #[tokio::test]
    async fn test_prompt() {
        let model = MockModel;
        let prompt = prompt::<MockModel, String>(model);

        let result = prompt.call("hello".to_string()).await.unwrap();
        assert_eq!(result, "Mock response: hello");
    }
}
