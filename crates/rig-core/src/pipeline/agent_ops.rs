use std::future::IntoFuture;

use crate::{
    completion::{self, CompletionModel},
    extractor::{ExtractionError, Extractor},
    message::Message,
    vector_store::{self, request::VectorSearchRequest},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
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
    In: Into<String> + WasmCompatSend + WasmCompatSync,
    T: WasmCompatSend + WasmCompatSync + for<'a> serde::Deserialize<'a>,
{
    type Input = In;
    type Output = Result<Vec<(f64, String, T)>, vector_store::VectorStoreError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let query: String = input.into();

        let req = VectorSearchRequest::builder()
            .query(query)
            .samples(self.n as u64)
            .build();

        let docs = self.index.top_n::<T>(req).await?.into_iter().collect();

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
    In: Into<String> + WasmCompatSend + WasmCompatSync,
    T: WasmCompatSend + WasmCompatSync + for<'a> serde::Deserialize<'a>,
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
    P: completion::Prompt + WasmCompatSend + WasmCompatSync,
    In: Into<String> + WasmCompatSend + WasmCompatSync,
{
    type Input = In;
    type Output = Result<String, completion::PromptError>;

    fn call(
        &self,
        input: Self::Input,
    ) -> impl std::future::Future<Output = Self::Output> + WasmCompatSend {
        self.prompt.prompt(input.into()).into_future()
    }
}

/// Create a new prompt operation.
///
/// The op will prompt the `model` with the input and return the response.
pub fn prompt<P, In>(model: P) -> Prompt<P, In>
where
    P: completion::Prompt,
    In: Into<String> + WasmCompatSend + WasmCompatSync,
{
    Prompt::new(model)
}

pub struct Extract<M, Input, Output>
where
    M: CompletionModel,
    Output: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + WasmCompatSend + WasmCompatSync,
{
    extractor: Extractor<M, Output>,
    _in: std::marker::PhantomData<Input>,
}

impl<M, Input, Output> Extract<M, Input, Output>
where
    M: CompletionModel,
    Output: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + WasmCompatSend + WasmCompatSync,
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
    Output: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + WasmCompatSend + WasmCompatSync,
    Input: Into<Message> + WasmCompatSend + WasmCompatSync,
{
    type Input = Input;
    type Output = Result<Output, ExtractionError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        self.extractor.extract(input).await
    }
}

/// Create a new extract operation.
///
/// The op will extract the structured data from the input using the provided `extractor`.
pub fn extract<M, Input, Output>(extractor: Extractor<M, Output>) -> Extract<M, Input, Output>
where
    M: CompletionModel,
    Output: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + WasmCompatSend + WasmCompatSync,
    Input: Into<String> + WasmCompatSend + WasmCompatSync,
{
    Extract::new(extractor)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::test_utils::{Foo, MockPromptModel, MockVectorStoreIndex};

    #[tokio::test]
    async fn test_lookup() {
        let index = MockVectorStoreIndex;
        let lookup = lookup::<MockVectorStoreIndex, String, Foo>(index, 1);

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
        let model = MockPromptModel;
        let prompt = prompt::<MockPromptModel, String>(model);

        let result = prompt.call("hello".to_string()).await.unwrap();
        assert_eq!(result, "Mock response: hello");
    }
}
