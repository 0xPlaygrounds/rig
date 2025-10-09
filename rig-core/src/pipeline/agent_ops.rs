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
            .build()?;

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
    use crate::message;
    use completion::{Prompt, PromptError};
    use vector_store::{VectorStoreError, VectorStoreIndex};

    pub struct MockModel;

    impl Prompt for MockModel {
        #[allow(refining_impl_trait)]
        async fn prompt(&self, prompt: impl Into<message::Message>) -> Result<String, PromptError> {
            let msg: message::Message = prompt.into();
            let prompt = match msg {
                message::Message::User { content } => match content.first() {
                    message::UserContent::Text(message::Text { text }) => text,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };
            Ok(format!("Mock response: {prompt}"))
        }
    }

    pub struct MockIndex;

    impl VectorStoreIndex for MockIndex {
        async fn top_n<T: for<'a> serde::Deserialize<'a> + WasmCompatSend>(
            &self,
            _req: VectorSearchRequest,
        ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
            let doc = serde_json::from_value(serde_json::json!({
                "foo": "bar",
            }))
            .unwrap();

            Ok(vec![(1.0, "doc1".to_string(), doc)])
        }

        async fn top_n_ids(
            &self,
            _req: VectorSearchRequest,
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
