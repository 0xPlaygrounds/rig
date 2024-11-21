pub mod agent_ops;
pub mod op;
pub mod try_op;
#[macro_use]
pub mod parallel;

use std::future::Future;

use agent_ops::{lookup, prompt as prompt_op};
pub use op::{map, passthrough, then, Op};
pub use try_op::TryOp;

use crate::{completion, vector_store};

pub struct PipelineBuilder<E> {
    _error: std::marker::PhantomData<E>,
}

impl<E> PipelineBuilder<E> {
    /// Chain a function to the current pipeline
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, Op};
    ///
    /// let chain = pipeline::new()
    ///    .map(|(x, y)| x + y)
    ///    .map(|z| format!("Result: {z}!"));
    ///
    /// let result = chain.call((1, 2)).await;
    /// assert_eq!(result, "Result: 3!");
    /// ```
    pub fn map<F, In, T>(self, f: F) -> impl Op<Input = In, Output = T>
    where
        F: Fn(In) -> T + Send + Sync,
        In: Send + Sync,
        T: Send + Sync,
        Self: Sized,
    {
        map(f)
    }

    /// Same as `map` but for asynchronous functions
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, Op};
    ///
    /// let chain = pipeline::new()
    ///     .then(|email: String| async move {
    ///         email.split('@').next().unwrap().to_string()
    ///     })
    ///     .then(|username: String| async move {
    ///         format!("Hello, {}!", username)
    ///     });
    ///
    /// let result = chain.call("bob@gmail.com".to_string()).await;
    /// assert_eq!(result, "Hello, bob!");
    /// ```
    pub fn then<F, In, Fut>(self, f: F) -> impl Op<Input = In, Output = Fut::Output>
    where
        F: Fn(In) -> Fut + Send + Sync,
        In: Send + Sync,
        Fut: Future + Send + Sync,
        Fut::Output: Send + Sync,
        Self: Sized,
    {
        then(f)
    }

    /// Chain an arbitrary operation to the current pipeline.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, Op};
    ///
    /// struct MyOp;
    ///
    /// impl Op for MyOp {
    ///     type Input = i32;
    ///     type Output = i32;
    ///
    ///     async fn call(&self, input: Self::Input) -> Self::Output {
    ///         input + 1
    ///     }
    /// }
    ///
    /// let chain = pipeline::new()
    ///    .chain(MyOp);
    ///
    /// let result = chain.call(1).await;
    /// assert_eq!(result, 2);
    /// ```
    pub fn chain<T>(self, op: T) -> impl Op<Input = T::Input, Output = T::Output>
    where
        T: Op,
        Self: Sized,
    {
        op
    }

    /// Chain a lookup operation to the current chain. The lookup operation expects the
    /// current chain to output a query string. The lookup operation will use the query to
    /// retrieve the top `n` documents from the index and return them with the query string.
    ///
    /// # Example
    /// ```rust
    /// use rig::chain::{self, Chain};
    ///
    /// let chain = chain::new()
    ///     .lookup(index, 2)
    ///     .chain(|(query, docs): (_, Vec<String>)| async move {
    ///         format!("User query: {}\n\nTop documents:\n{}", query, docs.join("\n"))
    ///     });
    ///
    /// let result = chain.call("What is a flurbo?".to_string()).await;
    /// ```
    pub fn lookup<I, In, T>(
        self,
        index: I,
        n: usize,
    ) -> impl Op<Input = In, Output = Result<Vec<T>, vector_store::VectorStoreError>>
    where
        I: vector_store::VectorStoreIndex,
        T: Send + Sync + for<'a> serde::Deserialize<'a>,
        In: Into<String> + Send + Sync,
        // E: From<vector_store::VectorStoreError> + Send + Sync,
        Self: Sized,
    {
        lookup(index, n)
    }

    /// Chain a prompt operation to the current chain. The prompt operation expects the
    /// current chain to output a string. The prompt operation will use the string to prompt
    /// the given agent (or any other type that implements the `Prompt` trait) and return
    /// the response.
    ///
    /// # Example
    /// ```rust
    /// use rig::chain::{self, Chain};
    ///
    /// let agent = &openai_client.agent("gpt-4").build();
    ///
    /// let chain = chain::new()
    ///    .map(|name| format!("Find funny nicknames for the following name: {name}!"))
    ///    .prompt(agent);
    ///
    /// let result = chain.call("Alice".to_string()).await;
    /// ```
    pub fn prompt<P, In>(
        self,
        prompt: P,
    ) -> impl Op<Input = In, Output = Result<String, completion::PromptError>>
    where
        P: completion::Prompt,
        In: Into<String> + Send + Sync,
        // E: From<completion::PromptError> + Send + Sync,
        Self: Sized,
    {
        prompt_op(prompt)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChainError {
    #[error("Failed to prompt agent: {0}")]
    PromptError(#[from] completion::PromptError),

    #[error("Failed to lookup documents: {0}")]
    LookupError(#[from] vector_store::VectorStoreError),
}

pub fn new() -> PipelineBuilder<ChainError> {
    PipelineBuilder {
        _error: std::marker::PhantomData,
    }
}

pub fn with_error<E>() -> PipelineBuilder<E> {
    PipelineBuilder {
        _error: std::marker::PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_ops::tests::{Foo, MockIndex, MockModel};
    use parallel::parallel;

    #[tokio::test]
    async fn test_prompt_pipeline() {
        let model = MockModel;

        let chain = super::new()
            .map(|input| format!("User query: {}", input))
            .prompt(model);

        let result = chain
            .call("What is a flurbo?")
            .await
            .expect("Failed to run chain");

        assert_eq!(result, "Mock response: User query: What is a flurbo?");
    }

    #[tokio::test]
    async fn test_prompt_pipeline_error() {
        let model = MockModel;

        let chain = super::with_error::<()>()
            .map(|input| format!("User query: {}", input))
            .prompt(model);

        let result = chain
            .try_call("What is a flurbo?")
            .await
            .expect("Failed to run chain");

        assert_eq!(result, "Mock response: User query: What is a flurbo?");
    }

    #[tokio::test]
    async fn test_lookup_pipeline() {
        let index = MockIndex;

        let chain = super::new()
            .lookup::<_, _, Foo>(index, 1)
            .map_ok(|docs| format!("Top documents:\n{}", docs[0].foo));

        let result = chain
            .try_call("What is a flurbo?")
            .await
            .expect("Failed to run chain");

        assert_eq!(
            result,
            "User query: What is a flurbo?\n\nTop documents:\nbar"
        );
    }

    #[tokio::test]
    async fn test_rag_pipeline() {
        let index = MockIndex;

        let chain = super::new()
            .chain(parallel!(passthrough(), lookup::<_, _, Foo>(index, 1),))
            .map(|(query, maybe_docs)| match maybe_docs {
                Ok(docs) => format!("User query: {}\n\nTop documents:\n{}", query, docs[0].foo),
                Err(err) => format!("Error: {}", err),
            })
            .prompt(MockModel);

        let result = chain
            .call("What is a flurbo?")
            .await
            .expect("Failed to run chain");

        assert_eq!(
            result,
            "Mock response: User query: What is a flurbo?\n\nTop documents:\nbar"
        );
    }
}
