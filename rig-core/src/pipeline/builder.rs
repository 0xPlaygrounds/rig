use std::future::Future;

use crate::{completion, vector_store};

use super::{agent_ops, op};

// pub struct PipelineBuilder<E> {
//     _error: std::marker::PhantomData<E>,
// }
pub struct PipelineBuilder;

impl PipelineBuilder {
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
    pub fn map<F, In, T>(self, f: F) -> op::Map<F, In>
    where
        F: Fn(In) -> T + Send + Sync,
        In: Send + Sync,
        T: Send + Sync,
        Self: Sized,
    {
        op::Map::new(f)
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
    pub fn then<F, In, Fut>(self, f: F) -> op::Then<F, In>
    where
        F: Fn(In) -> Fut + Send + Sync,
        In: Send + Sync,
        Fut: Future + Send + Sync,
        Fut::Output: Send + Sync,
        Self: Sized,
    {
        op::Then::new(f)
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
    pub fn chain<T>(self, op: T) -> T
    where
        T: op::Op,
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
    ) -> agent_ops::Lookup<I, In, T>
    where
        I: vector_store::VectorStoreIndex,
        T: Send + Sync + for<'a> serde::Deserialize<'a>,
        In: Into<String> + Send + Sync,
        Self: Sized,
    {
        agent_ops::Lookup::new(index, n)
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
    ) -> impl op::Op<Input = In, Output = Result<String, completion::PromptError>>
    where
        P: completion::Prompt,
        In: Into<String> + Send + Sync,
        Self: Sized,
    {
        agent_ops::prompt(prompt)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChainError {
    #[error("Failed to prompt agent: {0}")]
    PromptError(#[from] completion::PromptError),

    #[error("Failed to lookup documents: {0}")]
    LookupError(#[from] vector_store::VectorStoreError),
}

// pub fn new() -> PipelineBuilder<ChainError> {
//     PipelineBuilder {
//         _error: std::marker::PhantomData,
//     }
// }

// pub fn with_error<E>() -> PipelineBuilder<E> {
//     PipelineBuilder {
//         _error: std::marker::PhantomData,
//     }
// }

pub fn new() -> PipelineBuilder {
    PipelineBuilder
}

#[cfg(test)]
mod tests {    
    
    use super::*;
    use crate::pipeline::{op::Op, parallel::{parallel, Parallel}};
    use agent_ops::tests::{Foo, MockIndex, MockModel};

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

    // #[tokio::test]
    // async fn test_lookup_pipeline() {
    //     let index = MockIndex;

    //     let chain = super::new()
    //         .lookup::<_, _, Foo>(index, 1)
    //         .map_ok(|docs| format!("Top documents:\n{}", docs[0].foo));

    //     let result = chain
    //         .try_call("What is a flurbo?")
    //         .await
    //         .expect("Failed to run chain");

    //     assert_eq!(
    //         result,
    //         "User query: What is a flurbo?\n\nTop documents:\nbar"
    //     );
    // }

    #[tokio::test]
    async fn test_rag_pipeline() {
        let index = MockIndex;

        let chain = super::new()
            .chain(parallel!(op::passthrough(), agent_ops::lookup::<_, _, Foo>(index, 1),))
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

    #[tokio::test]
    async fn test_parallel_chain_compile_check() {
        let _ = super::new().chain(
            Parallel::new(
                op::map(|x: i32| x + 1),
                Parallel::new(
                    op::map(|x: i32| x * 3),
                    Parallel::new(
                        op::map(|x: i32| format!("{} is the number!", x)),
                        op::map(|x: i32| x == 1),
                    ),
                ),
            )
            .map(|(r1, (r2, (r3, r4)))| (r1, r2, r3, r4)),
        );
    }
}
