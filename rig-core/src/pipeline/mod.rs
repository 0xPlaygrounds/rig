//! This module defines a flexible pipeline API for defining a sequence of operations that 
//! may or may not use AI components (e.g.: semantic search, LLMs prompting, etc). 
//! 
//! The pipeline API was inspired by general orchestration pipelines such as Airflow, Dagster and Prefect,
//! but implemented with idiomatic Rust patterns and providing some AI-specific ops out-of-the-box along 
//! general combinators.
//! 
//! Pipelines are made up of one or more "ops", each of which must implement the [Op] trait.
//! The [Op] trait requires the implementation of only one method: `call`, which takes an input
//! and returns an output. The trait provides a wide range of combinators for chaining operations together.
//! One can think of a pipeline as a DAG (Directed Acyclic Graph) where each node is an operation and 
//! the edges represent the data flow between operations. When invoking the pipeline on some input, 
//! the input is passed to the root node of the DAG (i.e.: the first op defined in the pipeline) and 
//! the result of the leaf node is returned as the result of the full pipeline.
//! 
//! ## Basic Example
//! For example, the pipeline below takes a tuple of two integers, adds them together and then formats 
//! the result as a string using the [map](Op::map) combinator method, which applies a function to the 
//! output of the previous op:
//! ```rust
//! use rig::pipeline::{self, Op};
//! 
//! let pipeline = pipeline::new()
//!     // op1: add two numbers
//!     .map(|(x, y)| x + y)
//!     // op2: format result
//!     .map(|z| format!("Result: {z}!"));
//! 
//! let result = pipeline.call((1, 2)).await;
//! assert_eq!(result, "Result: 3!");
//! ```
//! 
//! This pipeline can be visualized as the following DAG:
//! ```text
//!    Input                     
//!      │                       
//!      ▼                       
//! ┌─────────┐                  
//! │   op1   │                  
//! └────┬────┘                  
//!      │                       
//!      ▼                       
//! ┌─────────┐                  
//! │   op2   │                  
//! └────┬────┘                  
//!      │                       
//!      ▼                       
//!    Output     
//! ```
//! 
//! ## Parallel Operations
//! The pipeline API also provides a [parallel!](crate::parallel!) and macro for running operations in parallel.
//! The macro takes a list of ops and turns them into a single op that will duplicate the input 
//! and run each op in parallel. The results of each op are then collected and returned as a tuple.
//! 
//! For example, the pipeline below runs two operations in parallel:
//! ```rust
//! use rig::{pipeline::{self, Op, map}, parallel};
//! 
//! let pipeline = pipeline::new()
//!     .chain(parallel!(
//!         // op1: add 1 to input
//!         map(|x| x + 1),
//!         // op2: subtract 1 from input
//!         map(|x| x - 1),
//!     ))
//!     // op3: format results
//!     .map(|(a, b)| format!("Results: {a}, {b}"));
//! 
//! let result = pipeline.call(1).await;
//! assert_eq!(result, "Result: 2, 0");
//! ```
//! 
//! Notes: 
//! - The [chain](Op::chain) method is similar to the [map](Op::map) method but it allows 
//! for chaining arbitrary operations, as long as they implement the [Op] trait. 
//! - [map] is a function that initializes a standalone [Map](self::op::Map) op without an existing pipeline/op.
//! 
//! The pipeline above can be visualized as the following DAG:
//! ```text                 
//!           Input            
//!             │              
//!      ┌──────┴──────┐       
//!      ▼             ▼       
//! ┌─────────┐   ┌─────────┐  
//! │   op1   │   │   op2   │  
//! └────┬────┘   └────┬────┘  
//!      └──────┬──────┘       
//!             ▼              
//!        ┌─────────┐         
//!        │   op3   │         
//!        └────┬────┘         
//!             │              
//!             ▼              
//!          Output           
//! ```

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

        assert_eq!(result, "Top documents:\nbar");
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
