//! This module defines a flexible pipeline API for defining a sequence of operations that
//! may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
//!
//! The pipeline API was inspired by general orchestration pipelines such as Airflow, Dagster and Prefect,
//! but implemented with idiomatic Rust patterns and providing some AI-specific ops out-of-the-box along
//! general combinators.
//!
//! Pipelines are made up of one or more operations, or "ops", each of which must implement the [Op] trait.
//! The [Op] trait requires the implementation of only one method: `call`, which takes an input
//! and returns an output. The trait provides a wide range of combinators for chaining operations together.
//!
//! One can think of a pipeline as a DAG (Directed Acyclic Graph) where each node is an operation and
//! the edges represent the data flow between operations. When invoking the pipeline on some input,
//! the input is passed to the root node of the DAG (i.e.: the first op defined in the pipeline).
//! The output of each op is then passed to the next op in the pipeline until the output reaches the
//! leaf node (i.e.: the last op defined in the pipeline). The output of the leaf node is then returned
//! as the result of the pipeline.
//!
//! ## Basic Example
//! For example, the pipeline below takes a tuple of two integers, adds them together and then formats
//! the result as a string using the [map](Op::map) combinator method, which applies a simple function
//! op to the output of the previous op:
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
//!          ┌─────────┐   ┌─────────┐         
//! Input───►│   op1   ├──►│   op2   ├──►Output
//!          └─────────┘   └─────────┘         
//! ```
//!
//! ## Parallel Operations
//! The pipeline API also provides a [parallel!](crate::parallel!) and macro for running operations in parallel.
//! The macro takes a list of ops and turns them into a single op that will duplicate the input
//! and run each op in concurrently. The results of each op are then collected and returned as a tuple.
//!
//! For example, the pipeline below runs two operations concurrently:
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
//!   for chaining arbitrary operations, as long as they implement the [Op] trait.
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
#[macro_use]
pub mod conditional;

use std::future::Future;

pub use op::{map, passthrough, then, Op};
pub use try_op::TryOp;

use crate::{completion, extractor::Extractor, vector_store};

pub struct PipelineBuilder<E> {
    _error: std::marker::PhantomData<E>,
}

impl<E> PipelineBuilder<E> {
    /// Add a function to the current pipeline
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, Op};
    ///
    /// let pipeline = pipeline::new()
    ///    .map(|(x, y)| x + y)
    ///    .map(|z| format!("Result: {z}!"));
    ///
    /// let result = pipeline.call((1, 2)).await;
    /// assert_eq!(result, "Result: 3!");
    /// ```
    pub fn map<F, Input, Output>(self, f: F) -> op::Map<F, Input>
    where
        F: Fn(Input) -> Output + Send + Sync,
        Input: Send + Sync,
        Output: Send + Sync,
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
    /// let pipeline = pipeline::new()
    ///     .then(|email: String| async move {
    ///         email.split('@').next().unwrap().to_string()
    ///     })
    ///     .then(|username: String| async move {
    ///         format!("Hello, {}!", username)
    ///     });
    ///
    /// let result = pipeline.call("bob@gmail.com".to_string()).await;
    /// assert_eq!(result, "Hello, bob!");
    /// ```
    pub fn then<F, Input, Fut>(self, f: F) -> op::Then<F, Input>
    where
        F: Fn(Input) -> Fut + Send + Sync,
        Input: Send + Sync,
        Fut: Future + Send + Sync,
        Fut::Output: Send + Sync,
        Self: Sized,
    {
        op::Then::new(f)
    }

    /// Add an arbitrary operation to the current pipeline.
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
    /// let pipeline = pipeline::new()
    ///    .chain(MyOp);
    ///
    /// let result = pipeline.call(1).await;
    /// assert_eq!(result, 2);
    /// ```
    pub fn chain<T>(self, op: T) -> T
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
    /// use rig::pipeline::{self, Op};
    ///
    /// let pipeline = pipeline::new()
    ///     .lookup(index, 2)
    ///     .pipeline(|(query, docs): (_, Vec<String>)| async move {
    ///         format!("User query: {}\n\nTop documents:\n{}", query, docs.join("\n"))
    ///     });
    ///
    /// let result = pipeline.call("What is a flurbo?".to_string()).await;
    /// ```
    pub fn lookup<I, Input, Output>(self, index: I, n: usize) -> agent_ops::Lookup<I, Input, Output>
    where
        I: vector_store::VectorStoreIndex,
        Output: Send + Sync + for<'a> serde::Deserialize<'a>,
        Input: Into<String> + Send + Sync,
        // E: From<vector_store::VectorStoreError> + Send + Sync,
        Self: Sized,
    {
        agent_ops::Lookup::new(index, n)
    }

    /// Add a prompt operation to the current pipeline/op. The prompt operation expects the
    /// current pipeline to output a string. The prompt operation will use the string to prompt
    /// the given `agent`, which must implements the [Prompt](completion::Prompt) trait and return
    /// the response.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, Op};
    ///
    /// let agent = &openai_client.agent("gpt-4").build();
    ///
    /// let pipeline = pipeline::new()
    ///    .map(|name| format!("Find funny nicknames for the following name: {name}!"))
    ///    .prompt(agent);
    ///
    /// let result = pipeline.call("Alice".to_string()).await;
    /// ```
    pub fn prompt<P, Input>(self, agent: P) -> agent_ops::Prompt<P, Input>
    where
        P: completion::Prompt,
        Input: Into<String> + Send + Sync,
        // E: From<completion::PromptError> + Send + Sync,
        Self: Sized,
    {
        agent_ops::Prompt::new(agent)
    }

    /// Add an extract operation to the current pipeline/op. The extract operation expects the
    /// current pipeline to output a string. The extract operation will use the given `extractor`
    /// to extract information from the string in the form of the type `T` and return it.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, Op};
    ///
    /// #[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
    /// struct Sentiment {
    ///     /// The sentiment score of the text (0.0 = negative, 1.0 = positive)
    ///     score: f64,
    /// }
    ///
    /// let extractor = &openai_client.extractor::<Sentiment>("gpt-4").build();
    ///
    /// let pipeline = pipeline::new()
    ///     .map(|text| format!("Analyze the sentiment of the following text: {text}!"))
    ///     .extract(extractor);
    ///
    /// let result: Sentiment = pipeline.call("I love ice cream!".to_string()).await?;
    /// assert!(result.score > 0.5);
    /// ```
    pub fn extract<M, Input, Output>(
        self,
        extractor: Extractor<M, Output>,
    ) -> agent_ops::Extract<M, Input, Output>
    where
        M: completion::CompletionModel,
        Output: schemars::JsonSchema + for<'a> serde::Deserialize<'a> + Send + Sync,
        Input: Into<String> + Send + Sync,
    {
        agent_ops::Extract::new(extractor)
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
            .map_ok(|docs| format!("Top documents:\n{}", docs[0].2.foo));

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
            .chain(parallel!(
                passthrough(),
                agent_ops::lookup::<_, _, Foo>(index, 1),
            ))
            .map(|(query, maybe_docs)| match maybe_docs {
                Ok(docs) => format!("User query: {}\n\nTop documents:\n{}", query, docs[0].2.foo),
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
