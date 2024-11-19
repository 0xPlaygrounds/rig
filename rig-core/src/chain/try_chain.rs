// use std::{marker::PhantomData};

use std::marker::PhantomData;

use futures::{Future, FutureExt, TryFutureExt};

use crate::{completion, vector_store};

use super::Chain;

pub trait TryChain: Send + Sync {
    type Input: Send;
    type Output: Send;
    type Error;

    fn try_call(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;

    /// Chain a function to the output of the current chain
    ///
    /// # Example
    /// ```rust
    /// use rig::chain::{self, Chain};
    ///
    /// let chain = chain::new()
    ///     .chain(|email: String| async move {
    ///         email.split('@').next().unwrap().to_string()
    ///     })
    ///     .chain(|username: String| async move {
    ///         format!("Hello, {}!", username)
    ///     });
    ///
    /// let result = chain.call("bob@gmail.com".to_string()).await;
    /// assert_eq!(result, "Hello, bob!");
    /// ```
    fn chain_ok<F, Fut>(self, f: F) -> ChainOk<Self, F>
    where
        F: Fn(Self::Output) -> Fut,
        Fut: Future,
        Self: Sized,
    {
        ChainOk::new(self, f)
    }

    /// Same as `chain` but for synchronous functions
    ///
    /// # Example
    /// ```rust
    /// use rig::chain::{self, Chain};
    ///
    /// let chain = chain::new()
    ///    .map(|(x, y)| x + y)
    ///    .map(|z| format!("Result: {z}!"));
    ///
    /// let result = chain.call((1, 2)).await;
    /// assert_eq!(result, "Result: 3!");
    /// ```
    fn map_ok<F, T>(self, f: F) -> MapOk<Self, F>
    where
        F: Fn(Self::Output) -> T,
        Self: Sized,
    {
        MapOk::new(self, f)
    }

    fn map_err<F, E>(self, f: F) -> MapErr<Self, F>
    where
        F: Fn(Self::Error) -> E,
        Self: Sized,
    {
        MapErr::new(self, f)
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
    fn lookup<I, T>(self, index: I, n: usize) -> LookupOk<Self, I, T>
    where
        I: vector_store::VectorStoreIndex,
        Self::Output: Into<String>,
        Self::Error: From<vector_store::VectorStoreError>,
        Self: Sized,
    {
        LookupOk::new(self, index, n)
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
    fn prompt<P>(self, prompt: P) -> PromptOk<Self, P>
    where
        P: completion::Prompt,
        Self::Output: Into<String>,
        Self::Error: From<completion::PromptError>,
        Self: Sized,
    {
        PromptOk::new(self, prompt)
    }
}

// #[derive(Debug, thiserror::Error)]
// pub enum ChainError {
//     #[error("Failed to prompt agent: {0}")]
//     PromptError(#[from] completion::PromptError),

//     #[error("Failed to lookup documents: {0}")]
//     LookupError(#[from] vector_store::VectorStoreError),

//     #[error("Failed to chain operation: {0}")]
//     ChainError(#[from] Box<dyn std::error::Error + Send + Sync>),
// }

impl<Ch, In, Out, E> TryChain for Ch
where
    Ch: Chain<Input = In, Output = Result<Out, E>>,
    In: Send,
    Out: Send,
{
    type Input = In;
    type Output = Out;
    type Error = E;

    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        self.call(input).await
    }
}

// pub struct Empty<T>(PhantomData<T>);

// impl<T: Send + Sync> Chain for Empty<T> {
//     type Input = T;
//     type Output = T;

//     async fn call(&self, value: Self::Input) -> Self::Output {
//         value
//     }
// }

// pub struct TryChained<Ch, F> {
//     chain: Ch,
//     f: F,
// }

// impl<Ch, F> TryChained<Ch, F> {
//     fn new(chain: Ch, f: F) -> Self {
//         Self { chain, f }
//     }
// }

// impl<Ch, F, Fut> Chain for TryChained<Ch, F>
// where
//     Ch: Chain,
//     F: Fn(Ch::Output) -> Fut + Send + Sync,
//     Fut: Future + Send,
// {
//     type Input = Ch::Input;
//     type Output = Fut::Output;

//     async fn call(&self, input: Self::Input) -> Self::Output {
//         let output = self.chain.call(input).await;
//         (self.f)(output).await
//     }
// }

pub struct ChainOk<Ch, F> {
    chain: Ch,
    f: F,
}

impl<Ch, F> ChainOk<Ch, F> {
    pub fn new(chain: Ch, f: F) -> Self {
        Self { chain, f }
    }
}

impl<Ch, F, Fut> TryChain for ChainOk<Ch, F>
where
    Ch: TryChain,
    F: Fn(Ch::Output) -> Fut + Send + Sync,
    Fut: Future + Send,
    Fut::Output: Send,
{
    type Input = Ch::Input;
    type Output = Fut::Output;
    type Error = Ch::Error;

    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        self.chain
            .try_call(input)
            .and_then(|value| (self.f)(value).map(Ok))
            .await
    }
}

pub struct MapOk<Ch, F> {
    chain: Ch,
    f: F,
}

impl<Ch, F> MapOk<Ch, F> {
    fn new(chain: Ch, f: F) -> Self {
        Self { chain, f }
    }
}

impl<Ch, F, T> TryChain for MapOk<Ch, F>
where
    Ch: TryChain,
    F: Fn(Ch::Output) -> T + Send + Sync,
    T: Send,
{
    type Input = Ch::Input;
    type Output = T;
    type Error = Ch::Error;

    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        self.chain
            .try_call(input)
            .map_ok(|value| (self.f)(value))
            .await
    }
}

pub struct MapErr<Ch, F> {
    chain: Ch,
    f: F,
}

impl<Ch, F> MapErr<Ch, F> {
    fn new(chain: Ch, f: F) -> Self {
        Self { chain, f }
    }
}

impl<Ch, F, E> TryChain for MapErr<Ch, F>
where
    Ch: TryChain,
    F: Fn(Ch::Error) -> E + Send + Sync,
    E: Send,
{
    type Input = Ch::Input;
    type Output = Ch::Output;
    type Error = E;

    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        self.chain
            .try_call(input)
            .map_err(|error| (self.f)(error))
            .await
    }
}

pub struct LookupOk<Ch, I: vector_store::VectorStoreIndex, T> {
    chain: Ch,
    index: I,
    n: usize,
    _t: PhantomData<T>,
}

impl<Ch, I: vector_store::VectorStoreIndex, T> LookupOk<Ch, I, T> {
    pub fn new(chain: Ch, index: I, n: usize) -> Self {
        Self {
            chain,
            index,
            n,
            _t: PhantomData,
        }
    }
}

impl<Ch, I, T> TryChain for LookupOk<Ch, I, T>
where
    I: vector_store::VectorStoreIndex,
    Ch: TryChain,
    Ch::Output: Into<String>,
    Ch::Error: From<vector_store::VectorStoreError>,
    T: Send + Sync + for<'a> serde::Deserialize<'a>,
{
    type Input = Ch::Input;
    type Output = (String, Vec<T>);
    type Error = Ch::Error;

    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        self.chain
            .try_call(input)
            .and_then(|query| async {
                let query: String = query.into();

                let docs = self
                    .index
                    .top_n::<T>(&query, self.n)
                    .await?
                    .into_iter()
                    .map(|(_, _, doc)| doc)
                    .collect();

                Ok((query, docs))
            })
            .await
    }
}

pub struct PromptOk<Ch, P: completion::Prompt> {
    chain: Ch,
    prompt: P,
}

impl<Ch, P: completion::Prompt> PromptOk<Ch, P> {
    pub fn new(chain: Ch, prompt: P) -> Self {
        Self { chain, prompt }
    }
}

impl<Ch, P> TryChain for PromptOk<Ch, P>
where
    Ch: TryChain,
    Ch::Output: Into<String>,
    Ch::Error: From<completion::PromptError>,
    P: completion::Prompt,
{
    type Input = Ch::Input;
    type Output = String;
    type Error = Ch::Error;

    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        self.chain
            .try_call(input)
            .and_then(|prompt| async {
                let prompt: String = prompt.into();

                Ok(self.prompt.prompt(&prompt).await?)
            })
            .await
    }
}

pub struct Empty<T, E>(PhantomData<(T, E)>);

impl<T: Send + Sync, E: Send + Sync> Default for Empty<T, E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: Send + Sync, E: Send + Sync> TryChain for Empty<T, E> {
    type Input = T;
    type Output = T;
    type Error = E;

    async fn try_call(&self, value: Self::Input) -> Result<Self::Output, Self::Error> {
        Ok(value)
    }
}
