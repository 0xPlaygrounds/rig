use std::{future::Future, marker::PhantomData};

use crate::{completion, vector_store};

pub trait Chain: Send + Sync {
    type Input: Send;
    type Output;

    fn call(self, input: Self::Input) -> impl std::future::Future<Output = Self::Output> + Send;

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
    fn chain<F, Fut>(self, f: F) -> Chained<Self, F>
    where
        F: Fn(Self::Output) -> Fut,
        Fut: Future,
        Self: Sized,
    {
        Chained::new(self, f)
    }

    // fn try_chain<F, Fut>(self, f: F) -> TryChained<Self, F>
    // where
    //     F: Fn(Self::Output) -> Fut,
    //     Fut: TryFuture,
    //     Self: Sized
    // {
    //     TryChained::new(self, f)
    // }

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
    fn map<F, T>(self, f: F) -> Map<Self, F>
    where
        F: Fn(Self::Output) -> T,
        Self: Sized,
    {
        Map::new(self, f)
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
    fn lookup<I: vector_store::VectorStoreIndex, T>(self, index: I, n: usize) -> Lookup<Self, I, T>
    where
        Self::Output: Into<String>,
        Self: Sized,
    {
        Lookup::new(self, index, n)
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
    fn prompt<P>(self, prompt: P) -> Prompt<Self, P>
    where
        P: completion::Prompt,
        Self::Output: Into<String>,
        Self: Sized,
    {
        Prompt::new(self, prompt)
    }
}

// pub struct Root<A, F> {
//     f: F,
//     _phantom: PhantomData<A>,
// }

// impl<A, F, Fut> Root<A, F>
// where
//     F: FnOnce(A) -> Fut,
//     Fut: Future,
// {
//     pub fn new(f: F) -> Self {
//         Self {
//             f,
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<A: Send + Sync, F, Fut> Chain for Root<A, F>
// where
//     F: FnOnce(A) -> Fut + Send + Sync,
//     Fut: Future + Send + Sync,
// {
//     type Input = A;
//     type Output = Fut::Output;

//     async fn call(self, input: Self::Input) -> Self::Output {
//         (self.f)(input).await
//     }
// }

pub struct Empty<T>(PhantomData<T>);

impl<T: Send + Sync> Chain for Empty<T> {
    type Input = T;
    type Output = T;

    async fn call(self, value: Self::Input) -> Self::Output {
        value
    }
}

pub struct Chained<Ch, F> {
    chain: Ch,
    f: F,
}

impl<Ch, F> Chained<Ch, F> {
    fn new(chain: Ch, f: F) -> Self {
        Self { chain, f }
    }
}

impl<Ch, F, Fut> Chain for Chained<Ch, F>
where
    Ch: Chain,
    F: Fn(Ch::Output) -> Fut + Send + Sync,
    Fut: Future + Send,
{
    type Input = Ch::Input;
    type Output = Fut::Output;

    async fn call(self, input: Self::Input) -> Self::Output {
        let output = self.chain.call(input).await;
        (self.f)(output).await
    }
}

// pub struct TryChained<Ch, F> {
//     chain: Ch,
//     f: F,
// }

// impl<Ch, F> TryChained<Ch, F> {
//     pub fn new(chain: Ch, f: F) -> Self {
//         Self {
//             chain,
//             f,
//         }
//     }
// }

// impl<Ch, F, Fut, T, E> Chain for TryChained<Ch, F>
// where
//     Ch: Chain,
//     F: Fn(Ch::Output) -> Fut + Send + Sync,
//     Fut: Future<Output = Result<T, E>>,
// {
//     type Input = Ch::Input;
//     type Output = Fut::Output;

//     async fn call(self, input: Self::Input) -> Self::Output {
//         let output = self.chain.call(input).await;
//         (self.f)(output).await
//     }
// }

pub struct Map<Ch, F> {
    chain: Ch,
    f: F,
}

impl<Ch, F> Map<Ch, F> {
    fn new(chain: Ch, f: F) -> Self {
        Self { chain, f }
    }
}

impl<Ch, F, T> Chain for Map<Ch, F>
where
    Ch: Chain,
    F: Fn(Ch::Output) -> T + Send + Sync,
    T: Send,
{
    type Input = Ch::Input;
    type Output = T;

    async fn call(self, input: Self::Input) -> Self::Output {
        let output = self.chain.call(input).await;
        (self.f)(output)
    }
}

pub struct Lookup<Ch, I: vector_store::VectorStoreIndex, T> {
    chain: Ch,
    index: I,
    n: usize,
    _t: PhantomData<T>,
}

impl<Ch, I: vector_store::VectorStoreIndex, T> Lookup<Ch, I, T>
where
    Ch: Chain,
{
    pub fn new(chain: Ch, index: I, n: usize) -> Self {
        Self {
            chain,
            index,
            n,
            _t: PhantomData,
        }
    }
}

impl<Ch, I, T> Chain for Lookup<Ch, I, T>
where
    I: vector_store::VectorStoreIndex,
    Ch: Chain,
    Ch::Output: Into<String>,
    T: Send + Sync + for<'a> serde::Deserialize<'a>,
{
    type Input = Ch::Input;
    type Output = (String, Vec<T>);

    async fn call(self, input: Self::Input) -> Self::Output {
        let query = self.chain.call(input).await.into();

        let docs = self
            .index
            .top_n::<T>(&query, self.n)
            .await
            .expect("Failed to get top n documents")
            .into_iter()
            .map(|(_, _, doc)| doc)
            .collect();

        (query, docs)
    }
}

pub struct Prompt<Ch, P: completion::Prompt> {
    chain: Ch,
    prompt: P,
}

impl<Ch, P: completion::Prompt> Prompt<Ch, P> {
    pub fn new(chain: Ch, prompt: P) -> Self {
        Self { chain, prompt }
    }
}

impl<Ch, P> Chain for Prompt<Ch, P>
where
    Ch: Chain,
    Ch::Output: Into<String>,
    P: completion::Prompt,
{
    type Input = Ch::Input;
    type Output = String;

    async fn call(self, input: Self::Input) -> Self::Output {
        let output = self.chain.call(input).await.into();

        self.prompt
            .prompt(&output)
            .await
            .expect("Failed to prompt agent")
    }
}

pub fn new<T>() -> Empty<T> {
    Empty(PhantomData)
}
