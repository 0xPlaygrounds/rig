use std::future::Future;

#[allow(unused_imports)] // Needed since this is used in a macro rule
use futures::join;

// ================================================================
// Core Op trait
// ================================================================
pub trait Op: Send + Sync {
    type Input: Send + Sync;
    type Output: Send + Sync;

    fn call(&self, input: Self::Input) -> impl Future<Output = Self::Output> + Send;

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
    fn map<F, T>(self, f: F) -> impl Op<Input = Self::Input, Output = T>
    where
        F: Fn(Self::Output) -> T + Send + Sync,
        T: Send + Sync,
        Self: Sized,
    {
        Sequential::new(self, map(f))
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
    fn then<F, Fut>(self, f: F) -> impl Op<Input = Self::Input, Output = Fut::Output>
    where
        F: Fn(Self::Output) -> Fut + Send + Sync,
        Fut: Future + Send + Sync,
        Fut::Output: Send + Sync,
        Self: Sized,
    {
        Sequential::new(self, then(f))
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
    fn chain<T>(self, op: T) -> impl Op<Input = Self::Input, Output = T::Output>
    where
        T: Op<Input = Self::Output>,
        Self: Sized,
    {
        Sequential::new(self, op)
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
    fn lookup<I, T>(
        self,
        index: I,
        n: usize,
    ) -> impl Op<Input = Self::Input, Output = Result<Vec<T>, vector_store::VectorStoreError>>
    where
        I: vector_store::VectorStoreIndex,
        T: Send + Sync + for<'a> serde::Deserialize<'a>,
        Self::Output: Into<String>,
        Self: Sized,
    {
        Sequential::new(self, Lookup::new(index, n))
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
    fn prompt<P>(
        self,
        prompt: P,
    ) -> impl Op<Input = Self::Input, Output = Result<String, completion::PromptError>>
    where
        P: completion::Prompt,
        Self::Output: Into<String>,
        Self: Sized,
    {
        Sequential::new(self, Prompt::new(prompt))
    }
}

impl<T: Op> Op for &T {
    type Input = T::Input;
    type Output = T::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (*self).call(input).await
    }
}

// ================================================================
// Op combinators
// ================================================================
pub struct Sequential<Op1, Op2> {
    prev: Op1,
    op: Op2,
}

impl<Op1, Op2> Sequential<Op1, Op2> {
    pub fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

impl<Op1, Op2> Op for Sequential<Op1, Op2>
where
    Op1: Op,
    Op2: Op<Input = Op1::Output>,
{
    type Input = Op1::Input;
    type Output = Op2::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        let prev = self.prev.call(input).await;
        self.op.call(prev).await
    }
}

use crate::{completion, vector_store};

use super::agent_ops::{Lookup, Prompt};

// ================================================================
// Core Op implementations
// ================================================================
pub struct Map<F, T> {
    f: F,
    _t: std::marker::PhantomData<T>,
}

impl<F, T> Map<F, T> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            _t: std::marker::PhantomData,
        }
    }
}

impl<F, T, Out> Op for Map<F, T>
where
    F: Fn(T) -> Out + Send + Sync,
    T: Send + Sync,
    Out: Send + Sync,
{
    type Input = T;
    type Output = Out;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (self.f)(input)
    }
}

pub fn map<F, T, Out>(f: F) -> impl Op<Input = T, Output = Out>
where
    F: Fn(T) -> Out + Send + Sync,
    T: Send + Sync,
    Out: Send + Sync,
{
    Map::new(f)
}

pub fn passthrough<T>() -> impl Op<Input = T, Output = T>
where
    T: Send + Sync,
{
    Map::new(|x| x)
}

pub struct Then<F, T> {
    f: F,
    _t: std::marker::PhantomData<T>,
}

impl<F, T> Then<F, T> {
    fn new(f: F) -> Self {
        Self {
            f,
            _t: std::marker::PhantomData,
        }
    }
}

impl<F, T, Fut> Op for Then<F, T>
where
    F: Fn(T) -> Fut + Send + Sync,
    T: Send + Sync,
    Fut: Future + Send,
    Fut::Output: Send + Sync,
{
    type Input = T;
    type Output = Fut::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (self.f)(input).await
    }
}

pub fn then<F, T, Fut>(f: F) -> impl Op<Input = T, Output = Fut::Output>
where
    F: Fn(T) -> Fut + Send + Sync,
    T: Send + Sync,
    Fut: Future + Send,
    Fut::Output: Send + Sync,
{
    Then::new(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sequential_constructor() {
        let op1 = map(|x: i32| x + 1);
        let op2 = map(|x: i32| x * 2);
        let op3 = map(|x: i32| x * 3);

        let pipeline = Sequential::new(Sequential::new(op1, op2), op3);

        let result = pipeline.call(1).await;
        assert_eq!(result, 12);
    }

    #[tokio::test]
    async fn test_sequential_chain() {
        let pipeline = map(|x: i32| x + 1)
            .map(|x| x * 2)
            .then(|x| async move { x * 3 });

        let result = pipeline.call(1).await;
        assert_eq!(result, 12);
    }

    // #[tokio::test]
    // async fn test_flatten() {
    //     let op = Parallel::new(
    //         Parallel::new(
    //             map(|x: i32| x + 1),
    //             map(|x: i32| x * 2),
    //         ),
    //         map(|x: i32| x * 3),
    //     );

    //     let pipeline = flatten::<_, (_, _, _)>(op);

    //     let result = pipeline.call(1).await;
    //     assert_eq!(result, (2, 2, 3));
    // }

    // #[tokio::test]
    // async fn test_parallel_macro() {
    //     let op1 = map(|x: i32| x + 1);
    //     let op2 = map(|x: i32| x * 3);
    //     let op3 = map(|x: i32| format!("{} is the number!", x));
    //     let op4 = map(|x: i32| x - 1);

    //     let pipeline = parallel!(op1, op2, op3, op4);

    //     let result = pipeline.call(1).await;
    //     assert_eq!(result, (2, 3, "1 is the number!".to_string(), 0));
    // }

    // #[tokio::test]
    // async fn test_parallel_join() {
    //     let op3 = map(|x: i32| format!("{} is the number!", x));

    //     let pipeline = Sequential::new(
    //         map(|x: i32| x + 1),
    //         then(|x| {
    //             // let op1 = map(|x: i32| x * 2);
    //             // let op2 = map(|x: i32| x * 3);
    //             let op3 = &op3;

    //             async move {
    //             join!(
    //                 (&map(|x: i32| x * 2)).call(x),
    //                 {
    //                     let op = map(|x: i32| x * 3);
    //                     op.call(x)
    //                 },
    //                 op3.call(x),
    //             )
    //         }}),
    //     );

    //     let result = pipeline.call(1).await;
    //     assert_eq!(result, (2, 3, "1 is the number!".to_string()));
    // }

    // #[test]
    // fn test_flatten() {
    //     let x = (1, (2, (3, 4)));
    //     let result = flatten!(0, 1, 1, 1, 1);
    //     assert_eq!(result, (1, 2, 3, 4));
    // }
}
