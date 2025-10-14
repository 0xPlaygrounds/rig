use crate::wasm_compat::*;
#[allow(unused_imports)] // Needed since this is used in a macro rule
use futures::join;
use futures::stream;
use std::future::Future;

// ================================================================
// Core Op trait
// ================================================================
pub trait Op: WasmCompatSend + WasmCompatSync {
    type Input: WasmCompatSend + WasmCompatSync;
    type Output: WasmCompatSend + WasmCompatSync;

    fn call(&self, input: Self::Input) -> impl Future<Output = Self::Output> + WasmCompatSend;

    /// Execute the current pipeline with the given inputs. `n` is the number of concurrent
    /// inputs that will be processed concurrently.
    fn batch_call<I>(
        &self,
        n: usize,
        input: I,
    ) -> impl Future<Output = Vec<Self::Output>> + WasmCompatSend
    where
        I: IntoIterator<Item = Self::Input> + WasmCompatSend,
        I::IntoIter: WasmCompatSend,
        Self: Sized,
    {
        use futures::stream::StreamExt;

        async move {
            stream::iter(input)
                .map(|input| self.call(input))
                .buffered(n)
                .collect()
                .await
        }
    }

    /// Chain a function `f` to the current op.
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
    fn map<F, Input>(self, f: F) -> Sequential<Self, Map<F, Self::Output>>
    where
        F: Fn(Self::Output) -> Input + WasmCompatSend + WasmCompatSync,
        Input: WasmCompatSend + WasmCompatSync,
        Self: Sized,
    {
        Sequential::new(self, Map::new(f))
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
    fn then<F, Fut>(self, f: F) -> Sequential<Self, Then<F, Fut::Output>>
    where
        F: Fn(Self::Output) -> Fut + Send + WasmCompatSync,
        Fut: Future + WasmCompatSend + WasmCompatSync,
        Fut::Output: WasmCompatSend + WasmCompatSync,
        Self: Sized,
    {
        Sequential::new(self, Then::new(f))
    }

    /// Chain an arbitrary operation to the current op.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, Op};
    ///
    /// struct AddOne;
    ///
    /// impl Op for AddOne {
    ///     type Input = i32;
    ///     type Output = i32;
    ///
    ///     async fn call(&self, input: Self::Input) -> Self::Output {
    ///         input + 1
    ///     }
    /// }
    ///
    /// let chain = pipeline::new()
    ///    .chain(AddOne);
    ///
    /// let result = chain.call(1).await;
    /// assert_eq!(result, 2);
    /// ```
    fn chain<T>(self, op: T) -> Sequential<Self, T>
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
    fn lookup<I, Input>(
        self,
        index: I,
        n: usize,
    ) -> Sequential<Self, Lookup<I, Self::Output, Input>>
    where
        I: vector_store::VectorStoreIndex,
        Input: WasmCompatSend + WasmCompatSync + for<'a> serde::Deserialize<'a>,
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
    fn prompt<P>(self, prompt: P) -> Sequential<Self, Prompt<P, Self::Output>>
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
    pub(crate) fn new(prev: Op1, op: Op2) -> Self {
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

use super::agent_ops::{Lookup, Prompt};
use crate::{completion, vector_store};

// ================================================================
// Core Op implementations
// ================================================================
pub struct Map<F, Input> {
    f: F,
    _t: std::marker::PhantomData<Input>,
}

impl<F, Input> Map<F, Input> {
    pub(crate) fn new(f: F) -> Self {
        Self {
            f,
            _t: std::marker::PhantomData,
        }
    }
}

impl<F, Input, Output> Op for Map<F, Input>
where
    F: Fn(Input) -> Output + WasmCompatSend + WasmCompatSync,
    Input: WasmCompatSend + WasmCompatSync,
    Output: WasmCompatSend + WasmCompatSync,
{
    type Input = Input;
    type Output = Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (self.f)(input)
    }
}

pub fn map<F, Input, Output>(f: F) -> Map<F, Input>
where
    F: Fn(Input) -> Output + WasmCompatSend + WasmCompatSync,
    Input: WasmCompatSend + WasmCompatSync,
    Output: WasmCompatSend + WasmCompatSync,
{
    Map::new(f)
}

pub struct Passthrough<T> {
    _t: std::marker::PhantomData<T>,
}

impl<T> Passthrough<T> {
    pub(crate) fn new() -> Self {
        Self {
            _t: std::marker::PhantomData,
        }
    }
}

impl<T> Op for Passthrough<T>
where
    T: WasmCompatSend + WasmCompatSync,
{
    type Input = T;
    type Output = T;

    async fn call(&self, input: Self::Input) -> Self::Output {
        input
    }
}

pub fn passthrough<T>() -> Passthrough<T>
where
    T: WasmCompatSend + WasmCompatSync,
{
    Passthrough::new()
}

pub struct Then<F, Input> {
    f: F,
    _t: std::marker::PhantomData<Input>,
}

impl<F, Input> Then<F, Input> {
    pub(crate) fn new(f: F) -> Self {
        Self {
            f,
            _t: std::marker::PhantomData,
        }
    }
}

impl<F, Input, Fut> Op for Then<F, Input>
where
    F: Fn(Input) -> Fut + WasmCompatSend + WasmCompatSync,
    Input: WasmCompatSend + WasmCompatSync,
    Fut: Future + WasmCompatSend,
    Fut::Output: WasmCompatSend + WasmCompatSync,
{
    type Input = Input;
    type Output = Fut::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (self.f)(input).await
    }
}

pub fn then<F, Input, Fut>(f: F) -> Then<F, Input>
where
    F: Fn(Input) -> Fut + WasmCompatSend + WasmCompatSync,
    Input: WasmCompatSend + WasmCompatSync,
    Fut: Future + WasmCompatSend,
    Fut::Output: WasmCompatSend + WasmCompatSync,
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
