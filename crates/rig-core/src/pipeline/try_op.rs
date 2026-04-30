use std::future::Future;

use futures::stream;
#[allow(unused_imports)] // Needed since this is used in a macro rule
use futures::try_join;

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

use super::op::{self};

// ================================================================
// Core TryOp trait
// ================================================================
pub trait TryOp: WasmCompatSend + WasmCompatSync {
    type Input: WasmCompatSend + WasmCompatSync;
    type Output: WasmCompatSend + WasmCompatSync;
    type Error: WasmCompatSend + WasmCompatSync;

    /// Execute the current op with the given input.
    fn try_call(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend;

    /// Execute the current op with the given inputs. `n` is the number of concurrent
    /// inputs that will be processed concurrently.
    /// If the op fails for one of the inputs, the entire operation will fail and the error will
    /// be returned.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, TryOp};
    ///
    /// let op = pipeline::new()
    ///    .map(|x: i32| if x % 2 == 0 { Ok(x + 1) } else { Err("x is odd") });
    ///
    /// // Execute the pipeline concurrently with 2 inputs
    /// let result = op.try_batch_call(2, vec![2, 4]).await;
    /// assert_eq!(result, Ok(vec![3, 5]));
    /// ```
    fn try_batch_call<I>(
        &self,
        n: usize,
        input: I,
    ) -> impl Future<Output = Result<Vec<Self::Output>, Self::Error>> + WasmCompatSend
    where
        I: IntoIterator<Item = Self::Input> + WasmCompatSend,
        I::IntoIter: WasmCompatSend,
        Self: Sized,
    {
        use stream::{StreamExt, TryStreamExt};

        async move {
            stream::iter(input)
                .map(|input| self.try_call(input))
                .buffered(n)
                .try_collect()
                .await
        }
    }

    /// Map the success return value (i.e., `Ok`) of the current op to a different value
    /// using the provided closure.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, TryOp};
    ///
    /// let op = pipeline::new()
    ///     .map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
    ///     .map_ok(|x| x * 2);
    ///
    /// let result = op.try_call(2).await;
    /// assert_eq!(result, Ok(4));
    /// ```
    fn map_ok<F, Output>(self, f: F) -> MapOk<Self, op::Map<F, Self::Output>>
    where
        F: Fn(Self::Output) -> Output + WasmCompatSend + WasmCompatSync,
        Output: WasmCompatSend + WasmCompatSync,
        Self: Sized,
    {
        MapOk::new(self, op::Map::new(f))
    }

    /// Map the error return value (i.e., `Err`) of the current op to a different value
    /// using the provided closure.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, TryOp};
    ///
    /// let op = pipeline::new()
    ///     .map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
    ///     .map_err(|err| format!("Error: {}", err));
    ///
    /// let result = op.try_call(1).await;
    /// assert_eq!(result, Err("Error: x is odd".to_string()));
    /// ```
    fn map_err<F, E>(self, f: F) -> MapErr<Self, op::Map<F, Self::Error>>
    where
        F: Fn(Self::Error) -> E + WasmCompatSend + WasmCompatSync,
        E: WasmCompatSend + WasmCompatSync,
        Self: Sized,
    {
        MapErr::new(self, op::Map::new(f))
    }

    /// Chain a function to the current op. The function will only be called
    /// if the current op returns `Ok`. The function must return a `Future` with value
    /// `Result<T, E>` where `E` is the same type as the error type of the current.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, TryOp};
    ///
    /// let op = pipeline::new()
    ///     .map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
    ///     .and_then(|x| async move { Ok(x * 2) });
    ///
    /// let result = op.try_call(2).await;
    /// assert_eq!(result, Ok(4));
    /// ```
    fn and_then<F, Fut, Output>(self, f: F) -> AndThen<Self, op::Then<F, Self::Output>>
    where
        F: Fn(Self::Output) -> Fut + WasmCompatSend + WasmCompatSync,
        Fut: Future<Output = Result<Output, Self::Error>> + WasmCompatSend + WasmCompatSync,
        Output: WasmCompatSend + WasmCompatSync,
        Self: Sized,
    {
        AndThen::new(self, op::Then::new(f))
    }

    /// Chain a function `f` to the current op. The function `f` will only be called
    /// if the current op returns `Err`. `f` must return a `Future` with value
    /// `Result<T, E>` where `T` is the same type as the output type of the current op.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, TryOp};
    ///
    /// let op = pipeline::new()
    ///     .map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
    ///     .or_else(|err| async move { Err(format!("Error: {}", err)) });
    ///
    /// let result = op.try_call(1).await;
    /// assert_eq!(result, Err("Error: x is odd".to_string()));
    /// ```
    fn or_else<F, Fut, E>(self, f: F) -> OrElse<Self, op::Then<F, Self::Error>>
    where
        F: Fn(Self::Error) -> Fut + WasmCompatSend + WasmCompatSync,
        Fut: Future<Output = Result<Self::Output, E>> + WasmCompatSend + WasmCompatSync,
        E: WasmCompatSend + WasmCompatSync,
        Self: Sized,
    {
        OrElse::new(self, op::Then::new(f))
    }

    /// Chain a new op `op` to the current op. The new op will be called with the success
    /// return value of the current op (i.e.: `Ok` value). The chained op can be any type that
    /// implements the `Op` trait.
    ///
    /// # Example
    /// ```rust
    /// use rig::pipeline::{self, TryOp};
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
    /// let op = pipeline::new()
    ///     .map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
    ///     .chain_ok(MyOp);
    ///
    /// let result = op.try_call(2).await;
    /// assert_eq!(result, Ok(3));
    /// ```
    fn chain_ok<T>(self, op: T) -> TrySequential<Self, T>
    where
        T: op::Op<Input = Self::Output>,
        Self: Sized,
    {
        TrySequential::new(self, op)
    }
}

impl<Op, T, E> TryOp for Op
where
    Op: super::Op<Output = Result<T, E>>,
    T: WasmCompatSend + WasmCompatSync,
    E: WasmCompatSend + WasmCompatSync,
{
    type Input = Op::Input;
    type Output = T;
    type Error = E;

    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        self.call(input).await
    }
}

// ================================================================
// TryOp combinators
// ================================================================
pub struct MapOk<Op1, Op2> {
    prev: Op1,
    op: Op2,
}

impl<Op1, Op2> MapOk<Op1, Op2> {
    pub(crate) fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

impl<Op1, Op2> op::Op for MapOk<Op1, Op2>
where
    Op1: TryOp,
    Op2: super::Op<Input = Op1::Output>,
{
    type Input = Op1::Input;
    type Output = Result<Op2::Output, Op1::Error>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        match self.prev.try_call(input).await {
            Ok(output) => Ok(self.op.call(output).await),
            Err(err) => Err(err),
        }
    }
}

pub struct MapErr<Op1, Op2> {
    prev: Op1,
    op: Op2,
}

impl<Op1, Op2> MapErr<Op1, Op2> {
    pub(crate) fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

// Result<T, E1> -> Result<T, E2>
impl<Op1, Op2> op::Op for MapErr<Op1, Op2>
where
    Op1: TryOp,
    Op2: super::Op<Input = Op1::Error>,
{
    type Input = Op1::Input;
    type Output = Result<Op1::Output, Op2::Output>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        match self.prev.try_call(input).await {
            Ok(output) => Ok(output),
            Err(err) => Err(self.op.call(err).await),
        }
    }
}

pub struct AndThen<Op1, Op2> {
    prev: Op1,
    op: Op2,
}

impl<Op1, Op2> AndThen<Op1, Op2> {
    pub(crate) fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

impl<Op1, Op2> op::Op for AndThen<Op1, Op2>
where
    Op1: TryOp,
    Op2: TryOp<Input = Op1::Output, Error = Op1::Error>,
{
    type Input = Op1::Input;
    type Output = Result<Op2::Output, Op1::Error>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        let output = self.prev.try_call(input).await?;
        self.op.try_call(output).await
    }
}

pub struct OrElse<Op1, Op2> {
    prev: Op1,
    op: Op2,
}

impl<Op1, Op2> OrElse<Op1, Op2> {
    pub(crate) fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

impl<Op1, Op2> op::Op for OrElse<Op1, Op2>
where
    Op1: TryOp,
    Op2: TryOp<Input = Op1::Error, Output = Op1::Output>,
{
    type Input = Op1::Input;
    type Output = Result<Op1::Output, Op2::Error>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        match self.prev.try_call(input).await {
            Ok(output) => Ok(output),
            Err(err) => self.op.try_call(err).await,
        }
    }
}

pub struct TrySequential<Op1, Op2> {
    prev: Op1,
    op: Op2,
}

impl<Op1, Op2> TrySequential<Op1, Op2> {
    pub(crate) fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

impl<Op1, Op2> op::Op for TrySequential<Op1, Op2>
where
    Op1: TryOp,
    Op2: op::Op<Input = Op1::Output>,
{
    type Input = Op1::Input;
    type Output = Result<Op2::Output, Op1::Error>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        match self.prev.try_call(input).await {
            Ok(output) => Ok(self.op.call(output).await),
            Err(err) => Err(err),
        }
    }
}

// TODO: Implement TryParallel
// pub struct TryParallel<Op1, Op2> {
//     op1: Op1,
//     op2: Op2,
// }

// impl<Op1, Op2> TryParallel<Op1, Op2> {
//     pub fn new(op1: Op1, op2: Op2) -> Self {
//         Self { op1, op2 }
//     }
// }

// impl<Op1, Op2> TryOp for TryParallel<Op1, Op2>
// where
//     Op1: TryOp,
//     Op2: TryOp<Input = Op1::Input, Output = Op1::Output, Error = Op1::Error>,
// {
//     type Input = Op1::Input;
//     type Output = (Op1::Output, Op2::Output);
//     type Error = Op1::Error;

//     #[inline]
//     async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
//         let (output1, output2) = tokio::join!(self.op1.try_call(input.clone()), self.op2.try_call(input));
//         Ok((output1?, output2?))
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::op::{map, then};

    #[tokio::test]
    async fn test_try_op() {
        let op = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") });
        let result = op.try_call(2).await.unwrap();
        assert_eq!(result, 2);
    }

    #[tokio::test]
    async fn test_map_ok_constructor() {
        let op1 = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") });
        let op2 = then(|x: i32| async move { x * 2 });
        let op3 = map(|x: i32| x - 1);

        let pipeline = MapOk::new(MapOk::new(op1, op2), op3);

        let result = pipeline.try_call(2).await.unwrap();
        assert_eq!(result, 3);
    }

    #[tokio::test]
    async fn test_map_ok_chain() {
        let pipeline = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
            .map_ok(|x| x * 2)
            .map_ok(|x| x - 1);

        let result = pipeline.try_call(2).await.unwrap();
        assert_eq!(result, 3);
    }

    #[tokio::test]
    async fn test_map_err_constructor() {
        let op1 = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") });
        let op2 = then(|err: &str| async move { format!("Error: {err}") });
        let op3 = map(|err: String| err.len());

        let pipeline = MapErr::new(MapErr::new(op1, op2), op3);

        let result = pipeline.try_call(1).await;
        assert_eq!(result, Err(15));
    }

    #[tokio::test]
    async fn test_map_err_chain() {
        let pipeline = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
            .map_err(|err| format!("Error: {err}"))
            .map_err(|err| err.len());

        let result = pipeline.try_call(1).await;
        assert_eq!(result, Err(15));
    }

    #[tokio::test]
    async fn test_and_then_constructor() {
        let op1 = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") });
        let op2 = then(|x: i32| async move { Ok(x * 2) });
        let op3 = map(|x: i32| Ok(x - 1));

        let pipeline = AndThen::new(AndThen::new(op1, op2), op3);

        let result = pipeline.try_call(2).await.unwrap();
        assert_eq!(result, 3);
    }

    #[tokio::test]
    async fn test_and_then_chain() {
        let pipeline = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
            .and_then(|x| async move { Ok(x * 2) })
            .and_then(|x| async move { Ok(x - 1) });

        let result = pipeline.try_call(2).await.unwrap();
        assert_eq!(result, 3);
    }

    #[tokio::test]
    async fn test_or_else_constructor() {
        let op1 = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") });
        let op2 = then(|err: &str| async move { Err(format!("Error: {err}")) });
        let op3 = map(|err: String| Ok::<i32, String>(err.len() as i32));

        let pipeline = OrElse::new(OrElse::new(op1, op2), op3);

        let result = pipeline.try_call(1).await.unwrap();
        assert_eq!(result, 15);
    }

    #[tokio::test]
    async fn test_or_else_chain() {
        let pipeline = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
            .or_else(|err| async move { Err(format!("Error: {err}")) })
            .or_else(|err| async move { Ok::<i32, String>(err.len() as i32) });

        let result = pipeline.try_call(1).await.unwrap();
        assert_eq!(result, 15);
    }
}
