use std::future::Future;

#[allow(unused_imports)] // Needed since this is used in a macro rule
use futures::try_join;

use super::op::{self, map, then};

// ================================================================
// Core TryOp trait
// ================================================================
pub trait TryOp: Send + Sync {
    type Input: Send + Sync;
    type Output: Send + Sync;
    type Error: Send + Sync;

    fn try_call(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;

    fn map_ok<F, T>(self, f: F) -> impl op::Op<Input = Self::Input, Output = Result<T, Self::Error>>
    where
        F: Fn(Self::Output) -> T + Send + Sync,
        T: Send + Sync,
        Self: Sized,
    {
        MapOk::new(self, map(f))
    }

    fn map_err<F, E>(
        self,
        f: F,
    ) -> impl TryOp<Input = Self::Input, Output = Self::Output, Error = E>
    where
        F: Fn(Self::Error) -> E + Send + Sync,
        E: Send + Sync,
        Self: Sized,
    {
        MapErr::new(self, map(f))
    }

    fn and_then<F, Fut, T>(
        self,
        f: F,
    ) -> impl TryOp<Input = Self::Input, Output = T, Error = Self::Error>
    where
        F: Fn(Self::Output) -> Fut + Send + Sync,
        Fut: Future<Output = Result<T, Self::Error>> + Send + Sync,
        T: Send + Sync,
        Self: Sized,
    {
        AndThen::new(self, then(f))
    }

    fn or_else<F, Fut, E>(
        self,
        f: F,
    ) -> impl TryOp<Input = Self::Input, Output = Self::Output, Error = E>
    where
        F: Fn(Self::Error) -> Fut + Send + Sync,
        Fut: Future<Output = Result<Self::Output, E>> + Send + Sync,
        E: Send + Sync,
        Self: Sized,
    {
        OrElse::new(self, then(f))
    }

    fn chain_ok<T>(self, op: T) -> impl TryOp<Input = Self::Input, Output = T::Output, Error = Self::Error>
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
    T: Send + Sync,
    E: Send + Sync,
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
    pub fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

// Result<T1, E> -> Result<T2, E>
// impl<Op1, Op2> TryOp for MapOk<Op1, Op2>
// where
//     Op1: TryOp,
//     Op2: super::Op<Input = Op1::Output>,
// {
//     type Input = Op1::Input;
//     type Output = Op2::Output;
//     type Error = Op1::Error;

//     #[inline]
//     async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
//         match self.prev.try_call(input).await {
//             Ok(output) => Ok(self.op.call(output).await),
//             Err(err) => Err(err),
//         }
//     }
// }

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
    pub fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

// Result<T, E1> -> Result<T, E2>
impl<Op1, Op2> TryOp for MapErr<Op1, Op2>
where
    Op1: TryOp,
    Op2: super::Op<Input = Op1::Error>,
{
    type Input = Op1::Input;
    type Output = Op1::Output;
    type Error = Op2::Output;

    #[inline]
    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
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
    pub fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

impl<Op1, Op2> TryOp for AndThen<Op1, Op2>
where
    Op1: TryOp,
    Op2: TryOp<Input = Op1::Output, Error = Op1::Error>,
{
    type Input = Op1::Input;
    type Output = Op2::Output;
    type Error = Op1::Error;

    #[inline]
    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        let output = self.prev.try_call(input).await?;
        self.op.try_call(output).await
    }
}

pub struct OrElse<Op1, Op2> {
    prev: Op1,
    op: Op2,
}

impl<Op1, Op2> OrElse<Op1, Op2> {
    pub fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

impl<Op1, Op2> TryOp for OrElse<Op1, Op2>
where
    Op1: TryOp,
    Op2: TryOp<Input = Op1::Error, Output = Op1::Output>,
{
    type Input = Op1::Input;
    type Output = Op1::Output;
    type Error = Op2::Error;

    #[inline]
    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
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
    pub fn new(prev: Op1, op: Op2) -> Self {
        Self { prev, op }
    }
}

impl<Op1, Op2> TryOp for TrySequential<Op1, Op2>
where
    Op1: TryOp,
    Op2: op::Op<Input = Op1::Output>,
{
    type Input = Op1::Input;
    type Output = Op2::Output;
    type Error = Op1::Error;

    #[inline]
    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
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
        let result = op.try_call(1).await.unwrap();
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
        let op2 = then(|err: &str| async move { format!("Error: {}", err) });
        let op3 = map(|err: String| err.len());

        let pipeline = MapErr::new(MapErr::new(op1, op2), op3);

        let result = pipeline.try_call(1).await;
        assert_eq!(result, Err(15));
    }

    #[tokio::test]
    async fn test_map_err_chain() {
        let pipeline = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
            .map_err(|err| format!("Error: {}", err))
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
        let op2 = then(|err: &str| async move { Err(format!("Error: {}", err)) });
        let op3 = map(|err: String| Ok::<i32, String>(err.len() as i32));

        let pipeline = OrElse::new(OrElse::new(op1, op2), op3);

        let result = pipeline.try_call(1).await.unwrap();
        assert_eq!(result, 15);
    }

    #[tokio::test]
    async fn test_or_else_chain() {
        let pipeline = map(|x: i32| if x % 2 == 0 { Ok(x) } else { Err("x is odd") })
            .or_else(|err| async move { Err(format!("Error: {}", err)) })
            .or_else(|err| async move { Ok::<i32, String>(err.len() as i32) });

        let result = pipeline.try_call(1).await.unwrap();
        assert_eq!(result, 15);
    }
}
