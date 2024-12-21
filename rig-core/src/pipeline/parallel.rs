use futures::{join, try_join};

use super::{Op, TryOp};

pub struct Parallel<Op1, Op2> {
    op1: Op1,
    op2: Op2,
}

impl<Op1, Op2> Parallel<Op1, Op2> {
    pub fn new(op1: Op1, op2: Op2) -> Self {
        Self { op1, op2 }
    }
}

impl<Op1, Op2> Op for Parallel<Op1, Op2>
where
    Op1: Op,
    Op1::Input: Clone,
    Op2: Op<Input = Op1::Input>,
{
    type Input = Op1::Input;
    type Output = (Op1::Output, Op2::Output);

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        join!(self.op1.call(input.clone()), self.op2.call(input))
    }
}

impl<Op1, Op2> TryOp for Parallel<Op1, Op2>
where
    Op1: TryOp,
    Op1::Input: Clone,
    Op2: TryOp<Input = Op1::Input, Error = Op1::Error>,
{
    type Input = Op1::Input;
    type Output = (Op1::Output, Op2::Output);
    type Error = Op1::Error;

    #[inline]
    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        try_join!(self.op1.try_call(input.clone()), self.op2.try_call(input))
    }
}

// See https://doc.rust-lang.org/src/core/future/join.rs.html#48
#[macro_export]
macro_rules! parallel_internal {
    // Last recursive step
    (
        // Accumulate a token for each future that has been expanded: "_ _ _".
        current_position: [
            $($underscores:tt)*
        ]
        // Accumulate values and their positions in the tuple: `_0th ()   _1st ( _ ) …`.
        values_and_positions: [
            $($acc:tt)*
        ]
        // Munch one value.
        munching: [
            $current:tt
        ]
    ) => (
        $crate::parallel_internal! {
            current_position: [
                $($underscores)*
                _
            ]
            values_and_positions: [
                $($acc)*
                $current ( $($underscores)* + )
            ]
            munching: []
        }
    );

    // Recursion step: map each value with its "position" (underscore count).
    (
        // Accumulate a token for each future that has been expanded: "_ _ _".
        current_position: [
            $($underscores:tt)*
        ]
        // Accumulate values and their positions in the tuple: `_0th ()   _1st ( _ ) …`.
        values_and_positions: [
            $($acc:tt)*
        ]
        // Munch one value.
        munching: [
            $current:tt
            $($rest:tt)+
        ]
    ) => (
        $crate::parallel_internal! {
            current_position: [
                $($underscores)*
                _
            ]
            values_and_positions: [
                $($acc)*
                $current ( $($underscores)* )
            ]
            munching: [
                $($rest)*
            ]
        }
    );

    // End of recursion: flatten the values.
    (
        current_position: [
            $($max:tt)*
        ]
        values_and_positions: [
            $(
                $val:tt ( $($pos:tt)* )
            )*
        ]
        munching: []
    ) => ({
        use $crate::pipeline::op::Op;

        $crate::parallel_op!($($val),*)
            .map(|output| {
                ($(
                    {
                        let $crate::tuple_pattern!(x $($pos)*) = output;
                        x
                    }
                ),+)
            })
    })
}

#[macro_export]
macro_rules! parallel_op {
    ($op1:tt, $op2:tt) => {
        $crate::pipeline::parallel::Parallel::new($op1, $op2)
    };
    ($op1:tt $(, $ops:tt)*) => {
        $crate::pipeline::parallel::Parallel::new(
            $op1,
            $crate::parallel_op!($($ops),*)
        )
    };
}

#[macro_export]
macro_rules! tuple_pattern {
    ($id:ident +) => {
        $id
    };
    ($id:ident) => {
        ($id, ..)
    };
    ($id:ident _ $($symbols:tt)*) => {
        (_, $crate::tuple_pattern!($id $($symbols)*))
    };
}

#[macro_export]
macro_rules! parallel {
    ($($es:expr),+ $(,)?) => {
        $crate::parallel_internal! {
            current_position: []
            values_and_positions: []
            munching: [
                $($es)+
            ]
        }
    };
}

// See https://doc.rust-lang.org/src/core/future/join.rs.html#48
#[macro_export]
macro_rules! try_parallel_internal {
    // Last recursive step
    (
        // Accumulate a token for each future that has been expanded: "_ _ _".
        current_position: [
            $($underscores:tt)*
        ]
        // Accumulate values and their positions in the tuple: `_0th ()   _1st ( _ ) …`.
        values_and_positions: [
            $($acc:tt)*
        ]
        // Munch one value.
        munching: [
            $current:tt
        ]
    ) => (
        $crate::try_parallel_internal! {
            current_position: [
                $($underscores)*
                _
            ]
            values_and_positions: [
                $($acc)*
                $current ( $($underscores)* + )
            ]
            munching: []
        }
    );

    // Recursion step: map each value with its "position" (underscore count).
    (
        // Accumulate a token for each future that has been expanded: "_ _ _".
        current_position: [
            $($underscores:tt)*
        ]
        // Accumulate values and their positions in the tuple: `_0th ()   _1st ( _ ) …`.
        values_and_positions: [
            $($acc:tt)*
        ]
        // Munch one value.
        munching: [
            $current:tt
            $($rest:tt)+
        ]
    ) => (
        $crate::try_parallel_internal! {
            current_position: [
                $($underscores)*
                _
            ]
            values_and_positions: [
                $($acc)*
                $current ( $($underscores)* )
            ]
            munching: [
                $($rest)*
            ]
        }
    );

    // End of recursion: flatten the values.
    (
        current_position: [
            $($max:tt)*
        ]
        values_and_positions: [
            $(
                $val:tt ( $($pos:tt)* )
            )*
        ]
        munching: []
    ) => ({
        use $crate::pipeline::try_op::TryOp;
        $crate::parallel_op!($($val),*)
            .map_ok(|output| {
                ($(
                    {
                        let $crate::tuple_pattern!(x $($pos)*) = output;
                        x
                    }
                ),+)
            })
    })
}

#[macro_export]
macro_rules! try_parallel {
    ($($es:expr),+ $(,)?) => {
        $crate::try_parallel_internal! {
            current_position: []
            values_and_positions: []
            munching: [
                $($es)+
            ]
        }
    };
}

pub use parallel;
pub use parallel_internal;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{
        self,
        op::{map, Sequential},
        passthrough, then,
    };

    #[tokio::test]
    async fn test_parallel() {
        let op1 = map(|x: i32| x + 1);
        let op2 = map(|x: i32| x * 3);
        let pipeline = Parallel::new(op1, op2);

        let result = pipeline.call(1).await;
        assert_eq!(result, (2, 3));
    }

    #[tokio::test]
    async fn test_parallel_nested() {
        let op1 = map(|x: i32| x + 1);
        let op2 = map(|x: i32| x * 3);
        let op3 = map(|x: i32| format!("{} is the number!", x));
        let op4 = map(|x: i32| x - 1);

        let pipeline = Parallel::new(Parallel::new(Parallel::new(op1, op2), op3), op4);

        let result = pipeline.call(1).await;
        assert_eq!(result, (((2, 3), "1 is the number!".to_string()), 0));
    }

    #[tokio::test]
    async fn test_parallel_nested_rev() {
        let op1 = map(|x: i32| x + 1);
        let op2 = map(|x: i32| x * 3);
        let op3 = map(|x: i32| format!("{} is the number!", x));
        let op4 = map(|x: i32| x == 1);

        let pipeline = Parallel::new(op1, Parallel::new(op2, Parallel::new(op3, op4)));

        let result = pipeline.call(1).await;
        assert_eq!(result, (2, (3, ("1 is the number!".to_string(), true))));
    }

    #[tokio::test]
    async fn test_sequential_and_parallel() {
        let op1 = map(|x: i32| x + 1);
        let op2 = map(|x: i32| x * 2);
        let op3 = map(|x: i32| x * 3);
        let op4 = map(|(x, y): (i32, i32)| x + y);

        let pipeline = Sequential::new(Sequential::new(op1, Parallel::new(op2, op3)), op4);

        let result = pipeline.call(1).await;
        assert_eq!(result, 10);
    }

    #[tokio::test]
    async fn test_parallel_chain_compile_check() {
        let _ = pipeline::new().chain(
            Parallel::new(
                map(|x: i32| x + 1),
                Parallel::new(
                    map(|x: i32| x * 3),
                    Parallel::new(
                        map(|x: i32| format!("{} is the number!", x)),
                        map(|x: i32| x == 1),
                    ),
                ),
            )
            .map(|(r1, (r2, (r3, r4)))| (r1, r2, r3, r4)),
        );
    }

    #[tokio::test]
    async fn test_parallel_pass_through() {
        let pipeline = then(|x| {
            let op = Parallel::new(Parallel::new(passthrough(), passthrough()), passthrough());

            async move {
                let ((r1, r2), r3) = op.call(x).await;
                (r1, r2, r3)
            }
        });

        let result = pipeline.call(1).await;
        assert_eq!(result, (1, 1, 1));
    }

    #[tokio::test]
    async fn test_parallel_macro() {
        let op2 = map(|x: i32| x * 2);

        let pipeline = parallel!(
            passthrough(),
            op2,
            map(|x: i32| format!("{} is the number!", x)),
            map(|x: i32| x == 1)
        );

        let result = pipeline.call(1).await;
        assert_eq!(result, (1, 2, "1 is the number!".to_string(), true));
    }

    #[tokio::test]
    async fn test_try_parallel_chain_compile_check() {
        let chain = pipeline::new().chain(
            Parallel::new(
                map(|x: i32| Ok::<_, String>(x + 1)),
                Parallel::new(
                    map(|x: i32| Ok::<_, String>(x * 3)),
                    Parallel::new(
                        map(|x: i32| Err::<i32, _>(format!("{} is the number!", x))),
                        map(|x: i32| Ok::<_, String>(x == 1)),
                    ),
                ),
            )
            .map_ok(|(r1, (r2, (r3, r4)))| (r1, r2, r3, r4)),
        );

        let response = chain.call(1).await;
        assert_eq!(response, Err("1 is the number!".to_string()));
    }

    #[tokio::test]
    async fn test_try_parallel_macro_ok() {
        let op2 = map(|x: i32| Ok::<_, String>(x * 2));

        let pipeline = try_parallel!(
            map(|x: i32| Ok::<_, String>(x)),
            op2,
            map(|x: i32| Ok::<_, String>(format!("{} is the number!", x))),
            map(|x: i32| Ok::<_, String>(x == 1))
        );

        let result = pipeline.try_call(1).await;
        assert_eq!(result, Ok((1, 2, "1 is the number!".to_string(), true)));
    }

    #[tokio::test]
    async fn test_try_parallel_macro_err() {
        let op2 = map(|x: i32| Ok::<_, String>(x * 2));

        let pipeline = try_parallel!(
            map(|x: i32| Ok::<_, String>(x)),
            op2,
            map(|x: i32| Err::<i32, _>(format!("{} is the number!", x))),
            map(|x: i32| Ok::<_, String>(x == 1))
        );

        let result = pipeline.try_call(1).await;
        assert_eq!(result, Err("1 is the number!".to_string()));
    }
}
