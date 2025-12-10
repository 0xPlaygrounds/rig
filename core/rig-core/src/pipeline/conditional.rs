/// Creates an `Op` that conditionally dispatches to one of multiple sub-ops
/// based on the variant of the input enum.
///
/// **Important Requirements**:
/// 1. The enum must be defined as a single-type-parameter wrapper, e.g.
///    ```rust
///    enum MyEnum<T> {
///        VariantA(T),
///        VariantB(T),
///    }
///    ```
///    This allows all variants to share the same inner type (`T`).
/// 2. All sub-ops must have the same `Input` type (this `T`) and the same `Output`.
///    That is, for each variant, the corresponding op must implement
///    `Op<Input = T, Output = Out>`.
///
/// # Example
/// ```rust
/// use rig::pipeline::*;
/// use rig::conditional;
/// use tokio;
///
/// #[tokio::main]
/// async fn main() {
///     #[derive(Debug)]
///     enum ExampleEnum<T> {
///         Variant1(T),
///         Variant2(T),
///     }
///
///     // Creates a pipeline Op that adds 1 if it’s Variant1, or doubles if it’s Variant2
///     let op1 = map(|x: i32| x + 1);
///     let op2 = map(|x: i32| x * 2);
///
///     let conditional = conditional!(ExampleEnum,
///         Variant1 => op1,
///         Variant2 => op2,
///     );
///
///     let result1 = conditional.call(ExampleEnum::Variant1(2)).await;
///     assert_eq!(result1, 3);
///
///     let result2 = conditional.call(ExampleEnum::Variant2(3)).await;
///     assert_eq!(result2, 6);
/// }
/// ```
#[macro_export]
macro_rules! conditional {
    ($enum:ident, $( $variant:ident => $op:expr ),+ $(,)?) => {
        {
            #[allow(non_snake_case)]
            struct ConditionalOp<$($variant),+> {
                $(
                    $variant: $variant,
                )+
            }

            impl<Value, Out, $($variant),+> Op for ConditionalOp<$($variant),+>
            where
                $($variant: Op<Input=Value, Output=Out>),+,
                Value: Send + Sync,
                Out: Send + Sync,
            {
                type Input = $enum<Value>;
                type Output = Out;

                fn call(&self, input: Self::Input) -> impl std::future::Future<Output=Self::Output> + Send {
                    async move {
                        match input {
                            $(
                                $enum::$variant(val) => self.$variant.call(val).await
                            ),+
                        }
                    }
                }
            }

            ConditionalOp { $($variant: $op),+ }
        }
    };
}

/// Creates a `TryOp` that conditionally dispatches to one of multiple sub-ops
/// based on the variant of the input enum, returning a `Result`.
///
/// **Important Requirements**:
/// 1. The enum must be defined as a single-type-parameter wrapper, e.g.
///    ```rust
///    enum MyEnum<T> {
///        VariantA(T),
///        VariantB(T),
///    }
///    ```
///    This allows all variants to share the same inner type (`T`).
/// 2. All sub-ops must have the same `Input` type (this `T`) and the same `Output`.
///    That is, for each variant, the corresponding op must implement
///    `TryOp<Input = T, Output = Out, Error = E>`.
///
/// # Example
/// ```rust
/// use rig::pipeline::*;
/// use rig::try_conditional;
/// use tokio;
///
/// #[tokio::main]
/// async fn main() {
///     #[derive(Debug)]
///     enum ExampleEnum<T> {
///         Variant1(T),
///         Variant2(T),
///     }
///
///     // Creates a pipeline TryOp that adds 1 or doubles, returning Ok(...) or Err(...)
///     let op1 = map(|x: i32| Ok::<_, String>(x + 1));
///     let op2 = map(|x: i32| Ok::<_, String>(x * 2));
///
///     let try_conditional = try_conditional!(ExampleEnum,
///         Variant1 => op1,
///         Variant2 => op2,
///     );
///
///     let result = try_conditional.try_call(ExampleEnum::Variant1(2)).await;
///     assert_eq!(result, Ok(3));
/// }
/// ```
#[macro_export]
macro_rules! try_conditional {
    ($enum:ident, $( $variant:ident => $op:expr ),+ $(,)?) => {
        {
            #[allow(non_snake_case)]
            struct TryConditionalOp<$( $variant ),+> {
                $( $variant: $variant ),+
            }

            impl<Value, Out, Err, $( $variant ),+> TryOp for TryConditionalOp<$( $variant ),+>
            where
                $( $variant: TryOp<Input=Value, Output=Out, Error=Err> ),+,
                Value: Send + Sync,
                Out: Send + Sync,
                Err: Send + Sync,
            {
                type Input = $enum<Value>;
                type Output = Out;
                type Error = Err;

                async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
                    match input {
                        $(
                            $enum::$variant(val) => self.$variant.try_call(val).await
                        ),+
                    }
                }
            }

            TryConditionalOp { $($variant: $op),+ }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::pipeline::*;

    #[tokio::test]
    async fn test_conditional_op() {
        enum ExampleEnum<T> {
            Variant1(T),
            Variant2(T),
        }

        let op1 = map(|x: i32| x + 1);
        let op2 = map(|x: i32| x * 2);

        let conditional = conditional!(ExampleEnum,
            Variant1 => op1,
            Variant2 => op2
        );

        let result1 = conditional.call(ExampleEnum::Variant1(2)).await;
        assert_eq!(result1, 3);

        let result2 = conditional.call(ExampleEnum::Variant2(3)).await;
        assert_eq!(result2, 6);
    }

    #[tokio::test]
    async fn test_try_conditional_op() {
        enum ExampleEnum<T> {
            Variant1(T),
            Variant2(T),
        }

        let op1 = map(|x: i32| Ok::<_, String>(x + 1));
        let op2 = map(|x: i32| Ok::<_, String>(x * 2));

        let try_conditional = try_conditional!(ExampleEnum,
            Variant1 => op1,
            Variant2 => op2
        );

        let result1 = try_conditional.try_call(ExampleEnum::Variant1(2)).await;
        assert_eq!(result1, Ok(3));

        let result2 = try_conditional.try_call(ExampleEnum::Variant2(3)).await;
        assert_eq!(result2, Ok(6));
    }
}
