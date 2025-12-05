#[macro_export]
macro_rules! assert_type_eq {
    ($x:ty, $($xs:ty),+ $(,)*) => {
        const _: fn() = || { $({
            trait TypeEq {
                type This: ?Sized;
            }

            impl<T: ?Sized> TypeEq for T {
                type This = Self;
            }

            fn assert_type_eq<T, U>()
            where
                T: ?Sized + TypeEq<This = U>,
                U: ?Sized,
            {}

            assert_type_eq::<$x, $xs>();
        })+ };
    };
}
