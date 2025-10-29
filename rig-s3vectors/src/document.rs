/// This is a 1:1 copy of `serde_json::json!`
/// to make creating `aws_smithy_types::Document` significantly less verbose
#[macro_export]
macro_rules! document {
    // Null
    (null) => {
        aws_smithy_types::Document::Null
    };

    // Boolean
    (true) => {
        aws_smithy_types::Document::Bool(true)
    };

    (false) => {
        aws_smithy_types::Document::Bool(false)
    };

    // Array - empty
    ([]) => {
        aws_smithy_types::Document::Array(vec![])
    };

    // Array - with elements
    ([ $($tt:tt)+ ]) => {
        aws_smithy_types::Document::Array($crate::document!(@array [] $($tt)+))
    };

    // Object - empty
    ({}) => {
        aws_smithy_types::Document::Object(std::collections::HashMap::new())
    };

    // Object - with entries
    ({ $($tt:tt)+ }) => {
        aws_smithy_types::Document::Object({
            let mut object = std::collections::HashMap::new();
            $crate::document!(@object object () ($($tt)+) ($($tt)+));
            object
        })
    };

    // Any other expression (numbers, strings, variables, etc.)
    ($other:expr) => {
        aws_smithy_types::Document::from($other)
    };

    // Array muncher - done with trailing comma
    (@array [$($elems:expr,)*]) => {
        vec![$($elems,)*]
    };

    // Array muncher - done without trailing comma
    (@array [$($elems:expr),*]) => {
        vec![$($elems),*]
    };

    // Array muncher - null element
    (@array [$($elems:expr,)*] null $($rest:tt)*) => {
        $crate::document!(@array [$($elems,)* $crate::document!(null)] $($rest)*)
    };

    // Array muncher - true element
    (@array [$($elems:expr,)*] true $($rest:tt)*) => {
        $crate::document!(@array [$($elems,)* $crate::document!(true)] $($rest)*)
    };

    // Array muncher - false element
    (@array [$($elems:expr,)*] false $($rest:tt)*) => {
        $crate::document!(@array [$($elems,)* $crate::document!(false)] $($rest)*)
    };

    // Array muncher - nested array
    (@array [$($elems:expr,)*] [$($array:tt)*] $($rest:tt)*) => {
        $crate::document!(@array [$($elems,)* $crate::document!([$($array)*])] $($rest)*)
    };

    // Array muncher - nested object
    (@array [$($elems:expr,)*] {$($map:tt)*} $($rest:tt)*) => {
        $crate::document!(@array [$($elems,)* $crate::document!({$($map)*})] $($rest)*)
    };

    // Array muncher - expression with comma
    (@array [$($elems:expr,)*] $next:expr, $($rest:tt)*) => {
        $crate::document!(@array [$($elems,)* $crate::document!($next),] $($rest)*)
    };

    // Array muncher - last expression
    (@array [$($elems:expr,)*] $last:expr) => {
        $crate::document!(@array [$($elems,)* $crate::document!($last)])
    };

    // Array muncher - handle trailing comma
    (@array [$($elems:expr),*] , $($rest:tt)*) => {
        aws_smithy_types::document!(@array [$($elems,)*] $($rest)*)
    };

    // Object muncher - done
    (@object $object:ident () () ()) => {};

    // Object muncher - insert entry with trailing comma
    (@object $object:ident [$($key:tt)+] ($value:expr) , $($rest:tt)*) => {
        $object.insert(($($key)+).into(), $value);
        $crate::document!(@object $object () ($($rest)*) ($($rest)*));
    };

    // Object muncher - insert last entry
    (@object $object:ident [$($key:tt)+] ($value:expr)) => {
        $object.insert(($($key)+).into(), $value);
    };

    // Object muncher - value is null
    (@object $object:ident ($($key:tt)+) (: null $($rest:tt)*) $copy:tt) => {
        $crate::document!(@object $object [$($key)+] ($crate::document!(null)) $($rest)*);
    };

    // Object muncher - value is true
    (@object $object:ident ($($key:tt)+) (: true $($rest:tt)*) $copy:tt) => {
        $crate::document!(@object $object [$($key)+] ($crate::document!(true)) $($rest)*);
    };

    // Object muncher - value is false
    (@object $object:ident ($($key:tt)+) (: false $($rest:tt)*) $copy:tt) => {
        $crate::document!(@object $object [$($key)+] (aws_smithy_types::document!(false)) $($rest)*);
    };

    // Object muncher - value is array
    (@object $object:ident ($($key:tt)+) (: [$($array:tt)*] $($rest:tt)*) $copy:tt) => {
        $crate::document!(@object $object [$($key)+] ($crate::document!([$($array)*])) $($rest)*);
    };

    // Object muncher - value is object
    (@object $object:ident ($($key:tt)+) (: {$($map:tt)*} $($rest:tt)*) $copy:tt) => {
        $crate::document!(@object $object [$($key)+] ($crate::document!({$($map)*})) $($rest)*);
    };

    // Object muncher - value is expression with comma
    (@object $object:ident ($($key:tt)+) (: $value:expr , $($rest:tt)*) $copy:tt) => {
        $crate::document!(@object $object [$($key)+] ($crate::document!($value)) , $($rest)*);
    };

    // Object muncher - value is last expression
    (@object $object:ident ($($key:tt)+) (: $value:expr) $copy:tt) => {
        $crate::document!(@object $object [$($key)+] ($crate::document!($value)));
    };

    // Object muncher - parenthesized key
    (@object $object:ident () (($key:expr) : $($rest:tt)*) $copy:tt) => {
        $crate::document!(@object $object ($key) (: $($rest)*) (: $($rest)*));
    };

    // Object muncher - munch token into key
    (@object $object:ident ($($key:tt)*) ($tt:tt $($rest:tt)*) $copy:tt) => {
        $crate::document!(@object $object ($($key)* $tt) ($($rest)*) ($($rest)*));
    };
}
