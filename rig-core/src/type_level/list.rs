use crate::type_level::{
    bit::{Bit, False, True},
    nat::{Nat, S, Z},
    sealed::Sealed,
};
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct Nil;

#[derive(Debug, Clone, Copy)]
pub struct Cons<Head, Tail>(PhantomData<(Head, Tail)>);

impl Sealed for Nil {}
impl<Head, Tail> Sealed for Cons<Head, Tail> where Tail: Sealed {}

trait ListImpl {
    type Last<X>;
    type Inits<X>: List;
}

#[macro_export]
macro_rules! List {
    () => { Nil };
    (...$Rest:ty) => { $Rest };
    ($A:ty) => { List![$A,] };
    ($A:ty, $($tok:tt)*) => {
        Cons<$A, $crate::List![$($tok)*]>
    }
}

pub trait List: Sealed + ListImpl {
    type Concat<Rhs: List>: List;
    type IsEmpty: Bit;
    type Len: Nat;
}

pub trait Empty: List + Sealed {}
pub trait NonEmpty: List + Sealed {
    type First;
    type Rest: List;
    type Last;
    type Inits: List;
}

impl ListImpl for Nil {
    type Last<X> = X;
    type Inits<X> = Nil;
}

impl List for Nil {
    type Concat<Rhs: List> = Rhs;
    type IsEmpty = True;
    type Len = Z;
}

impl Empty for Nil {}

impl<Head, Tail: List> ListImpl for Cons<Head, Tail> {
    type Last<X> = Tail::Last<Head>;
    type Inits<X> = Cons<X, Tail::Inits<Head>>;
}

impl<Head, Tail: List> List for Cons<Head, Tail> {
    type Concat<Rhs: List> = Cons<Head, Tail::Concat<Rhs>>;
    type IsEmpty = False;
    type Len = S<Tail::Len>;
}

impl<Head, Tail: List> NonEmpty for Cons<Head, Tail> {
    type First = Head;
    type Rest = Tail;
    type Last = Tail::Last<Head>;
    type Inits = Tail::Inits<Head>;
}

pub type Head<List> = <List as NonEmpty>::First;
pub type Tail<List> = <List as NonEmpty>::Rest;
pub type Last<List> = <List as NonEmpty>::Last;
pub type Inits<List> = <List as NonEmpty>::Inits;
pub type Concat<Lhs, Rhs> = <Lhs as List>::Concat<Rhs>;
pub type Len<TList> = <TList as List>::Len;
pub type IsEmpty<TList> = <TList as List>::IsEmpty;

#[cfg(test)]
mod test {
    use crate::assert_type_eq;

    use super::*;

    #[test]
    fn first() {
        assert_type_eq!(String, Head<List![String, f32, f32]>);
    }

    #[test]
    fn is_empty() {
        assert_type_eq!(True, IsEmpty<List![]>);
    }

    #[test]
    fn len() {
        assert_type_eq!(S<S<Z>>, Len<List![String, String]>);
    }
}
