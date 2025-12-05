/// Type-level church encoded natural numbers
/// Useful for handling type-level lists
use std::marker::PhantomData;

use crate::type_level::sealed::Sealed;

#[derive(Debug, Clone, Copy)]
pub struct Z;

#[derive(Debug, Clone, Copy)]
pub struct S<N>(PhantomData<N>);

impl Sealed for Z {}
impl<N> Sealed for S<N> where N: Sealed {}

trait ComputeSucc {
    type Output;
}

impl ComputeSucc for Z {
    type Output = S<Z>;
}

impl<N> ComputeSucc for S<N> {
    type Output = S<Self>;
}

pub type Succ<N> = <N as ComputeSucc>::Output;

trait ComputePred {
    type Output;
}

impl ComputePred for Z {
    type Output = Z;
}

impl<N> ComputePred for S<N> {
    type Output = N;
}

pub trait Nat {
    fn new() -> Self;
}

impl Nat for Z {
    fn new() -> Self {
        Self
    }
}

impl<N> Nat for S<N>
where
    N: Nat,
{
    fn new() -> Self {
        Self(PhantomData)
    }
}

impl Into<u32> for Z {
    fn into(self) -> u32 {
        0
    }
}

impl<N> Into<u32> for S<N>
where
    N: Nat + Into<u32>,
{
    fn into(self) -> u32 {
        1 + <N as Into<u32>>::into(N::new())
    }
}
