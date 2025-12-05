pub struct False;
pub struct True;

pub trait Bit {
    const AS_BYTE: u8;
    const AS_BOOL: bool;
}

impl Bit for False {
    const AS_BYTE: u8 = 0;
    const AS_BOOL: bool = false;
}

impl Bit for True {
    const AS_BYTE: u8 = 1;
    const AS_BOOL: bool = true;
}

trait ComputeNot {
    type Output;
}

impl ComputeNot for False {
    type Output = True;
}

impl ComputeNot for True {
    type Output = False;
}

pub type Not<Bit> = <Bit as ComputeNot>::Output;

trait ComputeAnd<Rhs> {
    type Output;
}

impl<Rhs> ComputeAnd<Rhs> for False {
    type Output = False;
}

impl ComputeAnd<False> for True {
    type Output = False;
}

impl ComputeAnd<True> for True {
    type Output = True;
}

pub type And<Lhs, Rhs> = <Lhs as ComputeAnd<Rhs>>::Output;

trait ComputeOr<Rhs> {
    type Output;
}

impl ComputeOr<False> for False {
    type Output = False;
}

impl ComputeOr<True> for False {
    type Output = True;
}

impl<Rhs> ComputeOr<Rhs> for True {
    type Output = True;
}

pub type Or<Lhs, Rhs> = <Lhs as ComputeOr<Rhs>>::Output;

trait IsTrue {}
impl IsTrue for True {}
