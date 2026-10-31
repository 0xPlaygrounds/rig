//! A tool call's arguments decoded into the tool's arguments type, the one
//! way every tool path decodes them. A model may send a whole number as a
//! float, such as `30.0`, which the schema check accepts as an integer
//! ([`validate`](crate::tool::validate)); where the type asks for an
//! integer, it gets that integer. A fraction, a number out of the type's
//! range and a negative number for an unsigned type are refused, never
//! rounded.
//!
//! ```
//! use serde::Deserialize;
//! use serde_json::json;
//!
//! #[derive(Deserialize)]
//! struct Args {
//!     limit: u32,
//!     scale: f64,
//! }
//!
//! let args: Args = rig_core::tool::decode_args(&json!({"limit": 30.0, "scale": 2.0}))?;
//! assert_eq!((args.limit, args.scale), (30, 2.0));
//! assert!(rig_core::tool::decode_args::<Args>(&json!({"limit": 1.5, "scale": 2.0})).is_err());
//! # Ok::<(), serde_json::Error>(())
//! ```

use serde::de::DeserializeOwned;
use serde_json::{Number, Value};

/// `args` as the arguments type `A`. Arguments that do not decode as sent
/// are decoded once more with each integral float in them as its integer,
/// one bound for a `serde_json::Value` or `f64` field too; arguments that
/// decode as sent reach `A` exactly so, every float in them a float.
///
/// # Errors
///
/// When `args` does not fit `A` either way: the error of the arguments as
/// sent.
pub fn decode_args<A: DeserializeOwned>(args: &Value) -> Result<A, serde_json::Error> {
    A::deserialize(args).or_else(|error| as_integers(args).ok_or(error))
}

/// `args` decoded with each integral float as its integer; `None` when it
/// has none or still does not decode.
fn as_integers<A: DeserializeOwned>(args: &Value) -> Option<A> {
    let mut whole = args.clone();
    integers(&mut whole)
        .then(|| A::deserialize(&whole).ok())
        .flatten()
}

/// Makes each float in `value` that is a [`Whole`] number that integer;
/// whether it made any.
fn integers(value: &mut Value) -> bool {
    match value {
        Value::Number(number) if number.is_f64() => match whole(number) {
            Some(Whole::Unsigned(unsigned)) => {
                *number = unsigned.into();
                true
            }
            Some(Whole::Signed(signed)) => {
                *number = signed.into();
                true
            }
            None => false,
        },
        Value::Array(items) => {
            let mut any = false;
            for item in items {
                any |= integers(item);
            }
            any
        }
        Value::Object(fields) => {
            let mut any = false;
            for field in fields.values_mut() {
                any |= integers(field);
            }
            any
        }
        _ => false,
    }
}

/// The whole number a JSON number is: an integer, or a float with no
/// fraction that a 64-bit integer holds exactly, such as `30.0`. What
/// counts as an integer, for the schema check and for decoding alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Whole {
    Signed(i64),
    Unsigned(u64),
}

/// The [`Whole`] number `number` is, if it is one.
pub(crate) fn whole(number: &Number) -> Option<Whole> {
    if let Some(unsigned) = number.as_u64() {
        return Some(Whole::Unsigned(unsigned));
    }
    if let Some(signed) = number.as_i64() {
        return Some(Whole::Signed(signed));
    }
    let float = number.as_f64().filter(|float| float.fract() == 0.0)?;
    // Both bounds are powers of two, exact in an `f64`, so each cast below
    // is of a whole number inside the integer's range, and exact.
    if (0.0..18_446_744_073_709_551_616.0).contains(&float) {
        Some(Whole::Unsigned(float as u64))
    } else if (-9_223_372_036_854_775_808.0..0.0).contains(&float) {
        Some(Whole::Signed(float as i64))
    } else {
        None
    }
}

#[cfg(test)]
mod tests;
