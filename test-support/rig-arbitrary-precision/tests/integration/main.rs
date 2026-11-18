//! Rig reads its data whole with serde_json's `arbitrary_precision` on. That
//! feature hands a float to serde's buffer for `flatten` and tagged or
//! untagged enums as a map, which a plain `f64` field does not read.
mod big_numbers;
mod catalog;
mod conversation;
mod effects;
mod providers;
