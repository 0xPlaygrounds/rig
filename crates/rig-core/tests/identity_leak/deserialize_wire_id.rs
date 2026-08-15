//! A `WireId` must not be deserializable.
//!
//! The empty-is-absent rule on the response metadata types is structural only
//! because `Option<WireId>` cannot be deserialized directly: every persistence
//! boundary is forced to read `Option<String>` and normalize, through a `Repr`
//! or through `streaming::deserialize_optional_wire_id`. A transparent
//! `Deserialize` would accept `""` and put the sentinel back from the JSON
//! side; a fallible one would turn a stored `""` into a hard load failure.
//!
//! Both hazards are argued on the type itself, but prose does not fail a
//! build: adding the derive would leave every existing test green — the
//! explicit `deserialize_with` attributes keep working — while letting the
//! *next* `Option<WireId>` field be declared without one. This cell is what
//! turns that rationale into a guard, matching `serialize_part_id.rs`.

fn main() {
    let _: rig_core::streaming::WireId = serde_json::from_str("\"msg_1\"").unwrap();
}
