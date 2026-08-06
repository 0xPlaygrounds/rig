//! A `PartId` must not be serializable: a minted identity that could enter
//! a serde-built request body would be a fabricated provider handle.

fn main() {
    let id = rig_core::streaming::MintKind::Reasoning.for_wire_index(0);
    let _ = serde_json::to_string(&id);
}
