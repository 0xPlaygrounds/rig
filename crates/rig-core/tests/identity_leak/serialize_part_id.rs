//! A `StreamPartId` must not be serializable: an accumulation key that
//! could enter a serde-built request body (or a persisted history) would
//! stop being opaque.

fn main() {
    let id = rig_core::streaming::MintKind::Reasoning.for_wire_index(0);
    let _ = serde_json::to_string(&id);
}
