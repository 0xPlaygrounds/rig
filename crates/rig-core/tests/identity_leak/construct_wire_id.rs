//! A `WireId` (the request-serializable identity) must be constructible only
//! through `PartId::into_wire_id`, which refuses minted identities — never
//! directly from a string.

fn main() {
    let _ = rig_core::streaming::WireId("fabricated".to_string());
}
