//! A `WireId` (the durable provider handle) must be constructible only
//! through `WireId::new`, which rejects the empty string at runtime — never
//! by direct struct construction, which could fabricate a handle from
//! nothing.

fn main() {
    let _ = rig_core::streaming::WireId("fabricated".to_string());
}
