//! A `StreamPartId`'s representation is private: pattern-matching cannot
//! extract the wire string (or any payload), so "no accessor into the
//! durable id space" is a property of the type, not a convention.

fn main() {
    let id = rig_core::streaming::StreamPartId::wire("rs_1");
    let rig_core::streaming::StreamPartId(repr) = &id;
    match id {
        rig_core::streaming::StreamPartId::Wire(wire) => drop(wire),
        _ => {}
    }
}
