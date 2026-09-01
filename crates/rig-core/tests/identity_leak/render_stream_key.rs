//! A `StreamPartId` has no rendering: the key is `Eq + Hash` bookkeeping
//! only, so nothing — not a serializer, not a public stream item, not a log
//! line — can observe it. (Debug formatting exists for diagnostics; a
//! `Display`/`render`-style accessor must not.)

fn main() {
    let id = rig_core::streaming::StreamPartId::wire("rs_1");
    let _ = id.render();
    let _ = format!("{id}");
}
