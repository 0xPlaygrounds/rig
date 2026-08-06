//! Compile-fail proof of the part-identity contract (#2258 A1).
//!
//! The contract: a [`rig_core::streaming::PartId::Minted`] identity has no
//! path into a request payload. That is enforced by the type system — no
//! `Serialize` impl on `PartId`, and `WireId`'s only constructor refusing
//! minted identities — so the proof is that the leaking programs below do
//! not compile. Deleting a serializer-side gate cannot silently regress:
//! there is no gate, and the leak itself is the compile error.

#[test]
fn minted_identities_cannot_reach_a_request() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/identity_leak/*.rs");
}
