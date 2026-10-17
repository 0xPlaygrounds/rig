//! Programs on rig's library crates alone, with no Bevy and no harness: each a module, some with a binary in `src/bin`. Their tests check that the harness resumes what they write.
//!
//! Each module was its own package; they share a crate so their tests
//! link one binary instead of one each.

pub mod session_compare_library;
pub mod session_doctor_library;
pub mod session_edit_library;
pub mod transcript_import_library;
