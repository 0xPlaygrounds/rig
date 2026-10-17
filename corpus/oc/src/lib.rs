//! The plugins of the one-conversation series (OC): one plugin per module, each written only against the harness's public API.
//!
//! Each module was its own package; they share a crate so their tests
//! link one binary instead of one each.

pub mod approval_batching;
pub mod architect_editor;
pub mod ask_parent;
pub mod auto_continue;
pub mod away_recap;
pub mod coverage_gap_nudge;
pub mod dataset_cli;
pub mod dependency_guard;
pub mod git_delta_context;
pub mod inline_images;
pub mod issue_references;
pub mod line_provenance;
pub mod message_ratings;
pub mod opaque_reasoning;
pub mod project_replace;
pub mod rate_limit_status;
pub mod reviewable_compaction;
pub mod session_bundle;
pub mod sql_query_tool;
pub mod steer_and_queue;
pub mod transcript_in_editor;
