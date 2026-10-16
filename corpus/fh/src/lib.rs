//! The plugins of the fresh held-out series (FH): one plugin per module, each written only against the harness's public API.
//!
//! Each module was its own package; they share a crate so their tests
//! link one binary instead of one each.

pub mod apply_patch_tool;
pub mod audit_log;
pub mod batch_provider;
pub mod checklist_runner;
pub mod correction_learner;
pub mod dirty_tree_guard;
pub mod early_tool_start;
pub mod editor_front;
pub mod encrypted_sessions;
pub mod import_sessions;
pub mod lsp_tools;
pub mod mentions_and_images;
pub mod next_step_suggestions;
pub mod open_pr;
pub mod prompt_cache_shaping;
pub mod remote_workspace;
pub mod rich_tool_rendering;
pub mod safety_judge;
pub mod side_question;
pub mod standing_team;
pub mod step_mode;
pub mod structured_compaction;
pub mod thinking_keywords;
pub mod transcript_search;
pub mod turn_timeline;
