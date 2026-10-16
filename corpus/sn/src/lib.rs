//! The plugins of the stack-next series (SN): one plugin per module, each written only against the harness's public API.
//!
//! Each module was its own package; they share a crate so their tests
//! link one binary instead of one each.

pub mod bookmarks;
pub mod consult_models;
pub mod context_gauge;
pub mod crash_resume;
pub mod file_annotations;
pub mod glob_rules;
pub mod hook_scripts;
pub mod large_change_guard;
pub mod leftover_detector;
pub mod model_params;
pub mod notebook_cells;
pub mod plugin_profiler;
pub mod race_models;
pub mod scope_fence;
pub mod server_side_state;
pub mod sessions_prune_cli;
pub mod stray_file_cleanup;
pub mod task_queue;
pub mod terminal_sessions;
pub mod tool_stats;
pub mod untrusted_quarantine;
pub mod usage_report_cli;
pub mod vim_input;
pub mod watch_ai_comments;
pub mod web_fetch;
