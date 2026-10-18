//! The plugins of the held-out series (H): one plugin per module, each written only against the harness's public API.
//!
//! Each module was its own package; they share a crate so their tests
//! link one binary instead of one each.

pub mod api_key_pool;
pub mod auto_session_name;
pub mod background_verify;
pub mod best_of_n;
pub mod blackboard;
pub mod cancellable_run;
pub mod command_palette;
pub mod command_provider;
pub mod conversation_surgery;
pub mod degenerate_stream_cutter;
pub mod dry_run;
pub mod exit_notes;
pub mod file_lease;
pub mod history_and_drafts;
pub mod hunk_review;
pub mod inline_shell;
pub mod raw_stream_log;
pub mod read_before_edit;
pub mod relay_pipeline;
pub mod repo_map;
pub mod script_tools;
pub mod session_recall;
pub mod spend_by_origin;
pub mod stale_read_elision;
pub mod stall_watchdog;
pub mod submit_result;
pub mod throughput_item;
pub mod tool_arg_repair;
pub mod working_set_carryover;
