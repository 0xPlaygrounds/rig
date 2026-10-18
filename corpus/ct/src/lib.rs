//! The plugins of the conversation-over-time series (CT): one plugin per module, each written only against the harness's public API.
//!
//! Each module was its own package; they share a crate so their tests
//! link one binary instead of one each.

pub mod ask_session_cli;
pub mod clarify_first;
pub mod compaction_archive;
pub mod conversation_chapters;
pub mod git_guardrails;
pub mod git_history_tools;
pub mod history_versions;
pub mod keep_last_compaction;
pub mod metrics_endpoint;
pub mod paste_chips;
pub mod process_reaper;
pub mod provider_side_tools;
pub mod reply_review_comments;
pub mod resume_subagent;
pub mod schema_adapter;
pub mod session_finder;
pub mod shell_companion;
pub mod status_command;
pub mod test_runner_tool;
pub mod timed_grants;
pub mod turn_navigation;
