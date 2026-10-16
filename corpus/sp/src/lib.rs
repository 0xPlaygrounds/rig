//! The plugins of the splits series (SP): one plugin per module, each written only against the harness's public API.
//!
//! Each module was its own package; they share a crate so their tests
//! link one binary instead of one each.

pub mod branch_cherry_pick;
pub mod branch_compare_view;
pub mod changelog_fragments;
pub mod churn_monitor;
pub mod commit_msg_hook;
pub mod dead_ends_ledger;
pub mod inbox_folder;
pub mod output_styles;
pub mod per_agent_compaction;
pub mod recent_sessions_start;
pub mod reply_prefill;
pub mod session_provenance;
pub mod session_tree_browser;
pub mod shell_env_policy;
pub mod structured_config_edit;
pub mod subagent_result_cache;
pub mod theme_files;
