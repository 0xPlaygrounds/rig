//! `/bridge` commands (`status`, `usage`, `reset`) and background UI notices.
//!
//! `/bridge status` and `/bridge reset` answer synchronously from the shared
//! [`Bridge`] state. `/bridge usage` queries each installed harness's official
//! quota interface in parallel off the UI thread (with a short timeout) and
//! delivers the formatted report via the bridge notice channel so the UI never
//! stalls.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy_ecs::message::Messages;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::ContextUse;
use rig_harness::rig_tools::process::{Process, Spawn};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::session::{Bridge, VendorStats};

#[cfg(test)]
mod tests;

/// Per-harness timeout for `/bridge usage` probes.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// A message queued from background bridge tasks for Rig's notice feed.
pub(crate) struct BridgeNotice {
    pub agent: Option<Entity>,
    pub warn: bool,
    pub text: String,
}

/// Shared handle held as a Bevy resource for `/bridge` commands.
#[derive(Resource, Clone)]
pub(crate) struct BridgeHandle {
    pub bridge: Arc<Bridge>,
    pub processes: Processes,
    pub installed: Vec<HarnessEntry>,
}

/// One configured harness and its resolved executable path (if installed).
#[derive(Clone, Debug)]
pub(crate) struct HarnessEntry {
    pub vendor: &'static str,
    pub label: &'static str,
    pub program: Option<PathBuf>,
    pub model_count: usize,
}

/// Receiver drained into Rig's [`Notice`] stream each frame.
#[derive(Resource)]
pub(crate) struct NoticeInbox(Mutex<mpsc::UnboundedReceiver<BridgeNotice>>);

impl NoticeInbox {
    pub(crate) fn new(rx: mpsc::UnboundedReceiver<BridgeNotice>) -> Self {
        Self(Mutex::new(rx))
    }
}

/// Drains queued bridge notices into Rig's UI notice stream.
pub(crate) fn drain_notices(
    inbox: Option<Res<NoticeInbox>>,
    mut notices: Option<MessageWriter<Notice>>,
) {
    let (Some(inbox), Some(notices)) = (inbox, notices.as_mut()) else {
        return;
    };
    let mut rx = inbox.0.lock().unwrap_or_else(PoisonError::into_inner);
    while let Ok(item) = rx.try_recv() {
        let notice = if item.warn {
            Notice::warn(item.agent, item.text)
        } else {
            Notice::info(item.agent, item.text)
        };
        notices.write(notice);
    }
}

/// Dispatches `/bridge [status | usage | reset]`.
pub(crate) fn command(
    In(args): In<CommandArgs>,
    handle: Option<Res<BridgeHandle>>,
    mut notices: MessageWriter<Notice>,
) {
    let Some(handle) = handle else {
        notices.write(Notice::warn(args.agent, "Bridge plugin is not active."));
        return;
    };
    let sub = args.args.split_whitespace().next().unwrap_or("status");
    match sub {
        "status" => {
            notices.write(Notice::info(args.agent, format_status(&handle)));
        }
        "reset" => {
            let stopped = handle.bridge.reset();
            let text = match stopped {
                0 => "No warm bridge sessions were running.".to_owned(),
                1 => "Stopped 1 warm bridge session and removed its scratch directory.".to_owned(),
                n => format!(
                    "Stopped {n} warm bridge sessions and removed their scratch directories."
                ),
            };
            notices.write(Notice::info(args.agent, text));
        }
        "usage" => {
            notices.write(Notice::info(args.agent, collect_usage(&handle)));
        }
        other => {
            notices.write(Notice::warn(
                args.agent,
                format!("Unknown `/bridge` subcommand `{other}`. Try `/bridge status`, `/bridge usage`, or `/bridge reset`."),
            ));
        }
    }
}

/// Registers `/bridge` and the notice-draining system when Rig's notice feed exists.
pub(crate) fn register(
    app: &mut App,
    handle: BridgeHandle,
    rx: mpsc::UnboundedReceiver<BridgeNotice>,
) {
    app.insert_resource(handle)
        .insert_resource(NoticeInbox::new(rx))
        .add_command(
            "bridge",
            "Bridge status, subscription usage, or reset: /bridge [status|usage|reset]",
            command,
        );
    if app.world().contains_resource::<Messages<Notice>>() {
        app.add_systems(Update, drain_notices);
    }
}

fn format_status(handle: &BridgeHandle) -> String {
    let snap = handle.bridge.snapshot();
    let mut lines = vec![
        format!(
            "Bridge status ({} warm session(s), scratch: {})",
            snap.warm.len(),
            snap.scratch_root.display()
        ),
        String::new(),
    ];
    for entry in &handle.installed {
        let state = match &entry.program {
            Some(path) => format!("{} ({} model(s))", path.display(), entry.model_count),
            None => "not installed".to_owned(),
        };
        let mut line = format!("- {}: {state}", entry.vendor);
        if let Some(stats) = snap.vendors.get(entry.vendor) {
            line.push_str(&format!(" | {}", format_session_stats(stats)));
            if let Some(note) = &stats.last_note {
                line.push_str(&format!(" | note: {note}"));
            }
        }
        lines.push(line);
    }
    if !snap.warm.is_empty() {
        lines.push(String::new());
        lines.push("Warm sessions:".to_owned());
        for warm in &snap.warm {
            let relist = if warm.relists {
                "relists"
            } else {
                "fixed-tools"
            };
            let ctx = warm
                .context_tokens
                .map(|tokens| {
                    format!(
                        ", ctx {}",
                        ContextUse {
                            tokens,
                            limit: warm.context_limit,
                        }
                    )
                })
                .unwrap_or_default();
            lines.push(format!(
                "- {}/{} ({} turn(s){ctx}, {relist})",
                warm.vendor, warm.model, warm.turns
            ));
        }
    }
    lines.push(String::new());
    lines.push("Subcommands: `/bridge status`, `/bridge usage`, `/bridge reset`.".to_owned());
    lines.join("\n")
}

fn format_session_stats(stats: &VendorStats) -> String {
    let cache_pct = stats
        .cached_input_tokens
        .saturating_mul(100)
        .checked_div(stats.input_tokens)
        .map(|pct| format!(" ({pct}% cached)"))
        .unwrap_or_default();
    let ctx = stats
        .last_context_tokens
        .map(|tokens| {
            format!(
                ", ctx {}",
                ContextUse {
                    tokens,
                    limit: stats.last_context_limit,
                }
            )
        })
        .unwrap_or_default();
    format!(
        "{} call(s), {} in{} / {} out{ctx}",
        stats.calls, stats.input_tokens, cache_pct, stats.output_tokens
    )
}

fn collect_usage(handle: &BridgeHandle) -> String {
    let snap = handle.bridge.snapshot();
    let scratch_root = handle.bridge.scratch_root().to_path_buf();
    let results: Vec<(&'static str, &'static str, String)> = std::thread::scope(|scope| {
        let jobs: Vec<_> = handle
            .installed
            .iter()
            .map(|entry| {
                let processes = handle.processes.clone();
                let scratch = scratch_root.clone();
                scope.spawn(move || {
                    let plan = match &entry.program {
                        None => "not installed".to_owned(),
                        Some(program) => {
                            probe_vendor_usage(&processes, entry.vendor, program, &scratch)
                        }
                    };
                    (entry.vendor, entry.label, plan)
                })
            })
            .collect();
        jobs.into_iter().filter_map(|job| job.join().ok()).collect()
    });

    let mut lines = vec![
        "Bridge subscription usage & session totals:".to_owned(),
        String::new(),
    ];
    for (vendor, label, plan) in results {
        let session = snap
            .vendors
            .get(vendor)
            .map(format_session_stats)
            .unwrap_or_else(|| "0 calls this session".to_owned());
        lines.push(format!(
            "- {label} (`{vendor}`): {plan} | session: {session}"
        ));
    }
    lines.join("\n")
}

fn probe_vendor_usage(
    processes: &Processes,
    vendor: &str,
    program: &Path,
    scratch_root: &Path,
) -> String {
    match vendor {
        "claude-code" => probe_claude_usage(processes, program),
        "codex" => probe_codex_usage(processes, program, scratch_root),
        "muse" => probe_muse_usage(processes, program),
        "grok-build" => probe_grok_usage(processes, program),
        "antigravity" | "copilot" => "unavailable (no official CLI/RPC quota API)".to_owned(),
        _ => "unavailable".to_owned(),
    }
}

/// Queries Claude Code's official `stream-json` `get_usage` control request.
fn probe_claude_usage(processes: &Processes, program: &Path) -> String {
    let mut command = processes.launcher().command(program);
    command
        .args([
            "--print",
            "--verbose",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--tools",
            "",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
        ])
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("CLAUDECODE")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let Ok(mut process) = Process::start_with_stdio(command) else {
        return "unavailable (could not start claude)".to_owned();
    };
    let (Some(mut stdin), Some(stdout)) = (process.take_stdin(), process.take_stdout()) else {
        return "unavailable".to_owned();
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let req = json!({
            "type": "control_request",
            "request_id": "rig-usage",
            "request": { "subtype": "get_usage" },
        });
        if writeln!(stdin, "{req}").is_err() || stdin.flush().is_err() {
            return;
        }
        let mut lines = BufReader::new(stdout).lines();
        while let Some(Ok(line)) = lines.next() {
            let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if msg.get("type").and_then(Value::as_str) == Some("control_response")
                && msg.pointer("/response/request_id").and_then(Value::as_str) == Some("rig-usage")
            {
                let _ = tx.send(msg);
                return;
            }
        }
    });
    let response = rx.recv_timeout(PROBE_TIMEOUT).ok();
    drop(process);
    let _ = reader.join();
    match response {
        Some(msg) => parse_claude_usage(&msg),
        None => "unavailable (timed out)".to_owned(),
    }
}

/// Formats a Claude Code `control_response` for `get_usage`.
pub(crate) fn parse_claude_usage(msg: &Value) -> String {
    let Some(body) = msg.pointer("/response/response") else {
        return "unavailable (no usage response)".to_owned();
    };
    if body.get("rate_limits_available").and_then(Value::as_bool) != Some(true) {
        return "unavailable (no subscription limits reported)".to_owned();
    }
    let Some(limits) = body.get("rate_limits") else {
        return "unavailable".to_owned();
    };
    let mut parts = Vec::new();
    if let Some(plan) = body.get("subscription_type").and_then(Value::as_str) {
        parts.push(format!("plan {plan}"));
    }
    for (key, label) in [
        ("five_hour", "5h"),
        ("seven_day", "weekly"),
        ("seven_day_opus", "opus weekly"),
        ("seven_day_sonnet", "sonnet weekly"),
    ] {
        if let Some(win) = limits.get(key)
            && let Some(pct) = win.get("utilization").and_then(Value::as_f64)
        {
            let reset = win
                .get("resets_at")
                .and_then(Value::as_str)
                .map(|ts| format!(" (resets {})", short_iso(ts)))
                .unwrap_or_default();
            parts.push(format!("{label} {pct:.0}% used{reset}"));
        }
    }
    if parts.is_empty() {
        "unavailable (no active quota windows)".to_owned()
    } else {
        parts.join(", ")
    }
}

/// Queries Codex's official `codex app-server` `account/rateLimits/read` RPC.
fn probe_codex_usage(processes: &Processes, program: &Path, scratch_root: &Path) -> String {
    let dir = scratch_root.join(format!(
        "rig-bridge-{}-codex-usage-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    if std::fs::create_dir_all(&dir).is_err() {
        return "unavailable (could not create probe dir)".to_owned();
    }
    let result = probe_codex_in_dir(processes, program, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn probe_codex_in_dir(processes: &Processes, program: &Path, dir: &Path) -> String {
    let auth = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from(std::env::var_os("HOME")?).join(".codex")))
        .map(|home| home.join("auth.json"));
    if let Some(auth) = auth.filter(|auth| auth.is_file()) {
        #[cfg(unix)]
        let _ = std::os::unix::fs::symlink(&auth, dir.join("auth.json"));
    }
    let mut command = processes.launcher().command(program);
    command
        .current_dir(dir)
        .args([
            "app-server",
            "-c",
            "analytics.enabled=false",
            "-c",
            "check_for_update_on_startup=false",
        ])
        .env("CODEX_HOME", dir)
        .env_remove("OPENAI_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let Ok(mut process) = Process::start_with_stdio(command) else {
        return "unavailable (could not start codex)".to_owned();
    };
    let (Some(mut stdin), Some(stdout)) = (process.take_stdin(), process.take_stdout()) else {
        return "unavailable".to_owned();
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let init = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": { "name": "rig", "version": env!("CARGO_PKG_VERSION") },
                "capabilities": {},
            },
        });
        let read = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "account/rateLimits/read",
            "params": {},
        });
        if writeln!(stdin, "{init}").is_err()
            || writeln!(stdin, "{read}").is_err()
            || stdin.flush().is_err()
        {
            return;
        }
        let mut lines = BufReader::new(stdout).lines();
        while let Some(Ok(line)) = lines.next() {
            let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if msg.get("id").and_then(Value::as_u64) == Some(2) {
                let _ = tx.send(msg);
                return;
            }
        }
    });
    let deadline = Instant::now() + PROBE_TIMEOUT;
    let response = rx
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .ok();
    drop(process);
    let _ = reader.join();
    match response {
        Some(msg) => parse_codex_usage(&msg),
        None => "unavailable (timed out)".to_owned(),
    }
}

/// Formats a Codex `account/rateLimits/read` JSON-RPC response.
pub(crate) fn parse_codex_usage(msg: &Value) -> String {
    if let Some(err) = msg.pointer("/error/message").and_then(Value::as_str) {
        return format!("unavailable ({err})");
    }
    let Some(limits) = msg
        .pointer("/result/rateLimitsByLimitId/codex")
        .or_else(|| msg.pointer("/result/rateLimits"))
    else {
        return "unavailable (not signed in or no limits reported)".to_owned();
    };
    let mut parts = Vec::new();
    if let Some(plan) = limits.get("planType").and_then(Value::as_str) {
        parts.push(format!("plan {plan}"));
    }
    for (key, fallback) in [("primary", "primary"), ("secondary", "secondary")] {
        if let Some(win) = limits.get(key)
            && let Some(pct) = win.get("usedPercent").and_then(Value::as_f64)
        {
            let label = match win.get("windowDurationMins").and_then(Value::as_u64) {
                Some(10080) => "weekly".to_owned(),
                Some(mins) if mins >= 60 && mins % 60 == 0 => format!("{}h", mins / 60),
                Some(mins) => format!("{mins}m"),
                None => fallback.to_owned(),
            };
            let reset = win
                .get("resetsAt")
                .and_then(Value::as_u64)
                .map(|secs| format!(" (resets in {})", format_relative_secs(secs)))
                .unwrap_or_default();
            parts.push(format!("{label} {pct:.0}% used{reset}"));
        }
    }
    if parts.is_empty() {
        "unavailable".to_owned()
    } else {
        parts.join(", ")
    }
}

/// Queries Muse's official `muse usage --json` CLI output.
fn probe_muse_usage(processes: &Processes, program: &Path) -> String {
    let mut command = processes.launcher().command(program);
    command.args(["usage", "--json"]);
    let output = Process::start(command, Spawn::default())
        .and_then(|process| process.output(Some(PROBE_TIMEOUT)));
    let Ok(output) = output else {
        return "unavailable (timed out or failed to run)".to_owned();
    };
    let text = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str::<Value>(&text)
        .map(|value| parse_muse_usage(&value))
        .unwrap_or_else(|_| "unavailable (invalid response)".to_owned())
}

/// Formats `muse usage --json`.
pub(crate) fn parse_muse_usage(value: &Value) -> String {
    let Some(allowance) = value.get("allowance") else {
        let reason = value
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or("not signed in");
        return format!("unavailable ({reason})");
    };
    let mut parts = Vec::new();
    if let Some(tier) = allowance.get("tier").and_then(Value::as_str) {
        parts.push(tier.to_owned());
    }
    for (key, label) in [("window", "window"), ("weekly", "weekly")] {
        if let Some(win) = allowance.get(key)
            && let Some(pct) = win.get("usedPercent").and_then(Value::as_f64)
        {
            let reset = win
                .get("resetsAtMs")
                .and_then(Value::as_u64)
                .map(|ms| format!(" (resets in {})", format_relative_secs(ms / 1000)))
                .unwrap_or_default();
            parts.push(format!("{label} {pct:.0}% used{reset}"));
        }
    }
    if parts.is_empty() {
        "unavailable".to_owned()
    } else {
        parts.join(", ")
    }
}

/// Queries Grok's official `grok usage --json` without starting Grok's daemon.
fn probe_grok_usage(processes: &Processes, program: &Path) -> String {
    let mut command = processes.launcher().command(program);
    command.args(["usage", "--json"]);
    let output = Process::start(command, Spawn::default())
        .and_then(|process| process.output(Some(PROBE_TIMEOUT)));
    let Ok(output) = output else {
        return "unavailable (timed out)".to_owned();
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let combined = format!("{stdout} {stderr}").to_lowercase();
        if combined.contains("background")
            || combined.contains("daemon")
            || combined.contains("service")
            || combined.contains("leader")
        {
            return "unavailable (Grok background service not running)".to_owned();
        }
        return "unavailable (Grok background service not running or not signed in)".to_owned();
    }
    serde_json::from_str::<Value>(&stdout)
        .map(|value| parse_grok_usage(&value))
        .unwrap_or_else(|_| "unavailable (Grok background service not running)".to_owned())
}

/// Formats `grok usage --json` output.
pub(crate) fn parse_grok_usage(value: &Value) -> String {
    if let Some(pct) = value
        .pointer("/usedPercent")
        .or_else(|| value.pointer("/used_percent"))
        .and_then(Value::as_f64)
    {
        return format!("{pct:.0}% used");
    }
    if let Some(obj) = value.as_object() {
        let mut parts = Vec::new();
        for (key, val) in obj {
            if let Some(pct) = val
                .get("usedPercent")
                .or_else(|| val.get("used_percent"))
                .or_else(|| val.get("utilization"))
                .and_then(Value::as_f64)
            {
                parts.push(format!("{key} {pct:.0}% used"));
            }
        }
        if !parts.is_empty() {
            return parts.join(", ");
        }
    }
    "available (run `grok usage` for details)".to_owned()
}

fn short_iso(ts: &str) -> &str {
    ts.get(..16).unwrap_or(ts)
}

fn format_relative_secs(target_epoch_secs: u64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    let delta = target_epoch_secs.saturating_sub(now);
    let days = delta / 86_400;
    let hours = (delta % 86_400) / 3600;
    let mins = (delta % 3600) / 60;
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m")
    }
}
