//! `/bridge` commands (`status`, `usage`, `reset`) and background signals.
//!
//! `/bridge status` and `/bridge reset` read shared [`Bridge`] state directly.
//! `/bridge usage` probes each installed harness's official quota interface in
//! parallel with a short timeout. Background signals also synchronize runtime
//! context windows onto active agent [`Connection`]s.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_ecs::message::Messages;
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_harness::rig_core::completion::ContextUse;
use rig_harness::rig_core::providers::registry::ProviderId;
use rig_harness::rig_ecs::stage::AgentLoop;
use rig_harness::rig_ecs::tasks::Wake;
use rig_harness::rig_tools::process::{Process, Spawn};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::harness::{Vendor, link};
use crate::rpc;
use crate::session::{Bridge, Dir, VendorStats};

#[cfg(test)]
mod tests;

/// Per-harness timeout for `/bridge usage` probes.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Background bridge signal drained on Bevy's `Update` schedule.
pub(crate) enum BridgeSignal {
    /// A notice for Rig's UI feed, optionally releasing a [`KeepRunning`] entity.
    Notice {
        agent: Option<Entity>,
        warn: bool,
        text: String,
        release: Option<Entity>,
    },
    /// An authoritative runtime context window reported by a harness.
    Window {
        vendor: Vendor,
        model: String,
        limit: u32,
    },
}

/// Shared handle held as a Bevy resource for `/bridge` commands.
#[derive(Resource, Clone)]
pub(crate) struct BridgeHandle {
    pub bridge: Arc<Bridge>,
    pub processes: Processes,
    pub installed: Vec<HarnessEntry>,
    pub signals: mpsc::UnboundedSender<BridgeSignal>,
}

/// One configured harness and its resolved executable path, if installed.
#[derive(Clone, Debug)]
pub(crate) struct HarnessEntry {
    pub vendor: Vendor,
    pub program: Option<PathBuf>,
    pub model_count: usize,
}

/// Receiver drained into Rig's UI, catalog, and active [`Connection`]s each frame.
#[derive(Resource)]
pub(crate) struct SignalInbox {
    rx: Mutex<mpsc::UnboundedReceiver<BridgeSignal>>,
    /// Agents whose `Connection` window was updated mid-turn and need one
    /// `Connection` refresh once `LastUsage` settles (clearing any stale
    /// `InputLimit` written from the call's pre-turn `SentTo` snapshot).
    pending_refresh: Mutex<HashSet<Entity>>,
}

impl SignalInbox {
    pub(crate) fn new(rx: mpsc::UnboundedReceiver<BridgeSignal>) -> Self {
        Self {
            rx: Mutex::new(rx),
            pending_refresh: Mutex::default(),
        }
    }
}

fn set_window(spec: &mut ModelSpec, limit: u32) {
    spec.context_window = Some(limit);
    spec.max_input_tokens = Some(limit);
}

/// Drains queued bridge signals into notices, the model catalog, and active
/// agent connections.
pub(crate) fn drain_signals(
    inbox: Option<Res<SignalInbox>>,
    mut notices: Option<MessageWriter<Notice>>,
    mut models: Option<ResMut<Models>>,
    wake: Option<Res<Wake>>,
    updated_usage: Query<(Entity, &Connection), Changed<LastUsage>>,
    agents: Query<(Entity, &Connection)>,
    mut commands: Commands,
) {
    let Some(inbox) = inbox else {
        return;
    };
    let mut pending = inbox
        .pending_refresh
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    // Run before draining new signals so `conn.spec` already holds the
    // updated window from an earlier frame rather than a stale pre-command value.
    for (entity, conn) in &updated_usage {
        if pending.remove(&entity) {
            commands.entity(entity).insert(Connection {
                spec: conn.spec.clone(),
                handler: conn.handler.clone(),
            });
        }
    }
    let mut rx = inbox.rx.lock().unwrap_or_else(PoisonError::into_inner);
    while let Ok(signal) = rx.try_recv() {
        match signal {
            BridgeSignal::Notice {
                agent,
                warn,
                text,
                release,
            } => {
                if let Some(notices) = notices.as_mut() {
                    let notice = if warn {
                        Notice::warn(agent, text)
                    } else {
                        Notice::info(agent, text)
                    };
                    notices.write(notice);
                }
                if let Some(holder) = release {
                    commands.entity(holder).try_despawn();
                    if let Some(wake) = wake.as_deref() {
                        wake.wake();
                    }
                }
            }
            BridgeSignal::Window {
                vendor,
                model,
                limit,
            } => {
                if let Some(models) = models.as_deref_mut()
                    && let Ok(pid) = ProviderId::declare(vendor.as_str())
                    && let Some(mut spec) = models.0.catalog().get_exact(pid, &model).cloned()
                    && spec.input_limit() != Some(limit)
                {
                    set_window(&mut spec, limit);
                    models.0.catalog_mut().insert(spec);
                }
                for (entity, conn) in &agents {
                    if conn.spec.provider.vendor() == vendor.as_str()
                        && conn.spec.id == model
                        && conn.spec.input_limit() != Some(limit)
                    {
                        let mut spec = (*conn.spec).clone();
                        set_window(&mut spec, limit);
                        commands.entity(entity).insert(Connection {
                            spec: Arc::new(spec),
                            handler: conn.handler.clone(),
                        });
                        pending.insert(entity);
                    }
                }
            }
        }
    }
}

/// Dispatches `/bridge [status | usage | reset]`.
pub(crate) fn command(
    In(args): In<CommandArgs>,
    handle: Option<Res<BridgeHandle>>,
    wake: Option<Res<Wake>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(handle) = handle else {
        notices.write(Notice::warn(args.agent, "Bridge plugin is not active."));
        return;
    };
    let notice = match args.args.split_whitespace().next().unwrap_or("status") {
        "status" => Notice::info(args.agent, format_status(&handle)),
        "usage" => {
            // Probe harnesses off the UI thread while keeping headless `-p` alive.
            let holder = commands.spawn(KeepRunning).id();
            let handle = handle.clone();
            let wake = wake.as_deref().cloned();
            let agent = args.agent;
            std::thread::spawn(move || {
                let text = collect_usage(&handle);
                let _ = handle.signals.send(BridgeSignal::Notice {
                    agent: Some(agent),
                    warn: false,
                    text,
                    release: Some(holder),
                });
                if let Some(wake) = wake {
                    wake.wake();
                }
            });
            return;
        }
        "reset" => {
            let text = match handle.bridge.reset() {
                0 => "No warm bridge sessions were running.".to_owned(),
                1 => "Stopped 1 warm bridge session and removed its scratch directory.".to_owned(),
                n => format!(
                    "Stopped {n} warm bridge sessions and removed their scratch directories."
                ),
            };
            Notice::info(args.agent, text)
        }
        other => Notice::warn(
            args.agent,
            format!(
                "Unknown `/bridge` subcommand `{other}`. Try `/bridge status`, `/bridge usage`, or `/bridge reset`."
            ),
        ),
    };
    notices.write(notice);
}

/// Registers `/bridge` and the signal-draining system.
pub(crate) fn register(
    app: &mut App,
    handle: BridgeHandle,
    rx: mpsc::UnboundedReceiver<BridgeSignal>,
) {
    app.insert_resource(handle)
        .insert_resource(SignalInbox::new(rx))
        .add_command(
            "bridge",
            "Bridge status, subscription usage, or reset: /bridge [status|usage|reset]",
            command,
        );
    if app.world().contains_resource::<Messages<Notice>>() {
        app.add_systems(Update, drain_signals.after(AgentLoop));
    }
}

fn format_ctx(tokens: Option<u64>, limit: Option<u32>) -> String {
    tokens
        .map(|tokens| format!(", ctx {}", ContextUse { tokens, limit }))
        .unwrap_or_default()
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
        if let Some(stats) = snap.vendors.get(&entry.vendor) {
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
            let ctx = format_ctx(warm.context_tokens, warm.context_limit);
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
    let ctx = format_ctx(stats.last_context_tokens, stats.last_context_limit);
    format!(
        "{} call(s), {} in{cache_pct} / {} out{ctx}",
        stats.calls, stats.input_tokens, stats.output_tokens
    )
}

fn collect_usage(handle: &BridgeHandle) -> String {
    let snap = handle.bridge.snapshot();
    let scratch_root = handle.bridge.scratch_root().to_path_buf();
    let results: Vec<(Vendor, String)> = std::thread::scope(|scope| {
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
                    (entry.vendor, plan)
                })
            })
            .collect();
        jobs.into_iter().filter_map(|job| job.join().ok()).collect()
    });

    let mut lines = vec![
        "Bridge subscription usage & session totals:".to_owned(),
        String::new(),
    ];
    for (vendor, plan) in results {
        let session = snap
            .vendors
            .get(&vendor)
            .map(format_session_stats)
            .unwrap_or_else(|| "0 calls this session".to_owned());
        lines.push(format!(
            "- {} (`{vendor}`): {plan} | session: {session}",
            vendor.label()
        ));
    }
    lines.join("\n")
}

fn probe_vendor_usage(
    processes: &Processes,
    vendor: Vendor,
    program: &Path,
    scratch_root: &Path,
) -> String {
    match vendor {
        Vendor::ClaudeCode => probe_claude_usage(processes, program),
        Vendor::Codex => probe_codex_usage(processes, program, scratch_root),
        Vendor::Muse => probe_muse_usage(processes, program),
        Vendor::GrokBuild => probe_grok_usage(processes, program),
        Vendor::Antigravity | Vendor::Copilot => {
            "unavailable (no official CLI/RPC quota API)".to_owned()
        }
    }
}

/// Queries Claude Code's official `stream-json` `get_usage` control request.
fn probe_claude_usage(processes: &Processes, program: &Path) -> String {
    let mut command = processes.launcher().command(program);
    command.args([
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
    ]);
    for name in crate::claude::INHERITED {
        command.env_remove(name);
    }
    let req = json!({
        "type": "control_request",
        "request_id": "rig-usage",
        "request": { "subtype": "get_usage" },
    });
    match rpc::probe(command, vec![req], PROBE_TIMEOUT, |msg| {
        msg.get("type").and_then(Value::as_str) == Some("control_response")
            && msg.pointer("/response/request_id").and_then(Value::as_str) == Some("rig-usage")
    }) {
        Ok(msg) => parse_claude_usage(&msg),
        Err(why) => format!("unavailable ({why})"),
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
    let mut parts: Vec<String> = body
        .get("subscription_type")
        .and_then(Value::as_str)
        .map(|plan| format!("plan {plan}"))
        .into_iter()
        .collect();
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
    let Ok(dir) = Dir::create(scratch_root) else {
        return "unavailable (could not create probe dir)".to_owned();
    };
    if let Some(auth) = crate::codex::signed_in().filter(|auth| auth.is_file()) {
        let _ = link(&auth, &dir.path().join("auth.json"));
    }
    let mut command = processes.launcher().command(program);
    command
        .current_dir(dir.path())
        .args([
            "app-server",
            "-c",
            "analytics.enabled=false",
            "-c",
            "check_for_update_on_startup=false",
        ])
        .env("CODEX_HOME", dir.path())
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENAI_BASE_URL");
    let requests = vec![
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": { "name": "rig", "version": env!("CARGO_PKG_VERSION") },
                "capabilities": {},
            },
        }),
        json!({ "method": "initialized" }),
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "account/rateLimits/read",
            "params": {},
        }),
    ];
    match rpc::probe(command, requests, PROBE_TIMEOUT, |msg| {
        msg.get("id").and_then(Value::as_u64) == Some(2)
    }) {
        Ok(msg) => parse_codex_usage(&msg),
        Err(why) => format!("unavailable ({why})"),
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
    let mut parts: Vec<String> = limits
        .get("planType")
        .and_then(Value::as_str)
        .map(|plan| format!("plan {plan}"))
        .into_iter()
        .collect();
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

/// Runs `program` with `args` and parses stdout as JSON.
fn probe_cli_json(processes: &Processes, program: &Path, args: &[&str]) -> Result<Value, String> {
    let mut command = processes.launcher().command(program);
    command
        .args(args)
        .env("MISE_OFFLINE", "1")
        .env("MUSE_NO_AUTO_UPDATE", "1")
        .env("GROK_DISABLE_AUTOUPDATER", "1");
    let output = Process::start(command, Spawn::default())
        .and_then(|process| process.output(Some(PROBE_TIMEOUT)))
        .map_err(|_| "timed out or failed to run".to_owned())?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{stdout} {stderr}"));
    }
    serde_json::from_str(&stdout).map_err(|_| "invalid response".to_owned())
}

/// Queries Muse's official `muse usage --json` CLI output.
fn probe_muse_usage(processes: &Processes, program: &Path) -> String {
    match probe_cli_json(processes, program, &["usage", "--json"]) {
        Ok(value) => parse_muse_usage(&value),
        Err(why) => format!("unavailable ({why})"),
    }
}

/// Formats `muse usage --json`.
pub(crate) fn parse_muse_usage(value: &Value) -> String {
    let Some(allowance) = value.get("allowance").filter(|v| !v.is_null()) else {
        let reason = value
            .get("reason")
            .and_then(Value::as_str)
            .map(|r| match r {
                "server_sent_no_usage" => "no usage data yet in this window",
                other => other,
            })
            .unwrap_or("not signed in");
        return format!("unavailable ({reason})");
    };
    let mut parts: Vec<String> = allowance
        .get("tier")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .into_iter()
        .collect();
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
    match probe_cli_json(processes, program, &["usage", "--json"]) {
        Ok(value) => parse_grok_usage(&value),
        Err(text) => format_grok_usage_error(&text),
    }
}

pub(crate) fn format_grok_usage_error(text: &str) -> String {
    let low = text.to_lowercase();
    if low.contains("/usage") || low.contains("can't show") {
        "unavailable (run `/usage` inside grok)".to_owned()
    } else if ["background", "daemon", "service", "leader"]
        .iter()
        .any(|word| low.contains(word))
    {
        "unavailable (Grok background service not running)".to_owned()
    } else {
        "unavailable (not signed in or unsupported CLI version)".to_owned()
    }
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
        let parts: Vec<String> = obj
            .iter()
            .filter_map(|(key, val)| {
                let pct = val
                    .get("usedPercent")
                    .or_else(|| val.get("used_percent"))
                    .or_else(|| val.get("utilization"))
                    .and_then(Value::as_f64)?;
                Some(format!("{key} {pct:.0}% used"))
            })
            .collect();
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
    let (days, hours, mins) = (delta / 86_400, (delta % 86_400) / 3600, (delta % 3600) / 60);
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m")
    }
}
