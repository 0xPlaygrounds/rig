//! `/bridge` commands (`status`, `usage`, `reset`) and background signals.
//!
//! `/bridge status` and `/bridge reset` read shared [`Bridge`] state directly.
//! `/bridge usage` probes each installed harness's official quota interface in
//! parallel with a short timeout. Background signals also synchronize runtime
//! context windows onto active agent [`Connection`]s.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::http::{HeaderValue, Method, Request, StatusCode, header};
use bevy_ecs::message::Messages;
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_harness::rig_core::completion::ContextUse;
use rig_harness::rig_core::http_client::{DynHttpClient, HttpClientExt};
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
    models: Option<Res<Models>>,
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
            let http = models.and_then(|models| models.0.transport().shared());
            let wake = wake.as_deref().cloned();
            let agent = args.agent;
            std::thread::spawn(move || {
                let text = collect_usage(&handle, http);
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
            let effort = warm
                .effort
                .map(|effort| format!(" [{}]", effort.as_str()))
                .unwrap_or_default();
            let ctx = format_ctx(warm.context_tokens, warm.context_limit);
            lines.push(format!(
                "- {}/{}{effort} ({} turn(s){ctx}, {relist})",
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

fn collect_usage(handle: &BridgeHandle, http: Option<DynHttpClient>) -> String {
    let snap = handle.bridge.snapshot();
    let scratch_root = handle.bridge.scratch_root().to_path_buf();
    let runtime = handle.bridge.runtime().clone();
    let results: Vec<(Vendor, String)> = std::thread::scope(|scope| {
        let jobs: Vec<_> = handle
            .installed
            .iter()
            .map(|entry| {
                let processes = handle.processes.clone();
                let scratch = scratch_root.clone();
                let http = http.clone();
                let runtime = runtime.clone();
                scope.spawn(move || {
                    let plan = match &entry.program {
                        None => "not installed".to_owned(),
                        Some(program) => probe_vendor_usage(
                            &processes,
                            &runtime,
                            http.as_ref(),
                            entry.vendor,
                            program,
                            &scratch,
                        ),
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
    runtime: &tokio::runtime::Handle,
    http: Option<&DynHttpClient>,
    vendor: Vendor,
    program: &Path,
    scratch_root: &Path,
) -> String {
    match vendor {
        Vendor::ClaudeCode => probe_claude_usage(processes, program),
        Vendor::Codex => probe_codex_usage(processes, program, scratch_root),
        Vendor::Muse => probe_muse_usage(processes, program),
        Vendor::GrokBuild => probe_grok_usage(processes, runtime, http, program),
        Vendor::Antigravity => probe_antigravity_usage(runtime, http),
        Vendor::Copilot => probe_copilot_usage(processes),
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

fn join_or(parts: Vec<String>, empty: &str) -> String {
    if parts.is_empty() {
        empty.to_owned()
    } else {
        parts.join(", ")
    }
}

fn reset_iso(ts: Option<&str>) -> String {
    ts.map(|ts| format!(" (resets {})", short_iso(ts)))
        .unwrap_or_default()
}

fn reset_rel(secs: Option<u64>) -> String {
    secs.map(|s| format!(" (resets in {})", format_relative_secs(s)))
        .unwrap_or_default()
}

fn read_json(path: Option<PathBuf>) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(path?).ok()?).ok()
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
            let reset = reset_iso(win.get("resets_at").and_then(Value::as_str));
            parts.push(format!("{label} {pct:.0}% used{reset}"));
        }
    }
    join_or(parts, "unavailable (no active quota windows)")
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
            "--disable",
            "daemon_auto_start",
            "-c",
            "analytics.enabled=false",
            "-c",
            "check_for_update_on_startup=false",
            "-c",
            "thread_unload_delay_secs=0",
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
            let reset = reset_rel(win.get("resetsAt").and_then(Value::as_u64));
            parts.push(format!("{label} {pct:.0}% used{reset}"));
        }
    }
    join_or(parts, "unavailable")
}

/// Runs `program` with `args` and `extra_env`, parsing stdout as JSON within
/// `timeout`. Returns `Err(None)` when the executable cannot be found.
fn probe_cli_json(
    processes: &Processes,
    program: &Path,
    args: &[&str],
    extra_env: &[(&str, &str)],
    timeout: Duration,
) -> Result<Value, Option<String>> {
    if timeout.is_zero() {
        return Err(Some("timed out".to_owned()));
    }
    let mut command = processes.launcher().command(program);
    command.args(args);
    for (key, value) in crate::harness::PROBE_ENV.iter().chain(extra_env) {
        command.env(key, value);
    }
    if let Ok(cwd) = std::env::current_dir() {
        crate::harness::clear_nested_tmpdir(&mut command, &cwd);
    }
    let process = match Process::start(command, Spawn::default()) {
        Ok(process) => process,
        Err(why) if why.kind() == std::io::ErrorKind::NotFound => return Err(None),
        Err(_) => return Err(Some("failed to run".to_owned())),
    };
    let output = process
        .output(Some(timeout))
        .map_err(|_| Some("timed out or failed to run".to_owned()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Some(format!("{stdout} {stderr}")));
    }
    serde_json::from_str(&stdout).map_err(|_| Some("invalid response".to_owned()))
}

/// Queries Muse's official `muse usage --json` CLI output.
fn probe_muse_usage(processes: &Processes, program: &Path) -> String {
    match probe_cli_json(processes, program, &["usage", "--json"], &[], PROBE_TIMEOUT) {
        Ok(value) => parse_muse_usage(&value),
        Err(Some(why)) => format!("unavailable ({why})"),
        Err(None) => "not installed".to_owned(),
    }
}

/// Formats `muse usage --json`.
pub(crate) fn parse_muse_usage(value: &Value) -> String {
    let Some(allowance) = value.get("allowance").filter(|v| !v.is_null()) else {
        let reason = match value.get("reason").and_then(Value::as_str) {
            Some("server_sent_no_usage") => "no usage data yet in this window",
            Some(other) => other,
            None => "not signed in",
        };
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
            let reset = reset_rel(
                win.get("resetsAtMs")
                    .and_then(Value::as_u64)
                    .map(|ms| ms / 1000),
            );
            parts.push(format!("{label} {pct:.0}% used{reset}"));
        }
    }
    join_or(parts, "unavailable")
}

enum FetchError {
    Unauthorized,
    Failed,
}

struct JsonProbe<'a> {
    method: Method,
    url: &'a str,
    bearer: &'a str,
    user_agent: Option<&'a str>,
    body: Option<&'static [u8]>,
    deadline: Instant,
}

/// Sends one HTTP JSON request over Rig's shared HTTP transport within a
/// single end-to-end `deadline`.
fn fetch_json(
    runtime: &tokio::runtime::Handle,
    http: &DynHttpClient,
    probe: JsonProbe<'_>,
) -> Result<Value, FetchError> {
    let remaining = probe.deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(FetchError::Failed);
    }
    let mut auth = HeaderValue::from_str(&format!("Bearer {}", probe.bearer))
        .map_err(|_| FetchError::Failed)?;
    auth.set_sensitive(true);
    let mut builder = Request::builder()
        .method(probe.method)
        .uri(probe.url)
        .header(header::AUTHORIZATION, auth)
        .header(header::ACCEPT, "application/json");
    if let Some(ua) = probe.user_agent {
        builder = builder.header(header::USER_AGENT, ua);
    }
    let payload = if let Some(bytes) = probe.body {
        builder = builder.header(header::CONTENT_TYPE, "application/json");
        Bytes::from_static(bytes)
    } else {
        Bytes::new()
    };
    let request = builder.body(payload).map_err(|_| FetchError::Failed)?;
    let http = http.clone();
    runtime.block_on(async move {
        tokio::time::timeout(remaining, async move {
            let response = http.send(request).await.map_err(|_| FetchError::Failed)?;
            let status = response.status();
            if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
                return Err(FetchError::Unauthorized);
            }
            if !status.is_success() {
                return Err(FetchError::Failed);
            }
            let bytes = response.into_body().await.map_err(|_| FetchError::Failed)?;
            serde_json::from_slice(&bytes).map_err(|_| FetchError::Failed)
        })
        .await
        .map_err(|_| FetchError::Failed)?
    })
}

/// Queries Grok's billing endpoint using the CLI's cached sign-in token,
/// falling back to `grok usage --json` within a single [`PROBE_TIMEOUT`] budget.
fn probe_grok_usage(
    processes: &Processes,
    runtime: &tokio::runtime::Handle,
    http: Option<&DynHttpClient>,
    program: &Path,
) -> String {
    const GROK_BILLING_URL: &str = "https://cli-chat-proxy.grok.com/v1/billing?format=credits";
    let deadline = Instant::now() + PROBE_TIMEOUT;
    if let Some(http) = http
        && let Some(token) = read_json(crate::grok::signed_in()).and_then(|v| grok_token(&v))
        && let Ok(value) = fetch_json(
            runtime,
            http,
            JsonProbe {
                method: Method::GET,
                url: GROK_BILLING_URL,
                bearer: &token,
                user_agent: None,
                body: None,
                deadline,
            },
        )
        && let Some(formatted) = parse_grok_billing(&value)
    {
        return formatted;
    }
    let remaining = deadline.saturating_duration_since(Instant::now());
    match probe_cli_json(processes, program, &["usage", "--json"], &[], remaining) {
        Ok(value) => parse_grok_usage(&value),
        Err(Some(text)) => format_grok_usage_error(&text),
        Err(None) => "not installed".to_owned(),
    }
}

/// Extracts the Grok Build bearer token from parsed `~/.grok/auth.json`.
pub(crate) fn grok_token(value: &Value) -> Option<String> {
    let key = |entry: &Value| {
        entry
            .get("key")
            .and_then(Value::as_str)
            .filter(|tok| !tok.is_empty())
            .map(str::to_owned)
    };
    if let Some(token) = key(value) {
        return Some(token);
    }
    let map = value.as_object()?;
    map.iter()
        .find_map(|(host, entry)| {
            let lower = host.to_ascii_lowercase();
            (lower.contains("x.ai") || lower.contains("grok"))
                .then(|| key(entry))
                .flatten()
        })
        .or_else(|| {
            (map.len() == 1)
                .then(|| map.values().find_map(key))
                .flatten()
        })
}

/// Formats Grok's `/v1/billing?format=credits` response.
pub(crate) fn parse_grok_billing(value: &Value) -> Option<String> {
    let config = value.get("config").unwrap_or(value);
    let pct = config
        .get("creditUsagePercent")
        .and_then(Value::as_f64)
        .or_else(|| {
            config
                .get("productUsage")
                .and_then(Value::as_array)?
                .iter()
                .find_map(|item| item.get("usagePercent").and_then(Value::as_f64))
        })?;
    let period = match config
        .pointer("/currentPeriod/type")
        .and_then(Value::as_str)
    {
        Some(kind) if kind.contains("WEEKLY") => "weekly",
        Some(kind) if kind.contains("MONTHLY") => "monthly",
        Some(kind) if kind.contains("DAILY") => "daily",
        _ => "period",
    };
    let reset = reset_iso(
        config
            .pointer("/currentPeriod/end")
            .or_else(|| config.get("billingPeriodEnd"))
            .and_then(Value::as_str),
    );
    Some(format!("{period} {pct:.0}% used{reset}"))
}

/// Queries Antigravity's quota endpoint using `agy`'s stored OAuth access token.
fn probe_antigravity_usage(
    runtime: &tokio::runtime::Handle,
    http: Option<&DynHttpClient>,
) -> String {
    let Some(http) = http else {
        return "unavailable (no HTTP client)".to_owned();
    };
    let Some((token, business)) =
        read_json(crate::antigravity::signed_in()).and_then(|v| antigravity_token(&v))
    else {
        return "unavailable (not signed in: run `agy`)".to_owned();
    };
    let hosts = if business {
        [
            "https://cloudcode-pa.googleapis.com",
            "https://daily-cloudcode-pa.googleapis.com",
        ]
    } else {
        [
            "https://daily-cloudcode-pa.googleapis.com",
            "https://cloudcode-pa.googleapis.com",
        ]
    };
    let ua = format!(
        "antigravity/hub/2.24.0 (aidev_client; os_type={}; arch={}; cl=963137146)",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    let deadline = Instant::now() + PROBE_TIMEOUT;
    for host in hosts {
        let url = format!("{host}/v1internal:fetchAvailableModels");
        match fetch_json(
            runtime,
            http,
            JsonProbe {
                method: Method::POST,
                url: &url,
                bearer: &token,
                user_agent: Some(&ua),
                body: Some(b"{}"),
                deadline,
            },
        ) {
            Ok(value) => {
                if let Some(formatted) = parse_antigravity_usage(&value) {
                    return formatted;
                }
            }
            Err(FetchError::Unauthorized) => break,
            Err(FetchError::Failed) => {}
        }
    }
    "unavailable (could not query Antigravity quota; run `agy` to refresh sign-in)".to_owned()
}

/// Extracts `(access_token, is_oauth_business)` from parsed Antigravity
/// `oauth_creds.json`.
pub(crate) fn antigravity_token(value: &Value) -> Option<(String, bool)> {
    let token = [
        value.pointer("/token/access_token"),
        value.get("access_token"),
    ]
    .into_iter()
    .flatten()
    .filter_map(Value::as_str)
    .find(|tok| !tok.is_empty())?
    .to_owned();
    let business = value.get("auth_method").and_then(Value::as_str) == Some("oauth-business");
    Some((token, business))
}

/// Summarizes Antigravity's `fetchAvailableModels` per-model `quotaInfo` by
/// model family (tightest remaining fraction per family).
pub(crate) fn parse_antigravity_usage(value: &Value) -> Option<String> {
    let models = value.get("models").and_then(Value::as_object)?;
    let (mut gemini, mut claude): (Option<(f64, Option<&str>)>, Option<(f64, Option<&str>)>) =
        (None, None);
    for (id, entry) in models {
        let Some(quota) = entry.get("quotaInfo") else {
            continue;
        };
        let reset = quota.get("resetTime").and_then(Value::as_str);
        // Proto3 JSON omits default `0.0`: a `quotaInfo` carrying `resetTime`
        // without `remainingFraction` means the model's quota is 100% exhausted.
        let rem = match quota.get("remainingFraction").and_then(Value::as_f64) {
            Some(rem) => rem,
            None if reset.is_some() => 0.0,
            None => continue,
        };
        let slot = if id.starts_with("gemini-") {
            &mut gemini
        } else if id.starts_with("claude-") {
            &mut claude
        } else {
            continue;
        };
        if slot.is_none_or(|(best, best_reset)| {
            rem < best || (rem == best && best_reset.is_none() && reset.is_some())
        }) {
            *slot = Some((rem, reset));
        }
    }
    let parts: Vec<String> = [("Gemini", gemini), ("Claude", claude)]
        .into_iter()
        .filter_map(|(label, slot)| {
            let (rem, reset) = slot?;
            let used = ((1.0 - rem.clamp(0.0, 1.0)) * 100.0).round();
            Some(format!("{label} {used:.0}% used{}", reset_iso(reset)))
        })
        .collect();
    (!parts.is_empty()).then(|| parts.join(", "))
}

/// Queries GitHub Copilot's subscription quota via `gh api --hostname github.com /copilot_internal/user`.
fn probe_copilot_usage(processes: &Processes) -> String {
    match probe_cli_json(
        processes,
        Path::new("gh"),
        &["api", "--hostname", "github.com", "/copilot_internal/user"],
        &[("GH_PROMPT_DISABLED", "1"), ("GH_NO_UPDATE_NOTIFIER", "1")],
        PROBE_TIMEOUT,
    ) {
        Ok(value) => parse_copilot_usage(&value),
        Err(None) => {
            "unavailable (install `gh` and run `gh auth login` for Copilot quota)".to_owned()
        }
        Err(Some(_)) => "unavailable (sign in with `gh auth login` for Copilot quota)".to_owned(),
    }
}

/// Formats GitHub's `/copilot_internal/user` quota response.
pub(crate) fn parse_copilot_usage(value: &Value) -> String {
    let mut parts: Vec<String> = value
        .get("copilot_plan")
        .and_then(Value::as_str)
        .map(|plan| format!("plan {plan}"))
        .into_iter()
        .collect();
    let reset = value
        .get("quota_reset_date")
        .or_else(|| value.get("quota_reset_date_utc"))
        .and_then(Value::as_str)
        .map(|ts| format!(" (resets {})", ts.get(..10).unwrap_or(ts)))
        .unwrap_or_default();
    if let Some(premium) = value.pointer("/quota_snapshots/premium_interactions") {
        if premium.get("unlimited").and_then(Value::as_bool) == Some(true) {
            parts.push("premium unlimited".to_owned());
        } else if let Some(pct_rem) = premium.get("percent_remaining").and_then(Value::as_f64) {
            let used = (100.0 - pct_rem.clamp(0.0, 100.0)).round();
            let count = match (
                premium.get("remaining").and_then(Value::as_i64),
                premium.get("entitlement").and_then(Value::as_i64),
            ) {
                (Some(rem), Some(total)) if total > 0 => format!(" [{}/{total} left]", rem.max(0)),
                _ => String::new(),
            };
            parts.push(format!("premium {used:.0}% used{count}{reset}"));
        }
    }
    join_or(parts, "unavailable")
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
