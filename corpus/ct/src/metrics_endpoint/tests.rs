use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

use rig_basics::approval::{ALLOW, ApprovalPlugin, ApprovalSystems, Approved};
use rig_core::completion::{Cost, Usage};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_core::tool::{ToolErrorKind, ToolOutput};
use rig_harness::ask::AskPlugin;
use rig_harness::harness_protocol::Invocation;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, answer_open, app_on, collect, first_agent, log_ended, run_until,
    scripted_models,
};
use serde_json::json;

use super::*;

/// The prompts, which no label may carry.
const FIRST: &str = "Probe the flux capacitor at /home/user/secret.txt";
const SECOND: &str = "Probe it once more, carefully";

/// One sample: its name, labels and value.
#[derive(Debug)]
struct Sample {
    name: String,
    labels: Vec<(String, String)>,
    value: f64,
}

/// Parses the Prometheus text format strictly: every sample's family is
/// declared by a `# TYPE` line before it, names and labels are well formed,
/// label values are quoted and escaped, values are numbers, and each
/// histogram's `+Inf` bucket equals its count.
fn parse(text: &str) -> Result<Vec<Sample>, String> {
    let name_ok = |name: &str| {
        let mut chars = name.chars();
        chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_' || c == ':')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
    };
    let mut types: Vec<(String, String)> = Vec::new();
    let mut samples = Vec::new();
    for line in text.lines().filter(|line| !line.is_empty()) {
        if let Some(comment) = line.strip_prefix("# ") {
            let mut words = comment.splitn(3, ' ');
            match (words.next(), words.next(), words.next()) {
                (Some("TYPE"), Some(name), Some(kind @ ("counter" | "gauge" | "histogram")))
                    if name_ok(name) =>
                {
                    types.push((name.to_owned(), kind.to_owned()));
                }
                (Some("HELP"), Some(name), Some(_)) if name_ok(name) => {}
                _ => return Err(format!("bad comment: {line}")),
            }
            continue;
        }
        let (series, value) = line.rsplit_once(' ').ok_or(format!("no value: {line}"))?;
        let value = match value {
            "+Inf" => f64::INFINITY,
            "-Inf" => f64::NEG_INFINITY,
            other => other.parse().map_err(|_| format!("bad value: {line}"))?,
        };
        let (name, mut rest) = series.split_once('{').unwrap_or((series, "}"));
        if !name_ok(name) {
            return Err(format!("bad name: {line}"));
        }
        let mut labels = Vec::new();
        while rest != "}" {
            let (label, after) = rest
                .split_once("=\"")
                .ok_or(format!("bad labels: {line}"))?;
            let mut chars = after.char_indices();
            let mut value = String::new();
            let end = loop {
                match chars.next() {
                    Some((_, '\\')) => match chars.next() {
                        Some((_, 'n')) => value.push('\n'),
                        Some((_, c @ ('\\' | '"'))) => value.push(c),
                        _ => return Err(format!("bad escape: {line}")),
                    },
                    Some((at, '"')) => break at,
                    Some((_, c)) => value.push(c),
                    None => return Err(format!("unterminated label: {line}")),
                }
            };
            if !name_ok(label.trim_start_matches(',')) {
                return Err(format!("bad label name: {line}"));
            }
            labels.push((label.trim_start_matches(',').to_owned(), value));
            rest = after.get(end + 1..).ok_or(format!("bad labels: {line}"))?;
        }
        let family = ["_bucket", "_sum", "_count"]
            .iter()
            .find_map(|suffix| {
                name.strip_suffix(suffix)
                    .filter(|base| types.iter().any(|(n, k)| n == base && k == "histogram"))
            })
            .unwrap_or(name);
        if !types.iter().any(|(declared, _)| declared == family) {
            return Err(format!("undeclared family: {line}"));
        }
        samples.push(Sample {
            name: name.to_owned(),
            labels,
            value,
        });
    }
    for (name, _) in types.iter().filter(|(_, kind)| kind == "histogram") {
        let bucket = format!("{name}_bucket");
        let count = format!("{name}_count");
        let inf: f64 = samples
            .iter()
            .filter(|s| s.name == bucket && s.labels.iter().any(|(l, v)| l == "le" && v == "+Inf"))
            .map(|s| s.value)
            .sum();
        let counted: f64 = samples
            .iter()
            .filter(|s| s.name == count)
            .map(|s| s.value)
            .sum();
        if (inf - counted).abs() > f64::EPSILON {
            return Err(format!(
                "{name}: +Inf bucket {inf} is not the count {counted}"
            ));
        }
    }
    Ok(samples)
}

/// The sum of the samples named `name` whose labels include `label`.
fn sum(samples: &[Sample], name: &str, label: Option<(&str, &str)>) -> f64 {
    samples
        .iter()
        .filter(|sample| sample.name == name)
        .filter(|sample| {
            label.is_none_or(|(l, v)| sample.labels.iter().any(|(sl, sv)| sl == l && sv == v))
        })
        .map(|sample| sample.value)
        .sum()
}

/// Scrapes `/metrics` with no frame of the app running, and how long it
/// took.
fn scrape(addr: SocketAddr) -> std::io::Result<(String, Duration)> {
    let began = Instant::now();
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n")?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok((response, began.elapsed()))
}

/// A reply's end with fixed token counts and a cost of $0.25.
fn finish(input: u64, output: u64, cached: u64) -> MockStreamEvent {
    let mut cost = Cost::default();
    cost.total = 0.25;
    let usage = Usage::new()
        .input_tokens(input)
        .output_tokens(output)
        .cached_input_tokens(cached)
        .cost(cost);
    MockStreamEvent::final_response(usage)
}

/// The probe tool: fails when called with `ok: false`.
fn probe(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    let ok = called.args.get("ok").and_then(serde_json::Value::as_bool) == Some(true);
    let result = if ok {
        ToolResult::success(ToolOutput::text("steady"))
    } else {
        ToolResult::failed(ToolExecutionError::new(
            ToolErrorKind::Other,
            "the probe broke",
        ))
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin("probe"), result);
    }
}

/// An ask rule: a risky probe needs the user's approval.
fn ask_rule(
    calls: Query<(Entity, &ToolRequest), (ReadyToStart, ToolCalls, Without<Approved>)>,
    mut commands: Commands,
) {
    for (call, request) in &calls {
        if request.call.function.arguments.get("ok") == Some(&json!(false)) {
            commands
                .entity(call)
                .queue(NeedsApproval::add("ask-rule", "A risky probe."));
        }
    }
}

/// Whether an approval was asked.
#[derive(Resource, Default)]
struct Asked(bool);

fn asked(_: On<Add<Ask>>, mut asked: ResMut<Asked>) {
    asked.0 = true;
}

/// What the scripted run showed: the scrape during the approval and how
/// long it took, the scrape after both turns with its head, and whether an
/// approval was open to answer.
struct Run {
    during: Vec<Sample>,
    took: Duration,
    head: String,
    after: Vec<Sample>,
    allowed: bool,
}

/// Two turns, two probes (the second failing, after an approval), four
/// scripted replies with fixed usage, scraped during and after.
fn scripted_run() -> Result<Run, String> {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "probe", json!({ "ok": true })),
            finish(100, 10, 40),
        ],
        vec![MockStreamEvent::text("steady"), finish(150, 5, 100)],
        vec![
            MockStreamEvent::tool_call("c2", "probe", json!({ "ok": false })),
            finish(200, 8, 150),
        ],
        vec![MockStreamEvent::text("it broke"), finish(260, 6, 200)],
    ]);
    let flags = vec![(METRICS.to_owned(), Some("127.0.0.1:0".to_owned()))];
    let invoked = Invoked {
        args: Invocation {
            flags,
            ..Invocation::default()
        },
        terminal: false,
    };
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(invoked)
        .insert_resource(scripted_models([("test/model", &model)]))
        .init_resource::<Diagnostics>()
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            AskPlugin,
            ApprovalPlugin,
            MetricsEndpointPlugin,
        ))
        .add_open_tool("probe", "Probe something.", ToolOptions::default(), probe)
        .add_systems(
            AgentPass,
            ask_rule.in_set(Stage::Gate).before(ApprovalSystems),
        )
        .init_resource::<Asked>()
        .add_observer(asked)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.world_mut().spawn(Answerer::default());
    app.finish();
    app.update();
    let addr = app
        .world()
        .get_resource::<MetricsEndpoint>()
        .ok_or("no endpoint")?
        .addr;
    let agent = first_agent(app.world_mut()).ok_or("no agent")?;
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    app.world_mut()
        .trigger(Deliver::new(agent, FIRST, DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    app.world_mut()
        .trigger(Deliver::new(agent, SECOND, DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| world.resource::<Asked>().0);
    app.update();

    // The approval is open and the app runs no frame while it is scraped.
    let (during, took) = scrape(addr).map_err(|failure| failure.to_string())?;
    let during = parse(during.split_once("\r\n\r\n").ok_or("no body")?.1)?;

    let allowed = answer_open(app.world_mut(), "Allow `probe`?", &Answer::choice(ALLOW));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    app.world_mut()
        .resource_mut::<Diagnostics>()
        .0
        .push_back(Logged {
            level: NoticeLevel::Error,
            target: "rig_demo".to_owned(),
            message: "a system failed".to_owned(),
            agent: None,
            plugin: Some("rig_demo::DemoPlugin".to_owned()),
        });
    app.update();

    let (after, _) = scrape(addr).map_err(|failure| failure.to_string())?;
    let (head, body) = after.split_once("\r\n\r\n").ok_or("no body")?;
    let after = parse(body)?;
    let head = head.to_owned();
    Ok(Run {
        during,
        took,
        head,
        after,
        allowed,
    })
}

#[test]
fn a_scrape_reports_the_scripted_run() {
    let run = scripted_run();
    assert!(run.is_ok(), "{:?}", run.as_ref().err());
    let Ok(Run {
        during,
        took,
        head,
        after,
        allowed,
    }) = run
    else {
        return;
    };
    let labels: Vec<&str> = after
        .iter()
        .flat_map(|s| s.labels.iter().map(|(_, v)| v.as_str()))
        .collect();

    assert!(
        head.starts_with("HTTP/1.1 200") && head.contains("version=0.0.4"),
        "{head}"
    );
    assert!(took < Duration::from_secs(1), "the scrape took {took:?}");
    assert!(allowed, "no approval was open");
    let waiting_during = sum(&during, registry::APPROVALS_WAITING, None);
    let busy_during = sum(&during, registry::AGENTS_BUSY, None);
    assert_eq!((waiting_during, busy_during), (1.0, 1.0));
    let model = Some(("model", "test/model"));
    assert_eq!(sum(&after, registry::MODEL_REQUESTS, model), 4.0);
    assert_eq!(sum(&after, registry::INPUT_TOKENS, model), 710.0);
    assert_eq!(sum(&after, registry::OUTPUT_TOKENS, model), 29.0);
    assert_eq!(sum(&after, registry::CACHED_TOKENS, model), 490.0);
    assert_eq!(sum(&after, registry::COST, model), 1.0);
    let tool = Some(("tool", "probe"));
    assert_eq!(sum(&after, registry::TOOL_CALLS, tool), 2.0);
    assert_eq!(sum(&after, registry::TOOL_ERRORS, tool), 1.0);
    assert_eq!(
        sum(&after, "rig_tool_call_duration_seconds_count", tool),
        2.0
    );
    assert_eq!(sum(&after, "rig_turn_duration_seconds_count", None), 2.0);
    assert_eq!(
        sum(&after, "rig_model_time_to_first_token_seconds_count", model),
        4.0
    );
    assert_eq!(sum(&after, registry::APPROVALS_WAITING, None), 0.0);
    assert_eq!(sum(&after, registry::AGENTS_BUSY, None), 0.0);
    let plugin = Some(("plugin", "rig_demo::DemoPlugin"));
    assert_eq!(sum(&after, registry::PLUGIN_ERRORS, plugin), 1.0);
    assert_eq!(
        after
            .iter()
            .filter(|s| s.name == registry::SINCE_ACTIVITY)
            .count(),
        1
    );
    let leaked =
        |label: &&str| label.contains(FIRST) || label.contains(SECOND) || label.contains("secret");
    assert!(!labels.iter().any(leaked), "{labels:?}");
}

#[test]
fn a_public_address_needs_the_opt_in() {
    let (mut app, _wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml("[metrics]\naddr = \"0.0.0.0:0\""))
        .add_plugins((AgentPlugin, SessionLogPlugin, MetricsEndpointPlugin));
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let refused = notices.iter().any(|notice| {
        notice.level == NoticeLevel::Error && notice.text.contains("metrics_public = true")
    });
    assert!(refused, "{notices:?}");
    assert!(app.world().get_resource::<MetricsEndpoint>().is_none());
}
