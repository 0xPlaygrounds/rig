use std::sync::Arc;

use rig_core::completion::{CompletionRequest, CompletionResponse, Usage};
use rig_core::effect::{EffectId, HandlerKey};
use rig_core::message::{Origin, Reasoning, ToolFunction, ToolName};
use rig_core::tool::{ExitStatus, ToolContext};
use rig_harness::harness_protocol::Invocation;
use rig_harness::rig_ecs::spawn::spawn_root;
use serde_json::{Value, json};

use super::*;
use crate::tests::stand_in::{SESSION_URL, StandIn};

pub(crate) mod stand_in;

/// What the command line gives.
fn invoked(flags: &[(&str, &str)]) -> Invoked {
    Invoked {
        args: Invocation {
            flags: flags
                .iter()
                .map(|(name, value)| ((*name).to_owned(), Some((*value).to_owned())))
                .collect(),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// No variable set.
fn no_env(_: &str) -> Option<String> {
    None
}

#[test]
fn without_a_token_or_a_target_it_is_off() {
    let token = Secrets::from_iter([(TOKEN_SECRET, "lin_api_key")]);
    let issue = invoked(&[(ISSUE_FLAG, "ENG-1")]);
    assert!(configure(Some(&issue), None, None, &no_env).is_none());
    assert!(configure(None, Some(&token), None, &no_env).is_none());
    let empty = Secrets::from_iter([(TOKEN_SECRET, " ")]);
    assert!(configure(Some(&issue), Some(&empty), None, &no_env).is_none());
    let half = Secrets::from_iter([(CLIENT_ID_SECRET, "client")]);
    assert!(configure(Some(&issue), Some(&half), None, &no_env).is_none());
    assert!(configure(Some(&issue), Some(&token), None, &no_env).is_some());
}

#[test]
fn options_beat_variables_and_a_session_beats_an_issue() {
    let token = Secrets::from_iter([(TOKEN_SECRET, "lin_api_key")]);
    let target = |invoked: &Invoked, env: &dyn Fn(&str) -> Option<String>| {
        configure(Some(invoked), Some(&token), None, env).map(|setup| setup.config.target)
    };
    let env = |name: &str| match name {
        ISSUE_ENV => Some("ENV-1".to_owned()),
        SESSION_ENV => Some("env-session".to_owned()),
        _ => None,
    };
    let issue_only = |name: &str| (name == ISSUE_ENV).then(|| "ENV-1".to_owned());
    assert_eq!(
        target(&invoked(&[(ISSUE_FLAG, "ENG-1")]), &env),
        Some(Target::Issue("ENG-1".to_owned()))
    );
    assert_eq!(
        target(
            &invoked(&[(ISSUE_FLAG, "ENG-1"), (SESSION_FLAG, "flag-session")]),
            &env
        ),
        Some(Target::Session("flag-session".to_owned()))
    );
    assert_eq!(
        target(&invoked(&[]), &env),
        Some(Target::Session("env-session".to_owned()))
    );
    assert_eq!(
        target(&invoked(&[]), &issue_only),
        Some(Target::Issue("ENV-1".to_owned()))
    );
}

#[test]
fn an_access_token_beats_client_credentials_which_take_the_scope_setting() {
    let issue = invoked(&[(ISSUE_FLAG, "ENG-1")]);
    let both = Secrets::from_iter([
        (TOKEN_SECRET, "token"),
        (CLIENT_ID_SECRET, "client"),
        (CLIENT_SECRET_SECRET, "secret"),
    ]);
    let credentials = |secrets: &Secrets, settings: Option<&LinearSettings>| {
        configure(Some(&issue), Some(secrets), settings, &no_env)
            .map(|setup| setup.config.credentials)
    };
    assert_eq!(
        credentials(&both, None),
        Some(Credentials::Token("token".to_owned()))
    );
    let client = Secrets::from_iter([
        (CLIENT_ID_SECRET, "client"),
        (CLIENT_SECRET_SECRET, "secret"),
    ]);
    let settings = LinearSettings {
        scope: "read".to_owned(),
        ..LinearSettings::default()
    };
    assert_eq!(
        credentials(&client, Some(&settings)),
        Some(Credentials::Client {
            id: "client".to_owned(),
            secret: "secret".to_owned(),
            scope: "read".to_owned(),
        })
    );
    assert_eq!(
        credentials(&client, None),
        Some(Credentials::Client {
            id: "client".to_owned(),
            secret: "secret".to_owned(),
            scope: DEFAULT_SCOPE.to_owned(),
        })
    );
}

#[test]
fn the_plan_file_replaces_the_settings_plan_and_a_bad_one_does_not() {
    let directory = std::env::temp_dir().join(format!("rig-linear-plan-{}", std::process::id()));
    std::fs::create_dir_all(&directory).ok();
    let good = directory.join("plan.toml");
    let bad = directory.join("bad.toml");
    std::fs::write(
        &good,
        "[[plan]]\ncontent = \"Triage\"\nskill = \"triage\"\n\n[[plan]]\ncontent = \"Ship\"\ncommand = \"gh pr create\"\n",
    )
    .ok();
    std::fs::write(&bad, "[[plan]]\ncontents = \"typo\"\n").ok();
    let token = Secrets::from_iter([(TOKEN_SECRET, "lin_api_key")]);
    let issue = invoked(&[(ISSUE_FLAG, "ENG-1")]);
    let settings = LinearSettings {
        plan: vec![PlanStep {
            content: "From settings".to_owned(),
            ..PlanStep::default()
        }],
        ..LinearSettings::default()
    };
    let plan = |path: &std::path::Path| {
        let path = path.display().to_string();
        let env = move |name: &str| (name == PLAN_ENV).then(|| path.clone());
        configure(Some(&issue), Some(&token), Some(&settings), &env).map(|setup| setup.plan)
    };
    assert_eq!(
        plan(&good),
        Some(vec![
            PlanStep {
                content: "Triage".to_owned(),
                skill: Some("triage".to_owned()),
                command: None,
            },
            PlanStep {
                content: "Ship".to_owned(),
                skill: None,
                command: Some("gh pr create".to_owned()),
            },
        ])
    );
    assert_eq!(plan(&bad), Some(settings.plan.clone()));
    assert_eq!(
        plan(&directory.join("missing.toml")),
        Some(settings.plan.clone())
    );
    std::fs::remove_dir_all(&directory).ok();
}

#[test]
fn the_settings_read_from_the_linear_table() {
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(
        "[linear]\napi_url = \"http://127.0.0.1:1/graphql\"\nheartbeat_minutes = 3\n\n\
         [[linear.plan]]\ncontent = \"Triage\"\nskill = \"triage\"\n",
    ))
    .add_plugins(LinearPlugin);
    let settings = app.world().get_resource::<LinearSettings>().cloned();
    assert_eq!(
        settings,
        Some(LinearSettings {
            api_url: "http://127.0.0.1:1/graphql".to_owned(),
            heartbeat_minutes: 3,
            scope: DEFAULT_SCOPE.to_owned(),
            plan: vec![PlanStep {
                content: "Triage".to_owned(),
                skill: Some("triage".to_owned()),
                command: None,
            }],
        })
    );
}

#[test]
fn off_it_declares_its_options_and_secrets_but_adds_no_tool_state_or_section() {
    let mut app = App::new();
    app.insert_resource(invoked(&[(ISSUE_FLAG, "ENG-1")]))
        .insert_resource(Secrets::default())
        .add_plugins(LinearPlugin);
    app.finish();
    app.update();
    assert!(!app.world().contains_resource::<Linear>());
    let mut flags = app.world_mut().query::<&Flag>();
    let names: Vec<&str> = flags.iter(app.world()).map(|flag| flag.name).collect();
    assert_eq!(names, [ISSUE_FLAG, SESSION_FLAG]);
    let mut secrets = app.world_mut().query::<&Secret>();
    let names: Vec<&str> = secrets
        .iter(app.world())
        .map(|secret| secret.name)
        .collect();
    assert_eq!(
        names,
        [TOKEN_SECRET, CLIENT_ID_SECRET, CLIENT_SECRET_SECRET]
    );
    let mut tools = app.world_mut().query::<&ToolDef>();
    assert_eq!(tools.iter(app.world()).count(), 0);
    let mut sections = app.world_mut().query::<&PromptSection>();
    assert_eq!(sections.iter(app.world()).count(), 0);
}

#[test]
fn a_reported_outcome_maps_to_its_activity_and_plan_end() {
    let summary = || "Summary.".to_owned();
    assert_eq!(
        outcome(OutcomeStatus::Done, summary()),
        (Content::Response(summary()), End::Done)
    );
    assert_eq!(
        outcome(OutcomeStatus::GaveUp, summary()),
        (Content::Error(summary()), End::Failed)
    );
    assert_eq!(
        outcome(OutcomeStatus::NeedsInput, summary()),
        (Content::Elicitation(summary()), End::NeedsInput)
    );
    let args: Result<OutcomeArgs, _> =
        serde_json::from_value(json!({ "status": "gave_up", "summary": "No." }));
    assert!(args.is_ok_and(|args| args.status == OutcomeStatus::GaveUp));
}

#[test]
fn without_a_report_the_end_is_the_stop_the_failure_or_the_last_reply() {
    let stopped = (Content::Error(STOPPED.to_owned()), End::Failed);
    assert_eq!(fallback(true, Some("boom"), Some("Hi.")), stopped);
    assert_eq!(
        fallback(false, Some("boom"), Some("Hi.")),
        (Content::Error("boom".to_owned()), End::Failed)
    );
    assert_eq!(
        fallback(false, None, Some("Hi.")),
        (Content::Response("Hi.".to_owned()), End::Done)
    );
    assert_eq!(fallback(false, None, None), stopped);
}

#[test]
fn the_prompt_section_names_the_issue_and_the_tool() {
    let text = section(Some("ENG-1"));
    assert!(text.contains("Linear issue ENG-1"), "{text}");
    assert!(text.contains(OUTCOME_TOOL), "{text}");
    assert!(text.contains("do not post a comment"), "{text}");
    assert!(section(None).contains("a Linear issue"));
}

/// An active app on the stand-in at `url`, for the issue `ENG-1`, with a
/// main agent `main`, its plan `plan` (TOML) and requests sent as soon as
/// they may.
fn active_app(url: &str, plan: &str, runtime: &tokio::runtime::Runtime) -> App {
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(&format!(
        "[linear]\napi_url = \"{url}\"\n{plan}"
    )))
    .insert_resource(Secrets::from_iter([(TOKEN_SECRET, "lin_api_key")]))
    .insert_resource(invoked(&[(ISSUE_FLAG, "ENG-1")]))
    .insert_resource(TokioRuntime(runtime.handle().clone()))
    .add_message::<EffectRecorded>()
    .add_plugins(LinearPlugin);
    if let Some(mut linear) = app.world_mut().get_resource_mut::<Linear>()
        && let Some((config, ..)) = &mut linear.pending
    {
        config.spacing = Duration::from_millis(5);
    }
    app.finish();
    spawn_root(app.world_mut(), AgentId("main".to_owned()));
    app
}

/// Updates `app` until the session's link is known, for at most 10 s.
fn wait_until_started(app: &mut App) {
    for _ in 0..1000 {
        app.update();
        if app
            .world()
            .get_resource::<Linear>()
            .is_some_and(|linear| linear.session_url.is_some())
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A record of the agent `scope`'s effect `id`.
fn record(id: u64, scope: &str, kind: EffectKind, outcome: Answered) -> EffectRecord {
    let mut record = EffectRecord::new(
        EffectId::from_raw(id),
        HandlerKey::from("handler"),
        kind,
        Ok(outcome),
    );
    record.scope = Some(scope.to_owned().into());
    record
}

/// A turn's model reply of `content`.
fn reply(id: u64, scope: &str, content: Vec<AssistantContent>) -> EffectRecord {
    let response = CompletionResponse::new(
        content,
        Usage::default(),
        Origin::new("anthropic", "test", "model"),
        Value::Null,
    );
    let kind = EffectKind::Completion {
        request: Arc::new(CompletionRequest::new("Go.")),
        stream: false,
    };
    let mut record = record(id, scope, kind, Answered::Completion(Arc::new(response)));
    record.purpose = Some(rig_harness::rig_ecs::turn::PURPOSE.into());
    record
}

/// A settled call of `tool` with `args` that output `output`, exiting with
/// `code` when given.
fn call(
    id: u64,
    scope: &str,
    tool: &str,
    args: Value,
    output: &str,
    code: Option<i32>,
) -> EffectRecord {
    let kind = EffectKind::ToolCall {
        name: tool.to_owned(),
        args: args.to_string(),
    };
    let result = ToolResult::success(ToolOutput::text(output));
    let mut record = record(id, scope, kind, Answered::ToolResult { result });
    if let Some(code) = code {
        let mut context = ToolContext::default();
        context.insert_result(ExitStatus::Exited(code)).ok();
        record.published = Some(context.into_result_context());
    }
    record
}

/// Sends `records` through `app` as the kernel would.
fn recorded(app: &mut App, records: Vec<EffectRecord>) {
    for record in records {
        app.world_mut().write_message(EffectRecorded(record));
    }
    app.update();
}

/// The `variables` of each request after the first `skip`.
fn variables(linear: &StandIn, skip: usize) -> Vec<Value> {
    linear
        .bodies()
        .into_iter()
        .skip(skip)
        .map(|body| body.get("variables").cloned().unwrap_or_default())
        .collect()
}

/// The plan's statuses as sent.
fn plan(statuses: &[(&str, &str)]) -> Value {
    Value::Array(
        statuses
            .iter()
            .map(|(content, status)| json!({ "content": content, "status": status }))
            .collect(),
    )
}

const PLAN: &str = "[[linear.plan]]\ncontent = \"Triage\"\nskill = \"triage\"\n\n\
                    [[linear.plan]]\ncontent = \"Ship\"\ncommand = \"gh pr create\"\n";

#[test]
fn a_run_shows_its_plan_notes_calls_links_and_last_reply() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::start();
    let mut app = active_app(&linear.url, PLAN, &runtime);
    let thinking = AssistantContent::Reasoning(Reasoning::new("Hmm."));
    recorded(
        &mut app,
        vec![
            reply(
                1,
                "main",
                vec![thinking, AssistantContent::text("Looking.")],
            ),
            call(
                2,
                "main",
                "read",
                json!({ "path": "/repo/skills/triage/SKILL.md" }),
                "# Triage",
                None,
            ),
            call(
                3,
                "helper",
                "shell",
                json!({ "command": "rm -rf build" }),
                "",
                Some(0),
            ),
            call(
                4,
                "main",
                "shell",
                json!({ "command": "gh pr create --fill" }),
                "https://github.com/acme/widgets/pull/7\n[exit code 0]",
                Some(0),
            ),
            call(
                5,
                "main",
                "shell",
                json!({ "command": "cargo test" }),
                "1 failed\n[exit code 101]",
                Some(101),
            ),
            reply(
                6,
                "main",
                vec![AssistantContent::text("Opened the pull request.")],
            ),
        ],
    );
    wait_until_started(&mut app);
    let mut sections = app.world_mut().query::<&PromptSection>();
    let texts: Vec<String> = sections
        .iter(app.world())
        .map(|section| section.text.clone())
        .collect();
    assert_eq!(texts, [section(Some("ENG-1"))]);
    let mut tools = app.world_mut().query::<&ToolDef>();
    let names: Vec<String> = tools
        .iter(app.world())
        .map(|tool| tool.0.name.to_string())
        .collect();
    assert_eq!(names, [OUTCOME_TOOL]);
    // The exit's plan would replace the one still queued.
    for _ in 0..1000 {
        if linear.requests().len() >= 9 {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    app.world_mut().write_message(AppExit::Success);
    app.update();
    let sent = variables(&linear, 2);
    let thought =
        |body: &str| json!({ "id": "session-1", "content": { "type": "thought", "body": body } });
    let expected = vec![
        json!({ "id": "session-1", "plan": plan(&[("Triage", "pending"), ("Ship", "pending")]) }),
        json!({ "id": "session-1", "content": { "type": "thought", "body": "Hmm." }, "ephemeral": true }),
        thought("Looking."),
        json!({ "id": "session-1", "plan": plan(&[("Triage", "completed"), ("Ship", "inProgress")]) }),
        json!({
            "id": "session-1",
            "content": {
                "type": "action",
                "action": "Ran",
                "parameter": "cargo test",
                "result": "exit code 101\n1 failed\n[exit code 101]",
            },
            "ephemeral": true,
        }),
        thought("Opened the pull request."),
        json!({ "id": "session-1" }),
        json!({ "id": "session-1", "content": { "type": "response", "body": "Opened the pull request." } }),
        json!({ "id": "session-1", "plan": plan(&[("Triage", "completed"), ("Ship", "completed")]) }),
    ];
    assert_eq!(sent, expected, "{:#?}", linear.bodies());
    let link_query = linear
        .bodies()
        .get(8)
        .map(|body| body.get("query").cloned().unwrap_or_default());
    assert_eq!(
        link_query,
        Some(json!(
            "mutation UpdateSession($id: String!) { agentSessionUpdate(id: $id, input: { \
             addedExternalUrls: [{ url: \"https://github.com/acme/widgets/pull/7\", label: \
             \"Pull request #7\" }] }) { success } }"
        ))
    );
    let mut links = app.world_mut().query::<&SessionLink>();
    let mut published: Vec<SessionLink> = links.iter(app.world()).cloned().collect();
    published.sort_by(|a, b| a.label.cmp(&b.label));
    assert_eq!(
        published,
        [
            SessionLink::new(SESSION_LINK_LABEL, SESSION_URL),
            SessionLink::new("Pull request #7", "https://github.com/acme/widgets/pull/7"),
        ]
    );
}

/// The tool's entity.
fn outcome_tool(app: &mut App) -> Option<Entity> {
    let mut tools = app.world_mut().query::<(Entity, &ToolDef)>();
    tools
        .iter(app.world())
        .find(|(_, tool)| tool.0.name.as_str() == OUTCOME_TOOL)
        .map(|(entity, _)| entity)
}

/// Calls the outcome tool as `agent` with `args`; the call's entity.
fn call_outcome(app: &mut App, agent: Entity, args: OutcomeArgs) -> Option<Entity> {
    let tool = outcome_tool(app)?;
    let id = app.world().get::<AgentId>(agent)?.clone();
    let call = app.world_mut().spawn(Decision::default()).id();
    let function = ToolFunction::new(ToolName::new(OUTCOME_TOOL).ok()?, json!({}));
    app.world_mut().trigger(ToolCalled {
        entity: tool,
        call,
        agent,
        agent_id: id,
        effect: EffectId::from_raw(99),
        parent: None,
        tool_call: ToolCall::from_wire("call-1", function),
        args,
    });
    app.update();
    Some(call)
}

#[test]
fn a_reported_outcome_posts_at_once_and_mutes_the_rest_of_the_turn() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::start();
    let mut app = active_app(&linear.url, PLAN, &runtime);
    wait_until_started(&mut app);
    let Some(main) = main_agent_of(&mut app) else {
        return;
    };
    let args = OutcomeArgs {
        status: OutcomeStatus::GaveUp,
        summary: " The tests need a database. ".to_owned(),
    };
    let call = call_outcome(&mut app, main, args);
    let answered = call
        .and_then(|call| app.world().get::<Decision>(call))
        .and_then(Decision::get)
        .is_some_and(|outcome| matches!(&outcome.0, Ok(Answered::ToolResult { result }) if result.is_success()));
    assert!(answered);
    recorded(
        &mut app,
        vec![reply(1, "main", vec![AssistantContent::text("Bye.")])],
    );
    app.world_mut().write_message(AppExit::Success);
    app.update();
    let sent = variables(&linear, 3);
    assert_eq!(
        sent,
        [
            json!({ "id": "session-1", "content": { "type": "error", "body": "The tests need a database." } }),
            json!({ "id": "session-1", "plan": plan(&[("Triage", "canceled"), ("Ship", "canceled")]) }),
        ],
        "{:#?}",
        linear.bodies()
    );
}

#[test]
fn only_the_main_agent_may_report_the_outcome() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::start();
    let mut app = active_app(&linear.url, "", &runtime);
    wait_until_started(&mut app);
    // Not the main agent: it has no `Agent` of its own.
    let helper = app.world_mut().spawn(AgentId("helper".to_owned())).id();
    let args = OutcomeArgs {
        status: OutcomeStatus::Done,
        summary: "Done.".to_owned(),
    };
    let call = call_outcome(&mut app, helper, args);
    let refused = call
        .and_then(|call| app.world().get::<Decision>(call))
        .and_then(Decision::get)
        .is_some_and(|outcome| matches!(&outcome.0, Ok(Answered::ToolResult { result }) if result.is_refused()));
    assert!(refused);
    assert!(
        app.world()
            .get_resource::<Linear>()
            .is_some_and(|linear| !linear.reported)
    );
}

#[test]
fn a_run_stopped_mid_turn_ends_as_stopped() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::start();
    let mut app = active_app(&linear.url, "", &runtime);
    wait_until_started(&mut app);
    recorded(
        &mut app,
        vec![reply(1, "main", vec![AssistantContent::text("Working.")])],
    );
    app.insert_resource(ExitReason::Signal);
    app.world_mut().write_message(AppExit::from_code(130));
    app.update();
    let last = linear.bodies().last().cloned().unwrap_or_default();
    assert_eq!(
        last.pointer("/variables/content"),
        Some(&json!({ "type": "error", "body": STOPPED }))
    );
}

/// The app's main agent.
fn main_agent_of(app: &mut App) -> Option<Entity> {
    let mut agents = app.world_mut().query::<(Entity, &AgentId)>();
    agents
        .iter(app.world())
        .find(|(_, id)| id.0 == "main")
        .map(|(entity, _)| entity)
}
