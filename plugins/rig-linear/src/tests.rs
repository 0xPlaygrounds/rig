use std::sync::Arc;

use rig_core::completion::{CompletionRequest, CompletionResponse, Usage};
use rig_core::effect::{EffectId, HandlerKey};
use rig_core::message::{Origin, Reasoning, ToolFunction, ToolName};
use rig_core::tool::{ExitStatus, ToolContext};
use rig_harness::harness_protocol::Invocation;
use rig_harness::rig_ecs::spawn::spawn_root;
use serde_json::{Value, json};

use super::*;
use crate::tests::stand_in::{SESSION_ID, SESSION_URL, StandIn, WEBHOOK_SECRET};

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
fn config_precedence_for_target_credentials_and_claim() {
    let issue = || Target::Issue("ENG-1".to_owned());
    let env = |name: &str| match name {
        ISSUE_ENV => Some("ENV-1".to_owned()),
        SESSION_ENV => Some("env-session".to_owned()),
        _ => None,
    };
    let issue_only = |name: &str| (name == ISSUE_ENV).then(|| "ENV-1".to_owned());

    // Target: flags beat env, session beats issue, blank is ignored.
    assert_eq!(target(None, &no_env), None);
    assert_eq!(target(Some(&invoked(&[(ISSUE_FLAG, " ")])), &no_env), None);
    assert_eq!(
        target(Some(&invoked(&[(ISSUE_FLAG, "ENG-1")])), &env),
        Some(Target::Issue("ENG-1".to_owned()))
    );
    assert_eq!(
        target(
            Some(&invoked(&[
                (ISSUE_FLAG, "ENG-1"),
                (SESSION_FLAG, "flag-session")
            ])),
            &env
        ),
        Some(Target::Session("flag-session".to_owned()))
    );
    assert_eq!(
        target(Some(&invoked(&[])), &env),
        Some(Target::Session("env-session".to_owned()))
    );
    assert_eq!(
        target(Some(&invoked(&[])), &issue_only),
        Some(Target::Issue("ENV-1".to_owned()))
    );

    // Credentials: missing/half-configured is off, token beats client creds.
    assert!(configure(issue(), false, None, None, &no_env).is_none());
    let half = Secrets::from_iter([(CLIENT_ID_SECRET, "client")]);
    assert!(configure(issue(), false, Some(&half), None, &no_env).is_none());

    let both = Secrets::from_iter([
        (TOKEN_SECRET, "lin_api_token_val"),
        (CLIENT_ID_SECRET, "client"),
        (CLIENT_SECRET_SECRET, "client_secret_val"),
    ]);
    let setup = configure(issue(), true, Some(&both), None, &no_env);
    assert!(setup.is_some());
    if let Some(setup) = setup {
        assert_eq!(
            setup.config.credentials,
            Credentials::Token("lin_api_token_val".to_owned())
        );
        assert!(setup.config.claim);
        // No webhook port unless an address is given.
        assert_eq!(setup.config.listen, None);
        // Both supplied secrets are redacted regardless of selected auth mode.
        assert_eq!(
            setup
                .redactor
                .text("tok=lin_api_token_val sec=client_secret_val"),
            "tok=[REDACTED] sec=[REDACTED]"
        );
    }

    // Webhook port: the variable beats the setting, and it needs the secret.
    let listening = LinearSettings {
        listen: Some("0.0.0.0:1".to_owned()),
        ..LinearSettings::default()
    };
    let listen_env = |name: &str| (name == LISTEN_ENV).then(|| "0.0.0.0:2".to_owned());
    let listen = |secrets: &Secrets, env: &dyn Fn(&str) -> Option<String>| {
        configure(issue(), false, Some(secrets), Some(&listening), env)
            .and_then(|setup| setup.config.listen)
    };
    assert_eq!(listen(&both, &listen_env), None);
    let signed = Secrets::from_iter([
        (TOKEN_SECRET, "lin_api_token_val"),
        (WEBHOOK_SECRET_SECRET, "webhook_secret_val"),
    ]);
    assert_eq!(
        listen(&signed, &no_env),
        Some(Listen {
            addr: "0.0.0.0:1".to_owned(),
            secret: "webhook_secret_val".to_owned(),
        })
    );
    assert_eq!(
        listen(&signed, &listen_env).map(|listen| listen.addr),
        Some("0.0.0.0:2".to_owned())
    );
    assert_eq!(
        configure(issue(), false, Some(&signed), None, &no_env)
            .map(|setup| setup.redactor.text("hook=webhook_secret_val")),
        Some("hook=[REDACTED]".to_owned())
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
        configure(issue(), false, Some(&client), Some(&settings), &no_env)
            .map(|setup| setup.config.credentials),
        Some(Credentials::Client {
            id: "client".to_owned(),
            secret: "secret".to_owned(),
            scope: "read".to_owned(),
        })
    );

    // Claim: flag or truthy env var.
    let claim_env =
        |value: &'static str| move |name: &str| (name == CLAIM_ENV).then(|| value.to_owned());
    assert!(claims(Some(&invoked(&[(CLAIM_FLAG, "")])), &no_env));
    for on in ["1", "true", "TRUE", " true "] {
        assert!(claims(None, &claim_env(on)), "{on}");
    }
    for off in ["", "0", "false", "yes"] {
        assert!(!claims(None, &claim_env(off)), "{off}");
    }
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
        let issue = Target::Issue("ENG-1".to_owned());
        configure(issue, false, Some(&token), Some(&settings), &env).map(|setup| setup.plan)
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
fn with_a_target_but_no_token_it_asks_for_no_secret_and_adds_no_tool_state_or_section() {
    let mut app = App::new();
    app.insert_resource(invoked(&[(ISSUE_FLAG, "ENG-1")]))
        .insert_resource(Secrets::default())
        .add_plugins(LinearPlugin);
    app.finish();
    app.update();
    assert!(!app.world().contains_resource::<Linear>());
    let mut flags = app.world_mut().query::<&Flag>();
    let names: Vec<&str> = flags.iter(app.world()).map(|flag| flag.name).collect();
    assert_eq!(names, [ISSUE_FLAG, SESSION_FLAG, CLAIM_FLAG]);
    let mut secrets = app.world_mut().query::<&Secret>();
    assert_eq!(secrets.iter(app.world()).count(), 0);
    let mut tools = app.world_mut().query::<&ToolDef>();
    assert_eq!(tools.iter(app.world()).count(), 0);
    let mut sections = app.world_mut().query::<&PromptSection>();
    assert_eq!(sections.iter(app.world()).count(), 0);
}

#[test]
fn a_question_offers_its_options_trimmed_without_blanks_or_repeats() {
    let options = [" sqlite ", "", "postgres", "sqlite", "  "].map(str::to_owned);
    let long = "x".repeat(300);
    let redactor = Redactor::default();
    let (content, _) = outcome(
        OutcomeStatus::NeedsInput,
        "Which?".to_owned(),
        &[options.as_slice(), &[long]].concat(),
        &redactor,
    );
    let options = match content {
        Content::Elicitation { options, .. } => options,
        _ => Vec::new(),
    };
    assert_eq!(
        options.get(..2),
        Some(["sqlite", "postgres"].map(str::to_owned).as_slice())
    );
    assert_eq!(options.len(), 3);
    assert!(
        options
            .get(2)
            .is_some_and(|long| long.chars().count() == 256)
    );
    let many: Vec<String> = (0..150).map(|at| at.to_string()).collect();
    let (content, _) = outcome(
        OutcomeStatus::NeedsInput,
        "Which?".to_owned(),
        &many,
        &redactor,
    );
    assert!(matches!(content, Content::Elicitation { options, .. } if options.len() == 100));
}

/// An active app on the stand-in at `url`, for `issue`, with a main agent
/// `main`, its plan `plan` (TOML), requests sent as soon as they may and
/// no webhook port ([`listen`] opens one).
fn active_app_for(url: &str, issue: &str, plan: &str, runtime: &tokio::runtime::Runtime) -> App {
    let mut app = App::new();
    app.insert_resource(Settings::from_toml(&format!(
        "[linear]\napi_url = \"{url}\"\n{plan}"
    )))
    .insert_resource(Secrets::from_iter([(TOKEN_SECRET, "lin_api_key")]))
    .insert_resource(invoked(&[(ISSUE_FLAG, issue)]))
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

fn active_app(url: &str, plan: &str, runtime: &tokio::runtime::Runtime) -> App {
    active_app_for(url, "ENG-1", plan, runtime)
}

/// Has `app`, before its first update, open a webhook port: its address.
fn listen(app: &mut App) -> String {
    let addr = stand_in::free_addr();
    if let Some(mut linear) = app.world_mut().get_resource_mut::<Linear>()
        && let Some((config, ..)) = &mut linear.pending
    {
        config.listen = Some(Listen {
            addr: addr.clone(),
            secret: WEBHOOK_SECRET.to_owned(),
        });
    }
    addr
}

/// Posts Ana's prompt `id` saying `body` (a stop when `None`) to the
/// webhook port at `addr`, signed.
fn ana_says(addr: &str, id: &str, body: Option<&str>) {
    let event = stand_in::prompted(
        SESSION_ID,
        id,
        "Ana",
        body.unwrap_or_default(),
        body.is_none(),
    );
    assert_eq!(
        stand_in::post(addr, &event, Some(WEBHOOK_SECRET)),
        Some(202)
    );
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
    let main = main_agent_of(&mut app);
    let mut sections = app.world_mut().query::<(&PromptSection, &SectionOf)>();
    let owned: Vec<(String, Entity)> = sections
        .iter(app.world())
        .map(|(section, owner)| (section.text.clone(), owner.0))
        .collect();
    assert_eq!(
        owned,
        main.map(|main| vec![(section(Some("ENG-1"), true), main)])
            .unwrap_or_default()
    );
    let mut tools = app.world_mut().query::<&ToolDef>();
    let names: Vec<String> = tools
        .iter(app.world())
        .map(|tool| tool.0.name.to_string())
        .collect();
    assert_eq!(names, [OUTCOME_TOOL]);
    // The exit's plan would replace the one still queued.
    for _ in 0..1000 {
        if linear.requests().len() >= 10 {
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
        thought(crate::send::WORKING),
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
        .get(9)
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
        options: Vec::new(),
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
    let sent = variables(&linear, 4);
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
        options: Vec::new(),
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

#[test]
fn a_question_with_options_is_posted_as_a_select() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::start();
    let mut app = active_app(&linear.url, "", &runtime);
    wait_until_started(&mut app);
    let Some(main) = main_agent_of(&mut app) else {
        return;
    };
    let args = OutcomeArgs {
        status: OutcomeStatus::NeedsInput,
        summary: "Which database?".to_owned(),
        options: vec!["sqlite".to_owned(), "postgres".to_owned()],
    };
    call_outcome(&mut app, main, args);
    app.world_mut().write_message(AppExit::Success);
    app.update();
    let last = variables(&linear, 0).last().cloned().unwrap_or_default();
    assert_eq!(
        last,
        json!({
            "id": "session-1",
            "content": { "type": "elicitation", "body": "Which database?" },
            "signal": "select",
            "signalMetadata": { "options": [
                { "label": "sqlite", "value": "sqlite" },
                { "label": "postgres", "value": "postgres" },
            ] },
        })
    );
}

/// Updates `app` until `done` holds, for at most 10 s.
fn update_until(app: &mut App, done: impl Fn(&App) -> bool) {
    for _ in 0..1000 {
        app.update();
        if done(app) {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The messages delivered to agents, as a test observer keeps them.
#[derive(Resource, Default)]
struct Delivered(Vec<(Entity, Incoming)>);

#[test]
fn a_persons_message_goes_to_the_main_agent_and_is_acknowledged() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::start();
    let mut app = active_app(&linear.url, "", &runtime);
    let addr = listen(&mut app);
    app.init_resource::<Delivered>().add_observer(
        |deliver: On<Deliver>, mut delivered: ResMut<Delivered>| {
            delivered.0.push((deliver.entity, deliver.message.clone()));
        },
    );
    wait_until_started(&mut app);
    ana_says(&addr, "p1", Some("Use sqlite, please.\nAnd keep it small."));
    update_until(&mut app, |app| {
        app.world()
            .get_resource::<Delivered>()
            .is_some_and(|delivered| !delivered.0.is_empty())
    });
    let main = main_agent_of(&mut app);
    let delivered = app
        .world_mut()
        .remove_resource::<Delivered>()
        .unwrap_or_default();
    let [(agent, message)] = delivered.0.as_slice() else {
        assert_eq!(delivered.0.len(), 1);
        return;
    };
    assert_eq!(Some(*agent), main);
    assert_eq!(message.text(), "Use sqlite, please.\nAnd keep it small.");
    assert_eq!(message.mode, DeliveryMode::Steer);
    assert_eq!(
        message.author,
        Author::plugin(ORIGIN).titled("Ana on Linear")
    );
    app.world_mut().write_message(AppExit::Success);
    app.update();
    let ack = json!({
        "id": "session-1",
        "content": { "type": "thought", "body": "Got your message: Use sqlite, please.…" },
    });
    assert!(
        variables(&linear, 0).contains(&ack),
        "{:#?}",
        linear.bodies()
    );
}

#[test]
fn a_stop_from_linear_exits_as_a_signal_and_posts_only_the_stop() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::start();
    let mut app = active_app(&linear.url, PLAN, &runtime);
    let addr = listen(&mut app);
    app.add_observer(|exit: On<Exit>, mut commands: Commands| {
        commands.insert_resource(exit.0.clone());
    });
    wait_until_started(&mut app);
    ana_says(&addr, "p1", None);
    update_until(&mut app, |app| {
        app.world().contains_resource::<ExitReason>()
    });
    assert_eq!(
        app.world().get_resource::<ExitReason>(),
        Some(&ExitReason::Signal)
    );
    // What the agent does as the exit stops it is not shown.
    recorded(
        &mut app,
        vec![
            reply(1, "main", vec![AssistantContent::text("Still going.")]),
            call(2, "main", "shell", json!({ "command": "ls" }), "", Some(0)),
        ],
    );
    app.world_mut().write_message(AppExit::from_code(130));
    app.update();
    let sent: Vec<Value> = variables(&linear, 0)
        .into_iter()
        .filter(|sent| sent.get("content").is_some() || sent.get("plan").is_some())
        .collect();
    let stopped = json!({
        "id": "session-1",
        "content": { "type": "error", "body": "Stopped at the request of Ana." },
    });
    let canceled =
        json!({ "id": "session-1", "plan": plan(&[("Triage", "canceled"), ("Ship", "canceled")]) });
    assert_eq!(
        sent.get(sent.len().saturating_sub(2)..),
        Some([canceled, stopped].as_slice()),
        "{sent:#?}"
    );
}

#[test]
fn the_prompt_section_is_scoped_to_main_and_omits_outcome_tool_when_disallowed() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::start();
    let mut app = active_app(&linear.url, "", &runtime);
    if let Some(main) = main_agent_of(&mut app) {
        app.world_mut()
            .entity_mut(main)
            .insert(ToolAccess::only(["read"]));
    }
    wait_until_started(&mut app);
    let main = main_agent_of(&mut app);
    let mut sections = app
        .world_mut()
        .query::<(&PromptSection, Option<&SectionOf>)>();
    let found: Vec<(String, Option<Entity>)> = sections
        .iter(app.world())
        .map(|(section, owner)| (section.text.clone(), owner.map(|owner| owner.0)))
        .collect();
    assert_eq!(found, vec![(section(Some("ENG-1"), false), main)]);
    app.world_mut().write_message(AppExit::Success);
    app.update();
}

#[test]
fn a_reload_reuses_the_saved_linear_session_and_a_switch_ends_the_outgoing_one() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::start();
    let mut app = active_app(&linear.url, "", &runtime);
    wait_until_started(&mut app);
    let saved = app
        .world()
        .get_resource::<SavedLinearSession>()
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        saved,
        SavedLinearSession {
            session_id: Some("session-1".to_owned()),
            issue: Some("ENG-1".to_owned()),
        }
    );
    // Reload skips posting a final activity.
    app.insert_resource(ExitReason::Reload);
    app.world_mut().write_message(AppExit::Success);
    app.update();
    assert!(
        !linear.bodies().iter().any(|body| matches!(
            body.pointer("/variables/content/type")
                .and_then(Value::as_str),
            Some("response" | "error" | "elicitation")
        )),
        "{:#?}",
        linear.bodies()
    );

    // Next process with the restored SavedLinearSession reattaches instead of creating a new one.
    let reloaded_server = StandIn::start();
    let mut reloaded = active_app(&reloaded_server.url, "", &runtime);
    reloaded.insert_resource(saved);
    wait_until_started(&mut reloaded);
    assert!(
        !reloaded_server
            .requests()
            .iter()
            .any(|req| req.query().contains("agentSessionCreateOnIssue")),
        "{:#?}",
        reloaded_server.bodies()
    );
    assert!(
        reloaded_server
            .requests()
            .iter()
            .any(|req| req.query().starts_with("query Session")),
        "{:#?}",
        reloaded_server.bodies()
    );
    // Switching sessions ends the outgoing Linear session.
    recorded(
        &mut reloaded,
        vec![reply(1, "main", vec![AssistantContent::text("Done here.")])],
    );
    reloaded.insert_resource(ExitReason::Switch(None));
    reloaded.world_mut().write_message(AppExit::Success);
    reloaded.update();
    assert_eq!(
        reloaded_server
            .bodies()
            .last()
            .and_then(|body| body.pointer("/variables/content").cloned()),
        Some(json!({ "type": "response", "body": "Done here." }))
    );
}

#[test]
fn reload_matches_uuid_issue_and_saves_session_when_startup_finishes_during_close() {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let linear = StandIn::scripted(|request, _| {
        if request.query().contains("agentSessionCreateOnIssue") {
            std::thread::sleep(Duration::from_millis(60));
        }
        None
    });
    // Target is a UUID while Linear returns display identifier "ENG-1".
    let mut app = active_app_for(&linear.url, "issue-uuid", "", &runtime);
    // Exit for reload immediately on the first frame while startup is still in flight.
    app.insert_resource(ExitReason::Reload);
    app.world_mut().write_message(AppExit::Success);
    app.update();
    let saved = app
        .world()
        .get_resource::<SavedLinearSession>()
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        saved,
        SavedLinearSession {
            session_id: Some("session-1".to_owned()),
            issue: Some("issue-uuid".to_owned()),
        }
    );

    // Reloading with the same UUID reattaches instead of opening a second session.
    let reloaded_server = StandIn::start();
    let mut reloaded = active_app_for(&reloaded_server.url, "issue-uuid", "", &runtime);
    reloaded.insert_resource(saved);
    wait_until_started(&mut reloaded);
    assert!(
        !reloaded_server
            .requests()
            .iter()
            .any(|req| req.query().contains("agentSessionCreateOnIssue")),
        "{:#?}",
        reloaded_server.bodies()
    );
    assert!(
        reloaded_server
            .requests()
            .iter()
            .any(|req| req.query().starts_with("query Session")),
        "{:#?}",
        reloaded_server.bodies()
    );
}
