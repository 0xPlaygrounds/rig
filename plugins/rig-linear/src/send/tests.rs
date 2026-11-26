use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::api::IssueState;
use crate::tests::stand_in::{
    self, APP_USER, Request, SESSION_ID, SESSION_URL, StandIn, activities, prompt,
};

/// The query of a lasting activity, as sent.
const ACTIVITY: &str = "mutation CreateActivity($id: String!, $content: JSONObject!) { \
agentActivityCreate(input: { agentSessionId: $id, content: $content }) { success } }";

/// The query of a passing activity, as sent.
const PASSING: &str = "mutation CreateActivity($id: String!, $content: JSONObject!, $ephemeral: \
Boolean) { agentActivityCreate(input: { agentSessionId: $id, content: $content, ephemeral: \
$ephemeral }) { success } }";

fn note(text: &str) -> Job {
    Job::Activity(Activity::note(text.to_owned()))
}

fn action(parameter: &str) -> Job {
    Job::Activity(Activity::action(
        "Ran".to_owned(),
        parameter.to_owned(),
        None,
    ))
}

fn link(label: &str) -> Job {
    Job::Links(vec![Link {
        label: label.to_owned(),
        url: format!("https://example.com/{label}"),
    }])
}

/// What a queue holds, in short.
fn kinds(queue: &VecDeque<Job>) -> Vec<String> {
    queue
        .iter()
        .map(|job| match job {
            Job::Activity(activity) => match &activity.content {
                Content::Action { parameter, .. } => format!("action {parameter}"),
                Content::Thought(body) => format!("note {body}"),
                content => format!("{content:?}"),
            },
            Job::Plan(plan) => format!("plan {plan}"),
            Job::Links(links) => {
                let labels: Vec<&str> = links.iter().map(|link| link.label.as_str()).collect();
                format!("links {}", labels.join(","))
            }
            Job::Close(_) => "close".to_owned(),
        })
        .collect()
}

#[test]
fn passing_activities_in_a_row_keep_only_the_last_before_a_lasting_one() {
    let mut queue = VecDeque::from([
        action("a"),
        action("b"),
        note("n"),
        action("c"),
        Job::Plan(json!([1])),
        action("d"),
    ]);
    coalesce(&mut queue);
    assert_eq!(
        kinds(&queue),
        ["action b", "note n", "plan [1]", "action d"]
    );
}

#[test]
fn only_the_last_plan_is_kept_and_links_are_sent_together() {
    let mut queue = VecDeque::from([
        Job::Plan(json!([1])),
        link("one"),
        note("n"),
        Job::Plan(json!([2])),
        link("two"),
    ]);
    coalesce(&mut queue);
    assert_eq!(kinds(&queue), ["links one,two", "note n", "plan [2]"]);
}

#[test]
fn the_close_ends_a_run_of_passing_activities() {
    let (done, _) = std::sync::mpsc::channel();
    let mut queue = VecDeque::from([action("a"), Job::Close(done), action("b")]);
    coalesce(&mut queue);
    assert_eq!(kinds(&queue), ["action a", "close", "action b"]);
}

/// A config for the stand-in at `url`, sending as soon as it may.
fn config(url: &str, credentials: Credentials, target: Target) -> Config {
    Config {
        api_url: url.to_owned(),
        credentials,
        target,
        plan: None,
        heartbeat: Duration::from_secs(3600),
        spacing: Duration::from_millis(5),
        poll: None,
        claim: false,
    }
}

/// Runs the task on `config` with `jobs` queued before it starts, then
/// the close; what it said it started, if it did.
fn run_with(config: Config, jobs: Vec<Job>) -> Option<Started> {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return None;
    };
    let (queue, queued) = async_channel::unbounded();
    let (tell, told) = async_channel::unbounded();
    for job in jobs {
        queue.try_send(job).ok();
    }
    let (done, closed) = std::sync::mpsc::channel();
    queue.try_send(Job::Close(done)).ok();
    let redactor = Redactor::new(Vec::new());
    runtime.spawn(run(config, rig_reqwest::shared(), redactor, queued, tell));
    closed.recv_timeout(Duration::from_secs(20)).ok()?;
    match told.try_recv().ok()? {
        Told::Started(started) => Some(started),
        _ => None,
    }
}

fn activity(content: serde_json::Value) -> serde_json::Value {
    json!({ "query": ACTIVITY, "variables": { "id": SESSION_ID, "content": content } })
}

fn passing(content: serde_json::Value) -> serde_json::Value {
    json!({
        "query": PASSING,
        "variables": { "id": SESSION_ID, "content": content, "ephemeral": true },
    })
}

#[test]
fn an_issue_gets_a_session_then_the_plan_then_the_coalesced_activities() {
    let linear = StandIn::start();
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Issue("ENG-1".to_owned()),
    );
    config.plan = Some(json!([{ "content": "Fix it", "status": "pending" }]));
    let started = run_with(config, vec![note("Looking."), action("ls"), action("make")]);
    assert_eq!(
        started,
        Some(Started {
            id: SESSION_ID.to_owned(),
            url: Some(SESSION_URL.to_owned()),
            issue: Some("ENG-1".to_owned()),
        })
    );
    let expected = vec![
        json!({
            "query": "query Issue($id: String!) { issue(id: $id) { id identifier url } }",
            "variables": { "id": "ENG-1" },
        }),
        json!({
            "query": "mutation CreateSession($issueId: String!) { agentSessionCreateOnIssue(input: \
                      { issueId: $issueId }) { success agentSession { id url activities(first: 50, \
                      filter: { type: { eq: \"prompt\" } }) { nodes { id createdAt } } } } }",
            "variables": { "issueId": "issue-uuid" },
        }),
        activity(json!({ "type": "thought", "body": WORKING })),
        json!({
            "query": "mutation UpdateSession($id: String!, $plan: JSONObject) { \
                      agentSessionUpdate(id: $id, input: { plan: $plan }) { success } }",
            "variables": {
                "id": SESSION_ID,
                "plan": [{ "content": "Fix it", "status": "pending" }],
            },
        }),
        activity(json!({ "type": "thought", "body": "Looking." })),
        passing(json!({ "type": "action", "action": "Ran", "parameter": "make" })),
    ];
    assert_eq!(linear.bodies(), expected);
    let requests = linear.requests();
    assert!(requests.iter().all(|request| {
        request.path == "/graphql"
            && request.authorization.as_deref() == Some("lin_api_key")
            && request.content_type.as_deref() == Some("application/json")
    }));
}

#[test]
fn client_credentials_are_exchanged_once_for_a_bearer_token() {
    let linear = StandIn::start();
    let credentials = Credentials::Client {
        id: "client".to_owned(),
        secret: "s3cret".to_owned(),
        scope: "read,app:assignable".to_owned(),
    };
    let target = Target::Session(SESSION_ID.to_owned());
    let started = run_with(config(&linear.url, credentials, target), vec![note("Hi.")]);
    assert_eq!(
        started,
        Some(Started {
            id: SESSION_ID.to_owned(),
            url: Some(SESSION_URL.to_owned()),
            issue: Some("ENG-1".to_owned()),
        })
    );
    let requests = linear.requests();
    let paths: Vec<&str> = requests
        .iter()
        .map(|request| request.path.as_str())
        .collect();
    assert_eq!(paths, ["/oauth/token", "/graphql", "/graphql", "/graphql"]);
    let token = requests.first().cloned().unwrap_or_default();
    assert_eq!(
        token.body,
        "grant_type=client_credentials&client_id=client&client_secret=s3cret&scope=read%2Capp%3Aassignable"
    );
    assert_eq!(
        token.content_type.as_deref(),
        Some("application/x-www-form-urlencoded")
    );
    assert_eq!(token.authorization, None);
    let rest = requests.get(1..).unwrap_or_default();
    assert!(
        rest.iter()
            .all(|request| request.authorization.as_deref() == Some("Bearer app-token"))
    );
    let bodies = linear.bodies();
    assert_eq!(
        bodies.get(1),
        Some(&json!({
            "query": "query Session($id: String!) { agentSession(id: $id) { id url issue { id \
                      identifier } activities(first: 50, filter: { type: { eq: \"prompt\" } }) \
                      { nodes { id createdAt } } answers: activities(first: 1, filter: { type: { \
                      in: [\"response\", \"error\", \"elicitation\"] } }) { nodes { createdAt } \
                      } } }",
            "variables": { "id": SESSION_ID },
        }))
    );
    assert_eq!(
        bodies.get(2),
        Some(&activity(json!({ "type": "thought", "body": WORKING })))
    );
    assert_eq!(
        bodies.get(3),
        Some(&activity(json!({ "type": "thought", "body": "Hi." })))
    );
}

/// Whether `request` adds an activity whose body is `body`.
fn adds(request: &Request, body: &str) -> bool {
    request
        .json()
        .pointer("/variables/content/body")
        .and_then(serde_json::Value::as_str)
        == Some(body)
}

#[test]
fn a_rate_limited_request_is_sent_again_after_its_retry_after() {
    let linear = StandIn::scripted(|request, index| {
        (adds(request, "first") && index == 2).then(|| {
            let wait = vec![("retry-after", "0".to_owned())];
            (429, wait, "{}".to_owned())
        })
    });
    let target = Target::Session(SESSION_ID.to_owned());
    let credentials = Credentials::Token("oauth-token".to_owned());
    run_with(
        config(&linear.url, credentials, target),
        vec![note("first"), note("second")],
    );
    let requests = linear.requests();
    let notes: Vec<bool> = requests
        .iter()
        .skip(2)
        .map(|request| adds(request, "first"))
        .collect();
    assert_eq!(notes, [true, true, false], "{requests:?}");
    assert!(
        requests
            .last()
            .is_some_and(|request| adds(request, "second"))
    );
    assert!(
        requests
            .iter()
            .all(|request| request.authorization.as_deref() == Some("Bearer oauth-token"))
    );
}

#[test]
fn a_failed_request_is_dropped_and_the_next_one_still_goes() {
    let linear = StandIn::scripted(|request, _| {
        adds(request, "first").then(|| (500, Vec::new(), "{}".to_owned()))
    });
    let target = Target::Session(SESSION_ID.to_owned());
    let credentials = Credentials::Token("lin_api_key".to_owned());
    run_with(
        config(&linear.url, credentials, target),
        vec![note("first"), note("second")],
    );
    let bodies: Vec<bool> = linear
        .requests()
        .iter()
        .skip(2)
        .map(|request| adds(request, "second"))
        .collect();
    assert_eq!(bodies, [false, true]);
}

#[test]
fn a_refused_session_is_warned_in_linears_words() {
    let said = ApiError::Reply("agent sessions disabled".to_owned());
    assert_eq!(
        refused(&said),
        "Linear refused the agent session: agent sessions disabled. Is the app configured for \
         agent sessions?"
    );
    assert_eq!(
        refused(&ApiError::Timeout),
        "Linear: the agent session could not start: no reply within 10 s"
    );
}

#[test]
fn a_session_that_cannot_start_answers_the_close_and_sends_nothing_more() {
    let linear = StandIn::scripted(|_, _| Some((401, Vec::new(), "{}".to_owned())));
    let target = Target::Issue("ENG-1".to_owned());
    let credentials = Credentials::Token("lin_api_key".to_owned());
    let started = run_with(config(&linear.url, credentials, target), vec![note("lost")]);
    assert_eq!(started, None);
    assert_eq!(linear.requests().len(), 1);
}

#[test]
fn a_quiet_session_gets_a_heartbeat_naming_the_last_action() {
    let linear = StandIn::start();
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    config.heartbeat = Duration::from_millis(300);
    let (queue, queued) = async_channel::unbounded();
    let (tell, _started) = async_channel::unbounded();
    queue.try_send(action("cargo test")).ok();
    runtime.spawn(run(
        config,
        rig_reqwest::shared(),
        Redactor::default(),
        queued,
        tell,
    ));
    std::thread::sleep(Duration::from_millis(1200));
    let (done, closed) = std::sync::mpsc::channel();
    queue.try_send(Job::Close(done)).ok();
    closed.recv_timeout(Duration::from_secs(10)).ok();
    let bodies = linear.bodies();
    let beat = passing(json!({ "type": "thought", "body": "Still working… Last: Ran cargo test" }));
    assert_eq!(bodies.get(3), Some(&beat), "{bodies:?}");
}

#[test]
fn no_heartbeat_follows_the_end_of_the_run() {
    let linear = StandIn::start();
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return;
    };
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    config.heartbeat = Duration::from_millis(200);
    let (queue, queued) = async_channel::unbounded();
    let (tell, _started) = async_channel::unbounded();
    let end = Activity::end(Content::Response("Done.".to_owned()));
    queue.try_send(Job::Activity(end)).ok();
    runtime.spawn(run(
        config,
        rig_reqwest::shared(),
        Redactor::default(),
        queued,
        tell,
    ));
    std::thread::sleep(Duration::from_millis(900));
    let (done, closed) = std::sync::mpsc::channel();
    queue.try_send(Job::Close(done)).ok();
    closed.recv_timeout(Duration::from_secs(10)).ok();
    assert_eq!(linear.requests().len(), 3);
}

#[test]
fn what_is_sent_is_redacted_and_the_token_never_shows() {
    let linear = StandIn::start();
    let target = Target::Session(SESSION_ID.to_owned());
    let credentials = Credentials::Token("lin_api_key_of_the_test".to_owned());
    run_with(
        config(&linear.url, credentials, target),
        vec![note("the key is lin_api_key_of_the_test")],
    );
    let bodies = linear.bodies();
    assert_eq!(
        bodies.get(2),
        Some(&activity(
            json!({ "type": "thought", "body": "the key is [REDACTED]" })
        ))
    );
}

#[test]
fn the_plan_is_redacted_too() {
    let linear = StandIn::start();
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key_of_the_test".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    config.plan = Some(json!([{ "content": "use lin_api_key_of_the_test", "status": "pending" }]));
    run_with(config, Vec::new());
    let plan = linear
        .bodies()
        .get(2)
        .and_then(|body| body.pointer("/variables/plan").cloned());
    assert_eq!(
        plan,
        Some(json!([{ "content": "use [REDACTED]", "status": "pending" }]))
    );
}

/// Runs the task on `config` for `for_how_long`, then closes it: what it
/// told after it started.
fn run_for(config: Config, for_how_long: Duration) -> Vec<Told> {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return Vec::new();
    };
    let (queue, queued) = async_channel::unbounded();
    let (tell, told) = async_channel::unbounded();
    runtime.spawn(run(
        config,
        rig_reqwest::shared(),
        Redactor::default(),
        queued,
        tell,
    ));
    std::thread::sleep(for_how_long);
    let (done, closed) = std::sync::mpsc::channel();
    queue.try_send(Job::Close(done)).ok();
    closed.recv_timeout(Duration::from_secs(10)).ok();
    std::iter::from_fn(|| told.try_recv().ok())
        .filter(|news| !matches!(news, Told::Started(_)))
        .collect()
}

/// Whether `request` reads the session's prompts.
fn reads(request: &Request) -> bool {
    request.query().starts_with("query Activities")
}

#[test]
fn only_prompts_after_the_last_answer_are_told_and_a_stop_ends_the_reading() {
    let read = std::sync::atomic::AtomicUsize::new(0);
    let linear = StandIn::scripted(move |request, _| {
        if request.query().starts_with("query Session") {
            return Some(stand_in::reply(json!({
                "agentSession": {
                    "id": SESSION_ID,
                    "url": SESSION_URL,
                    "issue": { "id": "issue-uuid", "identifier": "ENG-1" },
                    "activities": {
                        "nodes": [{ "id": "p1", "createdAt": "2026-01-01T00:00:00.000Z" }]
                    },
                    "answers": { "nodes": [{ "createdAt": "2026-01-01T00:00:30.000Z" }] },
                }
            })));
        }
        if !reads(request) {
            return None;
        }
        let first = prompt(
            "p1",
            "2026-01-01T00:00:00.000Z",
            "Ana",
            "Fix the bug.",
            false,
        );
        let nodes = match read.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
            // Newest first, with the one already present at attach.
            0 => vec![
                prompt(
                    "p3",
                    "2026-01-01T00:02:00.000Z",
                    "Ana",
                    "And add a test.",
                    false,
                ),
                prompt(
                    "p2",
                    "2026-01-01T00:01:00.000Z",
                    "Ana",
                    "Use sqlite.",
                    false,
                ),
                first,
            ],
            1 => vec![prompt("p4", "2026-01-01T00:03:00.000Z", "Ana", "", true)],
            _ => vec![prompt(
                "p5",
                "2026-01-01T00:04:00.000Z",
                "Ana",
                "Too late.",
                false,
            )],
        };
        Some(stand_in::reply(activities(&nodes)))
    });
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    config.poll = Some(Duration::from_millis(100));
    let told = run_for(config, Duration::from_millis(1500));
    let by = || Some("Ana".to_owned());
    assert_eq!(
        told,
        [
            Told::Prompt {
                body: "Use sqlite.".to_owned(),
                by: by(),
            },
            Told::Prompt {
                body: "And add a test.".to_owned(),
                by: by(),
            },
            Told::Stop { by: by() },
        ]
    );
    let reading: Vec<Value> = linear
        .requests()
        .iter()
        .filter(|request| reads(request))
        .map(|request| request.json())
        .collect();
    assert_eq!(reading.len(), 2, "{reading:#?}");
    let since = |at: usize| {
        reading
            .get(at)
            .and_then(|body| body.pointer("/variables/since").cloned())
    };
    assert_eq!(since(0), Some(json!("2026-01-01T00:00:00.000Z")));
    assert_eq!(since(1), Some(json!("2026-01-01T00:02:00.000Z")));
    let query = reading
        .first()
        .and_then(|body| body.get("query").cloned())
        .unwrap_or_default();
    assert_eq!(
        query,
        json!(
            "query Activities($id: String!, $after: String, $since: DateTimeOrDuration) { \
             agentSession(id: $id) { activities(first: 50, after: $after, filter: { type: { eq: \
             \"prompt\" }, createdAt: { gte: $since } }) { nodes { id createdAt signal user { \
             name app } content { __typename ... on AgentActivityPromptContent { body } } } pageInfo { \
             hasNextPage endCursor } } } }"
        )
    );
}

#[test]
fn a_failed_first_read_still_delivers_prompts_created_after_attach() {
    let read = std::sync::atomic::AtomicUsize::new(0);
    let linear = StandIn::scripted(move |request, _| {
        if request.query().starts_with("query Session") {
            return Some(stand_in::reply(json!({
                "agentSession": {
                    "id": SESSION_ID,
                    "url": SESSION_URL,
                    "issue": { "id": "issue-uuid", "identifier": "ENG-1" },
                    "activities": {
                        "nodes": [{ "id": "p1", "createdAt": "2026-01-01T00:00:00.000Z" }]
                    },
                    "answers": { "nodes": [{ "createdAt": "2026-01-01T00:00:01.000Z" }] },
                }
            })));
        }
        if !reads(request) {
            return None;
        }
        let old = prompt(
            "p1",
            "2026-01-01T00:00:00.000Z",
            "Ana",
            "Fix the bug.",
            false,
        );
        let newer = prompt("p2", "2026-01-01T00:00:05.000Z", "Ana", "Stop now.", true);
        Some(
            match read.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
                0 => (500, Vec::new(), "{}".to_owned()),
                _ => stand_in::reply(activities(&[newer, old])),
            },
        )
    });
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    config.poll = Some(Duration::from_millis(100));
    assert_eq!(
        run_for(config, Duration::from_millis(800)),
        [Told::Stop {
            by: Some("Ana".to_owned())
        }]
    );
}

#[test]
fn the_first_thought_goes_before_the_plan_and_no_prompt_is_read_when_polling_is_off() {
    let linear = StandIn::start();
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    config.plan = Some(json!([{ "content": "Fix it", "status": "pending" }]));
    run_with(config, Vec::new());
    let bodies = linear.bodies();
    assert_eq!(
        bodies.get(1),
        Some(&activity(json!({ "type": "thought", "body": WORKING })))
    );
    assert!(
        bodies
            .get(2)
            .is_some_and(|body| body.pointer("/variables/plan").is_some())
    );
    assert!(!linear.requests().iter().any(reads));
}

/// The `issueUpdate` inputs sent.
fn updates(linear: &StandIn) -> Vec<Value> {
    linear
        .bodies()
        .into_iter()
        .filter(|body| {
            body.get("query")
                .and_then(Value::as_str)
                .is_some_and(|query| query.contains("issueUpdate"))
        })
        .filter_map(|body| body.get("variables").cloned())
        .collect()
}

#[test]
fn a_claimed_issue_is_delegated_and_moved_to_its_first_started_state() {
    let linear = StandIn::start();
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Issue("ENG-1".to_owned()),
    );
    config.claim = true;
    run_with(config, vec![note("Looking.")]);
    assert_eq!(
        updates(&linear),
        [json!({
            "id": "issue-uuid",
            "input": { "delegateId": APP_USER, "stateId": "state-progress" },
        })]
    );
    let claim = linear.bodies().into_iter().find(|body| {
        body.get("query")
            .and_then(Value::as_str)
            .is_some_and(|query| query.starts_with("query Claim"))
    });
    assert_eq!(
        claim,
        Some(json!({
            "query": "query Claim($id: String!) { viewer { id } issue(id: $id) { id delegate { id } \
                      state { type } team { states(filter: { type: { eq: \"started\" } }) { nodes \
                      { id position } } } } }",
            "variables": { "id": "issue-uuid" },
        }))
    );
    assert!(
        linear
            .requests()
            .iter()
            .any(|request| adds(request, "Looking."))
    );
}

#[test]
fn an_issue_is_claimed_only_when_asked_and_only_as_far_as_it_needs() {
    let linear = StandIn::start();
    let config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Issue("ENG-1".to_owned()),
    );
    run_with(config, Vec::new());
    assert_eq!(updates(&linear), Vec::<Value>::new());
    assert!(
        !linear
            .requests()
            .iter()
            .any(|request| request.query().starts_with("query Claim"))
    );
    let state = |delegated: bool, state: &str| IssueState {
        viewer: APP_USER.to_owned(),
        delegated,
        state: Some(state.to_owned()),
        started: Some("state-progress".to_owned()),
    };
    assert_eq!(state(true, "started").claim(), None);
    assert_eq!(state(true, "completed").claim(), None);
    assert_eq!(state(true, "canceled").claim(), None);
    assert_eq!(
        state(false, "started").claim(),
        Some(json!({ "delegateId": APP_USER }))
    );
    assert_eq!(
        state(true, "triage").claim(),
        Some(json!({ "stateId": "state-progress" }))
    );
    let no_started = IssueState {
        started: None,
        ..state(true, "backlog")
    };
    assert_eq!(no_started.claim(), None);
}

#[test]
fn a_session_attached_to_claims_its_issue_and_a_failed_claim_stops_nothing() {
    let linear = StandIn::scripted(|request, _| {
        request
            .query()
            .contains("issueUpdate")
            .then(|| (500, Vec::new(), "{}".to_owned()))
    });
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    config.claim = true;
    run_with(config, vec![note("After the claim.")]);
    assert_eq!(updates(&linear).len(), 1);
    assert!(
        linear
            .requests()
            .last()
            .is_some_and(|request| adds(request, "After the claim."))
    );
}

#[test]
fn a_question_with_options_goes_with_the_select_signal() {
    let linear = StandIn::start();
    let question = Activity::end(Content::Elicitation {
        body: "Which database?".to_owned(),
        options: vec!["sqlite".to_owned(), "postgres".to_owned()],
    });
    run_with(
        config(
            &linear.url,
            Credentials::Token("lin_api_key".to_owned()),
            Target::Session(SESSION_ID.to_owned()),
        ),
        vec![Job::Activity(question)],
    );
    assert_eq!(
        linear.bodies().last(),
        Some(&json!({
            "query": "mutation CreateActivity($id: String!, $content: JSONObject!, $signal: \
                      AgentActivitySignal, $signalMetadata: JSONObject) { agentActivityCreate(input: \
                      { agentSessionId: $id, content: $content, signal: $signal, signalMetadata: \
                      $signalMetadata }) { success } }",
            "variables": {
                "id": SESSION_ID,
                "content": { "type": "elicitation", "body": "Which database?" },
                "signal": "select",
                "signalMetadata": { "options": [
                    { "label": "sqlite", "value": "sqlite" },
                    { "label": "postgres", "value": "postgres" },
                ] },
            },
        }))
    );
}

#[test]
fn a_failed_attach_answers_close_and_sends_nothing_more() {
    let linear = StandIn::scripted(|_, _| Some((401, Vec::new(), "{}".to_owned())));
    let target = Target::Session(SESSION_ID.to_owned());
    let credentials = Credentials::Token("lin_api_key".to_owned());
    let started = run_with(config(&linear.url, credentials, target), vec![note("lost")]);
    assert_eq!(started, None);
    assert_eq!(linear.requests().len(), 1);
}

#[test]
fn a_missing_attached_session_is_treated_as_a_startup_failure() {
    let linear = StandIn::scripted(|_, _| Some(stand_in::reply(json!({ "agentSession": null }))));
    let target = Target::Session(SESSION_ID.to_owned());
    let credentials = Credentials::Token("lin_api_key".to_owned());
    let started = run_with(config(&linear.url, credentials, target), vec![note("lost")]);
    assert_eq!(started, None);
    assert_eq!(linear.requests().len(), 1);
}

#[test]
fn startup_requests_respect_minimum_spacing() {
    let linear = StandIn::start();
    let mut cfg = config(
        &linear.url,
        Credentials::Client {
            id: "client".to_owned(),
            secret: "s3cret".to_owned(),
            scope: "read,write".to_owned(),
        },
        Target::Issue("ENG-1".to_owned()),
    );
    cfg.spacing = Duration::from_millis(80);
    let started_at = std::time::Instant::now();
    run_with(cfg, Vec::new());
    // 4 startup requests (token exchange, issue lookup, create session, WORKING) => >= 3 spacings.
    assert_eq!(linear.requests().len(), 4);
    assert!(started_at.elapsed() >= Duration::from_millis(220));
}

#[test]
fn retry_delay_honors_full_retry_after_and_enforces_spacing() {
    let spacing = Duration::from_secs(1);
    assert_eq!(retry_delay(Some(Duration::ZERO), spacing), spacing);
    assert_eq!(
        retry_delay(Some(Duration::from_millis(200)), spacing),
        spacing
    );
    assert_eq!(
        retry_delay(Some(Duration::from_secs(120)), spacing),
        Duration::from_secs(120)
    );
    assert_eq!(retry_delay(None, spacing), RETRY_WAIT);
    assert_eq!(
        retry_delay(None, Duration::from_secs(10)),
        Duration::from_secs(10)
    );
}

#[test]
fn synthetic_app_prompts_on_proactive_session_creation_are_ignored() {
    let linear = StandIn::scripted(|request, _| {
        if request.query().contains("agentSessionCreateOnIssue") {
            return Some(stand_in::reply(json!({
                "agentSessionCreateOnIssue": {
                    "success": true,
                    "agentSession": {
                        "id": SESSION_ID,
                        "url": SESSION_URL,
                        "activities": {
                            "nodes": [{ "id": "p0", "createdAt": "2026-01-01T00:00:00.000Z" }]
                        },
                    },
                }
            })));
        }
        if !reads(request) {
            return None;
        }
        let nodes = vec![
            json!({
                "id": "p2",
                "createdAt": "2026-01-01T00:00:02.000Z",
                "signal": Value::Null,
                "user": { "name": "Ana", "app": false },
                "content": { "__typename": "AgentActivityPromptContent", "body": "Human follow-up." },
            }),
            json!({
                "id": "p1",
                "createdAt": "2026-01-01T00:00:01.000Z",
                "signal": Value::Null,
                "user": { "name": "Rigged Intern", "app": true },
                "content": { "__typename": "AgentActivityPromptContent", "body": "Work on ENG-1" },
            }),
            json!({
                "id": "p0",
                "createdAt": "2026-01-01T00:00:00.000Z",
                "signal": Value::Null,
                "user": { "name": "Rigged Intern", "app": true },
                "content": { "__typename": "AgentActivityPromptContent", "body": "Work on ENG-1" },
            }),
        ];
        Some(stand_in::reply(activities(&nodes)))
    });
    let mut cfg = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Issue("ENG-1".to_owned()),
    );
    cfg.poll = Some(Duration::from_millis(80));
    let told = run_for(cfg, Duration::from_millis(400));
    assert_eq!(
        told,
        [Told::Prompt {
            body: "Human follow-up.".to_owned(),
            by: Some("Ana".to_owned()),
        }]
    );
}

/// A stand-in for a session that exists, whose prompts are `prompts` (each
/// `(id, createdAt, author, body)`, oldest first, an author `None` for the
/// app's synthetic one) and whose last answer is at `answered`.
fn existing_session(
    prompts: &'static [(
        &'static str,
        &'static str,
        Option<&'static str>,
        &'static str,
    )],
    answered: Option<&'static str>,
) -> StandIn {
    StandIn::scripted(move |request, _| {
        let node = |&(id, at, by, body): &(&str, &str, Option<&str>, &str)| {
            json!({
                "id": id,
                "createdAt": at,
                "signal": Value::Null,
                "user": { "name": by.unwrap_or("Rigged Intern"), "app": by.is_none() },
                "content": { "__typename": "AgentActivityPromptContent", "body": body },
            })
        };
        if request.query().starts_with("query Session") {
            let known: Vec<Value> = prompts
                .iter()
                .rev()
                .map(|&(id, at, _, _)| json!({ "id": id, "createdAt": at }))
                .collect();
            let answers: Vec<Value> = answered
                .into_iter()
                .map(|at| json!({ "createdAt": at }))
                .collect();
            return Some(stand_in::reply(json!({
                "agentSession": {
                    "id": SESSION_ID,
                    "url": SESSION_URL,
                    "issue": { "id": "issue-uuid", "identifier": "ENG-1" },
                    "activities": { "nodes": known },
                    "answers": { "nodes": answers },
                }
            })));
        }
        if !reads(request) {
            return None;
        }
        let since = request
            .json()
            .pointer("/variables/since")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let nodes: Vec<Value> = prompts
            .iter()
            .rev()
            .filter(|&&(_, at, _, _)| since.as_deref().is_none_or(|since| at >= since))
            .map(node)
            .collect();
        Some(stand_in::reply(activities(&nodes)))
    })
}

/// What a run on `target` of `linear` tells within half a second.
fn told_on(linear: &StandIn, target: Target) -> Vec<Told> {
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        target,
    );
    config.poll = Some(Duration::from_millis(80));
    run_for(config, Duration::from_millis(500))
}

#[test]
fn a_given_session_no_run_answered_tells_its_request_but_not_the_synthetic_prompt() {
    // As a webhook creates it: delegated (Linear's synthetic prompt), then a
    // person's mention.
    let linear = existing_session(
        &[
            ("p0", "2026-01-01T00:00:00.000Z", None, "Work on ENG-1"),
            (
                "p1",
                "2026-01-01T00:00:01.000Z",
                Some("Ana"),
                "Fix it, but leave auth alone.",
            ),
        ],
        None,
    );
    assert_eq!(
        told_on(&linear, Target::Session(SESSION_ID.to_owned())),
        [Told::Prompt {
            body: "Fix it, but leave auth alone.".to_owned(),
            by: Some("Ana".to_owned()),
        }]
    );
}

#[test]
fn a_given_session_tells_only_the_prompts_after_its_last_answer() {
    // A run answered the first request; a person answered it back since.
    let linear = existing_session(
        &[
            (
                "p1",
                "2026-01-01T00:00:00.000Z",
                Some("Ana"),
                "Fix the bug.",
            ),
            (
                "p2",
                "2026-01-01T00:02:00.000Z",
                Some("Ana"),
                "Use postgres.",
            ),
        ],
        Some("2026-01-01T00:01:00.000Z"),
    );
    assert_eq!(
        told_on(&linear, Target::Session(SESSION_ID.to_owned())),
        [Told::Prompt {
            body: "Use postgres.".to_owned(),
            by: Some("Ana".to_owned()),
        }]
    );
}

#[test]
fn a_saved_session_after_a_reload_tells_none_of_the_prompts_already_there() {
    let linear = existing_session(
        &[(
            "p1",
            "2026-01-01T00:00:00.000Z",
            Some("Ana"),
            "Fix the bug.",
        )],
        None,
    );
    assert_eq!(told_on(&linear, Target::Saved(SESSION_ID.to_owned())), []);
}
