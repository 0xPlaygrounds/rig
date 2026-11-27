use std::time::Duration;

use serde_json::{Value, json};

use super::*;
use crate::api::IssueState;
use crate::tests::stand_in::{
    self, APP_USER, Request, SESSION_ID, SESSION_URL, StandIn, WEBHOOK_SECRET, prompt, prompted,
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
fn coalesce_collapses_passing_runs_keeps_last_plan_merges_links_and_respects_close() {
    let mut queue = VecDeque::from([
        action("a"),
        action("b"),
        Job::Plan(json!([1])),
        link("one"),
        note("n"),
        action("c"),
        Job::Plan(json!([2])),
        link("two"),
        action("d"),
    ]);
    coalesce(&mut queue);
    assert_eq!(
        kinds(&queue),
        [
            "action b",
            "links one,two",
            "note n",
            "plan [2]",
            "action d"
        ]
    );

    let (done, _) = std::sync::mpsc::channel();
    let mut closed = VecDeque::from([action("a"), Job::Close(done), action("b")]);
    coalesce(&mut closed);
    assert_eq!(kinds(&closed), ["action a", "close", "action b"]);
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
        listen: None,
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

fn activity(content: Value) -> Value {
    json!({ "query": ACTIVITY, "variables": { "id": SESSION_ID, "content": content } })
}

fn passing(content: Value) -> Value {
    json!({
        "query": PASSING,
        "variables": { "id": SESSION_ID, "content": content, "ephemeral": true },
    })
}

/// Whether `request` adds an activity whose body is `body`.
fn adds(request: &Request, body: &str) -> bool {
    request
        .json()
        .pointer("/variables/content/body")
        .and_then(Value::as_str)
        == Some(body)
}

/// Runs the task on `config` with a webhook port, posts it `events` signed
/// with `secret` when given, then closes it: what it told after it started,
/// and the status of each post.
fn hear(mut config: Config, events: &[Value], secret: Option<&str>) -> (Vec<Told>, Vec<u16>) {
    let addr = stand_in::free_addr();
    config.listen = Some(Listen {
        addr: addr.clone(),
        secret: WEBHOOK_SECRET.to_owned(),
    });
    run_while(config, || {
        (events.iter())
            .map(|event| stand_in::post(&addr, event, secret).unwrap_or_default())
            .collect()
    })
}

/// Runs the task on `config` while `during` runs, then a little longer,
/// then closes it: what it told after it started, and what `during` gave.
fn run_while<T>(config: Config, during: impl FnOnce() -> T) -> (Vec<Told>, T) {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return (Vec::new(), during());
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
    let given = during();
    std::thread::sleep(Duration::from_millis(300));
    let (done, closed) = std::sync::mpsc::channel();
    queue.try_send(Job::Close(done)).ok();
    closed.recv_timeout(Duration::from_secs(10)).ok();
    let told = std::iter::from_fn(|| told.try_recv().ok())
        .filter(|news| !matches!(news, Told::Started(_)))
        .collect();
    (told, given)
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
                      filter: { type: { eq: \"prompt\" } }) { nodes { id createdAt signal user { name app \
                      } content { __typename ... on AgentActivityPromptContent { body } } } } } } }",
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
    assert!(
        !requests
            .iter()
            .any(|r| r.query().starts_with("query Claim"))
    );
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
                      { nodes { id createdAt signal user { name app } content { __typename ... on \
                      AgentActivityPromptContent { body } } } } answers: activities(first: 1, filter: \
                      { type: { \
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
fn startup_failures_answer_close_and_send_nothing_more() {
    let cases = [
        (
            Target::Issue("ENG-1".to_owned()),
            (401, Vec::new(), "{}".to_owned()),
        ),
        (
            Target::Session(SESSION_ID.to_owned()),
            (401, Vec::new(), "{}".to_owned()),
        ),
        (
            Target::Session(SESSION_ID.to_owned()),
            stand_in::reply(json!({ "agentSession": null })),
        ),
    ];
    for (target, response) in cases {
        let linear = StandIn::scripted(move |_, _| Some(response.clone()));
        let credentials = Credentials::Token("lin_api_key".to_owned());
        let started = run_with(config(&linear.url, credentials, target), vec![note("lost")]);
        assert_eq!(started, None);
        assert_eq!(linear.requests().len(), 1);
    }
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
fn token_is_redacted_from_both_plan_and_activities() {
    let linear = StandIn::start();
    let mut config = config(
        &linear.url,
        Credentials::Token("lin_api_key_of_the_test".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    config.plan = Some(json!([{ "content": "use lin_api_key_of_the_test", "status": "pending" }]));
    run_with(config, vec![note("the key is lin_api_key_of_the_test")]);
    let bodies = linear.bodies();
    assert_eq!(
        bodies
            .get(2)
            .and_then(|body| body.pointer("/variables/plan").cloned()),
        Some(json!([{ "content": "use [REDACTED]", "status": "pending" }]))
    );
    assert_eq!(
        bodies.get(3),
        Some(&activity(
            json!({ "type": "thought", "body": "the key is [REDACTED]" })
        ))
    );
}

#[test]
fn the_unanswered_prompts_then_each_one_heard_are_told_once_until_a_stop() {
    let linear = existing_session(
        &[
            (
                "p1",
                "2026-01-01T00:00:00.000Z",
                Some("Ana"),
                "Fix the bug.",
            ),
            ("p2", "2026-01-01T00:02:00.000Z", Some("Ana"), "Use sqlite."),
        ],
        Some("2026-01-01T00:01:00.000Z"),
    );
    let config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    let events = [
        prompted(SESSION_ID, "p2", "Ana", "Use sqlite.", false),
        prompted(SESSION_ID, "p3", "Ana", "And add a test.", false),
        // Linear delivers a webhook again when it got no answer in time.
        prompted(SESSION_ID, "p3", "Ana", "And add a test.", false),
        prompted("other-session", "p9", "Bo", "Not this run's.", false),
        prompted(SESSION_ID, "p4", "Ana", "", true),
        prompted(SESSION_ID, "p5", "Ana", "Too late.", false),
    ];
    let (told, statuses) = hear(config, &events, Some(WEBHOOK_SECRET));
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
    assert_eq!(statuses, [202; 6]);
}

#[test]
fn a_webhook_not_signed_with_the_secret_or_not_fresh_is_refused() {
    let linear = StandIn::start();
    let config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Session(SESSION_ID.to_owned()),
    );
    let event = prompted(SESSION_ID, "p1", "Ana", "Delete everything.", false);
    let mut stale = event.clone();
    if let Some(sent) = stale.get_mut("webhookTimestamp") {
        *sent = json!(sent.as_f64().unwrap_or_default() - 120_000.0);
    }
    let mut created = event.clone();
    if let Some(action) = created.get_mut("action") {
        *action = json!("created");
    }
    let addr = stand_in::free_addr();
    let mut config = config;
    config.listen = Some(Listen {
        addr: addr.clone(),
        secret: WEBHOOK_SECRET.to_owned(),
    });
    let (told, statuses) = run_while(config, || {
        [
            stand_in::post(&addr, &event, None),
            stand_in::post(&addr, &event, Some("not-the-secret")),
            stand_in::post(&addr, &stale, Some(WEBHOOK_SECRET)),
            stand_in::post(&addr, &created, Some(WEBHOOK_SECRET)),
        ]
    });
    assert_eq!(told, []);
    assert_eq!(statuses, [Some(401), Some(401), Some(401), Some(204)]);
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
    assert!(
        linear
            .requests()
            .iter()
            .any(|request| adds(request, "Looking."))
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
        if request.query().starts_with("query Session") {
            let known: Vec<Value> = prompts
                .iter()
                .rev()
                .map(|&(id, at, by, body)| prompt(id, at, by, body, false))
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
        None
    })
}

/// What a run on `target` of `linear` tells as it attaches.
fn told_on(linear: &StandIn, target: Target) -> Vec<Told> {
    let config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        target,
    );
    run_while(config, || ()).0
}

#[test]
fn the_prompts_on_a_session_the_run_creates_are_not_told_but_later_ones_are() {
    let linear = StandIn::scripted(|request, _| {
        request.query().contains("agentSessionCreateOnIssue").then(|| {
            stand_in::reply(json!({
                "agentSessionCreateOnIssue": {
                    "success": true,
                    "agentSession": {
                        "id": SESSION_ID,
                        "url": SESSION_URL,
                        "activities": { "nodes": [
                            prompt("p0", "2026-01-01T00:00:00.000Z", None, "Work on ENG-1", false),
                        ] },
                    },
                }
            }))
        })
    });
    let config = config(
        &linear.url,
        Credentials::Token("lin_api_key".to_owned()),
        Target::Issue("ENG-1".to_owned()),
    );
    let events = [
        prompted(SESSION_ID, "p0", "Rigged Intern", "Work on ENG-1", false),
        prompted(SESSION_ID, "p1", "Ana", "Human follow-up.", false),
    ];
    assert_eq!(
        hear(config, &events, Some(WEBHOOK_SECRET)).0,
        [Told::Prompt {
            body: "Human follow-up.".to_owned(),
            by: Some("Ana".to_owned()),
        }]
    );
}

#[test]
fn a_given_session_no_run_answered_tells_its_request_but_not_the_synthetic_prompt() {
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

#[test]
fn startup_requests_retry_after_429() {
    let attempts = std::sync::atomic::AtomicUsize::new(0);
    let linear = StandIn::scripted(move |request, _| {
        (request.query().starts_with("query Issue")
            && attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0)
            .then(|| (429, vec![("retry-after", "0".to_owned())], "{}".to_owned()))
    });
    let started = run_with(
        config(
            &linear.url,
            Credentials::Token("lin_api_key".to_owned()),
            Target::Issue("ENG-1".to_owned()),
        ),
        vec![note("After retry.")],
    );
    assert!(started.is_some());
    assert!(
        linear
            .requests()
            .iter()
            .any(|request| adds(request, "After retry."))
    );
}
