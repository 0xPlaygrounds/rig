use std::time::Duration;

use serde_json::json;

use super::*;
use crate::tests::stand_in::{Request, SESSION_ID, SESSION_URL, StandIn};

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
    }
}

/// Runs the task on `config` with `jobs` queued before it starts, then
/// the close; what it said it started, if it did.
fn run_with(config: Config, jobs: Vec<Job>) -> Option<Started> {
    let Ok(runtime) = tokio::runtime::Runtime::new() else {
        return None;
    };
    let (queue, queued) = async_channel::unbounded();
    let (tell, started) = async_channel::unbounded();
    for job in jobs {
        queue.try_send(job).ok();
    }
    let (done, closed) = std::sync::mpsc::channel();
    queue.try_send(Job::Close(done)).ok();
    let redactor = Redactor::new(Vec::new());
    runtime.spawn(run(config, rig_reqwest::shared(), redactor, queued, tell));
    closed.recv_timeout(Duration::from_secs(20)).ok()?;
    started.try_recv().ok()
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
                      { issueId: $issueId }) { success agentSession { id url } } }",
            "variables": { "issueId": "issue-uuid" },
        }),
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
            url: Some(SESSION_URL.to_owned()),
            issue: Some("ENG-1".to_owned()),
        })
    );
    let requests = linear.requests();
    let paths: Vec<&str> = requests
        .iter()
        .map(|request| request.path.as_str())
        .collect();
    assert_eq!(paths, ["/oauth/token", "/graphql", "/graphql"]);
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
            "query": "query Session($id: String!) { agentSession(id: $id) { id url issue { identifier } } }",
            "variables": { "id": SESSION_ID },
        }))
    );
    assert_eq!(
        bodies.get(2),
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
        (adds(request, "first") && index == 1).then(|| {
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
        .skip(1)
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
        .skip(1)
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
    assert_eq!(bodies.get(2), Some(&beat), "{bodies:?}");
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
    assert_eq!(linear.requests().len(), 2);
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
        bodies.get(1),
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
        .get(1)
        .and_then(|body| body.pointer("/variables/plan").cloned());
    assert_eq!(
        plan,
        Some(json!([{ "content": "use [REDACTED]", "status": "pending" }]))
    );
}
