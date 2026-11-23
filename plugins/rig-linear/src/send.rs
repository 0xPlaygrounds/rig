//! The task that talks to Linear, on rig's tokio runtime: it signs in,
//! creates or attaches to the session, then sends what the app queues, one
//! request at a time, at most one a second. Consecutive passing activities
//! are coalesced, a 429 is retried after its `Retry-After`, a heartbeat
//! keeps the session from going stale, and every failure is a warning.

use std::collections::VecDeque;
use std::time::Duration;

use async_channel::{Receiver, Sender};
use bevy_log::warn;
use rig_core::http_client::DynHttpClient;
use serde_json::Value;
use tokio::time::Instant;

use crate::activity::{Activity, Content, Redactor, head};
use crate::api::{Api, ApiError, Credentials, Link, Session, authorization};

#[cfg(test)]
mod tests;

/// How many times a rate-limited request is sent again.
const RETRIES: usize = 3;

/// The wait after a 429 that names none.
const RETRY_WAIT: Duration = Duration::from_secs(5);

/// The longest wait after a 429 that names one.
const RETRY_MOST: Duration = Duration::from_secs(60);

/// What the heartbeat says.
const STILL_WORKING: &str = "Still working…";

/// One piece of work for the task, in the order queued.
#[derive(Debug)]
pub(crate) enum Job {
    /// Add an activity.
    Activity(Activity),
    /// Replace the plan.
    Plan(Value),
    /// Add external links.
    Links(Vec<Link>),
    /// Send what came before, then end, telling `done`.
    Close(std::sync::mpsc::Sender<()>),
}

/// Who to send to and how.
#[derive(Clone, Debug)]
pub(crate) struct Config {
    /// The GraphQL endpoint.
    pub(crate) api_url: String,
    /// How the requests are signed.
    pub(crate) credentials: Credentials,
    /// The session to create or attach to.
    pub(crate) target: Target,
    /// The plan as it starts, if there is one.
    pub(crate) plan: Option<Value>,
    /// The quiet after which a heartbeat goes.
    pub(crate) heartbeat: Duration,
    /// The least time between two requests.
    pub(crate) spacing: Duration,
}

/// The session a run shows on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// A new session on this issue (identifier or id).
    Issue(String),
    /// The session with this id, which exists.
    Session(String),
}

/// What the task tells the app once the session is there.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Started {
    /// Where a person sees the session.
    pub(crate) url: Option<String>,
    /// The issue's identifier, when known.
    pub(crate) issue: Option<String>,
}

/// Drops from `queue` each passing activity a later passing activity
/// replaces before any lasting activity or the close comes, keeps only the
/// last plan, and merges every queued link into the first links job.
pub(crate) fn coalesce(queue: &mut VecDeque<Job>) {
    let mut replaced = false;
    let mut later_plan = false;
    let mut keep = vec![true; queue.len()];
    for (at, job) in queue.iter().enumerate().rev() {
        match job {
            Job::Activity(activity) if activity.ephemeral => {
                if replaced && let Some(keep) = keep.get_mut(at) {
                    *keep = false;
                }
                replaced = true;
            }
            Job::Activity(_) | Job::Close(_) => replaced = false,
            Job::Plan(_) => {
                if later_plan && let Some(keep) = keep.get_mut(at) {
                    *keep = false;
                }
                later_plan = true;
            }
            Job::Links(_) => {}
        }
    }
    let mut kept = keep.into_iter();
    queue.retain(|_| kept.next().unwrap_or(true));
    let mut links = Vec::new();
    for job in queue.iter_mut() {
        if let Job::Links(more) = job {
            links.append(more);
        }
    }
    let mut first = true;
    queue.retain_mut(|job| match job {
        Job::Links(empty) if first && !links.is_empty() => {
            first = false;
            empty.append(&mut links);
            true
        }
        Job::Links(_) => false,
        _ => true,
    });
}

/// The task: starts the session, tells `started`, and sends `jobs` until
/// the close or until the app is gone.
pub(crate) async fn run(
    config: Config,
    http: DynHttpClient,
    redactor: Redactor,
    jobs: Receiver<Job>,
    started: Sender<Started>,
) {
    let mut sender = match start(&config, &http, &redactor).await {
        Ok(sender) => sender,
        Err(why) => {
            warn!("{}", redactor.text(&refused(&why)));
            // Nothing more is sent this run: answer the close, if one comes,
            // at once.
            while let Ok(job) = jobs.recv().await {
                if let Job::Close(done) = job {
                    done.send(()).ok();
                    return;
                }
            }
            return;
        }
    };
    let link = sender.session.url.clone();
    if let Some(url) = &link {
        eprintln!("Linear agent session: {url}");
    }
    started
        .send(Started {
            url: link,
            issue: sender.session.issue.clone(),
        })
        .await
        .ok();
    if let Some(plan) = config.plan.clone() {
        sender.send(Job::Plan(plan)).await;
    }
    sender.serve(jobs, config.heartbeat).await;
}

/// The warning for a session that could not start: Linear's own words when
/// it refused, which name a missing scope or an app not set up for agent
/// sessions.
pub(crate) fn refused(why: &ApiError) -> String {
    match why {
        ApiError::Reply(said) => format!(
            "Linear refused the agent session: {said}. Is the app configured for agent sessions?"
        ),
        why => format!("Linear: the agent session could not start: {why}"),
    }
}

/// Signs in and creates or attaches to the session.
async fn start(
    config: &Config,
    http: &DynHttpClient,
    redactor: &Redactor,
) -> Result<Connected, ApiError> {
    let token = match &config.credentials {
        Credentials::Token(token) => token.clone(),
        Credentials::Client { id, secret, scope } => {
            let token_url = crate::api::token_url(&config.api_url);
            Api::exchange(http, &token_url, id, secret, scope).await?
        }
    };
    let api = Api::new(http.clone(), config.api_url.clone(), authorization(&token));
    let session = match &config.target {
        Target::Issue(issue) => api.create_session(issue).await?,
        Target::Session(id) => match api.session(id).await {
            Ok(session) => session,
            Err(why) => {
                warn!(
                    "Linear: agent session {id} could not be read: {}",
                    redactor.text(&why.to_string())
                );
                Session {
                    id: id.clone(),
                    ..Session::default()
                }
            }
        },
    };
    let mut redactor = redactor.clone();
    redactor.add(token);
    Ok(Connected {
        api,
        session,
        redactor,
        spacing: config.spacing,
        last: None,
        last_action: None,
        ended: false,
    })
}

/// The task once the session is there.
struct Connected {
    api: Api,
    session: Session,
    redactor: Redactor,
    spacing: Duration,
    /// When the last request went.
    last: Option<Instant>,
    /// The last tool call sent, for the heartbeat.
    last_action: Option<String>,
    /// Whether the last activity sent ended the run, so no heartbeat goes.
    ended: bool,
}

impl Connected {
    /// Sends the jobs as they come, until the close.
    async fn serve(&mut self, jobs: Receiver<Job>, heartbeat: Duration) {
        let mut queue = VecDeque::new();
        loop {
            while let Ok(job) = jobs.try_recv() {
                queue.push_back(job);
            }
            if queue.is_empty() {
                let quiet = self
                    .last
                    .map_or(heartbeat, |last| heartbeat.saturating_sub(last.elapsed()));
                match tokio::time::timeout(quiet, jobs.recv()).await {
                    Ok(Ok(job)) => queue.push_back(job),
                    Ok(Err(_)) => return,
                    Err(_) if self.ended => self.last = Some(Instant::now()),
                    Err(_) => queue.push_back(Job::Activity(self.heartbeat())),
                }
                continue;
            }
            coalesce(&mut queue);
            if matches!(queue.front(), Some(Job::Close(_))) {
                if let Some(Job::Close(done)) = queue.pop_front() {
                    done.send(()).ok();
                }
                return;
            }
            self.pace().await;
            while let Ok(job) = jobs.try_recv() {
                queue.push_back(job);
            }
            coalesce(&mut queue);
            let Some(job) = queue.pop_front() else {
                continue;
            };
            if let Job::Close(done) = job {
                done.send(()).ok();
                return;
            }
            self.send(job).await;
        }
    }

    /// The heartbeat: still working, and the last tool call if any.
    fn heartbeat(&self) -> Activity {
        Activity::passing(match &self.last_action {
            Some(action) => format!("{STILL_WORKING} Last: {action}"),
            None => STILL_WORKING.to_owned(),
        })
    }

    /// Waits until the next request may go.
    async fn pace(&self) {
        if let Some(last) = self.last {
            tokio::time::sleep_until(last + self.spacing).await;
        }
    }

    /// Sends `job`, again after each 429 up to [`RETRIES`] times; warns
    /// when it fails.
    async fn send(&mut self, job: Job) {
        let job = match job {
            Job::Activity(activity) => Job::Activity(self.redactor.activity(activity)),
            Job::Links(links) => Job::Links(
                links
                    .into_iter()
                    .map(|link| Link {
                        label: self.redactor.text(&link.label),
                        url: self.redactor.text(&link.url),
                    })
                    .collect(),
            ),
            Job::Plan(mut plan) => {
                self.redactor.json(&mut plan);
                Job::Plan(plan)
            }
            job => job,
        };
        let mut tries = 0;
        loop {
            let sent = match &job {
                Job::Activity(activity) => {
                    self.api.create_activity(&self.session.id, activity).await
                }
                Job::Plan(plan) => {
                    self.api
                        .update_session(&self.session.id, Some(plan), &[])
                        .await
                }
                Job::Links(links) => self.api.update_session(&self.session.id, None, links).await,
                Job::Close(_) => Ok(()),
            };
            self.last = Some(Instant::now());
            match sent {
                Err(ApiError::RateLimited(wait)) if tries < RETRIES => {
                    tries += 1;
                    let wait = wait.unwrap_or(RETRY_WAIT).min(RETRY_MOST);
                    tokio::time::sleep(wait).await;
                }
                Err(why) => {
                    warn!(
                        "Linear: a request failed: {}",
                        self.redactor.text(&why.to_string())
                    );
                    return;
                }
                Ok(()) => break,
            }
        }
        if let Job::Activity(activity) = &job {
            self.ended = activity.content.ends();
            if let Content::Action {
                action, parameter, ..
            } = &activity.content
            {
                self.last_action = Some(head(&format!("{action} {parameter}"), 200));
            }
        }
    }
}
