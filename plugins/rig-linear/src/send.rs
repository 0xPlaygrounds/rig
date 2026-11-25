//! The task that talks to Linear, on rig's tokio runtime: it signs in,
//! creates or attaches to the session, says it is working, claims the issue
//! when asked, then sends what the app queues and reads the session's new
//! prompts back every so often, one request at a time, at most one a
//! second. Consecutive passing activities are coalesced, a 429 is retried
//! after its `Retry-After`, a heartbeat keeps the session from going stale,
//! and every failure is a warning.

use std::collections::{HashSet, VecDeque};
use std::time::Duration;

use async_channel::{Receiver, Sender};
use bevy_log::warn;
use rig_core::http_client::DynHttpClient;
use serde_json::Value;
use tokio::time::Instant;

use crate::activity::{Activity, Content, Redactor, head};
use crate::api::{Api, ApiError, Credentials, Link, Seen, Session, authorization};

#[cfg(test)]
mod tests;

/// How many times a rate-limited request is sent again.
const RETRIES: usize = 3;

/// The wait after a 429 that names none.
const RETRY_WAIT: Duration = Duration::from_secs(5);

/// How long a rate-limited request waits before trying again: honors the
/// server's `Retry-After` in full and never goes below `spacing`.
pub(crate) fn retry_delay(wait: Option<Duration>, spacing: Duration) -> Duration {
    wait.unwrap_or(RETRY_WAIT).max(spacing)
}

/// What the heartbeat says.
const STILL_WORKING: &str = "Still working…";

/// Enforces the minimum delay between two Linear API requests.
struct Pacer {
    spacing: Duration,
    last: Option<Instant>,
}

impl Pacer {
    fn new(spacing: Duration) -> Self {
        Self {
            spacing,
            last: None,
        }
    }

    /// Waits until the next request may go.
    async fn pace(&self) {
        if let Some(last) = self.last {
            tokio::time::sleep_until(last + self.spacing).await;
        }
    }

    /// Notes that a request just finished.
    fn mark(&mut self) {
        self.last = Some(Instant::now());
    }
}

/// The thought that goes as soon as the session is there.
pub(crate) const WORKING: &str = "Working on it.";

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
    /// How often the session's new prompts are read; `None` for never.
    pub(crate) poll: Option<Duration>,
    /// Whether the issue is delegated to the app and moved to started.
    pub(crate) claim: bool,
}

/// The session a run shows on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// A new session on this issue (identifier or id).
    Issue(String),
    /// The session with this id, which exists.
    Session(String),
}

/// What the task tells the app.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Told {
    /// The session is there.
    Started(Started),
    /// A person wrote `body` in the session.
    Prompt {
        body: String,
        /// The person's name.
        by: Option<String>,
    },
    /// A person asked the agent to stop.
    Stop {
        /// The person's name.
        by: Option<String>,
    },
}

/// What the task tells the app once the session is there.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Started {
    /// The agent session's id.
    pub(crate) id: String,
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

/// The task: starts the session, tells `told` so, says it is working,
/// claims the issue when asked, and sends `jobs` until the close or until
/// the app is gone, telling `told` each new prompt it reads back.
pub(crate) async fn run(
    config: Config,
    http: DynHttpClient,
    redactor: Redactor,
    jobs: Receiver<Job>,
    told: Sender<Told>,
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
    told.send(Told::Started(Started {
        id: sender.session.id.clone(),
        url: link,
        issue: sender.session.issue.clone(),
    }))
    .await
    .ok();
    sender.pacer.pace().await;
    sender
        .send(Job::Activity(Activity::note(WORKING.to_owned())))
        .await;
    if let Some(every) = config.poll {
        sender.watch = Some(Watch::new(every, &sender.session.initial_prompts));
    }
    if let Some(plan) = config.plan.clone() {
        sender.pacer.pace().await;
        sender.send(Job::Plan(plan)).await;
    }
    if config.claim {
        sender.claim().await;
    }
    sender.serve(jobs, config.heartbeat, &told).await;
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

/// Signs in and creates or attaches to the session, pacing every request.
async fn start(
    config: &Config,
    http: &DynHttpClient,
    redactor: &Redactor,
) -> Result<Connected, ApiError> {
    let mut pacer = Pacer::new(config.spacing);
    let token = match &config.credentials {
        Credentials::Token(token) => token.clone(),
        Credentials::Client { id, secret, scope } => {
            let token_url = crate::api::token_url(&config.api_url);
            let token = Api::exchange(http, &token_url, id, secret, scope).await?;
            pacer.mark();
            token
        }
    };
    let api = Api::new(http.clone(), config.api_url.clone(), authorization(&token));
    let session = match &config.target {
        Target::Issue(issue) => {
            pacer.pace().await;
            let (issue_id, identifier) = api.issue(issue).await?;
            pacer.mark();
            pacer.pace().await;
            let session = api.create_session_on_issue(&issue_id, identifier).await?;
            pacer.mark();
            session
        }
        Target::Session(id) => {
            pacer.pace().await;
            let session = api.session(id).await?;
            pacer.mark();
            session
        }
    };
    let mut redactor = redactor.clone();
    redactor.add(token);
    Ok(Connected {
        api,
        session,
        redactor,
        pacer,
        last_write: None,
        last_action: None,
        ended: false,
        watch: None,
    })
}

/// The most pages of new prompts one read goes through.
const PAGES: usize = 4;

/// What the task knows of the session's prompts, to read only those created
/// after the run created or attached to the session.
#[derive(Debug)]
struct Watch {
    /// How often it reads.
    every: Duration,
    /// When it reads next.
    next: Instant,
    /// The newest prompt's `createdAt`.
    since: Option<String>,
    /// The ids of the prompts already known (including those present at
    /// attachment).
    seen: HashSet<String>,
    /// Whether the last read failed, so the next failure warns no more.
    failing: bool,
}

impl Watch {
    fn new(every: Duration, initial: &[(String, String)]) -> Self {
        let mut seen = HashSet::new();
        let mut since: Option<String> = None;
        for (id, created_at) in initial {
            seen.insert(id.clone());
            if since.as_ref().is_none_or(|cur| cur < created_at) {
                since = Some(created_at.clone());
            }
        }
        Self {
            every,
            next: Instant::now(),
            since,
            seen,
            failing: false,
        }
    }

    /// Takes in `read`, newest first: what the new ones tell, oldest first.
    fn take(&mut self, mut read: Vec<Seen>) -> Vec<Told> {
        read.retain(|seen| self.seen.insert(seen.id.clone()));
        read.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        if let Some(newest) = read.last()
            && self
                .since
                .as_ref()
                .is_none_or(|since| *since < newest.created_at)
        {
            self.since = Some(newest.created_at.clone());
        }
        read.into_iter()
            .filter_map(|seen| match (seen.stop, seen.from_app, seen.prompt) {
                (true, _, _) => Some(Told::Stop { by: seen.by }),
                (false, false, Some(body)) if !body.trim().is_empty() => Some(Told::Prompt {
                    body: body.trim().to_owned(),
                    by: seen.by,
                }),
                _ => None,
            })
            .collect()
    }
}

/// The task once the session is there.
struct Connected {
    api: Api,
    session: Session,
    redactor: Redactor,
    pacer: Pacer,
    /// When the last request that adds to the session went, for the
    /// heartbeat.
    last_write: Option<Instant>,
    /// The last tool call sent, for the heartbeat.
    last_action: Option<String>,
    /// Whether the last activity sent ended the run, so no heartbeat goes.
    ended: bool,
    /// The session's prompts, when they are read back.
    watch: Option<Watch>,
}

impl Connected {
    /// Sends the jobs as they come, until the close, and reads the new
    /// prompts when it is time, telling `told` what they say.
    async fn serve(&mut self, jobs: Receiver<Job>, heartbeat: Duration, told: &Sender<Told>) {
        let mut queue = VecDeque::new();
        loop {
            while let Ok(job) = jobs.try_recv() {
                queue.push_back(job);
            }
            coalesce(&mut queue);
            if matches!(queue.front(), Some(Job::Close(_))) {
                if let Some(Job::Close(done)) = queue.pop_front() {
                    done.send(()).ok();
                }
                return;
            }
            if self
                .watch
                .as_ref()
                .is_some_and(|watch| watch.next <= Instant::now())
            {
                self.pacer.pace().await;
                for news in self.poll().await {
                    told.send(news).await.ok();
                }
                continue;
            }
            if queue.is_empty() {
                let quiet = self
                    .last_write
                    .map_or(heartbeat, |last| heartbeat.saturating_sub(last.elapsed()));
                let poll = (self.watch.as_ref())
                    .map(|watch| watch.next.saturating_duration_since(Instant::now()));
                let wait = poll.map_or(quiet, |poll| poll.min(quiet));
                match tokio::time::timeout(wait, jobs.recv()).await {
                    Ok(Ok(job)) => queue.push_back(job),
                    Ok(Err(_)) => return,
                    // The read is due, not the heartbeat.
                    Err(_) if wait < quiet => {}
                    Err(_) if self.ended => self.last_write = Some(Instant::now()),
                    Err(_) => queue.push_back(Job::Activity(self.heartbeat())),
                }
                continue;
            }
            self.pacer.pace().await;
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

    /// Reads the session's prompts since the last read: what the new ones
    /// tell. A stop ends the reading. A failed read warns, once until one
    /// succeeds, and the next read tries again.
    async fn poll(&mut self) -> Vec<Told> {
        let Some(mut watch) = self.watch.take() else {
            return Vec::new();
        };
        let since = watch.since.clone();
        let mut read = Vec::new();
        let mut after = None;
        let mut failed = None;
        for page in 0..PAGES {
            if page > 0 {
                self.pacer.pace().await;
            }
            let got = (self.api)
                .activities(&self.session.id, since.as_deref(), after.as_deref())
                .await;
            self.pacer.mark();
            match got {
                Ok(page) => {
                    read.extend(page.activities);
                    after = page.next;
                }
                Err(why) => {
                    failed = Some(why);
                    break;
                }
            }
            if after.is_none() {
                break;
            }
        }
        watch.next = Instant::now() + watch.every;
        let told = match failed {
            None => {
                watch.failing = false;
                watch.take(read)
            }
            Some(why) => {
                if let ApiError::RateLimited(wait) = &why {
                    let wait = retry_delay(*wait, self.pacer.spacing);
                    watch.next = Instant::now() + wait.max(watch.every);
                }
                if !std::mem::replace(&mut watch.failing, true) {
                    warn!(
                        "Linear: the session's prompts could not be read: {}",
                        self.redactor.text(&why.to_string())
                    );
                }
                Vec::new()
            }
        };
        if !told.iter().any(|news| matches!(news, Told::Stop { .. })) {
            self.watch = Some(watch);
        }
        told
    }

    /// Delegates the session's issue to the app when it has no delegate,
    /// and moves it to its team's first `started` state unless it is
    /// started, completed or canceled. A failure warns.
    async fn claim(&mut self) {
        let Some(issue) = self.session.issue_id.clone() else {
            return;
        };
        self.pacer.pace().await;
        let state = self.api.issue_state(&issue).await;
        self.pacer.mark();
        let claimed = match state.map(|state| state.claim()) {
            Ok(Some(input)) => {
                self.pacer.pace().await;
                let updated = self.api.update_issue(&issue, input).await;
                self.pacer.mark();
                updated
            }
            Ok(None) => Ok(()),
            Err(why) => Err(why),
        };
        if let Err(why) = claimed {
            warn!(
                "Linear: the issue was not claimed: {}",
                self.redactor.text(&why.to_string())
            );
        }
    }

    /// The heartbeat: still working, and the last tool call if any.
    fn heartbeat(&self) -> Activity {
        Activity::passing(match &self.last_action {
            Some(action) => format!("{STILL_WORKING} Last: {action}"),
            None => STILL_WORKING.to_owned(),
        })
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
            self.pacer.mark();
            self.last_write = self.pacer.last;
            match sent {
                Err(ApiError::RateLimited(wait)) if tries < RETRIES => {
                    tries += 1;
                    tokio::time::sleep(retry_delay(wait, self.pacer.spacing)).await;
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
