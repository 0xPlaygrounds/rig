//! The task that talks to Linear, on rig's tokio runtime: it signs in,
//! creates or attaches to the session, says it is working, claims the issue
//! when asked, then sends what the app queues, one request at a time, at
//! most one a second, and tells the app the prompts the webhook port hears
//! of. Consecutive passing activities are coalesced, a 429 is retried after
//! its `Retry-After`, a heartbeat keeps the session from going stale, and
//! every failure is a warning.

use std::collections::{HashSet, VecDeque};
use std::time::Duration;

use async_channel::{Receiver, Sender};
use bevy_log::warn;
use rig_core::http_client::DynHttpClient;
use serde_json::Value;
use tokio::time::Instant;

use crate::activity::{Activity, Content, Redactor, head};
use crate::api::{Api, ApiError, Credentials, Link, Seen, Session, authorization};
use crate::webhook::{Listener, Prompted};

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
    /// The port that hears of the session's new prompts; `None` for none.
    pub(crate) listen: Option<Listen>,
    /// Whether the issue is delegated to the app and moved to started.
    pub(crate) claim: bool,
}

/// Where the webhooks of the session's prompts come in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Listen {
    /// The address to bind, such as `0.0.0.0:8787`.
    pub(crate) addr: String,
    /// The webhook's signing secret.
    pub(crate) secret: String,
}

/// The session a run shows on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    /// A new session on this issue (identifier or id).
    Issue(String),
    /// The session with this id, which exists, given to the run: its prompts
    /// that came after the session's last answer (response, error or
    /// question) are the run's request, and are told as it attaches.
    Session(String),
    /// The session with this id, which this rig session showed on before a
    /// reload: every prompt on it at attach time was told already.
    Saved(String),
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

/// The task: opens the webhook port when asked, starts the session, tells
/// `told` so and the prompts no run answered yet, says it is working,
/// claims the issue when asked, and sends `jobs` until the close or until
/// the app is gone, telling `told` each new prompt the port hears of.
pub(crate) async fn run(
    config: Config,
    http: DynHttpClient,
    redactor: Redactor,
    jobs: Receiver<Job>,
    told: Sender<Told>,
) {
    let (heard, prompts) = async_channel::unbounded();
    let listener = match &config.listen {
        Some(listen) => {
            match crate::webhook::listen(&listen.addr, listen.secret.clone(), heard).await {
                Ok(listener) => {
                    eprintln!(
                        "Linear webhooks: http://{}{}",
                        listener.addr,
                        crate::webhook::PATH
                    );
                    Some(listener)
                }
                Err(why) => {
                    warn!(
                        "Linear: the webhook port {} was not opened, so no prompt reaches the run: {why}",
                        listen.addr
                    );
                    None
                }
            }
        }
        None => None,
    };
    let (mut sender, unanswered) = match start(&config, &http, &redactor).await {
        Ok(started) => started,
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
    for news in unanswered {
        told.send(news).await.ok();
    }
    sender.pacer.pace().await;
    sender
        .send(Job::Activity(Activity::note(WORKING.to_owned())))
        .await;
    if let Some(listener) = listener {
        sender.prompts = Some((listener, prompts));
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

/// Runs `call`, pacing before it and retrying on 429 up to [`RETRIES`] times.
async fn paced_retry<T, F, Fut>(pacer: &mut Pacer, mut call: F) -> Result<T, ApiError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, ApiError>>,
{
    let mut tries = 0;
    loop {
        pacer.pace().await;
        let result = call().await;
        pacer.mark();
        match result {
            Err(ApiError::RateLimited(wait)) if tries < RETRIES => {
                tries += 1;
                tokio::time::sleep(retry_delay(wait, pacer.spacing)).await;
            }
            other => return other,
        }
    }
}

/// Signs in and creates or attaches to the session, pacing every request
/// and honoring 429 `Retry-After` up to [`RETRIES`] times: the task, and
/// what the session's prompts no run answered yet tell.
async fn start(
    config: &Config,
    http: &DynHttpClient,
    redactor: &Redactor,
) -> Result<(Connected, Vec<Told>), ApiError> {
    let mut pacer = Pacer::new(config.spacing);
    let token = match &config.credentials {
        Credentials::Token(token) => token.clone(),
        Credentials::Client { id, secret, scope } => {
            let token_url = crate::api::token_url(&config.api_url);
            paced_retry(&mut pacer, || {
                Api::exchange(http, &token_url, id, secret, scope)
            })
            .await?
        }
    };
    let api = Api::new(http.clone(), config.api_url.clone(), authorization(&token));
    let session = match &config.target {
        Target::Issue(issue) => {
            let (issue_id, identifier) = paced_retry(&mut pacer, || api.issue(issue)).await?;
            paced_retry(&mut pacer, || {
                api.create_session_on_issue(&issue_id, identifier.clone())
            })
            .await?
        }
        Target::Session(id) | Target::Saved(id) => {
            paced_retry(&mut pacer, || api.session(id)).await?
        }
    };
    let mut redactor = redactor.clone();
    redactor.add(token);
    let catch_up = matches!(config.target, Target::Session(_));
    let (inbox, unanswered) = Inbox::attach(&session, catch_up);
    let connected = Connected {
        api,
        session,
        redactor,
        pacer,
        last_write: None,
        last_action: None,
        ended: false,
        inbox,
        prompts: None,
    };
    Ok((connected, unanswered))
}

/// What the task knows of the session's prompts, to tell each new one
/// once.
#[derive(Debug)]
struct Inbox {
    /// The session's id.
    session: String,
    /// The ids of the prompts already known (including those present at
    /// attachment).
    seen: HashSet<String>,
    /// Whether prompts are still told: a stop ends that.
    open: bool,
    /// Whether a prompt of another session came, so the next one warns no
    /// more.
    strayed: bool,
}

impl Inbox {
    /// The inbox of `session`, which knows the prompts on it at attach
    /// time, and what those no run answered yet tell when `catch_up`.
    fn attach(session: &Session, catch_up: bool) -> (Self, Vec<Told>) {
        let mut inbox = Self {
            session: session.id.clone(),
            seen: HashSet::new(),
            open: true,
            strayed: false,
        };
        let told = (session.unanswered(catch_up).into_iter())
            .filter_map(|seen| inbox.take(seen))
            .collect();
        (inbox.seen).extend(session.initial_prompts.iter().map(|seen| seen.id.clone()));
        (inbox, told)
    }

    /// What `prompted` tells, when it is a new prompt of the session.
    fn receive(&mut self, prompted: Prompted) -> Option<Told> {
        if prompted.session != self.session {
            if !std::mem::replace(&mut self.strayed, true) {
                warn!(
                    "Linear: a webhook for the agent session {} reached the run of {}",
                    prompted.session, self.session
                );
            }
            return None;
        }
        self.take(prompted.seen)
    }

    /// What `seen` tells, unless it is known already, it is an app's, it is
    /// blank, or a stop came before it.
    fn take(&mut self, seen: Seen) -> Option<Told> {
        if !self.seen.insert(seen.id.clone()) || !self.open {
            return None;
        }
        let told = match (seen.stop, seen.from_app, seen.prompt) {
            (true, _, _) => Told::Stop { by: seen.by },
            (false, false, Some(body)) if !body.trim().is_empty() => Told::Prompt {
                body: body.trim().to_owned(),
                by: seen.by,
            },
            _ => return None,
        };
        self.open = !matches!(told, Told::Stop { .. });
        Some(told)
    }
}

/// The next prompt `prompts` hears of; never without a port.
async fn next(prompts: Option<&Receiver<Prompted>>) -> Option<Prompted> {
    match prompts {
        Some(prompts) => prompts.recv().await.ok(),
        None => std::future::pending().await,
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
    /// The prompts of the session already known.
    inbox: Inbox,
    /// The webhook port and the prompts it hears of, when it is open.
    prompts: Option<(Listener, Receiver<Prompted>)>,
}

impl Connected {
    /// Sends the jobs as they come, until the close, telling `told` the
    /// new prompts as the webhook port hears of them.
    async fn serve(&mut self, jobs: Receiver<Job>, heartbeat: Duration, told: &Sender<Told>) {
        let mut queue = VecDeque::new();
        loop {
            while let Ok(job) = jobs.try_recv() {
                queue.push_back(job);
            }
            self.tell_heard(told).await;
            coalesce(&mut queue);
            if matches!(queue.front(), Some(Job::Close(_))) {
                if let Some(Job::Close(done)) = queue.pop_front() {
                    done.send(()).ok();
                }
                return;
            }
            if queue.is_empty() {
                let quiet = self
                    .last_write
                    .map_or(heartbeat, |last| heartbeat.saturating_sub(last.elapsed()));
                let heard = self.prompts.as_ref().map(|(_, prompts)| prompts.clone());
                tokio::select! {
                    waited = tokio::time::timeout(quiet, jobs.recv()) => match waited {
                        Ok(Ok(job)) => queue.push_back(job),
                        Ok(Err(_)) => return,
                        Err(_) if self.ended => self.last_write = Some(Instant::now()),
                        Err(_) => queue.push_back(Job::Activity(self.heartbeat())),
                    },
                    Some(prompted) = next(heard.as_ref()) => {
                        if let Some(news) = self.inbox.receive(prompted) {
                            told.send(news).await.ok();
                        }
                    }
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

    /// Tells `told` the new prompts the webhook port heard of so far.
    async fn tell_heard(&mut self, told: &Sender<Told>) {
        let Some((_, prompts)) = &self.prompts else {
            return;
        };
        while let Ok(prompted) = prompts.try_recv() {
            if let Some(news) = self.inbox.receive(prompted) {
                told.send(news).await.ok();
            }
        }
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
