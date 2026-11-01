//! Which of the plan's models the signed-in account may use, as the backend
//! lists them: asked once per run at first need, on the IO pool, and once
//! more after the backend said a model it was sent was not found. Until the
//! answer comes, listings say the plan's models are being checked and
//! requests wait for it. A list that could not be had is refused out loud,
//! never guessed, and not kept. A fetch nobody's request waits for retries
//! a transient failure itself, as a model call is retried; after a failure
//! the next request asks again, once any wait the backend asked for is
//! over, and so does a listing a while later.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

use bevy_ecs::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::catalog::{ModelSpec, ModelStatus};
use rig_core::client::chatgpt_models::{LIST_TIMEOUT, account_models};
use rig_core::client::signed_in::failed_sign_in;
use rig_core::effect::{Answered, EffectId, EffectKind};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::http_client::DynHttpClient;
use rig_core::providers::chatgpt::{self, auth::Authenticator};
use rig_core::serve::{AfterServe, BeforeServe, Intercept};
use rig_core::wasm_compat::{sleep, timeout};

use rig_ecs::agent::{ActiveTurn, Notice};
use rig_ecs::model::{Connection, ModelRefused, Models};
use rig_ecs::retry::{RetryPolicy, Verdict};
use rig_ecs::tasks::Wake;

use super::{PROVIDER, TITLE};

/// What the backend answered: the slugs the account may use, or why it
/// could not be asked, as a request refused for it reports it.
type Asked = Result<Arc<[String]>, ErrorReport>;

/// How hard asking tries.
#[derive(Clone, Copy, Debug)]
pub(super) struct Pace {
    /// How a fetch nobody's request waits for retries a transient failure.
    pub(super) retry: RetryPolicy,
    /// How long after a failure a listing asks again, at least.
    pub(super) cooldown: Duration,
}

/// Two retries, as a model call is retried, and a listing asks again 10 s
/// after a failure, which keeps the listing that follows each answer from
/// asking again at once.
impl Default for Pace {
    fn default() -> Self {
        Self {
            retry: RetryPolicy {
                max_retries: 2,
                ..RetryPolicy::DEFAULT
            },
            cooldown: LIST_TIMEOUT,
        }
    }
}

/// The account's models, shared by the sign-in, its models' requests and
/// the app.
#[derive(Default)]
pub(super) struct Offered {
    pace: Pace,
    state: Mutex<State>,
    /// Bumped each time an answer comes, so the app marks [`Models`]
    /// changed and listings are built again.
    answers: AtomicU64,
    /// Wakes the app's loop when an answer comes.
    wake: Mutex<Option<Wake>>,
}

#[derive(Default)]
struct State {
    asked: Asking,
    /// Which sign-in an answer is for: a fetch started before the last
    /// sign-in or sign-out is dropped.
    epoch: u64,
    /// Whether the one fetch after a model not found was started.
    again: bool,
}

#[derive(Default)]
enum Asking {
    #[default]
    NotYet,
    /// A fetch runs; each request waiting for it holds one of these.
    Running(Vec<async_channel::Sender<Asked>>),
    /// The account's models.
    Answered(Arc<[String]>),
    /// The last fetch failed, when, and how long the backend asked not to
    /// be asked again.
    Failed(ErrorReport, Instant, Duration),
}

impl Offered {
    /// No answer yet, asked at `pace`.
    #[cfg(test)]
    pub(super) fn paced(pace: Pace) -> Self {
        Self {
            pace,
            ..Self::default()
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Wakes `wake` when an answer comes.
    pub(super) fn wake_with(&self, wake: Wake) {
        if let Ok(mut held) = self.wake.lock() {
            *held = Some(wake);
        }
    }

    /// How many answers came so far.
    pub(super) fn answers(&self) -> u64 {
        self.answers.load(Ordering::Acquire)
    }

    /// The answer, if it came, or why the last fetch failed; asking starts
    /// on the IO pool if it had not, or if the failure is older than the
    /// [`Pace`]'s cooldown and the wait the backend asked for.
    fn now(self: &Arc<Self>, auth: &Authenticator, http: &DynHttpClient) -> Option<Asked> {
        let mut state = self.state();
        match &state.asked {
            Asking::Answered(listed) => Some(Ok(Arc::clone(listed))),
            Asking::Failed(failure, at, wait) if at.elapsed() < self.pace.cooldown.max(*wait) => {
                Some(Err(failure.clone()))
            }
            Asking::Running(_) => None,
            Asking::NotYet | Asking::Failed(..) => {
                self.start(&mut state, (auth, http), Vec::new(), true);
                None
            }
        }
    }

    /// Whether the account offers `spec`, at once: [`ModelStatus::Checking`]
    /// until the answer comes.
    pub(super) fn status(
        self: &Arc<Self>,
        spec: &ModelSpec,
        auth: &Authenticator,
        http: &DynHttpClient,
    ) -> ModelStatus {
        match self.now(auth, http) {
            None => ModelStatus::Checking,
            Some(asked) => match refusal(&asked, &spec.id) {
                None => ModelStatus::Usable,
                Some(why @ Refusal::NotOffered) => {
                    ModelStatus::NotOffered(why.text(&spec.reference()))
                }
                Some(why @ Refusal::Unasked(_)) => {
                    ModelStatus::Unavailable(why.text(&spec.reference()))
                }
            },
        }
    }

    /// Why a model of the plan `spec` names is refused, once the answer
    /// leaves it out, worded after `Cannot use <model>:`; `None` while the
    /// answer is not known or could not be had, so a model can connect and
    /// its request waits for the answer or asks again.
    pub(super) fn refused(
        self: &Arc<Self>,
        spec: &ModelSpec,
        auth: &Authenticator,
        http: &DynHttpClient,
    ) -> Option<String> {
        let listed = self.now(auth, http)?.ok()?;
        let offered = listed.contains(&spec.id);
        (!offered).then(|| Refusal::NotOffered.reason())
    }

    /// Why the model `slug` is refused, if the answer came and leaves it out.
    pub(super) fn known_refusal(&self, slug: &str) -> Option<Refusal> {
        match &self.state().asked {
            Asking::Answered(listed) if !listed.iter().any(|listed| listed == slug) => {
                Some(Refusal::NotOffered)
            }
            _ => None,
        }
    }

    /// The answer, waiting for it, asking first if nobody had or the last
    /// fetch failed, once the wait the backend asked for is over; the
    /// failure itself when that wait is longer than a model call would.
    async fn answer(self: &Arc<Self>, auth: &Authenticator, http: &DynHttpClient) -> Asked {
        loop {
            let (sender, waiting) = async_channel::bounded(1);
            let left = {
                let mut state = self.state();
                let left = match &state.asked {
                    Asking::Answered(listed) => return Ok(Arc::clone(listed)),
                    Asking::Failed(failure, at, wait) => {
                        let left = wait.saturating_sub(at.elapsed());
                        if left > RetryPolicy::DEFAULT.max_asked_wait {
                            return Err(failure.clone());
                        }
                        left
                    }
                    Asking::NotYet | Asking::Running(_) => Duration::ZERO,
                };
                match &mut state.asked {
                    Asking::Running(waiting) => waiting.push(sender),
                    _ if left.is_zero() => {
                        self.start(&mut state, (auth, http), vec![sender], false)
                    }
                    _ => {}
                }
                left
            };
            if !left.is_zero() {
                sleep(left).await;
                continue;
            }
            return waiting.recv().await.unwrap_or_else(|_| {
                let ended = "the model list request ended";
                Err(ErrorReport::new(ErrorKind::Cancelled, ended))
            });
        }
    }

    /// Asks again once in this run, after the backend did not find a model
    /// the account was listed with.
    fn ask_again(self: &Arc<Self>, auth: &Authenticator, http: &DynHttpClient) {
        let mut state = self.state();
        if state.again || !matches!(state.asked, Asking::Answered(_)) {
            return;
        }
        state.again = true;
        self.start(&mut state, (auth, http), Vec::new(), true);
    }

    /// Forgets the answer, as a sign-in or sign-out may change the account,
    /// so listings are built again.
    pub(super) fn forget(&self) {
        let mut state = self.state();
        let epoch = state.epoch.wrapping_add(1);
        *state = State {
            epoch,
            ..State::default()
        };
        self.answers.fetch_add(1, Ordering::AcqRel);
    }

    /// Starts the one fetch on the IO pool, which answers `waiting` too:
    /// retried as the [`Pace`] says when `retries`, since no request does.
    fn start(
        self: &Arc<Self>,
        state: &mut State,
        (auth, http): (&Authenticator, &DynHttpClient),
        waiting: Vec<async_channel::Sender<Asked>>,
        retries: bool,
    ) {
        state.asked = Asking::Running(waiting);
        let (epoch, offered) = (state.epoch, Arc::clone(self));
        let (auth, http) = (auth.clone(), http.clone());
        let policy = self.pace.retry;
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let asking = async move {
            let mut retried = 0;
            loop {
                let asked = fetch(&auth, &http).await;
                let verdict = match &asked {
                    Err(failure) if retries => policy.verdict(failure, retried, SystemTime::now()),
                    _ => Verdict::Final,
                };
                let Verdict::Retry(wait) = verdict else {
                    return offered.answered(epoch, asked);
                };
                retried += 1;
                sleep(wait).await;
            }
        };
        pool.spawn(asking).detach();
    }

    /// Keeps `asked`, unless a sign-in came since it was asked, and tells
    /// the requests waiting for it and the app.
    fn answered(&self, epoch: u64, asked: Asked) {
        let waiting = {
            let mut state = self.state();
            if state.epoch != epoch {
                return;
            }
            let kept = match &asked {
                Ok(listed) => Asking::Answered(Arc::clone(listed)),
                Err(failure) => {
                    let wait = failure.retry_after(SystemTime::now()).unwrap_or_default();
                    Asking::Failed(failure.clone(), Instant::now(), wait)
                }
            };
            let previous = std::mem::replace(&mut state.asked, kept);
            // Counted under the lock: whoever sees the answer sees it counted.
            self.answers.fetch_add(1, Ordering::AcqRel);
            match previous {
                Asking::Running(waiting) => waiting,
                _ => Vec::new(),
            }
        };
        for waiter in waiting {
            waiter.try_send(asked.clone()).ok();
        }
        if let Ok(wake) = self.wake.lock()
            && let Some(wake) = wake.as_ref()
        {
            wake.wake();
        }
    }
}

/// The account's models, asked once within the list's deadline, which also
/// covers a refresh of the credential that waits for other rig processes.
async fn fetch(auth: &Authenticator, http: &DynHttpClient) -> Asked {
    let asked = timeout(LIST_TIMEOUT, async {
        match auth.auth_context(http).await {
            Ok(context) => account_models(&context, http)
                .await
                .map(Arc::from)
                .map_err(|error| ErrorReport::from(&error)),
            Err(error) => Err(failed_sign_in(error)),
        }
    });
    asked.await.unwrap_or_else(|_| {
        let why = format!("no answer in {} s", LIST_TIMEOUT.as_secs());
        Err(ErrorReport::new(ErrorKind::Timeout, why).with_retryable(true))
    })
}

/// Why a model is refused by an answer.
pub(super) enum Refusal {
    /// The account does not offer it.
    NotOffered,
    /// The backend could not be asked, for this reason.
    Unasked(ErrorReport),
}

impl Refusal {
    /// The reason, worded after `Cannot use <model>:`.
    fn reason(&self) -> String {
        match self {
            Self::NotOffered => format!("not offered to this {TITLE} account"),
            Self::Unasked(failure) => format!(
                "could not ask {TITLE} which models this account may use ({})",
                failure.message
            ),
        }
    }

    /// The refusal of the model `reference`, whole.
    fn text(&self, reference: &str) -> String {
        match self {
            Self::NotOffered => format!("{reference} is {}", self.reason()),
            Self::Unasked(_) => self.reason(),
        }
    }

    /// The report of a request refused for this reason: the failure's
    /// own, with its class and any wait it asked for, when the backend
    /// could not be asked.
    fn report(self, reference: &str) -> ErrorReport {
        let message = self.text(reference);
        match self {
            Self::NotOffered => ErrorReport::new(ErrorKind::Provider, message),
            Self::Unasked(failure) => ErrorReport { message, ..failure },
        }
    }
}

/// Why `asked` refuses the model `slug`, if it does.
fn refusal(asked: &Asked, slug: &str) -> Option<Refusal> {
    match asked {
        Ok(listed) if listed.iter().any(|listed| listed == slug) => None,
        Ok(_) => Some(Refusal::NotOffered),
        Err(failure) => Some(Refusal::Unasked(failure.clone())),
    }
}

/// The layer on each of the plan's models: a request waits for the answer
/// and is refused, unsent, when it does not offer the model; a reply the
/// backend refused as `model_not_found` has the list asked again, once.
pub(super) struct Gate {
    pub(super) offered: Arc<Offered>,
    pub(super) spec: ModelSpec,
    pub(super) auth: Authenticator,
    pub(super) http: DynHttpClient,
}

impl Intercept for Gate {
    fn name(&self) -> String {
        "account-models".to_owned()
    }

    /// Checks the model the request is sent to: one it names itself, as
    /// the ChatGPT wire resolves it, else the connected one.
    async fn before(&self, _: EffectId, kind: &EffectKind) -> BeforeServe {
        let asked = self.offered.answer(&self.auth, &self.http).await;
        let named = match kind {
            EffectKind::Completion { request, .. } => chatgpt::requested_model(request),
            _ => None,
        };
        let sent = named.unwrap_or(&self.spec.id);
        match refusal(&asked, sent) {
            None => BeforeServe::Proceed,
            Some(why) => BeforeServe::Deny(why.report(&format!("{PROVIDER}/{sent}"))),
        }
    }

    async fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        outcome: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        if let Err(report) = outcome
            && report.code.as_deref() == Some("model_not_found")
        {
            self.offered.ask_again(&self.auth, &self.http);
        }
        AfterServe::Keep
    }
}

/// Marks the [`Models`] changed once an answer came, so pickers, completions
/// and editors list what the account offers, and agents refused before
/// connect again.
pub(super) fn show_answers(
    session: Res<super::Session>,
    mut models: ResMut<Models>,
    mut seen: Local<u64>,
) {
    let answers = session.offered.answers();
    if answers != *seen {
        *seen = answers;
        models.set_changed();
    }
}

/// Refuses, with an error notice, the connection of an idle agent to one
/// of the plan's models the answer refuses, such as after a model was not
/// found and the list was asked again: its front shows the model refused,
/// and choosing it again is refused too. An agent whose turn runs is
/// refused once it ends; its request is refused unsent meanwhile.
pub(super) fn refuse_unoffered(
    session: Res<super::Session>,
    agents: Query<(Entity, &Connection), (Without<ActiveTurn>, Without<ModelRefused>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (agent, connection) in &agents {
        refuse(agent, connection, &session, &mut commands, &mut notices);
    }
}

/// Refuses `agent`'s `connection` if it is one of the plan's models the
/// answer refuses.
fn refuse(
    agent: Entity,
    Connection { spec, .. }: &Connection,
    session: &super::Session,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    if spec.provider.vendor() != PROVIDER {
        return;
    }
    let Some(why) = session.offered.known_refusal(&spec.id) else {
        return;
    };
    let why = format!("Cannot use {}: {}.", spec.reference(), why.reason());
    commands
        .entity(agent)
        .remove::<Connection>()
        .insert(ModelRefused(why.clone()));
    notices.write(Notice::error(agent, why));
}
