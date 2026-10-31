//! Which of the plan's models the signed-in account may use, as the backend
//! lists them: asked once per run at first need, on the IO pool, and once
//! more after the backend said a model it was sent was not found. Until the
//! answer comes, listings say the plan's models are being checked and
//! requests wait for it; a failure is refused out loud, never guessed.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use bevy_ecs::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::catalog::{ModelSpec, ModelStatus};
use rig_core::effect::{Answered, EffectId, EffectKind};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::http_client::DynHttpClient;
use rig_core::providers::chatgpt::auth::Authenticator;
use rig_core::providers::chatgpt::models::{LIST_TIMEOUT, account_models};
use rig_core::serve::{AfterServe, BeforeServe, Intercept};
use rig_core::wasm_compat::timeout;

use rig_ecs::model::Models;
use rig_ecs::tasks::Wake;

use super::TITLE;

/// What the backend answered: the slugs the account may use, or why it
/// could not be asked.
type Asked = Result<Arc<[String]>, String>;

/// The account's models, shared by the sign-in, its models' requests and
/// the app.
#[derive(Default)]
pub(super) struct Offered {
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
    Answered(Asked),
}

impl Offered {
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

    /// The answer, if it came; asking starts on the IO pool if it had not.
    fn now(self: &Arc<Self>, auth: &Authenticator, http: &DynHttpClient) -> Option<Asked> {
        let mut state = self.state();
        match &state.asked {
            Asking::Answered(asked) => Some(asked.clone()),
            Asking::Running(_) => None,
            Asking::NotYet => {
                self.start(&mut state, auth, http);
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
            Some(asked) => match refusal(&asked, spec) {
                None => ModelStatus::Usable,
                Some(Refusal::NotOffered(why)) => ModelStatus::NotOffered(why),
                Some(Refusal::Unasked(why)) => ModelStatus::Unavailable(why),
            },
        }
    }

    /// Why `spec` is refused, once the answer is known; `None` while it is
    /// not, so a model can connect and its request waits.
    pub(super) fn refused(
        self: &Arc<Self>,
        spec: &ModelSpec,
        auth: &Authenticator,
        http: &DynHttpClient,
    ) -> Option<String> {
        let asked = self.now(auth, http)?;
        refusal(&asked, spec).map(Refusal::into_text)
    }

    /// The answer, waiting for it, asking first if nobody had.
    async fn answer(self: &Arc<Self>, auth: &Authenticator, http: &DynHttpClient) -> Asked {
        let waiting = {
            let mut state = self.state();
            if let Asking::NotYet = state.asked {
                self.start(&mut state, auth, http);
            }
            match &mut state.asked {
                Asking::Answered(asked) => return asked.clone(),
                Asking::Running(waiting) => {
                    let (sender, receiver) = async_channel::bounded(1);
                    waiting.push(sender);
                    receiver
                }
                Asking::NotYet => return Err("the model list was not asked for".to_owned()),
            }
        };
        waiting
            .recv()
            .await
            .unwrap_or_else(|_| Err("the model list request ended".to_owned()))
    }

    /// Asks again once in this run, after the backend did not find a model
    /// the account was listed with.
    fn ask_again(self: &Arc<Self>, auth: &Authenticator, http: &DynHttpClient) {
        let mut state = self.state();
        if state.again || !matches!(state.asked, Asking::Answered(_)) {
            return;
        }
        state.again = true;
        self.start(&mut state, auth, http);
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

    /// Starts the one request on the IO pool.
    fn start(self: &Arc<Self>, state: &mut State, auth: &Authenticator, http: &DynHttpClient) {
        state.asked = Asking::Running(Vec::new());
        let (epoch, offered) = (state.epoch, Arc::clone(self));
        let (auth, http) = (auth.clone(), http.clone());
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let asking = async move {
            // The credential may need a refresh, which waits for other rig
            // processes too: one deadline covers it and the list.
            let asked = timeout(LIST_TIMEOUT, async {
                match auth.auth_context(&http).await {
                    Ok(context) => account_models(&context, &http)
                        .await
                        .map(Arc::from)
                        .map_err(|error| error.to_string()),
                    Err(error) => Err(format!("the sign-in gave no credential ({error})")),
                }
            });
            let asked = asked
                .await
                .unwrap_or_else(|_| Err(format!("no answer in {} s", LIST_TIMEOUT.as_secs())));
            offered.answered(epoch, asked);
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
            let previous = std::mem::replace(&mut state.asked, Asking::Answered(asked.clone()));
            match previous {
                Asking::Running(waiting) => waiting,
                _ => Vec::new(),
            }
        };
        for waiter in waiting {
            waiter.try_send(asked.clone()).ok();
        }
        self.answers.fetch_add(1, Ordering::AcqRel);
        if let Ok(wake) = self.wake.lock()
            && let Some(wake) = wake.as_ref()
        {
            wake.wake();
        }
    }
}

/// Why a model is refused by an answer.
enum Refusal {
    /// The account does not offer it.
    NotOffered(String),
    /// The backend could not be asked.
    Unasked(String),
}

impl Refusal {
    fn into_text(self) -> String {
        match self {
            Self::NotOffered(why) | Self::Unasked(why) => why,
        }
    }
}

/// Why `asked` refuses `spec`, if it does.
fn refusal(asked: &Asked, spec: &ModelSpec) -> Option<Refusal> {
    match asked {
        Ok(listed) if listed.contains(&spec.id) => None,
        Ok(_) => Some(Refusal::NotOffered(format!(
            "{} is not offered to this {TITLE} account",
            spec.reference()
        ))),
        Err(why) => Some(Refusal::Unasked(format!(
            "could not ask {TITLE} which models this account may use ({why}); /login asks again"
        ))),
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

    async fn before(&self, _: EffectId, _: &EffectKind) -> BeforeServe {
        let asked = self.offered.answer(&self.auth, &self.http).await;
        match refusal(&asked, &self.spec) {
            None => BeforeServe::Proceed,
            Some(why) => BeforeServe::Deny(
                ErrorReport::new(ErrorKind::Provider, why.into_text()).with_retryable(false),
            ),
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
    offered: Res<super::Session>,
    mut models: ResMut<Models>,
    mut seen: Local<u64>,
) {
    let answers = offered.offered.answers();
    if answers != *seen {
        *seen = answers;
        models.set_changed();
    }
}
