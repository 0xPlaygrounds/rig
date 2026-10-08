//! `/login` and `/logout`: signing in to the ChatGPT plan from the agent,
//! a plugin of its own that the core knows only as [`SignIns`] on its
//! [`ModelConnector`]. `/login` runs the sign-in on the IO pool: in the browser, which it opens on
//! the sign-in page and whose URL it shows as a notice, or with a device
//! code shown as a notice when no browser can be assumed (no graphical
//! session, or an SSH login), when the browser's callback ports are taken,
//! or when asked with `--device`. `/logout` forgets the credential. The
//! credential is kept in `RIG_HOME/auth/chatgpt.json`, and every model
//! call of the plan reads it, refreshed when it has expired, before its
//! request.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::{fs, io};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use crossbeam_channel::{Receiver, Sender};
use rig::harness_protocol::Home;
use rig_core::catalog::{Catalog, ModelSpec};
use rig_core::driver::DynModel;
use rig_core::effect::{EffectKind, HandlerDescriptor, family};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::operation::Completion;
use rig_core::providers::chatgpt::{
    self,
    auth::{
        AuthError, AuthSource, Authenticator, BrowserSignInPrompt, DeviceCodeHandler,
        DeviceCodePrompt,
    },
};
use rig_core::providers::registry::{ConnectError, ConnectOptions};
use rig_core::serve::adapters::ModelAdapter;
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};

use crate::core::agent::{ActiveTurn, Agent, Connection, Interrupt, ModelChoice, Notice, SetModel};
use crate::core::calls::{Done, Running, Wake, poll_calls};
use crate::core::commands::{AppCommandsExt, CommandArgs};
use crate::core::effects::Effects;
use crate::core::models::{self, ModelConnector, SignIns};
use crate::core::turn::PollCalls;

/// The provider `/login` signs in to: the ChatGPT plan, the catalog's
/// `chatgpt` vendor, which is also what `/login` takes.
const PROVIDER: &str = chatgpt::PROVIDER_NAME;

/// How the user knows [`PROVIDER`].
const TITLE: &str = "ChatGPT";

/// The model a fresh sign-in switches to: the plan's latest frontier
/// model, first in Codex's own model list.
const FRONTIER_MODEL: &str = "chatgpt/gpt-6.1-sol";

/// Adds `/login` and `/logout`, and connects the plan's models with the
/// signed-in credential.
#[derive(Default)]
pub struct LoginPlugin;

impl Plugin for LoginPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ModelConnector::new(ChatGptSignIn))
            .add_command(
                "login",
                "Sign in with your ChatGPT plan: /login chatgpt opens the browser (--device shows \
                 a code to enter instead); /login again or Esc cancels",
                on_login,
            )
            .add_command("logout", "Forget a sign-in: /logout chatgpt", on_logout)
            .add_systems(
                Update,
                (
                    poll_calls::<SignedInResult, Done<SignedInResult>>,
                    show_login_prompts,
                )
                    .in_set(PollCalls),
            )
            .add_observer(on_signed_in)
            .add_observer(cancel_on_interrupt);
    }
}

/// Where the credential is kept.
fn auth_file() -> PathBuf {
    Home::from_env().auth(PROVIDER)
}

/// Whether `spec` is one of the plan's models.
fn of_plan(spec: &ModelSpec) -> bool {
    spec.provider.vendor() == PROVIDER
}

/// The process's one reader of the credential, so concurrent calls
/// refresh it once. It never starts a sign-in.
fn session() -> &'static Authenticator {
    static SESSION: OnceLock<Authenticator> = OnceLock::new();
    SESSION.get_or_init(|| {
        Authenticator::new(
            AuthSource::OAuth,
            Some(auth_file()),
            DeviceCodeHandler::default(),
            false,
        )
    })
}

/// The plan's models, connected with the signed-in credential.
struct ChatGptSignIn;

impl SignIns for ChatGptSignIn {
    fn serves(&self, spec: &ModelSpec) -> bool {
        of_plan(spec) && auth_file().is_file()
    }

    fn handler(&self, spec: &'static ModelSpec) -> Result<ErasedHandler, ConnectError> {
        // Built without a key only for the handler's description.
        let unsigned = Catalog::builtin().connect_with(spec, ConnectOptions::new().api_key(""))?;
        let descriptor = Serve::descriptor(&ModelAdapter::<Completion>::new(
            models::reference(spec),
            unsigned,
        ));
        Ok(ErasedHandler::new(SignedIn { spec, descriptor }))
    }

    fn plan(&self, spec: &ModelSpec) -> Option<&'static str> {
        of_plan(spec).then_some(TITLE)
    }
}

/// A sign-in waiting for the user, on an entity of its own whose
/// [`Running<SignedInResult>`] is the flow. Despawning it cancels the flow.
#[derive(Component)]
pub struct PendingLogin {
    /// The agent that asked.
    pub agent: Entity,
    /// What the flow asks the user to do.
    prompts: Receiver<LoginPrompt>,
}

/// What a sign-in asks the user to do.
enum LoginPrompt {
    /// Sign in on the page the browser was asked to open.
    Browser(BrowserSignInPrompt),
    /// Enter a code at a URL.
    DeviceCode(DeviceCodePrompt),
    /// The browser sign-in could not listen for its callback, so a device
    /// code follows.
    BrowserUnavailable(String),
}

/// How a sign-in asks the user to authorize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    Browser,
    DeviceCode,
}

impl Method {
    /// The browser when one can be assumed and `--device` was not asked for.
    fn choose(device_asked: bool) -> Self {
        if device_asked || !graphical_session() {
            Self::DeviceCode
        } else {
            Self::Browser
        }
    }
}

/// Whether a browser opened here would reach the user: not over SSH, and on
/// Linux and the BSDs, inside an X11 or Wayland session.
fn graphical_session() -> bool {
    let set = |name: &str| std::env::var_os(name).is_some_and(|value| !value.is_empty());
    if set("SSH_CONNECTION") || set("SSH_TTY") {
        return false;
    }
    if cfg!(any(target_os = "macos", windows)) {
        return true;
    }
    set("DISPLAY") || set("WAYLAND_DISPLAY")
}

/// How a sign-in ended: `Err` says why it failed.
pub struct SignedInResult(Result<(), String>);

/// Whether `text` names [`PROVIDER`], or is empty; otherwise a notice
/// says what it takes.
fn named_provider(agent: Entity, text: &str, notices: &mut MessageWriter<Notice>) -> bool {
    let text = text.trim();
    let named = text.is_empty() || text == PROVIDER;
    if !named {
        notices.write(Notice::error(
            agent,
            format!("No sign-in for `{text}`. Sign in to: {PROVIDER}."),
        ));
    }
    named
}

/// `/login`: starts a sign-in for the agent, or cancels the one already
/// waiting. Refused while a turn runs. `--device` after the name asks for
/// the device code.
fn on_login(
    In(args): In<CommandArgs>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    pending: Query<(Entity, &PendingLogin)>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(busy) = agents.get(agent) else {
        return;
    };
    let mut device_asked = false;
    let words: Vec<&str> = args
        .args
        .split_whitespace()
        .filter(|word| {
            let flag = *word == "--device";
            device_asked |= flag;
            !flag
        })
        .collect();
    if !named_provider(agent, &words.join(" "), &mut notices) {
        return;
    }
    if let Some((login, _)) = pending.iter().next() {
        commands.entity(login).despawn();
        notices.write(Notice::info(agent, format!("{TITLE} sign-in cancelled.")));
        return;
    }
    if busy {
        notices.write(Notice::info(
            agent,
            "A turn is running. Press Esc to stop it, then /login.",
        ));
        return;
    }
    let (sender, prompts) = crossbeam_channel::unbounded();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((
        Name::new(format!("login:{PROVIDER}")),
        PendingLogin { agent, prompts },
        Running::spawn(
            pool,
            &wake,
            sign_in_flow(Method::choose(device_asked), sender, wake.clone()),
        ),
    ));
    notices.write(Notice::info(agent, format!("Signing in to {TITLE}…")));
}

/// Runs the sign-in by `method`, sending what the user must do
/// through `prompts`, and keeps the credential readable by the owner alone.
/// A browser sign-in whose callback ports are taken falls back to the
/// device code.
async fn sign_in_flow(method: Method, prompts: Sender<LoginPrompt>, wake: Wake) -> SignedInResult {
    let file = auth_file();
    let prompt = move |prompt: LoginPrompt| {
        prompts.send(prompt).ok();
        wake.wake();
    };
    let device_prompt = prompt.clone();
    let handler = DeviceCodeHandler::new(move |code| device_prompt(LoginPrompt::DeviceCode(code)));
    let flow = async {
        let auth = Authenticator::new(AuthSource::OAuth, Some(file), handler, true);
        let http = rig_reqwest::shared();
        let browser = match method {
            Method::Browser => {
                let prompt = prompt.clone();
                auth.sign_in_with_browser(&http, move |page| prompt(LoginPrompt::Browser(page)))
                    .await
            }
            Method::DeviceCode => auth.sign_in_with_device_code(&http).await,
        };
        match browser {
            Err(AuthError::Io(error))
                if method == Method::Browser && error.kind() == io::ErrorKind::AddrInUse =>
            {
                prompt(LoginPrompt::BrowserUnavailable(error.to_string()));
                auth.sign_in_with_device_code(&http).await
            }
            other => other,
        }
        .map(drop)
        .map_err(|error| error.to_string())
    };
    SignedInResult(flow.await)
}

/// Shows what a sign-in asks the user to do.
fn show_login_prompts(logins: Query<&PendingLogin>, mut notices: MessageWriter<Notice>) {
    let (title, name) = (TITLE, PROVIDER);
    for login in &logins {
        for prompt in login.prompts.try_iter() {
            let text = match prompt {
                LoginPrompt::Browser(page) if page.browser_launched => format!(
                    "Opening your browser to sign in to {title}. If it doesn't open, visit {}\n\
                     Waiting for it; Esc or /login {name} cancels.",
                    page.authorize_url,
                ),
                LoginPrompt::Browser(page) => format!(
                    "Sign in to {title}: open {}\n\
                     Waiting for it; Esc or /login {name} cancels.",
                    page.authorize_url,
                ),
                LoginPrompt::DeviceCode(code) => format!(
                    "Sign in to {title}: open {} and enter the code {}\n\
                     Waiting for it; Esc or /login {name} cancels.",
                    code.verification_uri, code.user_code,
                ),
                LoginPrompt::BrowserUnavailable(why) => {
                    format!("Cannot sign in through the browser ({why}); using a device code.")
                }
            };
            notices.write(Notice::info(login.agent, text));
        }
    }
}

/// Reports how a sign-in ended, and connects the agents whose chosen model
/// it serves and could not connect before.
fn on_signed_in(
    done: On<Add<Done<SignedInResult>>>,
    logins: Query<(&PendingLogin, &Done<SignedInResult>)>,
    unconnected: Query<(Entity, &ModelChoice), (With<Agent>, Without<Connection>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((login, Done(SignedInResult(result)))) = logins.get(done.entity) else {
        return;
    };
    match result {
        Ok(()) => {
            notices.write(Notice::info(
                login.agent,
                format!(
                    "Signed in to {TITLE}; switching to {FRONTIER_MODEL}. /model lists its \
                     other models; the credential is in {}.",
                    auth_file().display()
                ),
            ));
            for (agent, choice) in &unconnected {
                if agent != login.agent && models::resolve(&choice.0).is_some_and(of_plan) {
                    commands.entity(agent).insert(choice.clone());
                }
            }
            commands.trigger(SetModel {
                entity: login.agent,
                model: FRONTIER_MODEL.to_owned(),
            });
        }
        Err(why) => {
            notices.write(Notice::error(
                login.agent,
                format!("{TITLE} sign-in failed: {why}"),
            ));
        }
    }
    commands.entity(done.entity).despawn();
}

/// Esc cancels the agent's waiting sign-ins.
fn cancel_on_interrupt(
    interrupt: On<Interrupt>,
    pending: Query<(Entity, &PendingLogin)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (login, pending) in &pending {
        if pending.agent == interrupt.entity {
            commands.entity(login).despawn();
            notices.write(Notice::info(
                pending.agent,
                format!("{TITLE} sign-in cancelled."),
            ));
        }
    }
}

/// `/logout`: deletes the credential and forgets the plan's connected
/// models. Refused while a turn runs.
fn on_logout(
    In(args): In<CommandArgs>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    mut effects: ResMut<Effects>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(busy) = agents.get(agent) else {
        return;
    };
    if !named_provider(agent, &args.args, &mut notices) {
        return;
    }
    if busy {
        notices.write(Notice::info(
            agent,
            "A turn is running. Press Esc to stop it, then /logout.",
        ));
        return;
    }
    effects.forget_vendor(PROVIDER);
    let notice = match fs::remove_file(auth_file()) {
        Ok(()) => Notice::info(agent, format!("Signed out of {TITLE}.")),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            Notice::info(agent, format!("Not signed in to {TITLE}."))
        }
        Err(error) => Notice::error(
            agent,
            format!("Could not delete {}: {error}.", auth_file().display()),
        ),
    };
    notices.write(notice);
}

/// A catalog model whose requests carry the signed-in credential. Each
/// request reads the credential, refreshing it when it has expired, and
/// connects the model with it, so a long session outlives its token.
struct SignedIn {
    spec: &'static ModelSpec,
    descriptor: HandlerDescriptor,
}

impl SignedIn {
    /// The model, connected with the current credential.
    async fn connect(&self) -> Result<DynModel<Completion>, ErrorReport> {
        let http = rig_reqwest::shared();
        let context = session().auth_context(&http).await.map_err(|error| {
            ErrorReport::new(
                ErrorKind::Provider,
                format!(
                    "the {TITLE} sign-in did not give a credential ({error}); /login \
                         {PROVIDER} signs in again"
                ),
            )
            .with_retryable(false)
        })?;
        let mut options = ConnectOptions::new()
            .api_key(context.access_token)
            .http(http);
        if let Some(account_id) = context.account_id {
            options = options.account_id(account_id);
        }
        Catalog::builtin()
            .connect_with(self.spec, options)
            .map_err(|error| ErrorReport::new(ErrorKind::Internal, error.to_string()))
    }
}

impl Serve for SignedIn {
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        self.descriptor.clone()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        match self.connect().await {
            Ok(model) => {
                ModelAdapter::<Completion>::new(models::reference(self.spec), model)
                    .serve(kind, dispatch)
                    .await
            }
            Err(report) => Reply::Outcome(Err(report)),
        }
    }
}
