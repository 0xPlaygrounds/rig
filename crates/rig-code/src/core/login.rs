//! Signing in to a provider's subscription from the agent. [`SignIn`] runs
//! the provider's sign-in on the IO pool: in the browser, which it opens on
//! the sign-in page and whose URL it shows as a notice, or with a device
//! code shown as a notice when no browser can be assumed (no graphical
//! session, or an SSH login), when the browser's callback ports are taken,
//! or when asked with `--device`. [`SignOut`] forgets the credential. The
//! credential is kept in `RIG_HOME/auth/<provider>.json`, and every model
//! call of that provider reads it, refreshed when it has expired, before
//! its request. ChatGPT is the one provider so far.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::{fs, io};

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use crossbeam_channel::{Receiver, Sender};
use rig::code_protocol::Home;
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
use rig_core::serve::{Dispatch, Reply, Serve};

use super::agent::{ActiveTurn, Agent, Connection, Interrupt, ModelChoice, Notice};
use super::calls::{Done, Running, Wake};
use super::effects::Effects;
use super::models;

/// A provider `/login` signs in to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoginProvider {
    /// The ChatGPT plan's models, the catalog's `chatgpt` vendor.
    ChatGpt,
}

impl LoginProvider {
    /// Every provider, in the order `/login` lists them.
    pub const ALL: [Self; 1] = [Self::ChatGpt];

    /// The catalog vendor, which is also what `/login` takes.
    pub fn name(self) -> &'static str {
        match self {
            Self::ChatGpt => chatgpt::PROVIDER_NAME,
        }
    }

    /// The provider `name` names.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|provider| provider.name() == name)
    }

    /// The provider whose sign-in serves `spec`.
    pub fn of(spec: &ModelSpec) -> Option<Self> {
        Self::named(spec.provider.vendor())
    }

    /// How the user knows it.
    pub fn title(self) -> &'static str {
        match self {
            Self::ChatGpt => "ChatGPT",
        }
    }

    /// Where its credential is kept.
    pub fn auth_file(self) -> PathBuf {
        Home::from_env().auth(self.name())
    }

    /// Whether it holds a credential.
    pub fn signed_in(self) -> bool {
        self.auth_file().is_file()
    }

    /// The process's one reader of the credential, so concurrent calls
    /// refresh it once. It never starts a sign-in.
    fn session(self) -> &'static Authenticator {
        static CHATGPT: OnceLock<Authenticator> = OnceLock::new();
        match self {
            Self::ChatGpt => CHATGPT.get_or_init(|| {
                Authenticator::new(
                    AuthSource::OAuth,
                    Some(self.auth_file()),
                    DeviceCodeHandler::default(),
                    false,
                )
            }),
        }
    }

    /// `spec` as an effect handler that signs each request with this
    /// provider's credential.
    pub(crate) fn model_handler(self, spec: &'static ModelSpec) -> Result<SignedIn, ConnectError> {
        // Built without a key only for the handler's description.
        let unsigned = Catalog::builtin().connect_with(spec, ConnectOptions::new().api_key(""))?;
        let descriptor = Serve::descriptor(&ModelAdapter::<Completion>::new(
            models::reference(spec),
            unsigned,
        ));
        Ok(SignedIn {
            provider: self,
            spec,
            descriptor,
        })
    }
}

/// Sign in to `provider` (a [`LoginProvider`] name) for the agent, or
/// cancel the sign-in already waiting for it. Refused while a turn runs.
/// The sign-in is in the browser when one can be assumed, otherwise with a
/// device code; `--device` after the name asks for the device code.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SignIn {
    /// The agent whose notices show the sign-in.
    pub entity: Entity,
    /// The provider, optionally followed by `--device`.
    pub provider: String,
}

/// Forget the credential of `provider` (a [`LoginProvider`] name).
/// Refused while a turn runs.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SignOut {
    /// The agent whose notices report it.
    pub entity: Entity,
    /// The provider.
    pub provider: String,
}

/// A sign-in waiting for the user, on an entity of its own whose
/// [`Running<SignedInResult>`] is the flow. Despawning it cancels the flow.
#[derive(Component)]
pub struct PendingLogin {
    /// The agent that asked.
    pub agent: Entity,
    /// The provider.
    pub provider: LoginProvider,
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

/// The name `text` gives, the only provider when it is empty, or a notice
/// listing them.
fn named_provider(
    agent: Entity,
    text: &str,
    notices: &mut MessageWriter<Notice>,
) -> Option<LoginProvider> {
    let text = text.trim();
    let found = match (text, LoginProvider::ALL.as_slice()) {
        ("", [only]) => Some(*only),
        (name, _) => LoginProvider::named(name),
    };
    if found.is_none() {
        let names: Vec<&str> = LoginProvider::ALL
            .iter()
            .map(|provider| provider.name())
            .collect();
        notices.write(Notice::error(
            agent,
            format!("No sign-in for `{text}`. Sign in to: {}.", names.join(", ")),
        ));
    }
    found
}

/// Starts a sign-in, or cancels the one already waiting for the provider.
pub(crate) fn on_sign_in(
    sign_in: On<SignIn>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    pending: Query<(Entity, &PendingLogin)>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = sign_in.entity;
    let Ok(busy) = agents.get(agent) else {
        return;
    };
    let mut device_asked = false;
    let words: Vec<&str> = sign_in
        .provider
        .split_whitespace()
        .filter(|word| {
            let flag = *word == "--device";
            device_asked |= flag;
            !flag
        })
        .collect();
    let Some(provider) = named_provider(agent, &words.join(" "), &mut notices) else {
        return;
    };
    if let Some((login, _)) = pending.iter().find(|(_, login)| login.provider == provider) {
        commands.entity(login).despawn();
        notices.write(Notice::info(
            agent,
            format!("{} sign-in cancelled.", provider.title()),
        ));
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
        Name::new(format!("login:{}", provider.name())),
        PendingLogin {
            agent,
            provider,
            prompts,
        },
        Running::spawn(
            pool,
            &wake,
            sign_in_flow(provider, Method::choose(device_asked), sender, wake.clone()),
        ),
    ));
    notices.write(Notice::info(
        agent,
        format!("Signing in to {}…", provider.title()),
    ));
}

/// Runs `provider`'s sign-in by `method`, sending what the user must do
/// through `prompts`, and keeps the credential readable by the owner alone.
/// A browser sign-in whose callback ports are taken falls back to the
/// device code.
async fn sign_in_flow(
    provider: LoginProvider,
    method: Method,
    prompts: Sender<LoginPrompt>,
    wake: Wake,
) -> SignedInResult {
    let file = provider.auth_file();
    let prompt = move |prompt: LoginPrompt| {
        prompts.send(prompt).ok();
        wake.wake();
    };
    let device_prompt = prompt.clone();
    let handler = DeviceCodeHandler::new(move |code| device_prompt(LoginPrompt::DeviceCode(code)));
    let flow = async {
        private_dir(&file).map_err(|error| error.to_string())?;
        let auth = Authenticator::new(AuthSource::OAuth, Some(file.clone()), handler, true);
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
        .map_err(|error| error.to_string())?;
        private_file(&file).map_err(|error| error.to_string())
    };
    SignedInResult(flow.await)
}

/// Creates the directory of `file`, readable by the owner alone on Unix.
fn private_dir(file: &Path) -> io::Result<()> {
    let Some(dir) = file.parent() else {
        return Ok(());
    };
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)
}

/// Makes `file` readable by the owner alone on Unix.
fn private_file(file: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(file, fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    let _ = file;
    Ok(())
}

/// Shows what a sign-in asks the user to do.
pub(crate) fn show_login_prompts(logins: Query<&PendingLogin>, mut notices: MessageWriter<Notice>) {
    for login in &logins {
        let (title, name) = (login.provider.title(), login.provider.name());
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
pub(crate) fn on_signed_in(
    done: On<Add<Done<SignedInResult>>>,
    logins: Query<(&PendingLogin, &Done<SignedInResult>)>,
    unconnected: Query<(Entity, &ModelChoice), (With<Agent>, Without<Connection>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((login, Done(SignedInResult(result)))) = logins.get(done.entity) else {
        return;
    };
    let title = login.provider.title();
    match result {
        Ok(()) => {
            notices.write(Notice::info(
                login.agent,
                format!(
                    "Signed in to {title}. /model lists its models; the credential is in {}.",
                    login.provider.auth_file().display()
                ),
            ));
            for (agent, choice) in &unconnected {
                if models::resolve(&choice.0).and_then(LoginProvider::of) == Some(login.provider) {
                    commands.entity(agent).insert(choice.clone());
                }
            }
        }
        Err(why) => {
            notices.write(Notice::error(
                login.agent,
                format!("{title} sign-in failed: {why}"),
            ));
        }
    }
    commands.entity(done.entity).despawn();
}

/// Esc cancels the agent's waiting sign-ins.
pub(crate) fn cancel_on_interrupt(
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
                format!("{} sign-in cancelled.", pending.provider.title()),
            ));
        }
    }
}

/// Deletes the provider's credential and forgets its connected models.
pub(crate) fn on_sign_out(
    sign_out: On<SignOut>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    mut effects: ResMut<Effects>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = sign_out.entity;
    let Ok(busy) = agents.get(agent) else {
        return;
    };
    let Some(provider) = named_provider(agent, &sign_out.provider, &mut notices) else {
        return;
    };
    if busy {
        notices.write(Notice::info(
            agent,
            "A turn is running. Press Esc to stop it, then /logout.",
        ));
        return;
    }
    effects.forget_vendor(provider.name());
    let notice = match fs::remove_file(provider.auth_file()) {
        Ok(()) => Notice::info(agent, format!("Signed out of {}.", provider.title())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            Notice::info(agent, format!("Not signed in to {}.", provider.title()))
        }
        Err(error) => Notice::error(
            agent,
            format!(
                "Could not delete {}: {error}.",
                provider.auth_file().display()
            ),
        ),
    };
    notices.write(notice);
}

/// A catalog model whose requests carry the signed-in credential. Each
/// request reads the credential, refreshing it when it has expired, and
/// connects the model with it, so a long session outlives its token.
pub(crate) struct SignedIn {
    provider: LoginProvider,
    spec: &'static ModelSpec,
    descriptor: HandlerDescriptor,
}

impl SignedIn {
    /// The model, connected with the current credential.
    async fn connect(&self) -> Result<DynModel<Completion>, ErrorReport> {
        let http = rig_reqwest::shared();
        let context = self
            .provider
            .session()
            .auth_context(&http)
            .await
            .map_err(|error| {
                ErrorReport::new(
                    ErrorKind::Provider,
                    format!(
                        "the {} sign-in did not give a credential ({error}); /login {} signs \
                         in again",
                        self.provider.title(),
                        self.provider.name()
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
