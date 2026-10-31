//! `/login` and `/logout` ([`LoginPlugin`]): a sign-in with the ChatGPT
//! plan, which the kernel knows only as a [`SignIn`] in its [`Models`], and
//! the API keys of the other providers.
//!
//! - `/login` without arguments lets the user pick the ChatGPT plan or a
//!   provider that takes an API key. `/login chatgpt` signs in, and
//!   `/login <provider>` asks for the provider's key, hidden as it is
//!   typed.
//! - The ChatGPT sign-in runs on the IO pool: in the browser, which it
//!   opens and whose URL it shows as a notice, or with a device code when
//!   no browser can be assumed (no graphical session, or SSH), when the
//!   callback ports are taken, or with `--device`.
//! - `/logout` lets the user pick what to remove of what `/login` stored,
//!   or takes its name. Signing out of ChatGPT asks OpenAI to revoke the
//!   credential's tokens, then deletes it, whether or not OpenAI
//!   confirmed, and says which. Environment variables are never touched.
//! - The ChatGPT credential is kept in `RIG_HOME/auth/chatgpt.json`. Every
//!   model call of the plan reads it before its request, refreshed when it
//!   has expired. Calls and sign-ins share one authenticator, and every rig
//!   process holds the file's lock while it refreshes or signs in, so a
//!   refresh never overwrites a newer sign-in. The file is replaced whole.
//! - The plan's models are those the backend lists for the signed-in
//!   account, asked once per run when first needed.

use std::path::PathBuf;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use crossbeam_channel::{Receiver, Sender};
use rig_core::catalog::{ModelSpec, ModelStatus, SignIn, Transport};
use rig_core::client::signed_in::SignedInModel;
use rig_core::providers::chatgpt::{
    self,
    auth::{
        AuthError, AuthSource, Authenticator, DeviceCodeHandler, DeviceCodePrompt, FileTokenStore,
        SignInMethod, SignInPrompt, SignedOut,
    },
};
use rig_core::providers::registry::ConnectError;
use rig_core::serve::ErasedHandler;
use rig_harness::services::RigHome;

use rig_ecs::agent::{ActiveTurn, Agent, Notice, StopTurn};
use rig_ecs::model::{ChoiceKind, Connection, ModelChoice, Models, SetModel};
use rig_ecs::tasks::{Done, PollTasks, Running, Wake};
use rig_harness::commands::{AppCommandsExt, CommandArgs};

/// The provider `/login` signs in to: the ChatGPT plan, the catalog's
/// `chatgpt` vendor, which is also what `/login` and `/logout` take for it.
const PROVIDER: &str = chatgpt::PROVIDER_NAME;

/// How the user knows [`PROVIDER`].
const TITLE: &str = chatgpt::PLAN_TITLE;

mod keys;
mod offered;

use offered::Offered;

/// Adds `/login` and `/logout`, and connects the plan's models with the
/// signed-in credential and each provider's models with its saved key.
#[derive(Default)]
pub struct LoginPlugin;

impl Plugin for LoginPlugin {
    fn build(&self, app: &mut App) {
        // One reader of the credential for every request, so concurrent
        // calls refresh it once. It never starts a sign-in on its own.
        let store = FileTokenStore::new(RigHome::of(app.world()).auth());
        let file = store.path(PROVIDER);
        let auth = Authenticator::new(
            AuthSource::OAuth,
            store,
            DeviceCodeHandler::default(),
            false,
        );
        let offered = Arc::new(Offered::default());
        let session = Session {
            auth,
            file,
            offered,
        };
        app.world_mut()
            .get_resource_or_init::<Models>()
            .0
            .add_sign_in(ChatGptSignIn(session.clone()));
        app.insert_resource(session);
        keys::add(app);
        app.add_command(
            "login",
            "Sign in: pick the ChatGPT plan or a provider's API key; /login chatgpt opens the \
                 browser (--device shows a code instead), /login <provider> asks for its key",
            on_login,
        )
        .add_command(
            "logout",
            "Remove what /login stored: pick it, or /logout chatgpt or /logout <provider>",
            on_logout,
        )
        .add_systems(PreStartup, wake_on_answers)
        .add_systems(Update, show_login_prompts.in_set(PollTasks))
        .add_systems(First, offered::show_answers)
        .add_systems(Update, offered::refuse_unoffered)
        .add_observer(on_signed_in)
        .add_observer(on_signed_out)
        .add_observer(cancel_on_interrupt);
    }
}

/// The authenticator every call and sign-in of the plan goes through, the
/// file it keeps the credential in, and the models the account may use.
#[derive(Resource, Clone)]
struct Session {
    auth: Authenticator,
    file: PathBuf,
    offered: Arc<Offered>,
}

/// Has the app's loop woken when the account's models are known.
fn wake_on_answers(session: Res<Session>, wake: Option<Res<Wake>>) {
    if let Some(wake) = wake {
        session.offered.wake_with(wake.clone());
    }
}

/// The plan's models the account may use, connected with the signed-in
/// credential while it is kept.
struct ChatGptSignIn(Session);

impl SignIn for ChatGptSignIn {
    fn handler(
        &self,
        spec: &ModelSpec,
        transport: &Transport,
    ) -> Option<Result<ErasedHandler, ConnectError>> {
        if self.plan(spec).is_none() || !self.0.file.is_file() {
            return None;
        }
        let Some(http) = transport.shared() else {
            return Some(Err(ConnectError::NoHttpClient));
        };
        let Session { auth, offered, .. } = &self.0;
        if let Some(why) = offered.refused(spec, auth, &http) {
            return Some(Err(ConnectError::Provider(why)));
        }
        let model = SignedInModel::new(spec.clone(), auth.clone(), http.clone());
        let gate = offered::Gate {
            offered: Arc::clone(offered),
            spec: spec.clone(),
            auth: auth.clone(),
            http,
        };
        Some(model.map(|model| ErasedHandler::new(model).layered(gate)))
    }

    fn plan(&self, spec: &ModelSpec) -> Option<&'static str> {
        (spec.provider.vendor() == PROVIDER).then_some(TITLE)
    }

    fn offers(&self, spec: &ModelSpec, transport: &Transport) -> ModelStatus {
        match transport.shared() {
            Some(http) => self.0.offered.status(spec, &self.0.auth, &http),
            None => ModelStatus::Usable,
        }
    }
}

/// A sign-in waiting for the user, on an entity of its own whose
/// [`Running`] task is the flow until it ends; `--print` waits for it as
/// for any `Running` task. Despawning it cancels the flow.
#[derive(Component)]
pub struct PendingLogin {
    /// The agent that asked.
    pub agent: Entity,
    /// What the flow asks the user to do, worded for a notice.
    prompts: Receiver<String>,
}

/// How a sign-in ended: `Err` says why it failed.
pub(crate) struct SignInEnded(Result<(), String>);

/// `/login`: cancels the ChatGPT sign-in already waiting, else lets the
/// user pick what to sign in to, starts the ChatGPT sign-in
/// (`/login chatgpt`, `--device` after it asking for the device code), or
/// asks for a provider's API key (`/login <provider>`). A sign-in is
/// refused while a turn runs.
fn on_login(
    In(args): In<CommandArgs>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    pending: Query<(Entity, &PendingLogin)>,
    (wake, session, keys, models): (Res<Wake>, Res<Session>, Res<keys::Keys>, Res<Models>),
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
    let named = words.join(" ");
    if let Some((login, _)) = pending.iter().next()
        && (named.is_empty() || named == PROVIDER)
    {
        commands.entity(login).despawn();
        notices.write(Notice::info(agent, format!("{TITLE} sign-in cancelled.")));
        return;
    }
    if named.is_empty() {
        commands.spawn(keys::login_picker(agent, &models.0));
        return;
    }
    if named != PROVIDER {
        keys::ask_key(
            agent,
            &named,
            (&keys, &models.0),
            &mut commands,
            &mut notices,
        );
        return;
    }
    if busy {
        notices.write(Notice::turn_running(agent));
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
            sign_in_flow(
                session.auth.clone(),
                SignInMethod::detect(device_asked),
                sender,
                wake.clone(),
            ),
        ),
    ));
    notices.write(Notice::info(agent, format!("Signing in to {TITLE}…")));
}

/// Runs the sign-in by `method` on `session`, sending what the user must do
/// through `prompts`. The credential file is written readable by the owner
/// alone.
async fn sign_in_flow(
    session: Authenticator,
    method: SignInMethod,
    prompts: Sender<String>,
    wake: Wake,
) -> SignInEnded {
    let say = move |text: String| {
        prompts.send(text).ok();
        wake.wake();
    };
    let device_say = say.clone();
    let handler = DeviceCodeHandler::new(move |code: DeviceCodePrompt| {
        device_say(format!(
            "Sign in to {TITLE}: open {} and enter the code {}\n{WAITING}",
            code.verification_uri, code.user_code,
        ));
    });
    let signed_in = session
        .with_device_code_handler(handler)
        .sign_in(&rig_reqwest::shared(), method, |prompt| {
            say(match prompt {
                SignInPrompt::Browser(page) if page.browser_launched => format!(
                    "Opening your browser to sign in to {TITLE}. If it doesn't open, visit {}\n\
                     {WAITING}",
                    page.authorize_url,
                ),
                SignInPrompt::Browser(page) => {
                    format!("Sign in to {TITLE}: open {}\n{WAITING}", page.authorize_url)
                }
                SignInPrompt::BrowserUnavailable(why) => {
                    format!("Cannot sign in through the browser ({why}); using a device code.")
                }
            });
        })
        .await;
    SignInEnded(signed_in.map(drop).map_err(|error| error.to_string()))
}

/// The line after a sign-in prompt.
const WAITING: &str = "Waiting for it; Esc or /login cancels.";

/// Shows what a sign-in asks the user to do.
fn show_login_prompts(logins: Query<&PendingLogin>, mut notices: MessageWriter<Notice>) {
    for login in &logins {
        for text in login.prompts.try_iter() {
            notices.write(Notice::info(login.agent, text));
        }
    }
}

/// Reports how a sign-in ended, and connects the agents whose chosen model
/// it serves and could not connect before.
fn on_signed_in(
    done: On<Add<Done<SignInEnded>>>,
    logins: Query<(&PendingLogin, &Done<SignInEnded>)>,
    unconnected: Query<(Entity, &ModelChoice), (With<Agent>, Without<Connection>)>,
    (models, session): (Res<Models>, Res<Session>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((login, Done(SignInEnded(result)))) = logins.get(done.entity) else {
        return;
    };
    match result {
        Ok(()) => {
            session.offered.forget();
            let model = format!("{PROVIDER}/{}", chatgpt::DEFAULT_MODEL);
            notices.write(Notice::info(
                login.agent,
                format!(
                    "Signed in to {TITLE}; switching to {model}. /model lists its \
                     other models; the credential is in {}.",
                    session.file.display()
                ),
            ));
            for (agent, choice) in &unconnected {
                let catalog = models.0.catalog();
                let of_plan = catalog
                    .resolve(&choice.0)
                    .is_ok_and(|found| found.spec.provider.vendor() == PROVIDER);
                if agent != login.agent && of_plan {
                    commands.entity(agent).insert(choice.clone());
                }
            }
            commands.trigger(SetModel {
                entity: login.agent,
                model,
                kind: ChoiceKind::ForThisRun,
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
    interrupt: On<StopTurn>,
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

/// `/logout`: lets the user pick what to remove of what `/login` stored,
/// removes a provider's saved key (`/logout <provider>`), or signs out of
/// ChatGPT on the IO pool (`/logout chatgpt`, [`Authenticator::sign_out`]):
/// the tokens are revoked and the credential deleted, so the plan's models
/// no longer connect, and its connected models' requests fail. Signing out
/// is refused while a turn runs.
fn on_logout(
    In(args): In<CommandArgs>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    (session, keys, wake): (Res<Session>, Res<keys::Keys>, Res<Wake>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let Ok(busy) = agents.get(agent) else {
        return;
    };
    let named = args.args.trim();
    if named.is_empty() {
        let signed_in = session.file.is_file().then_some((TITLE, PROVIDER));
        match keys::logout_picker(agent, signed_in, &keys) {
            Some(picker) => {
                commands.spawn(picker);
            }
            None => {
                notices.write(Notice::info(agent, keys::NOTHING_STORED));
            }
        }
        return;
    }
    if named != PROVIDER {
        commands.trigger(keys::RemoveKey {
            agent,
            vendor: named.to_owned(),
        });
        return;
    }
    if busy {
        notices.write(Notice::turn_running(agent));
        return;
    }
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let session = session.auth.clone();
    let sign_out = async move { SignOutEnded(session.sign_out(&rig_reqwest::shared()).await) };
    commands.spawn((
        Name::new(format!("logout:{PROVIDER}")),
        PendingLogout { agent },
        Running::spawn(pool, &wake, sign_out),
    ));
}

/// A sign-out under way, on an entity of its own whose [`Running`] task
/// ends as a [`Done<SignOutEnded>`].
#[derive(Component)]
struct PendingLogout {
    /// The agent that asked.
    agent: Entity,
}

/// How a sign-out ended.
struct SignOutEnded(Result<SignedOut, AuthError>);

/// Says how a sign-out ended.
fn on_signed_out(
    done: On<Add<Done<SignOutEnded>>>,
    logouts: Query<(&PendingLogout, &Done<SignOutEnded>)>,
    session: Res<Session>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((logout, Done(SignOutEnded(ended)))) = logouts.get(done.entity) else {
        return;
    };
    let agent = logout.agent;
    session.offered.forget();
    notices.write(match ended {
        Ok(SignedOut::NotSignedIn) => Notice::info(agent, format!("Not signed in to {TITLE}.")),
        Ok(SignedOut::Forgotten { revoked: Ok(()) }) => Notice::info(
            agent,
            format!("Signed out of {TITLE}; OpenAI revoked the credential."),
        ),
        Ok(SignedOut::Forgotten { revoked: Err(why) }) => Notice::warn(
            agent,
            format!(
                "Signed out of {TITLE} here, but OpenAI did not confirm revoking the \
                 credential ({why}); it lapses when it expires."
            ),
        ),
        Err(error) => Notice::error(
            agent,
            format!(
                "Could not sign out of {TITLE} ({}): {error}.",
                session.file.display()
            ),
        ),
    });
    commands.entity(done.entity).despawn();
}

#[cfg(test)]
mod tests;
