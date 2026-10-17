//! `/login` and `/logout` for the ChatGPT plan, which the kernel knows only
//! as a [`SignIn`] in its [`Models`].
//!
//! - `/login` signs in on the IO pool: in the browser, which it opens and
//!   whose URL it shows as a notice, or with a device code when no browser
//!   can be assumed (no graphical session, or SSH), when the callback ports
//!   are taken, or with `--device`.
//! - `/logout` asks OpenAI to revoke the credential's tokens, then deletes
//!   it, whether or not OpenAI confirmed, and says which.
//! - The credential is kept in `RIG_HOME/auth/chatgpt.json`. Every model
//!   call of the plan reads it before its request, refreshed when it has
//!   expired. Calls and sign-ins share one authenticator, and every rig
//!   process holds the file's lock while it refreshes or signs in, so a
//!   refresh never overwrites a newer sign-in. The file is replaced whole.

use std::path::PathBuf;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use crossbeam_channel::{Receiver, Sender};
use rig_core::catalog::{ModelSpec, SignIn, Transport};
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
use rig_harness::harness_protocol::Home;

use rig_ecs::agent::{ActiveTurn, Agent, Notice, StopTurn};
use rig_ecs::model::{Connection, ModelChoice, Models, SetModel};
use rig_ecs::tasks::{Done, PollTasks, Running, Wake};
use rig_harness::commands::{AppCommandsExt, CommandArgs};

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
pub struct ChatgptLoginPlugin;

impl Plugin for ChatgptLoginPlugin {
    fn build(&self, app: &mut App) {
        // One reader of the credential for every request, so concurrent
        // calls refresh it once. It never starts a sign-in on its own.
        let session = Authenticator::new(
            AuthSource::OAuth,
            FileTokenStore::new(Home::from_env().auth()),
            DeviceCodeHandler::default(),
            false,
        );
        app.world_mut()
            .get_resource_or_init::<Models>()
            .0
            .add_sign_in(ChatGptSignIn(session.clone()));
        app.insert_resource(Session(session));
        app.add_command(
            "login",
            "Sign in with your ChatGPT plan: /login chatgpt opens the browser (--device shows \
                 a code to enter instead); /login again or Esc cancels",
            on_login,
        )
        .add_command(
            "logout",
            "Sign out: /logout chatgpt revokes the credential and deletes it",
            on_logout,
        )
        .add_systems(Update, show_login_prompts.in_set(PollTasks))
        .add_observer(on_signed_in)
        .add_observer(on_signed_out)
        .add_observer(cancel_on_interrupt);
    }
}

/// Where the credential is kept.
fn auth_file() -> PathBuf {
    FileTokenStore::new(Home::from_env().auth()).path(PROVIDER)
}

/// The authenticator every call and sign-in of the plan goes through.
#[derive(Resource)]
struct Session(Authenticator);

/// The plan's models, connected with the signed-in credential while it is
/// kept.
struct ChatGptSignIn(Authenticator);

impl SignIn for ChatGptSignIn {
    fn handler(
        &self,
        spec: &ModelSpec,
        transport: &Transport,
    ) -> Option<Result<ErasedHandler, ConnectError>> {
        let model = || {
            let http = transport.shared().ok_or(ConnectError::NoHttpClient)?;
            SignedInModel::new(spec.clone(), self.0.clone(), http)
        };
        let signed_in = self.plan(spec).is_some() && auth_file().is_file();
        signed_in.then(|| model().map(ErasedHandler::new))
    }

    fn plan(&self, spec: &ModelSpec) -> Option<&'static str> {
        (spec.provider.vendor() == PROVIDER).then_some(TITLE)
    }
}

/// A sign-in waiting for the user, on an entity of its own whose
/// [`Running`] task is the flow until it ends as a
/// [`Done<SignInEnded>`]; `--print` waits for it as for any `Running`
/// task. Despawning it cancels the flow.
#[derive(Component)]
pub struct PendingLogin {
    /// The agent that asked.
    pub agent: Entity,
    /// What the flow asks the user to do, worded for a notice.
    prompts: Receiver<String>,
}

/// How a sign-in ended: `Err` says why it failed.
pub struct SignInEnded(Result<(), String>);

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
    session: Res<Session>,
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
                session.0.clone(),
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
const WAITING: &str = "Waiting for it; Esc or /login chatgpt cancels.";

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
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((login, Done(SignInEnded(result)))) = logins.get(done.entity) else {
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

/// `/logout`: signs out on the IO pool ([`Authenticator::sign_out`]): the
/// tokens are revoked and the credential deleted, so the plan's models no
/// longer connect, and its connected models' requests fail. Refused while
/// a turn runs.
fn on_logout(
    In(args): In<CommandArgs>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    session: Res<Session>,
    wake: Res<Wake>,
    mut commands: Commands,
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
        notices.write(Notice::turn_running(agent));
        return;
    }
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let session = session.0.clone();
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
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((logout, Done(SignOutEnded(ended)))) = logouts.get(done.entity) else {
        return;
    };
    let agent = logout.agent;
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
                auth_file().display()
            ),
        ),
    });
    commands.entity(done.entity).despawn();
}
