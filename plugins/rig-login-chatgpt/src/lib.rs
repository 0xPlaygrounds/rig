//! `/login` and `/logout`: signing in to the ChatGPT plan from the agent,
//! a plugin of its own that the core knows only as a [`SignIn`] in its
//! [`Models`]. `/login` runs the sign-in on the IO pool: in the browser, which it opens on
//! the sign-in page and whose URL it shows as a notice, or with a device
//! code shown as a notice when no browser can be assumed (no graphical
//! session, or an SSH login), when the browser's callback ports are taken,
//! or when asked with `--device`. `/logout` forgets the credential. The
//! credential is kept in `RIG_HOME/auth/chatgpt.json`, and every model
//! call of the plan reads it, refreshed when it has expired, before its
//! request.

use std::path::PathBuf;
use std::{fs, io};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use crossbeam_channel::{Receiver, Sender};
use rig_core::catalog::{ModelSpec, SignIn};
use rig_core::providers::chatgpt::{
    self, SignedInModel,
    auth::{
        AuthSource, Authenticator, DeviceCodeHandler, DeviceCodePrompt, SignInMethod, SignInPrompt,
    },
};
use rig_core::providers::registry::ConnectError;
use rig_core::serve::ErasedHandler;
use rig_harness::harness_protocol::Home;

use rig_ecs::agent::{ActiveTurn, Agent, Interrupt, Notice};
use rig_ecs::calls::{Done, PollCalls, Running, Wake};
use rig_ecs::commands::{AppCommandsExt, CommandArgs};
use rig_ecs::model::{Connection, ModelChoice, Models, SetModel};

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
        // calls refresh it once. It never starts a sign-in.
        let session = Authenticator::new(
            AuthSource::OAuth,
            Some(auth_file()),
            DeviceCodeHandler::default(),
            false,
        );
        app.world_mut()
            .get_resource_or_init::<Models>()
            .0
            .add_sign_in(ChatGptSignIn(session));
        app.add_command(
            "login",
            "Sign in with your ChatGPT plan: /login chatgpt opens the browser (--device shows \
                 a code to enter instead); /login again or Esc cancels",
            on_login,
        )
        .add_command("logout", "Forget a sign-in: /logout chatgpt", on_logout)
        .add_systems(Update, show_login_prompts.in_set(PollCalls))
        .add_observer(on_signed_in)
        .add_observer(cancel_on_interrupt);
    }
}

/// Where the credential is kept.
fn auth_file() -> PathBuf {
    Home::from_env().auth(PROVIDER)
}

/// The plan's models, connected with the signed-in credential while it is
/// kept.
struct ChatGptSignIn(Authenticator);

impl SignIn for ChatGptSignIn {
    fn handler(&self, spec: &ModelSpec) -> Option<Result<ErasedHandler, ConnectError>> {
        let model = || SignedInModel::new(spec.clone(), self.0.clone(), rig_reqwest::shared());
        let signed_in = self.plan(spec).is_some() && auth_file().is_file();
        signed_in.then(|| model().map(ErasedHandler::new))
    }

    fn plan(&self, spec: &ModelSpec) -> Option<&'static str> {
        (spec.provider.vendor() == PROVIDER).then_some(TITLE)
    }
}

/// A sign-in waiting for the user, on an entity of its own whose
/// [`Running`] task is the flow until it ends as a
/// [`Done<SignedInResult>`]; `--print` waits for it as for any `Running`
/// task. Despawning it cancels the flow.
#[derive(Component)]
pub struct PendingLogin {
    /// The agent that asked.
    pub agent: Entity,
    /// What the flow asks the user to do, worded for a notice.
    prompts: Receiver<String>,
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
            sign_in_flow(SignInMethod::detect(device_asked), sender, wake.clone()),
        ),
    ));
    notices.write(Notice::info(agent, format!("Signing in to {TITLE}…")));
}

/// Runs the sign-in by `method`, sending what the user must do through
/// `prompts`. The credential file is written readable by the owner alone.
async fn sign_in_flow(method: SignInMethod, prompts: Sender<String>, wake: Wake) -> SignedInResult {
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
    let auth = Authenticator::new(AuthSource::OAuth, Some(auth_file()), handler, true);
    let signed_in = auth
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
    SignedInResult(signed_in.map(drop).map_err(|error| error.to_string()))
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
    done: On<Add<Done<SignedInResult>>>,
    logins: Query<(&PendingLogin, &Done<SignedInResult>)>,
    unconnected: Query<(Entity, &ModelChoice), (With<Agent>, Without<Connection>)>,
    models: Res<Models>,
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

/// `/logout`: deletes the credential, so the plan's models no longer
/// connect, and its connected models' requests fail. Refused while a turn
/// runs.
fn on_logout(
    In(args): In<CommandArgs>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
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
