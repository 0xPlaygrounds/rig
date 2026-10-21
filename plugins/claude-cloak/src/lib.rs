//! Opt-in Anthropic subscription authentication on a separate catalog vendor.
//!
//! Available on Linux, where browser login uses `xdg-open` and credentials use
//! private Unix file permissions.
//!
//! ```no_run
//! use claude_cloak::ClaudeCloakPlugin;
//! use rig_harness::prelude::*;
//!
//! let mut app = App::new();
//! app.add_plugins(ClaudeCloakPlugin);
//! ```

#![cfg(target_os = "linux")]

mod auth;
mod names;
mod policy;
mod schema;
#[cfg(test)]
mod tests;
mod transport;

use rig_harness::prelude::*;
use rig_harness::rig_core::{
    catalog::{Catalog, ModelSpec, Transport},
    operation::Completion,
    providers::registry::{ConnectError, ConnectOptions, ProviderId},
    serve::{ErasedHandler, adapters::ModelAdapter},
};

/// Adds `anthropic-oauth/*` without replacing the API-key Anthropic provider.
#[derive(Default)]
pub struct ClaudeCloakPlugin;

impl Plugin for ClaudeCloakPlugin {
    fn build(&self, app: &mut App) {
        let Ok(vendor) = ProviderId::declare("anthropic-oauth") else {
            return;
        };
        let mut models = app.world_mut().get_resource_or_init::<Models>();
        let specs: Vec<_> = models
            .0
            .catalog()
            .iter()
            .filter(|spec| spec.provider.vendor() == "anthropic")
            .cloned()
            .collect();
        for mut spec in specs {
            spec.provider = vendor;
            models.0.catalog_mut().insert(spec);
        }
        models.0.provide(vendor, connect);
        app.register_required_components::<Agent, policy::OAuthSession>()
            .keeps_saved::<policy::OAuthSession>()
            .add_systems(
                AgentPass,
                policy::identify_requests.in_set(PrepareSystems::Edit),
            )
            .add_command(
                "anthropic-oauth",
                "Separate Anthropic subscription: login | logout",
                command,
            )
            .add_systems(Update, completed);
    }
}

fn connect(
    spec: &ModelSpec,
    catalog: &Catalog,
    transport: &Transport,
) -> Result<ErasedHandler, ConnectError> {
    let client = transport::new().map_err(|error| ConnectError::Provider(error.to_string()))?;
    connect_with_client(spec, catalog, transport, client)
}

fn connect_with_client(
    spec: &ModelSpec,
    catalog: &Catalog,
    transport: &Transport,
    client: rig_harness::rig_core::http_client::DynHttpClient,
) -> Result<ErasedHandler, ConnectError> {
    let mut upstream = spec.clone();
    upstream.provider = ProviderId::catalog("anthropic")
        .ok_or_else(|| ConnectError::Provider("Anthropic provider unavailable".into()))?;
    // The placeholder is removed before I/O; credentials are loaded and refreshed per request.
    let options = ConnectOptions::new()
        .api_key("oauth-transport-managed")
        .base_url("https://api.anthropic.com")
        .http(transport.wrap(client));
    let model = catalog.connect_with(&upstream, options)?;
    Ok(
        names::decode_handler(ErasedHandler::new(ModelAdapter::<Completion>::new(
            spec.reference(),
            model,
        )))
        .layered(policy::RequestPolicy {
            adaptive_only: spec.compat.adaptive_thinking,
            sampling_allowed: spec.sampling
                != Some(rig_harness::rig_core::catalog::Sampling::Never),
            forced_tools_allowed: !spec.compat.rejects_forced_tool_choice,
            max_output_tokens: spec.max_output_tokens,
        }),
    )
}

#[derive(Component)]
struct Pending(Entity);

type LoginResult = Result<&'static str, OperationError>;

#[derive(Debug, thiserror::Error)]
enum OperationError {
    #[error(transparent)]
    Auth(#[from] auth::AuthError),
    #[error("Anthropic OAuth operation stopped.")]
    Stopped,
}

fn command(
    In(args): In<CommandArgs>,
    runtime: Res<TokioRuntime>,
    wake: Res<Wake>,
    pending: Query<&Pending>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if pending.iter().next().is_some() {
        notices.write(Notice::error(
            args.agent,
            "An Anthropic OAuth operation is already running.",
        ));
        return;
    }
    let login = match args.args.trim() {
        "login" => true,
        "logout" => false,
        _ => {
            notices.write(Notice::info(args.agent, "Usage: /anthropic-oauth login | logout. Login explicitly opens your browser. Select anthropic-oauth/<model> separately."));
            return;
        }
    };
    let task = runtime.0.spawn(async move {
        if login {
            auth::login().await
        } else {
            auth::logout()
        }
        .map(|()| {
            if login {
                "Anthropic OAuth signed in. Select an anthropic-oauth model with /model."
            } else {
                "Anthropic OAuth signed out."
            }
        })
        .map_err(OperationError::from)
    });
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    commands.spawn((
        Pending(args.agent),
        Running::spawn(pool, &wake, async move {
            task.await.unwrap_or(Err(OperationError::Stopped))
        }),
    ));
    notices.write(Notice::info(
        args.agent,
        if login {
            "Opening browser for Anthropic OAuth sign-in."
        } else {
            "Signing out of Anthropic OAuth."
        },
    ));
}

fn completed(
    pending: Query<(Entity, &Pending, &Done<LoginResult>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (entity, pending, result) in &pending {
        notices.write(match &result.0 {
            Ok(text) => Notice::info(pending.0, *text),
            Err(error) => Notice::error(pending.0, error.to_string()),
        });
        commands.entity(entity).despawn();
    }
}
