//! The API keys `/login <provider>` saves, one file per provider in
//! `RIG_HOME/auth/keys/<vendor>.json`, readable by the owner alone and
//! replaced whole. A saved key connects its provider's models before the
//! environment's: the provider is reached through [`SavedKey`], which reads
//! the file each time a model connects, and with no file any more connects
//! as the environment says. Saving or removing a key reconnects the idle
//! agents on that provider's models.

use std::path::PathBuf;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_core::catalog::{Catalog, Connector, ModelSpec, Provide, Transport};
use rig_core::providers::chatgpt::auth::{FileTokenStore, TokenStore};
use rig_core::providers::copilot;
use rig_core::providers::registry::{ConnectError, ConnectOptions, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_ecs::agent::{ActiveTurn, Agent, Notice};
use rig_ecs::model::{Connection, ModelChoice, Models};
use rig_harness::commands::RunCommand;
use rig_harness::harness_protocol::Home;
use rig_harness::pick::{PickItem, Picked, Picker};
use rig_harness::services::RigHome;
use serde::{Deserialize, Serialize};

/// What `/logout` says when `/login` stored nothing.
pub(super) const NOTHING_STORED: &str = "Nothing stored to remove. /logout removes only what \
    /login saved; environment variables are unchanged.";

/// Adds the saved keys ([`Keys`]), each provider with one reached through
/// [`SavedKey`], and saves the key typed in `/login <provider>`'s picker.
pub(super) fn add(app: &mut App) {
    let keys = Keys::at(&RigHome::of(app.world()));
    let models = &mut app.world_mut().get_resource_or_init::<Models>().0;
    for vendor in keys.vendors() {
        provide(models, &vendor, &keys);
    }
    app.insert_resource(keys)
        .add_observer(save_typed_key)
        .add_observer(remove_key);
}

/// The saved API keys, as files in one directory.
#[derive(Resource, Clone, Debug)]
pub(crate) struct Keys {
    dir: PathBuf,
    store: FileTokenStore,
}

/// A saved key's file.
#[derive(Serialize, Deserialize)]
struct Saved {
    key: String,
}

/// Why a key could not be saved or removed.
type StoreError = Box<dyn std::error::Error + Send + Sync>;

impl Keys {
    /// The keys of `home`, in `auth/keys/`.
    pub(crate) fn at(home: &Home) -> Self {
        let dir = home.auth().join("keys");
        let store = FileTokenStore::new(&dir);
        Self { dir, store }
    }

    /// The file `vendor`'s key is saved in.
    pub(crate) fn path(&self, vendor: &str) -> PathBuf {
        self.store.path(vendor)
    }

    /// `vendor`'s saved key, if any.
    pub(crate) fn load(&self, vendor: &str) -> Option<String> {
        let record = std::fs::read_to_string(self.path(vendor)).ok()?;
        let saved: Saved = serde_json::from_str(&record).ok()?;
        Some(saved.key)
    }

    /// Saves `key` as `vendor`'s, replacing the file whole. One file holds
    /// one key, so no lock is needed: the last save wins.
    pub(crate) fn save(&self, vendor: &str, key: &str) -> Result<(), StoreError> {
        let record = serde_json::to_string(&Saved {
            key: key.to_owned(),
        })?;
        bevy_tasks::block_on(self.store.save(vendor, &record))
    }

    /// Removes `vendor`'s key; whether there was one.
    pub(crate) fn remove(&self, vendor: &str) -> Result<bool, StoreError> {
        let had = self.path(vendor).is_file();
        bevy_tasks::block_on(self.store.delete(vendor))?;
        Ok(had)
    }

    /// The vendors with a saved key, sorted.
    pub(crate) fn vendors(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut vendors: Vec<String> = entries
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let vendor = path.file_stem()?.to_str()?;
                let json = path
                    .extension()
                    .is_some_and(|extension| extension == "json");
                json.then(|| vendor.to_owned())
            })
            .collect();
        vendors.sort();
        vendors
    }
}

/// Reaches a provider's models with its saved key, read when a model
/// connects, else as the environment says.
struct SavedKey(Keys);

impl Provide for SavedKey {
    fn handler(
        &self,
        spec: &ModelSpec,
        catalog: &Catalog,
        transport: &Transport,
    ) -> Result<ErasedHandler, ConnectError> {
        let options = match self.0.load(spec.provider.vendor()) {
            Some(key) => ConnectOptions::new().api_key(key),
            None => ConnectOptions::new(),
        };
        options.handler(spec, catalog, transport)
    }
}

/// Reaches `vendor`'s models through [`SavedKey`], unless a provider is
/// registered for them already: this one, or another plugin's.
fn provide(connector: &mut Connector, vendor: &str, keys: &Keys) {
    if let Some(id) = ProviderId::catalog(vendor)
        && !connector.is_provided(id)
    {
        connector.provide(id, SavedKey(keys.clone()));
    }
}

/// The providers a key alone reaches, by vendor, sorted: those whose
/// preset reads a key from the environment and needs nothing else, with a
/// catalog model that calls tools and no sign-in plan. Copilot is left
/// out: its key is a session token that expires within the hour.
pub(crate) fn key_vendors(connector: &Connector) -> Vec<&'static str> {
    let reached_by_key = |id: &ProviderId| {
        id.requires_credential()
            && !id.api_key_envs().is_empty()
            && id.vendor() != copilot::PROVIDER_NAME
            && id
                .config("key")
                .is_some_and(|config| !config.base_url().is_empty())
    };
    let mut vendors: Vec<&'static str> = ProviderId::all()
        .filter(reached_by_key)
        .map(|id| id.vendor())
        .filter(|vendor| {
            let specs = connector.catalog().iter();
            let mut models = specs.filter(|spec| spec.provider.vendor() == *vendor);
            models.any(|spec| spec.tools && connector.plan(spec).is_none())
        })
        .collect();
    vendors.sort_unstable();
    vendors.dedup();
    vendors
}

/// The variables `vendor`'s key is read from, as a sentence's words.
fn key_envs(vendor: &str) -> String {
    let envs = ProviderId::catalog(vendor).map(|id| id.api_key_envs());
    envs.unwrap_or_default().join(" or ")
}

/// `/login`'s picker: the ChatGPT plan, then each provider a key reaches.
pub(super) fn login_picker(agent: Entity, connector: &Connector) -> Picker {
    let plan = PickItem {
        label: "ChatGPT plan".to_owned(),
        command: format!("login {}", super::PROVIDER),
        detail: Some("sign in in the browser".to_owned()),
        checked: false,
    };
    let keys = key_vendors(connector).into_iter().map(|vendor| PickItem {
        label: vendor.to_owned(),
        command: format!("login {vendor}"),
        detail: Some(format!("API key, used before {}", key_envs(vendor))),
        checked: false,
    });
    Picker {
        items: std::iter::once(plan).chain(keys).collect(),
        ..Picker::new("Sign in").for_agent(agent)
    }
}

/// On `/login <provider>`'s picker: the provider whose key it asks for.
#[derive(Component)]
struct KeyEntry {
    agent: Entity,
    vendor: &'static str,
}

/// Asks for `named`'s API key in a picker that hides it, or says which
/// providers take one.
pub(super) fn ask_key(
    agent: Entity,
    named: &str,
    (keys, connector): (&Keys, &Connector),
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    let vendors = key_vendors(connector);
    let Some(vendor) = vendors.iter().copied().find(|vendor| *vendor == named) else {
        let why = format!(
            "No sign-in for `{named}`. /login takes {} or one of: {}.",
            super::PROVIDER,
            vendors.join(", ")
        );
        notices.write(Notice::error(agent, why));
        return;
    };
    let body = format!(
        "Paste or type the key. It is saved in {}, readable only by you, and used before {}.",
        keys.path(vendor).display(),
        key_envs(vendor),
    );
    let title = format!("API key for {vendor}");
    let picker = Picker::new(title).for_agent(agent).body(body).secret();
    commands.spawn((picker, KeyEntry { agent, vendor }));
}

/// Saves the key typed in `/login <provider>`'s picker, reconnects the
/// provider's idle agents with it, and lets the asking agent pick one of
/// the provider's models when it has no model connected.
fn save_typed_key(
    picked: On<Picked>,
    entries: Query<&KeyEntry>,
    keys: Res<Keys>,
    mut models: ResMut<Models>,
    agents: Query<
        (
            Entity,
            Option<&Connection>,
            Option<&ModelChoice>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(&KeyEntry { agent, vendor }) = entries.get(picked.entity) else {
        return;
    };
    let typed = picked
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.text.as_deref());
    let Some(key) = typed.map(str::trim).filter(|key| !key.is_empty()) else {
        notices.write(Notice::info(
            agent,
            format!("No API key saved for {vendor}."),
        ));
        return;
    };
    let path = keys.path(vendor);
    if let Err(why) = keys.save(vendor, key) {
        let why = format!(
            "Could not save the API key for {vendor} in {}: {why}.",
            path.display()
        );
        notices.write(Notice::error(agent, why));
        return;
    }
    provide(&mut models.0, vendor, &keys);
    reconnect(vendor, &agents, &mut commands);
    notices.write(Notice::info(
        agent,
        format!(
            "Saved the API key for {vendor} in {}; it is used before {}.",
            path.display(),
            key_envs(vendor)
        ),
    ));
    let Ok((_, connected, choice, _)) = agents.get(agent) else {
        return;
    };
    let catalog = models.0.catalog();
    let chose_vendor = choice.is_some_and(|choice| {
        let found = catalog.resolve(&choice.0);
        found.is_ok_and(|found| found.spec.provider.vendor() == vendor)
    });
    if connected.is_none() && !chose_vendor {
        commands.trigger(RunCommand::new(agent, "model".to_owned()));
    }
}

/// Removes a provider's saved key, for `/logout <provider>`.
#[derive(Event)]
pub(super) struct RemoveKey {
    /// The agent that asked.
    pub(super) agent: Entity,
    /// The provider.
    pub(super) vendor: String,
}

/// Removes the key and reconnects the provider's idle agents as the
/// environment says.
fn remove_key(
    remove: On<RemoveKey>,
    keys: Res<Keys>,
    mut models: ResMut<Models>,
    agents: Query<
        (
            Entity,
            Option<&Connection>,
            Option<&ModelChoice>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let (agent, vendor) = (remove.agent, remove.vendor.as_str());
    notices.write(match keys.remove(vendor) {
        Ok(true) => {
            models.set_changed();
            reconnect(vendor, &agents, &mut commands);
            Notice::info(
                agent,
                format!("Removed the API key for {vendor}. Environment variables are unchanged."),
            )
        }
        Ok(false) => Notice::info(
            agent,
            format!("No API key is saved for `{vendor}`; /logout lists what is."),
        ),
        Err(why) => Notice::error(
            agent,
            format!(
                "Could not remove the API key for {vendor} ({}): {why}.",
                keys.path(vendor).display()
            ),
        ),
    });
}

/// Drops the connections of the idle agents on `vendor`'s models, which
/// the kernel connects again once [`Models`] changed, with the key now
/// saved or none.
fn reconnect(
    vendor: &str,
    agents: &Query<
        (
            Entity,
            Option<&Connection>,
            Option<&ModelChoice>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    commands: &mut Commands,
) {
    for (agent, connection, _, busy) in agents {
        if !busy && connection.is_some_and(|connection| connection.spec.provider.vendor() == vendor)
        {
            commands.entity(agent).remove::<Connection>();
        }
    }
}

/// `/logout`'s picker: the ChatGPT sign-in when `signed_in` names it, then
/// each saved key; `None` when nothing is stored.
pub(super) fn logout_picker(
    agent: Entity,
    signed_in: Option<(&str, &str)>,
    keys: &Keys,
) -> Option<Picker> {
    let plan = signed_in.map(|(title, provider)| PickItem {
        label: format!("{title} sign-in"),
        command: format!("logout {provider}"),
        detail: Some("revoked, then deleted".to_owned()),
        checked: false,
    });
    let saved = keys.vendors().into_iter().map(|vendor| PickItem {
        label: format!("API key for {vendor}"),
        command: format!("logout {vendor}"),
        detail: Some(keys.path(&vendor).display().to_string()),
        checked: false,
    });
    let items: Vec<PickItem> = plan.into_iter().chain(saved).collect();
    (!items.is_empty()).then(|| Picker {
        items,
        ..Picker::new("Remove a stored credential").for_agent(agent)
    })
}

#[cfg(test)]
mod tests;
