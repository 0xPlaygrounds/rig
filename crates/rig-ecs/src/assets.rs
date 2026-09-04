//! Prompts and tool definitions as assets (the `assets` feature): a
//! [`Prompt`] is a Markdown or text file, a [`ToolDefinitions`] a JSON
//! array of `ToolDefinition`s, each with a `bevy_asset` loader; a handle
//! on an agent ([`PromptHandle`], [`ToolsHandle`]) becomes the agent's
//! [`Preamble`] and its [`Grant`]s — to the bound handlers the definitions
//! name, in file order — the tick the asset is loaded, once (the marker
//! [`Applied`] says so). The host adds `bevy_asset::AssetPlugin` first,
//! then [`AssetsPlugin`].

use std::marker::PhantomData;

use bevy_app::{App, Plugin, Update};
use bevy_asset::{Asset, AssetApp, AssetLoader, Assets, Handle, LoadContext, io::Reader};
use bevy_ecs::prelude::*;
use bevy_reflect::TypePath;
use rig_core::{completion::ToolDefinition, effect::FamilyDescriptor};

use crate::{
    agent::{Grant, OrderCounter, Preamble},
    bus::Bound,
    systems::next_order_in,
};

/// A prompt: the file's text.
#[derive(Asset, TypePath, Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// The text, as the file has it.
    pub text: String,
}

/// Loads a `.md` or `.txt` file as a [`Prompt`].
#[derive(Debug, Default, Clone, Copy, TypePath)]
pub struct PromptLoader;

impl AssetLoader for PromptLoader {
    type Asset = Prompt;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _context: &mut LoadContext<'_>,
    ) -> Result<Prompt, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = String::from_utf8(bytes).map_err(std::io::Error::other)?;
        Ok(Prompt { text })
    }

    fn extensions(&self) -> &[&str] {
        &["md", "txt", "prompt"]
    }
}

/// Tool definitions: a JSON array of `{ name, description, parameters }`.
#[derive(Asset, TypePath, Debug, Clone, PartialEq)]
pub struct ToolDefinitions {
    /// The definitions, in file order.
    pub tools: Vec<ToolDefinition>,
}

/// Loads a `.tools.json` (or any `.json`) file as [`ToolDefinitions`].
#[derive(Debug, Default, Clone, Copy, TypePath)]
pub struct ToolDefinitionsLoader;

impl AssetLoader for ToolDefinitionsLoader {
    type Asset = ToolDefinitions;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _context: &mut LoadContext<'_>,
    ) -> Result<ToolDefinitions, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let tools = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
        Ok(ToolDefinitions { tools })
    }

    fn extensions(&self) -> &[&str] {
        &["json"]
    }
}

/// The prompt an agent reads its [`Preamble`] from.
#[derive(Component, Debug, Clone)]
pub struct PromptHandle(pub Handle<Prompt>);

/// The definitions an agent's [`Grant`]s come from.
#[derive(Component, Debug, Clone)]
pub struct ToolsHandle(pub Handle<ToolDefinitions>);

/// The asset `A` was applied to this agent: the systems apply once.
#[derive(Component, Debug)]
pub struct Applied<A: Asset>(PhantomData<fn() -> A>);

impl<A: Asset> Default for Applied<A> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// A loaded [`Prompt`] on an agent becomes its [`Preamble`].
pub fn apply_prompts(
    mut commands: Commands,
    prompts: Res<Assets<Prompt>>,
    agents: Query<(Entity, &PromptHandle), Without<Applied<Prompt>>>,
) {
    for (agent, handle) in &agents {
        let Some(prompt) = prompts.get(&handle.0) else {
            continue;
        };
        commands.entity(agent).insert((
            Preamble(Some(prompt.text.trim_end().to_owned())),
            Applied::<Prompt>::default(),
        ));
    }
}

/// Loaded [`ToolDefinitions`] on an agent become its [`Grant`]s: one per
/// definition, in file order, to the bound handler whose descriptor is
/// the tool of that name. A definition no handler serves is no grant
/// (logged): a definition names a tool, the handler is what runs it.
pub fn grant_tools(
    mut commands: Commands,
    definitions: Res<Assets<ToolDefinitions>>,
    agents: Query<(Entity, &ToolsHandle), Without<Applied<ToolDefinitions>>>,
    bound: Query<(Entity, &Bound)>,
    mut orders: ResMut<OrderCounter>,
) {
    for (agent, handle) in &agents {
        let Some(definitions) = definitions.get(&handle.0) else {
            continue;
        };
        for definition in &definitions.tools {
            let tool = bound
                .iter()
                .find_map(|(entity, bound)| match &bound.descriptor.family {
                    FamilyDescriptor::Tool { name, .. } if *name == definition.name => Some(entity),
                    FamilyDescriptor::Tool { .. }
                    | FamilyDescriptor::Completion { .. }
                    | FamilyDescriptor::Embed { .. }
                    | FamilyDescriptor::Rerank { .. }
                    | FamilyDescriptor::Memory { .. }
                    | FamilyDescriptor::Retrieve { .. }
                    | FamilyDescriptor::Custom { .. } => None,
                });
            match tool {
                Some(tool) => {
                    commands.spawn((Grant(tool), next_order_in(&mut orders), ChildOf(agent)));
                }
                None => tracing::warn!(
                    tool = definition.name,
                    "a tool definition no handler serves: not granted"
                ),
            }
        }
        commands
            .entity(agent)
            .insert(Applied::<ToolDefinitions>::default());
    }
}

/// Registers the two assets and their loaders, and the systems that apply
/// them. After `bevy_asset::AssetPlugin`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssetsPlugin;

impl Plugin for AssetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Prompt>()
            .register_asset_loader(PromptLoader)
            .init_asset::<ToolDefinitions>()
            .register_asset_loader(ToolDefinitionsLoader)
            .add_systems(Update, (apply_prompts, grant_tools));
    }
}
