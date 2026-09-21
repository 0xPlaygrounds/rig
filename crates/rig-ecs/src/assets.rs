//! Prompt and tool-definition asset loaders, enabled by the `assets` feature.
//!
//! Loaded handles apply preambles and grants once in [`AssetsSet`] during
//! `Update`, before the rig schedule. Install the asset and agent plugins before
//! [`AssetsPlugin`]; later asset changes are not reapplied.
//!
//! ```
//! use rig_ecs::assets::PromptAsset;
//! let prompt = PromptAsset { text: "Answer briefly.".into() };
//! ```

use std::marker::PhantomData;

use bevy_app::{App, Plugin, Update};
use bevy_asset::{Asset, AssetApp, AssetLoader, Assets, Handle, LoadContext, io::Reader};
use bevy_ecs::prelude::*;
use bevy_reflect::TypePath;
use rig_core::{completion::ToolDefinition, effect::FamilyDescriptor};

use crate::{
    agent::{Grant, Preamble, RunCounter},
    bus::Bound,
};

/// Prompt file text. Applying it to an agent trims trailing whitespace before
/// setting the preamble.
#[derive(Asset, TypePath, Debug, Clone, PartialEq, Eq)]
pub struct PromptAsset {
    /// The text, as the file has it.
    pub text: String,
}

/// Loads `.md`, `.txt`, and `.prompt` files as [`PromptAsset`], returning an I/O
/// error for unreadable files or invalid UTF-8.
#[derive(Debug, Default, Clone, Copy, TypePath)]
pub struct PromptLoader;

impl AssetLoader for PromptLoader {
    type Asset = PromptAsset;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _context: &mut LoadContext<'_>,
    ) -> Result<PromptAsset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = String::from_utf8(bytes).map_err(std::io::Error::other)?;
        Ok(PromptAsset { text })
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

/// Loads a `.json` array as [`ToolDefinitions`], returning an I/O error for
/// unreadable files or invalid JSON.
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
pub struct PromptHandle(pub Handle<PromptAsset>);

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

/// A loaded [`PromptAsset`] on an agent becomes its [`Preamble`].
pub fn apply_prompts(
    mut commands: Commands,
    prompts: Res<Assets<PromptAsset>>,
    agents: Query<(Entity, &PromptHandle), Without<Applied<PromptAsset>>>,
) {
    for (agent, handle) in &agents {
        let Some(prompt) = prompts.get(&handle.0) else {
            continue;
        };
        commands.entity(agent).insert((
            Preamble(Some(prompt.text.trim_end().to_owned())),
            Applied::<PromptAsset>::default(),
        ));
    }
}

/// Apply loaded tool definitions once by spawning [`Grant`] links in file order
/// to bound handlers with matching tool names. Warn and skip unmatched definitions.
pub fn grant_tools(
    mut commands: Commands,
    definitions: Res<Assets<ToolDefinitions>>,
    agents: Query<(Entity, &ToolsHandle), Without<Applied<ToolDefinitions>>>,
    bound: Query<(Entity, &Bound)>,
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
                    commands.spawn((Grant(tool), ChildOf(agent)));
                }
                None => log::warn!(
                    "a tool definition no handler serves: not granted: {}",
                    definition.name
                ),
            }
        }
        commands
            .entity(agent)
            .insert(Applied::<ToolDefinitions>::default());
    }
}

/// The set the applying systems run in, in `Update` (before `RigSchedule`).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetsSet;

/// Registers the two assets and their loaders, and the systems that apply
/// them, in [`AssetsSet`]. Install after `bevy_asset::AssetPlugin` and the agent
/// plugin. Panics if the agent's [`RunCounter`] resource is absent.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssetsPlugin;

impl Plugin for AssetsPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.world().contains_resource::<RunCounter>(),
            "AssetsPlugin needs AgentPlugin first: its grants and preambles are the agent's"
        );
        app.init_asset::<PromptAsset>()
            .register_asset_loader(PromptLoader)
            .init_asset::<ToolDefinitions>()
            .register_asset_loader(ToolDefinitionsLoader)
            .add_systems(Update, (apply_prompts, grant_tools).in_set(AssetsSet));
    }
}
