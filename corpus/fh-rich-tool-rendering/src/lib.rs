//! FH-19: rich tool call rendering in the terminal view (`rig-tui`).
//!
//! The built-in tools' calls get looks of their own, in place of the
//! default plugins' ([`TOOLS`]): `edit` and `write` as a coloured,
//! highlighted diff that collapses after [`COLLAPSED_LINES`] lines, `shell`
//! as `$ command` with its output as it comes, then the last
//! [`SHELL_TAIL`] lines and `exit N · 4.2s`, `read` as one line with the
//! lines read, `search` as a count with the first matches. The terminal
//! view draws a call through its look while the model still streams its
//! arguments, and keeps each call's expanded state; Ctrl+O expands the
//! call selected (Alt+Up), or with none, expands or collapses every call.
//! Any other tool keeps its own renderer or the default look; the looks
//! never change what is saved or sent.

mod highlight;
mod looks;

use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{AssistantContent, Message};
use rig_tui::{
    AppKeyBindingsExt, AppToolRenderersExt, InputAction, KeyBinding, KeyPress, KeyPressed,
    ToolRenderer, TranscriptItem, TranscriptView, TuiPlugin,
};

pub use looks::{COLLAPSED_LINES, SHELL_TAIL};

/// The tools this plugin draws: the default plugins' looks of them are
/// turned off.
pub const TOOLS: [&str; 5] = ["edit", "write", "shell", "read", "search"];

/// Draws the built-in tools' calls the way they read best.
#[derive(Default)]
pub struct RichToolRenderingPlugin;

impl Plugin for RichToolRenderingPlugin {
    fn build(&self, _: &mut App) {}

    fn finish(&self, app: &mut App) {
        // Every plugin's `build` ran: the default looks exist and are
        // turned off here, so their names are free.
        let world = app.world_mut();
        let mut renderers = world.query::<(Entity, &ToolRenderer)>();
        let taken: Vec<Entity> = renderers
            .iter(world)
            .filter(|(_, renderer)| TOOLS.contains(&renderer.tool.as_str()))
            .map(|(entity, _)| entity)
            .collect();
        for renderer in taken {
            world.entity_mut(renderer).insert(Disabled);
        }
        for tool in TOOLS {
            app.add_tool_renderer(tool, looks::render);
        }
        if app.is_plugin_added::<TuiPlugin>() {
            // A global binding shadows the input's own Ctrl+O.
            let toggle = KeyBinding::new(KeyPress::ctrl('o'), "Expand or collapse tool calls");
            app.add_key_binding(toggle, toggle_calls);
        }
    }
}

/// Ctrl+O: the selected item expanded or collapsed, as the view does it;
/// with none selected, every tool call of the agent shown collapsed when
/// one is expanded, else every one expanded.
fn toggle_calls(
    pressed: On<KeyPressed>,
    agents: Query<&Conversation>,
    mut view: ResMut<TranscriptView>,
    mut actions: MessageWriter<InputAction>,
) {
    if view.selected.is_some() {
        actions.write(InputAction::Expand);
        return;
    }
    let Some(conversation) = pressed.agent.and_then(|agent| agents.get(agent).ok()) else {
        return;
    };
    let calls: Vec<TranscriptItem> = conversation
        .messages()
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant) => Some(assistant.content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(TranscriptItem::Call(call.id.clone())),
            _ => None,
        })
        .collect();
    if calls.iter().any(|call| view.expanded.contains(call)) {
        for call in &calls {
            view.expanded.remove(call);
        }
    } else {
        view.expanded.extend(calls);
    }
}

#[cfg(test)]
mod tests;
