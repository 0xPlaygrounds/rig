//! The status line's parts in `/plugins`: each one the terminal view knows
//! ([`StatusParts`]) with whether the running `[status] hide` shows it, and
//! a switch that writes `[status] hide` and the running setting at once.

use std::path::Path;

use bevy_ecs::system::SystemParam;
use rig_harness::harness_protocol::config;
use rig_harness::prelude::*;
use rig_tui::status::{StatusParts, StatusSettings};

use super::{Node, Part, Tree, hidden_in, parts_shown, shown_again, write_hide};

/// The parts the status line knows and its running `[status]`, while it
/// runs.
#[derive(SystemParam)]
pub(super) struct Status<'w> {
    parts: Option<Res<'w, StatusParts>>,
    settings: Option<ResMut<'w, StatusSettings>>,
}

impl Status<'_> {
    /// Each part, shown or not; none without the status line.
    pub(super) fn parts(&self) -> Vec<Part> {
        let (Some(parts), Some(settings)) = (&self.parts, &self.settings) else {
            return Vec::new();
        };
        (parts.iter())
            .map(|part| Part {
                name: part.name,
                label: part.label,
                shown: !settings.hide.iter().any(|name| name == part.name),
            })
            .collect()
    }

    /// Whether the running setting hides `name`.
    fn hides(&self, name: &str) -> bool {
        (self.settings.as_ref()).is_some_and(|settings| settings.hide.iter().any(|it| it == name))
    }

    /// The parts of `tree` whose ticks in `on` differ from the running
    /// setting: each part's name and whether to show it.
    pub(super) fn switches(&self, tree: &Tree, on: &[bool]) -> Vec<(&'static str, bool)> {
        (parts_shown(tree, on))
            .filter(|(part, shown)| *shown == self.hides(part.name))
            .map(|(part, shown)| (part.name, shown))
            .collect()
    }

    /// `on` with the parts of `tree` ticked as the running setting shows
    /// them.
    pub(super) fn as_running(&self, tree: &Tree, on: &[bool]) -> Vec<bool> {
        (tree.nodes.iter().zip(on))
            .map(|((node, _), now)| match (node, tree.part(node)) {
                (Node::Parts, _) => true,
                (_, Some(part)) => !self.hides(part.name),
                _ => *now,
            })
            .collect()
    }

    /// The names the running setting or the settings file at `path` hides,
    /// the file read even when the status line does not run.
    pub(super) fn hidden(&self, path: &Path) -> Vec<String> {
        let mut hidden = (self.settings.as_ref()).map_or_else(Vec::new, |it| it.hide.clone());
        for name in hidden_in(path) {
            if !hidden.contains(&name) {
                hidden.push(name);
            }
        }
        hidden
    }

    /// Shows at once every part [`Self::hidden`] names, saying so for
    /// `agent`, or why the file at `path` cannot be written.
    pub(super) fn show_all(
        &mut self,
        path: &Path,
        agent: Entity,
        notices: &mut MessageWriter<Notice>,
    ) {
        let hidden = self.hidden(path);
        if hidden.is_empty() {
            return;
        }
        let changes: Vec<(&str, bool)> = hidden.iter().map(|name| (name.as_str(), true)).collect();
        notices.write(shown_again(
            path,
            &hidden,
            agent,
            self.switch(path, &changes),
        ));
    }

    /// Applies the switches of the parts of `tree` ticked in `on` with
    /// [`Self::switch`], when there are any. Returns the error and the
    /// file it names when it cannot be written.
    pub(super) fn apply(
        &mut self,
        path: &Path,
        tree: &Tree,
        on: &[bool],
    ) -> Result<(), config::Error> {
        let switches = self.switches(tree, on);
        if switches.is_empty() {
            return Ok(());
        }
        self.switch(path, &switches)
    }

    /// Shows each part `changes` names `true` and hides each named `false`,
    /// in `[status] hide` of the settings file at `path`, keeping every
    /// other line, name and comment, then in the running setting. Nothing
    /// changes when the file cannot be written.
    ///
    /// # Errors
    ///
    /// When the file does not read, its `[status]` is not a table or its
    /// `hide` not a list, or it cannot be written, naming the file.
    pub(super) fn switch(
        &mut self,
        path: &Path,
        changes: &[(&str, bool)],
    ) -> Result<(), config::Error> {
        write_hide(path, changes)?;
        if let Some(settings) = &mut self.settings {
            let mut hide = settings.hide.clone();
            switch(&mut hide, changes);
            if hide != settings.hide {
                settings.hide = hide;
            }
        }
        Ok(())
    }
}

/// `names` with each name of `changes` shown (`true`) out of it and each
/// hidden (`false`) at its end.
fn switch(names: &mut Vec<String>, changes: &[(&str, bool)]) {
    for (name, shown) in changes {
        let has = names.iter().any(|it| it == name);
        match (shown, has) {
            (true, true) => names.retain(|it| it != name),
            (false, false) => names.push((*name).to_owned()),
            _ => {}
        }
    }
}
