//! `/plugins`: rig's plugins and the extras of `RIG_HOME/settings.toml`,
//! read from that file in this process, each with whether it is on and
//! whether this build runs it (its `PluginSource` entity) and which plugin
//! loaded it, if one did (its `ProvidedBy`). In the terminal view it is a
//! picker with a tree: crates at the top, then areas, then
//! plugins by short name, branches folded, each plugin's own choice ticked
//! and each branch ticked while `[plugins] off` leaves it on, with the
//! highlighted row's detail beside it ([`PickPreview`]). A crate or an area
//! of one plugin is that plugin's row, unless `off` names it. Enter ticks
//! and unticks a plugin's own choice, and switches a branch over its
//! plugins as the view does ([`PickItem::parent`]); each row that differs
//! from the file shows `+ on` or `- off`, worked out by making the changes
//! to a copy of the file. Leaving with changes asks whether to reload now
//! ([`ReloadStatus::ask`]), at the next start, to keep editing or to leave
//! without writing, and adds up what the changes take away; either of the
//! first two writes every change in one edit of the file
//! ([`config::turn`]). `/plugins defaults` asks the same about going back
//! to rig's defaults, branches turned back on. Without a terminal it lists
//! them. While the status line runs, a last branch holds its parts,
//! settings rather than plugins: ticking one writes `[status] hide` and
//! changes the running line at once, outside the question, and
//! `/plugins defaults` shows them all again.
//!
//! `rig plugin enable` and `rig plugin disable` change the same file from a
//! shell, with the same rules.

use rig_harness::harness_protocol::config::{
    self, BUILT_IN, BuiltIn, Extra, Plugins, SettingsFile,
};
use rig_harness::harness_protocol::{self, Home};
use rig_harness::pick::{Highlighted, PickPreview, Picked, Picker, Ticked};
use rig_harness::prelude::*;
use rig_harness::services::RigHome;

use crate::reload::{Rebuild, ReloadStatus};

#[cfg(feature = "tui")]
mod parts;

/// `/plugins`.
#[derive(Default)]
pub struct PluginsPlugin;

impl Plugin for PluginsPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "plugins",
            "List every plugin, on and off, and turn them on and off in settings.toml; \
             `/plugins defaults` goes back to rig's defaults",
            plugins,
        );
        let mark = (preview, unhighlighted, mark);
        #[cfg(feature = "tui")]
        let mark = mark.before(rig_tui::ChoiceSystems);
        app.add_systems(Update, mark);
    }
}

/// One plugin as `/plugins` shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    /// Its type path, which settings.toml names.
    plugin: String,
    /// Its crate, the top of its place in the tree.
    krate: String,
    /// What the list shows: a built-in's short name, an extra's crate.
    name: String,
    /// Whether its own choice in settings.toml is on.
    on: bool,
    /// Whether rig's defaults turn it on here: an extra always.
    default: bool,
    /// The `target_os` values it builds on, when this is none of them.
    unbuilt: Option<&'static [&'static str]>,
    /// Its state, short: `on`, `off`, or why it is off.
    state: String,
    /// What goes when it is off, for an essential plugin.
    essential: Option<&'static str>,
    /// Everything about it, for the preview.
    about: String,
}

/// A part of the status line, a setting rather than a plugin.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Part {
    /// Its items' name, which `[status] hide` takes.
    name: &'static str,
    /// What the list shows.
    label: &'static str,
    /// Whether the running `[status] hide` leaves it out.
    shown: bool,
}

/// A row of the tree.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Node {
    /// A branch, by its name in `[plugins] off`, shown as `label`.
    Branch { name: String, label: String },
    /// The plugin of [`Tree::rows`] at this position.
    Plugin(usize),
    /// The branch of the status line's parts.
    Parts,
    /// The part of [`Tree::parts`] at this position.
    Part(usize),
}

impl Node {
    /// Whether it is of the status line's parts, which apply at once and
    /// are never among the changes of plugins.
    fn setting(&self) -> bool {
        matches!(self, Node::Parts | Node::Part(_))
    }
}

/// The label of the branch of the status line's parts.
const PARTS: &str = "Status line parts";

/// The `/plugins` list against settings.toml: its plugins, its tree, and
/// the file as read with what it turns on, which changes are tried on.
#[derive(Clone, Debug)]
struct Tree {
    rows: Vec<Row>,
    /// The status line's parts, as the list opened.
    parts: Vec<Part>,
    /// The tree's rows in its order, each with the position of its branch.
    nodes: Vec<(Node, Option<usize>)>,
    file: SettingsFile,
    before: Plugins,
}

/// What ticking a [`Tree`]'s rows makes of settings.toml.
struct Trial<'a> {
    /// The changes to write: a plugin or a branch and whether it is on.
    changes: Vec<(&'a str, bool)>,
    /// What the file turns on with them, or why it would not read.
    after: Result<Plugins, config::Error>,
    /// The plugins ticked on that do not build here, which stay off.
    unbuilt: Vec<&'a Row>,
}

impl Tree {
    /// The tree of the plugins `before` that `file` turns on, with whether
    /// `running` holds each and the plugin that added it, and the status
    /// line's `parts`.
    fn new(
        file: SettingsFile,
        before: Plugins,
        running: &[(String, Option<String>)],
        parts: Vec<Part>,
    ) -> Self {
        let rows = rows(&before, running);
        let mut nodes = nodes(&rows, &before.off);
        if !parts.is_empty() {
            let branch = nodes.len();
            nodes.push((Node::Parts, None));
            nodes.extend((0..parts.len()).map(|part| (Node::Part(part), Some(branch))));
        }
        Tree {
            rows,
            parts,
            nodes,
            file,
            before,
        }
    }

    /// The row of the plugin at node `node`.
    fn row(&self, node: &Node) -> Option<&Row> {
        match node {
            Node::Plugin(row) => self.rows.get(*row),
            Node::Branch { .. } | Node::Parts | Node::Part(_) => None,
        }
    }

    /// The status line's part at node `node`.
    fn part(&self, node: &Node) -> Option<&Part> {
        match node {
            Node::Part(part) => self.parts.get(*part),
            _ => None,
        }
    }

    /// Whether settings.toml has each node on: a plugin's own choice, a
    /// branch out of `[plugins] off`.
    fn on(&self) -> Vec<bool> {
        (self.nodes.iter())
            .map(|(node, _)| match node {
                Node::Branch { name, .. } => !self.before.off.contains(name),
                Node::Plugin(_) => self.row(node).is_some_and(|row| row.on),
                Node::Parts => true,
                Node::Part(_) => self.part(node).is_some_and(|part| part.shown),
            })
            .collect()
    }

    /// rig's defaults: each plugin's default, every branch and part on.
    fn defaults(&self) -> Vec<bool> {
        (self.nodes.iter())
            .map(|(node, _)| match node {
                Node::Plugin(_) => self.row(node).is_some_and(|row| row.default),
                _ => true,
            })
            .collect()
    }

    /// How many plugins and branches `on` changes.
    fn changed(&self, on: &[bool]) -> usize {
        (self.nodes.iter().zip(self.on()).zip(on))
            .filter(|(((node, _), was), now)| !node.setting() && was != *now)
            .count()
    }

    /// What ticking the nodes as `on` makes of settings.toml, tried on a
    /// copy of it.
    fn trial(&self, on: &[bool]) -> Trial<'_> {
        let mut changes = Vec::new();
        let mut unbuilt = Vec::new();
        let nodes = self.nodes.iter().zip(self.on()).zip(on);
        for (((node, _), _), now) in nodes.filter(|((_, was), now)| was != *now) {
            match (node, self.row(node)) {
                (Node::Parts | Node::Part(_), _) => {}
                (Node::Branch { name, .. }, _) => changes.push((name.as_str(), *now)),
                (_, Some(row)) if *now && row.unbuilt.is_some() => unbuilt.push(row),
                (_, Some(row)) => changes.push((row.plugin.as_str(), *now)),
                (_, None) => {}
            }
        }
        let mut file = self.file.clone();
        let after = config::turn(&mut file, &changes).and_then(|_| config::plugins(&file));
        Trial {
            changes,
            after,
            unbuilt,
        }
    }

    /// What the preview says of node `node`.
    fn about(&self, node: usize) -> String {
        match self.nodes.get(node) {
            Some((Node::Branch { name, .. }, _)) => Some(self.branch_about(name)),
            Some((Node::Parts, _)) => Some(PARTS_ABOUT.to_owned()),
            Some((node @ Node::Part(_), _)) => self.part(node).map(Part::about),
            Some((node, _)) => self.row(node).map(|row| row.about.clone()),
            None => None,
        }
        .unwrap_or_default()
    }

    /// What the preview says of the branch `name`.
    fn branch_about(&self, name: &str) -> String {
        let held: Vec<&str> = (self.rows.iter())
            .filter(|row| config::holds(name, &row.krate, &row.plugin))
            .map(|row| row.name.as_str())
            .collect();
        let state = if self.before.off.iter().any(|off| off == name) {
            "Off: [plugins] off names it, which keeps each of its plugins off whatever its own \
             choice. Enter turns it back on, and each plugin comes back as its own choice says."
        } else {
            "On. Enter turns it off, into [plugins] off, and each plugin keeps its own choice for \
             when it is back on; on with every plugin off, Enter turns each of them on."
        };
        format!(
            "{name}\nBranch of {} plugins: {}.\n\n{state}\n",
            held.len(),
            held.join(", ")
        )
    }
}

/// What the preview says of the branch of the status line's parts.
const PARTS_ABOUT: &str = "Status line parts\nSettings, not plugins: the parts of the status line \
under the input, as [status] hide in settings.toml leaves them in or out.\n\nEnter on a part \
shows or hides it at once, and writes [status] hide; unticking the branch hides them all. A part a \
plugin draws without declaring it shows here once it is drawn.\n";

impl Part {
    /// What the preview says of it.
    fn about(&self) -> String {
        format!(
            "{}\nStatus line part `{}`, a setting\n\nEnter shows or hides it at once, and \
             writes [status] hide in settings.toml. Other plugins still read it.\n",
            self.label, self.name
        )
    }
}

impl Trial<'_> {
    /// What the file turns on with the changes, when it reads.
    fn after(&self) -> Option<&Plugins> {
        self.after.as_ref().ok()
    }
}

/// Whether `plugins` turns `plugin` on.
fn runs(plugins: &Plugins, plugin: &str) -> bool {
    plugins.on.iter().any(|on| on == plugin)
}

/// The `/plugins` list: its tree, against settings.toml, and the agent it
/// is for.
#[derive(Component)]
struct PluginsPicker {
    tree: Tree,
    agent: Entity,
}

/// The question asked on leaving the list with changes: the list's tree,
/// which nodes are ticked, the node highlighted, whether a reload can run
/// and the answers offered.
#[derive(Component)]
struct LeaveQuestion {
    tree: Tree,
    on: Vec<bool>,
    agent: Entity,
    at: usize,
    reload: Reload,
    answers: Vec<Leave>,
}

/// An answer to [`LeaveQuestion`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leave {
    /// Write the changes and rebuild once no turn runs.
    Reload,
    /// Write the changes for the next start.
    Later,
    /// Back to the list, with the ticks.
    Edit,
    /// Close the list and write nothing.
    Discard,
}

impl Leave {
    /// Its label in the question.
    fn label(self) -> &'static str {
        match self {
            Leave::Reload => "Reload now",
            Leave::Later => "Later, at the next start",
            Leave::Edit => "Keep editing",
            Leave::Discard => "Leave without writing",
        }
    }
}

/// `/plugins`: the picker on an interactive terminal, else the list;
/// `/plugins defaults`: the question about going back to rig's defaults.
fn plugins(
    In(args): In<CommandArgs>,
    home: Option<Res<RigHome>>,
    invoked: Option<Res<Invoked>>,
    running: Query<(&Name, Option<&ProvidedBy>), With<PluginSource>>,
    slash_commands: Query<&Name, With<SlashCommand>>,
    #[cfg(feature = "tui")] mut status: parts::Status,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let defaults = match args.args.as_str() {
        "" => false,
        "defaults" => true,
        other => {
            notices.write(Notice::error(
                args.agent,
                format!("/plugins takes nothing or `defaults`, not `{other}`."),
            ));
            return;
        }
    };
    let home = home.map_or_else(Home::from_env, |home| home.0.clone());
    let read = SettingsFile::read(&home.settings())
        .and_then(|file| config::plugins(&file).map(|chosen| (file, chosen)));
    let (file, chosen) = match read {
        Ok(read) => read,
        Err(why) => {
            // An unknown plugin's refusal names its own fix.
            let fix = match &why {
                config::Error::Unavailable { plugin, .. } => {
                    format!("`rig plugin disable {plugin}` in a shell turns it off.")
                }
                _ => "Edit the file to fix it.".to_owned(),
            };
            notices.write(Notice::error(
                args.agent,
                format!(
                    "{} does not read, so /plugins cannot tell which plugins are on: {why}\n{fix}",
                    home.settings().display()
                ),
            ));
            return;
        }
    };
    let short =
        |name: &Name| config::built_in(name).map_or(name.to_string(), |it| it.name.to_owned());
    let running: Vec<(String, Option<String>)> = (running.iter())
        .map(|(name, by)| {
            let by = by
                .and_then(|by| running.get(by.0).ok())
                .map(|(name, _)| name);
            (name.to_string(), by.map(short))
        })
        .collect();
    #[cfg(feature = "tui")]
    let parts = status.parts();
    #[cfg(not(feature = "tui"))]
    let parts = Vec::new();
    let tree = Tree::new(file, chosen, &running, parts);
    let interactive = invoked.is_some_and(|invoked| invoked.interactive());
    if defaults {
        let on = tree.defaults();
        #[cfg(feature = "tui")]
        let hides = !status.hidden(&home.settings()).is_empty();
        #[cfg(not(feature = "tui"))]
        let hides = !hidden_in(&home.settings()).is_empty();
        if tree.changed(&on) == 0 && !hides {
            notices.write(Notice::info(
                args.agent,
                format!("{} already runs rig's defaults.", home.settings().display()),
            ));
            return;
        }
        // The question needs a terminal to show it, and without one
        // nothing is written, `[status] hide` included.
        if !interactive {
            let mut commands: Vec<String> = (tree.trial(&on).changes.iter())
                .map(|(name, on)| {
                    let verb = if *on { "enable" } else { "disable" };
                    format!("  rig plugin {verb} {name}")
                })
                .collect();
            if hides {
                commands.push(format!(
                    "  delete the `hide` line under [status] in {}",
                    home.settings().display()
                ));
            }
            notices.write(Notice::error(
                args.agent,
                format!(
                    "/plugins defaults asks before it writes, which needs rig on a terminal. \
                     These go back to rig's defaults:\n{}",
                    commands.join("\n")
                ),
            ));
            return;
        }
        #[cfg(feature = "tui")]
        if hides {
            status.show_all(&home.settings(), args.agent, &mut notices);
        }
        #[cfg(not(feature = "tui"))]
        if hides {
            let hidden = hidden_in(&home.settings());
            let changes: Vec<(&str, bool)> =
                hidden.iter().map(|name| (name.as_str(), true)).collect();
            let written = write_hide(&home.settings(), &changes);
            notices.write(shown_again(&home.settings(), &hidden, args.agent, written));
        }
        if tree.changed(&on) == 0 {
            return;
        }
        let reload = Reload::here(&slash_commands);
        ask(&mut commands, (tree, on), args.agent, 0, reload);
        return;
    }
    if !interactive {
        let rows = &tree.rows;
        let names = rows.iter().map(|row| row.name.len()).max().unwrap_or(0);
        let paths = rows.iter().map(|row| row.plugin.len()).max().unwrap_or(0);
        let lines: Vec<String> = (rows.iter())
            .map(|row| {
                format!(
                    "{:<names$}  {:<paths$}  {}",
                    row.name, row.plugin, row.state
                )
            })
            .collect();
        notices.write(Notice::info(args.agent, lines.join("\n")));
        return;
    }
    let on = tree.on();
    list(&mut commands, tree, &on, args.agent, 0);
}

/// Opens the list of `tree` for `agent` with `on` ticked and node `at`
/// selected.
fn list(commands: &mut Commands, tree: Tree, on: &[bool], agent: Entity, at: usize) {
    let mut picker = Picker::new("Plugins")
        .for_agent(agent)
        .body(body(&tree, on))
        .multiple();
    picker.selected = at;
    let details = details(&tree, on);
    picker.items = (tree.nodes.iter().zip(on).zip(details))
        .map(|(((node, parent), ticked), detail)| {
            let (label, more) = match (node, tree.row(node), tree.part(node)) {
                (Node::Branch { name, label }, ..) => (label.as_str(), name.clone()),
                (Node::Parts, ..) => (PARTS, "[status] hide".to_owned()),
                (_, _, Some(part)) => (part.label, format!("[status] hide: {}", part.name)),
                // Typing a type path finds its plugin.
                (_, Some(row), _) => (row.name.as_str(), format!("{}: {}", row.plugin, row.state)),
                (_, None, None) => ("", String::new()),
            };
            PickItem {
                checked: *ticked,
                more: Some(more),
                parent: *parent,
                ..PickItem::new(label, "").detail(detail)
            }
        })
        .collect();
    // The terminal view gives the preview its half of the screen from the
    // first frame only when the picker comes with one.
    let shown = tree.about(at);
    let mut list = commands.spawn((
        picker,
        PickPreview::Text(shown),
        PluginsPicker { tree, agent },
    ));
    #[cfg(feature = "tui")]
    list.insert(rig_tui::ChoiceKeys(vec![(
        rig_tui::crossterm::event::KeyCode::Enter,
        rig_tui::ChoiceAction::Tick,
    )]));
    list.observe(leave);
}

/// The list's body: its keys and how many rows `on` changes.
fn body(tree: &Tree, on: &[bool]) -> String {
    let keys = "Enter ticks or unticks a plugin or a branch, → and ← open and close branches, \
                Esc leaves, type to find. Status line parts apply at once.";
    match tree.changed(on) {
        0 => format!("{keys} Nothing is written until you leave."),
        1 => format!("{keys} 1 change: leaving asks whether to reload now or later."),
        changes => {
            format!("{keys} {changes} changes: leaving asks whether to reload now or later.")
        }
    }
}

/// What the list shows beside each node while `on` is ticked: its state,
/// or how the ticks change it.
fn details(tree: &Tree, on: &[bool]) -> Vec<String> {
    let trial = tree.trial(on);
    let was = tree.on();
    (tree.nodes.iter().zip(on).zip(was))
        .map(|(((node, _), now), was)| match (node, tree.row(node)) {
            (Node::Parts | Node::Part(_), _) => part_detail(tree, node, on),
            (Node::Branch { name, .. }, _) => branch_detail(tree, name, (was, *now), &trial),
            (_, Some(row)) => detail(tree, row, *now, &trial),
            (_, None) => String::new(),
        })
        .collect()
}

/// The names `[status] hide` in the settings file at `path` hides; none
/// when the file does not read.
fn hidden_in(path: &std::path::Path) -> Vec<String> {
    let file = SettingsFile::read(path).ok();
    (file.as_ref())
        .and_then(|file| file.document.get("status")?.get("hide")?.as_array())
        .into_iter()
        .flatten()
        .filter_map(|name| name.as_str().map(str::to_owned))
        .collect()
}

/// Shows each part `changes` names `true` and hides each named `false`, in
/// `[status] hide` of the settings file at `path`, keeping every other
/// line, name and comment.
///
/// # Errors
///
/// When the file does not read, its `[status]` is not a table or its `hide`
/// not a list, or it cannot be written, naming the file.
fn write_hide(path: &std::path::Path, changes: &[(&str, bool)]) -> Result<(), config::Error> {
    config::edit(path, |file| {
        let hides = changes.iter().any(|(_, shown)| !shown);
        if !hides && file.document.get("status").is_none() {
            return Ok((false, ()));
        }
        let path = file.path.clone();
        let problem = |problem: &str| config::Error::Table {
            path: path.clone(),
            table: "status".to_owned(),
            problem: problem.to_owned(),
        };
        let table =
            config::table(&mut file.document, "status").ok_or_else(|| problem("not a table"))?;
        if table.get("hide").is_some_and(|hide| !hide.is_array()) {
            return Err(problem("its hide is not a list"));
        }
        let changed = changes.iter().fold(false, |changed, (name, shown)| {
            let switched = if *shown {
                config::drop_name(table, "hide", name)
            } else {
                config::add_name(table, "hide", name)
            };
            switched | changed
        });
        Ok((changed, ()))
    })?;
    Ok(())
}

/// What `/plugins defaults` tells `agent` once it has written the file at
/// `path` to show every part of `hidden` again, or why it could not.
fn shown_again(
    path: &std::path::Path,
    hidden: &[String],
    agent: Entity,
    written: Result<(), config::Error>,
) -> Notice {
    match written {
        Ok(()) => Notice::info(
            agent,
            format!(
                "{}: every status line part is shown again; [status] hide named {}.",
                path.display(),
                hidden.join(", ")
            ),
        ),
        Err(why) => Notice::error(agent, format!("{} is unchanged: {why}", path.display())),
    }
}

/// Which of the status line's parts `on` shows: each part's node ticked
/// under its branch ticked.
fn parts_shown<'a>(tree: &'a Tree, on: &'a [bool]) -> impl Iterator<Item = (&'a Part, bool)> {
    let ticked = |node: usize| on.get(node).copied().unwrap_or(false);
    let branch = tree.nodes.iter().position(|(node, _)| *node == Node::Parts);
    let branch = branch.is_some_and(ticked);
    (tree.nodes.iter().enumerate())
        .filter_map(move |(at, (node, _))| Some((tree.part(node)?, branch && ticked(at))))
}

/// What the list shows beside the status line's part, or their branch,
/// at `node` while `on` is ticked.
fn part_detail(tree: &Tree, node: &Node, on: &[bool]) -> String {
    let mut parts = parts_shown(tree, on);
    if let Some(part) = tree.part(node) {
        let shown = parts.find(|(other, _)| *other == part);
        return match shown {
            Some((_, true)) => "shown".to_owned(),
            _ => "hidden".to_owned(),
        };
    }
    let (count, hidden) = parts.fold((0, 0), |(count, hidden), (_, shown)| {
        (count + 1, hidden + usize::from(!shown))
    });
    match hidden {
        0 => "all shown".to_owned(),
        hidden => format!("{hidden} of {count} hidden"),
    }
}

/// What the list shows beside the branch `name`, on in settings.toml or
/// not and ticked or not (`was`, `now`) as `trial` tries.
fn branch_detail(tree: &Tree, name: &str, (was, now): (bool, bool), trial: &Trial) -> String {
    match (was, now) {
        (true, false) => "- off".to_owned(),
        (false, true) => "+ on".to_owned(),
        (false, false) => "off ([plugins] off)".to_owned(),
        (true, true) => {
            let plugins = trial.after().unwrap_or(&tree.before);
            let held = tree.before.held(name);
            let on = held.iter().filter(|plugin| runs(plugins, plugin)).count();
            format!("{on} of {} on", held.len())
        }
    }
}

/// What the list shows beside `row`, ticked or not (`now`), as `trial`
/// tries: its state, or how the ticks change it.
fn detail(tree: &Tree, row: &Row, now: bool, trial: &Trial) -> String {
    if let Some(targets) = row.unbuilt.filter(|_| now && !row.on) {
        return format!("+ on, but it builds on {} only", targets.join("/"));
    }
    let Some(after) = trial.after() else {
        return row.state.clone();
    };
    let plugin = row.plugin.as_str();
    let (was, will) = (runs(&tree.before, plugin), runs(after, plugin));
    let branch = after.off_with(plugin);
    let moved = branch != tree.before.off_with(plugin);
    let what = (row.essential.filter(|_| was && !will))
        .map(|what| format!(", takes away {what}"))
        .unwrap_or_default();
    match (row.on, now, branch) {
        (true, false, _) => format!("- off{what}"),
        (false, true, Some(branch)) if !will => {
            format!("+ on, still off with its branch {branch}")
        }
        (false, true, _) => "+ on".to_owned(),
        // The branch is the row above, or the preview names it.
        _ if was != will && moved => {
            let word = if will { "+ on" } else { "- off" };
            format!("{word} with its branch{what}")
        }
        _ => row.state.clone(),
    }
}

/// Which of `count` nodes `positions` ticks.
fn ticks(count: usize, positions: &[usize]) -> Vec<bool> {
    (0..count).map(|index| positions.contains(&index)).collect()
}

/// Every plugin, rig's and then the extras, in the tree's order
/// ([`config::tree_order`]), with their own choice in `chosen`, the
/// branch keeping each off, and whether `running` holds each and the
/// plugin that added it.
fn rows(chosen: &Plugins, running: &[(String, Option<String>)]) -> Vec<Row> {
    let on = |plugin: &str| chosen.own.iter().any(|on| on == plugin);
    let off = |plugin: &str| chosen.off_with(plugin).map(str::to_owned);
    let runs = |plugin: &str| {
        let found = running.iter().find(|(runs, _)| runs == plugin);
        (found.is_some(), found.and_then(|(_, by)| by.as_deref()))
    };
    let built_ins = (BUILT_IN.iter()).map(|built_in| {
        let plugin = built_in.plugin;
        built_in_row(built_in, (on(plugin), off(plugin)), runs(plugin))
    });
    let mut names: Vec<String> = (BUILT_IN.iter())
        .map(|built_in| built_in.name.to_owned())
        .chain(
            chosen
                .extra
                .iter()
                .map(|extra| extra_name(extra, &chosen.extra)),
        )
        .collect();
    for (index, extra) in chosen.extra.iter().enumerate() {
        unique(&mut names, BUILT_IN.len() + index, extra);
    }
    let extras = (chosen
        .extra
        .iter()
        .zip(names.into_iter().skip(BUILT_IN.len())))
    .map(|(extra, name)| {
        let plugin = &extra.plugin;
        extra_row(extra, name, (on(plugin), off(plugin)), runs(plugin))
    });
    // Every extra's crate is after rig's.
    config::tree_order(built_ins.chain(extras).collect(), |row: &Row| {
        (row.krate.as_str(), row.plugin.as_str())
    })
}

/// The tree over `rows`, in the tree's order: a branch for each crate and
/// each area of several plugins, or that `off` names, before what it
/// holds, and a node for each plugin, each with the position of its
/// branch.
fn nodes(rows: &[Row], off: &[String]) -> Vec<(Node, Option<usize>)> {
    let mut nodes = Vec::new();
    // The crate and area of the row before, and the node of each.
    let mut krate: Option<(&str, Option<usize>)> = None;
    let mut area: Option<(&str, Option<usize>)> = None;
    for (index, row) in rows.iter().enumerate() {
        let mut branch = |name: String, label: &str, parent: Option<usize>, holds: usize| {
            (holds > 1 || off.contains(&name)).then(|| {
                let label = label.to_owned();
                nodes.push((Node::Branch { name, label }, parent));
                nodes.len() - 1
            })
        };
        if krate.is_none_or(|(name, _)| name != row.krate) {
            let holds = rows.iter().filter(|other| other.krate == row.krate).count();
            krate = Some((
                row.krate.as_str(),
                branch(row.krate.clone(), &row.krate, None, holds),
            ));
            area = None;
        }
        let parent = krate.and_then(|(_, node)| node);
        match config::area(&row.plugin) {
            Some(name) if area.is_none_or(|(was, _)| was != name) => {
                let holds = (rows.iter())
                    .filter(|other| {
                        other.krate == row.krate && config::area(&other.plugin) == Some(name)
                    })
                    .count();
                let node = branch(format!("{}/{name}", row.krate), name, parent, holds);
                area = Some((name, node));
            }
            Some(_) => {}
            None => area = None,
        }
        let parent = area.and_then(|(_, node)| node).or(parent);
        nodes.push((Node::Plugin(index), parent));
    }
    nodes
}

/// Whether "Reload now" can run here, else why not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reload {
    /// `/reload` is a command and the launcher started rig.
    Can,
    /// `/reload` is not a command: `ReloadPlugin` is off.
    NoPlugin,
    /// The launcher did not start rig.
    NoLauncher,
}

impl Reload {
    /// Whether a reload can run, with `/reload` among `slash_commands` or
    /// not.
    fn here(slash_commands: &Query<&Name, With<SlashCommand>>) -> Self {
        let plugin = slash_commands.iter().any(|name| name.as_str() == "/reload");
        match (plugin, harness_protocol::launcher().is_some()) {
            (false, _) => Reload::NoPlugin,
            (true, false) => Reload::NoLauncher,
            (true, true) => Reload::Can,
        }
    }

    /// Why "Reload now" is not offered, when it is not.
    fn why_not(self) -> Option<&'static str> {
        match self {
            Reload::Can => None,
            Reload::NoPlugin => Some(
                "Reloading now needs the Reload plugin, which is off; the changes apply when \
                 rig starts again.",
            ),
            Reload::NoLauncher => Some(
                "Reloading now needs rig started with `rig`; the changes apply when it starts \
                 again.",
            ),
        }
    }

    /// When changes written for later apply.
    fn later(self) -> &'static str {
        match self {
            Reload::Can => "They apply when rig starts again, or at /reload.",
            Reload::NoPlugin | Reload::NoLauncher => "They apply when rig starts again.",
        }
    }
}

/// An area's title, from its module name: `coding_tools` is `Coding tools`.
fn title(area: &str) -> String {
    let words = area.replace('_', " ");
    let mut chars = words.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

/// Where a plugin of `krate` sits in the tree, for its preview: the crate,
/// and the area's title and branch name when it has one.
fn place(krate: &str, plugin: &str) -> String {
    match config::area(plugin) {
        Some(area) => format!("{} · {krate}/{area}", title(area)),
        None => krate.to_owned(),
    }
}

/// Renames the extra `extra` at `at` in `names` while another row has its
/// name: to its type path, then to its crate and type path, which no other
/// row has since an extra's type path is its own and no name but these
/// holds `: `.
fn unique(names: &mut [String], at: usize, extra: &Extra) {
    let candidates = [
        extra.plugin.clone(),
        format!("{}: {}", extra.krate, extra.plugin),
    ];
    for candidate in candidates {
        let Some(name) = names.get(at) else {
            return;
        };
        let taken = (names.iter().enumerate()).any(|(other, it)| other != at && it == name);
        if !taken {
            return;
        }
        if let Some(name) = names.get_mut(at) {
            *name = candidate;
        }
    }
}

/// A plugin's own choice in settings.toml, and the branch of `[plugins]
/// off` that keeps it off, if one does.
type Choice = (bool, Option<String>);

/// Whether this build runs a plugin, and the plugin that added it, if one
/// did.
type Runs<'a> = (bool, Option<&'a str>);

/// The row of rig's plugin `built_in`, its own choice `on` and the branch
/// keeping it off.
fn built_in_row(built_in: &BuiltIn, (on, branch_off): Choice, (runs, by): Runs) -> Row {
    let runs_on = on && branch_off.is_none() || by.is_some();
    let mut state = if built_in.available() {
        state(on, branch_off.as_deref(), (runs, by))
    } else {
        format!("off here: {} only", built_in.targets.join("/"))
    };
    if let Some(what) = built_in.essential.filter(|_| !runs_on) {
        state.push_str(&format!(" [warning: takes away {what}]"));
    }
    let mut about = format!(
        "{}\n{}\n{}\n\n{}\n",
        built_in.name,
        place(built_in.krate, built_in.plugin),
        built_in.plugin,
        states(Some(built_in.on), (on, branch_off.as_deref()), (runs, by)),
    );
    if !built_in.targets.is_empty() {
        about.push_str(&format!(
            "Builds on {} only.\n",
            built_in.targets.join(" and ")
        ));
    }
    about.push_str(&branch(branch_off.as_deref().filter(|_| by.is_none())));
    about.push_str(reload(runs_on, runs));
    match (built_in.essential, runs_on) {
        (Some(what), true) => {
            about.push_str(&format!("Essential: turning it off takes away {what}.\n"));
        }
        (Some(what), false) => {
            about.push_str(&format!("Warning: off, which takes away {what}.\n"));
        }
        (None, _) => {}
    }
    about.push('\n');
    about.push_str(built_in.about);
    Row {
        plugin: built_in.plugin.to_owned(),
        krate: built_in.krate.to_owned(),
        name: built_in.name.to_owned(),
        on,
        default: built_in.on && built_in.available(),
        unbuilt: (!built_in.available()).then_some(built_in.targets),
        state,
        essential: built_in.essential,
        about,
    }
}

/// The name `extra` is shown by before [`unique`]: its crate when it is the
/// crate's only extra, else its crate and type name when no other of the
/// crate's has that type name, else its type path.
fn extra_name(extra: &Extra, extras: &[Extra]) -> String {
    let last = |plugin: &str| plugin.rsplit("::").next().unwrap_or(plugin).to_owned();
    let mut others = (extras.iter())
        .filter(|other| other.krate == extra.krate && other.plugin != extra.plugin)
        .peekable();
    if others.peek().is_none() {
        return extra.krate.clone();
    }
    let name = last(&extra.plugin);
    if others.any(|other| last(&other.plugin) == name) {
        return extra.plugin.clone();
    }
    format!("{} {name}", extra.krate)
}

/// The row of the extra plugin `extra`, shown as `name`, its own choice
/// `on` and the branch keeping it off.
fn extra_row(extra: &Extra, name: String, (on, branch_off): Choice, (runs, by): Runs) -> Row {
    let about = format!(
        "{name}\nYour plugins · {}\n{}\n\n{}\n{}{}\nFrom {}.",
        place(&extra.krate, &extra.plugin),
        extra.plugin,
        states(None, (on, branch_off.as_deref()), (runs, by)),
        branch(branch_off.as_deref().filter(|_| by.is_none())),
        reload(on && branch_off.is_none() || by.is_some(), runs),
        extra.source,
    );
    Row {
        plugin: extra.plugin.clone(),
        krate: extra.krate.clone(),
        name,
        on,
        default: true,
        unbuilt: None,
        state: state(on, branch_off.as_deref(), (runs, by)),
        essential: None,
        about,
    }
}

/// A plugin's short state: on or off in settings.toml, off with the
/// branch `branch_off` whatever its own choice `on`, and whether that
/// waits for the next reload because this build runs it or not. One
/// another plugin added is on whatever settings.toml says.
fn state(on: bool, branch_off: Option<&str>, (runs, by): Runs) -> String {
    if let Some(by) = by {
        return format!("on, added by {by}");
    }
    let word = match branch_off {
        Some(branch) => format!("off with its branch {branch}"),
        None if on => "on".to_owned(),
        None => "off".to_owned(),
    };
    if (on && branch_off.is_none()) == runs {
        word
    } else {
        format!("{word} · next reload")
    }
}

/// What the preview says of the branch `branch_off` that keeps a plugin
/// off, if one does.
fn branch(branch_off: Option<&str>) -> String {
    branch_off.map_or_else(String::new, |branch| {
        format!(
            "Its branch {branch} is off ([plugins] off), which keeps it off whatever its own \
             choice. Enter changes only its own choice; Enter on {branch} turns the branch \
             back on.\n"
        )
    })
}

/// What the next reload changes for a plugin, as a line, or nothing.
fn reload(on: bool, runs: bool) -> &'static str {
    match (on, runs) {
        (true, false) => "Takes effect at the next reload.\n",
        (false, true) => "Runs until the next reload.\n",
        _ => "",
    }
}

/// A plugin's default, when it has one, its own choice in settings.toml
/// and the branch keeping it off, and whether this build runs it and which
/// plugin added it.
fn states(
    default: Option<bool>,
    (on, branch_off): (bool, Option<&str>),
    (runs, by): Runs,
) -> String {
    let word = |on: bool| if on { "on" } else { "off" };
    let mut parts = Vec::new();
    if let Some(default) = default {
        parts.push(format!("default {}", word(default)));
    }
    parts.push(format!("{} in settings", word(on)));
    if let Some(branch) = branch_off {
        parts.push(format!("off with its branch {branch}"));
    }
    parts.push(match by {
        Some(by) => format!("running now, added by {by}"),
        None if runs => "running now".to_owned(),
        None => "not running".to_owned(),
    });
    parts.join(" · ")
}

/// Shows the highlighted row's detail beside each `/plugins` picker.
fn preview(
    pickers: Query<
        (Entity, &PluginsPicker, &Highlighted, Option<&PickPreview>),
        Changed<Highlighted>,
    >,
    mut commands: Commands,
) {
    for (entity, picker, highlighted, shown) in &pickers {
        if highlighted.0 >= picker.tree.nodes.len() {
            continue;
        }
        let preview = PickPreview::Text(picker.tree.about(highlighted.0));
        if shown != Some(&preview) {
            commands.entity(entity).insert(preview);
        }
    }
}

/// What the preview says while no plugin is highlighted.
const NONE_HIGHLIGHTED: &str = "No plugin matches the filter.";

/// Says so in the preview of each `/plugins` picker whose filter left no
/// row to highlight; [`preview`] shows the plugin again once one is.
fn unhighlighted(
    mut removed: RemovedComponents<Highlighted>,
    pickers: Query<(), (With<PluginsPicker>, Without<Highlighted>)>,
    mut commands: Commands,
) {
    for entity in removed.read() {
        if pickers.contains(entity) {
            let none = PickPreview::Text(NONE_HIGHLIGHTED.to_owned());
            commands.entity(entity).insert(none);
        }
    }
}

/// Draws `+ on`, `- off` and the count of changes on each list as the
/// user ticks, and shows or hides the status line's parts ticked at once;
/// a part that cannot be written is ticked back, with an error notice.
fn mark(
    lists: Query<(Entity, &PluginsPicker, &Picker, &Ticked), Changed<Ticked>>,
    #[cfg(feature = "tui")] (home, highlights, mut status, mut notices): (
        Option<Res<RigHome>>,
        Query<&Highlighted>,
        parts::Status,
        MessageWriter<Notice>,
    ),
    mut commands: Commands,
) {
    for (entity, list, shown, ticked) in &lists {
        let on = ticks(list.tree.nodes.len(), &ticked.0);
        #[cfg(feature = "tui")]
        {
            let path = (home.as_ref())
                .map_or_else(Home::from_env, |home| home.0.clone())
                .settings();
            if let Err(why) = status.apply(&path, &list.tree, &on) {
                notices.write(Notice::error(
                    list.agent,
                    format!("{} is unchanged: {why}", path.display()),
                ));
                let on = status.as_running(&list.tree, &on);
                let at =
                    (highlights.get(entity)).map_or(shown.selected, |highlighted| highlighted.0);
                commands.entity(entity).try_despawn();
                self::list(&mut commands, list.tree.clone(), &on, list.agent, at);
                continue;
            }
        }
        let mut picker = shown.clone();
        picker.body = body(&list.tree, &on);
        for (item, detail) in picker.items.iter_mut().zip(details(&list.tree, &on)) {
            item.detail = Some(detail);
        }
        if picker != *shown {
            commands.entity(entity).insert(picker);
        }
    }
}

/// On leaving the list: the status line's parts ticked since the last
/// frame shown or hidden, then the question when what is ticked differs
/// from settings.toml, else nothing. A front that answers with the items
/// picked rather than ticks them gives those; one that closes it without
/// ever saying what is ticked never showed it.
fn leave(
    picked: On<Picked>,
    lists: Query<(
        &PluginsPicker,
        &Picker,
        Option<&Ticked>,
        Option<&Highlighted>,
    )>,
    slash_commands: Query<&Name, With<SlashCommand>>,
    #[cfg(feature = "tui")] (home, mut status, mut notices): (
        Option<Res<RigHome>>,
        parts::Status,
        MessageWriter<Notice>,
    ),
    mut commands: Commands,
) {
    let Ok((list, shown, ticked, highlighted)) = lists.get(picked.entity) else {
        return;
    };
    let count = list.tree.nodes.len();
    let on = match (&picked.chosen, ticked) {
        (Some(chosen), _) => ticks(count, &chosen.items),
        (None, Some(ticked)) => ticks(count, &ticked.0),
        (None, None) => return,
    };
    #[cfg(feature = "tui")]
    {
        let path = (home.as_ref())
            .map_or_else(Home::from_env, |home| home.0.clone())
            .settings();
        if let Err(why) = status.apply(&path, &list.tree, &on) {
            notices.write(Notice::error(
                list.agent,
                format!("{} is unchanged: {why}", path.display()),
            ));
        }
    }
    if list.tree.changed(&on) == 0 {
        return;
    }
    let at = highlighted.map_or(shown.selected, |highlighted| highlighted.0);
    let reload = Reload::here(&slash_commands);
    ask(
        &mut commands,
        (list.tree.clone(), on),
        list.agent,
        at,
        reload,
    );
}

/// The short names of the plugins and the branches among `changes` turned
/// on (`on`) or off, in `tree`. A plugin turned on that a branch keeps off
/// in `after` says so.
fn turned<'a>(
    (tree, after): (&'a Tree, Option<&Plugins>),
    changes: &[(&'a str, bool)],
    on: bool,
) -> (Vec<String>, Vec<&'a str>) {
    let changed = changes.iter().filter(|(_, turned)| *turned == on);
    let name = |plugin: &str| (tree.rows.iter()).find(|row| row.plugin == plugin);
    let plugins = changed.clone().filter_map(|(plugin, _)| name(plugin));
    let branches = changed.filter(|(plugin, _)| name(plugin).is_none());
    let branch = |row: &Row| {
        let after = after.filter(|after| on && !runs(after, &row.plugin))?;
        after.off_with(&row.plugin)
    };
    (
        plugins
            .map(|row| match branch(row) {
                Some(branch) => format!("{} (stays off with its branch {branch})", row.name),
                None => row.name.clone(),
            })
            .collect(),
        branches.map(|(branch, _)| *branch).collect(),
    )
}

/// What `trial` takes away beyond the changes themselves, in `tree` with
/// `on` ticked: the plugins that go off or come on with a branch, the
/// essentials that go with every command that brings each back, and the
/// plugins ticked on that do not build here.
fn sum(tree: &Tree, on: &[bool], trial: &Trial, after: &Plugins) -> Vec<String> {
    let own: Vec<(&Row, bool)> = (tree.nodes.iter().zip(on))
        .filter_map(|((node, _), on)| Some((tree.row(node)?, *on)))
        .collect();
    let mut lines = Vec::new();
    for (word, gone) in [("off", true), ("on", false)] {
        let moving: Vec<&Row> = (own.iter())
            .filter(|(row, now)| row.on == *now)
            .filter(|(row, _)| {
                let (was, will) = (runs(&tree.before, &row.plugin), runs(after, &row.plugin));
                was != will && was == gone
            })
            .map(|(row, _)| *row)
            .collect();
        let branched: Vec<&str> = moving.iter().map(|row| row.name.as_str()).collect();
        if !branched.is_empty() {
            lines.push(format!(
                "With their branch, {word}: {}",
                branched.join(", ")
            ));
        }
    }
    for (row, _) in &own {
        let plugin = row.plugin.as_str();
        let gone = runs(&tree.before, plugin) && !runs(after, plugin);
        if let Some(what) = row.essential.filter(|_| gone) {
            // Every branch that holds it, and its own choice when that is off.
            let own = after.own.iter().any(|on| on == plugin);
            let back: Vec<String> = (after.off.iter())
                .filter(|branch| config::holds(branch, &row.krate, plugin))
                .map(String::as_str)
                .chain((!own).then_some(plugin))
                .map(|back| format!("  rig plugin enable {back}"))
                .collect();
            let these = if back.len() == 1 {
                "this brings"
            } else {
                "these bring"
            };
            lines.push(format!(
                "\nTurning off {} takes away {what}. In a shell, {these} it back:\n{}",
                row.name,
                back.join("\n")
            ));
        }
    }
    for row in &trial.unbuilt {
        let targets = row.unbuilt.unwrap_or_default().join(" and ");
        lines.push(format!(
            "\n{} stays off: it builds on {targets} only.",
            row.name
        ));
    }
    lines
}

/// Asks whether to reload now with what `on` changes in `tree`, at the
/// next start, to go back to the list at node `at`, or to leave without
/// writing. It offers to reload now only where `reload` says one can run,
/// and else says why; changes that leave the file unreadable offer only to
/// keep editing or leave.
fn ask(
    commands: &mut Commands,
    (tree, on): (Tree, Vec<bool>),
    agent: Entity,
    at: usize,
    reload: Reload,
) {
    let trial = tree.trial(&on);
    let mut lines = Vec::new();
    for (word, want) in [("on", true), ("off", false)] {
        let (plugins, branches) = turned((&tree, trial.after()), &trial.changes, want);
        if !plugins.is_empty() {
            lines.push(format!("Turn {word}: {}", plugins.join(", ")));
        }
        if !branches.is_empty() {
            let back = if want { "back on" } else { "off" };
            lines.push(format!("Turn {back} the branches: {}", branches.join(", ")));
        }
    }
    let answers: Vec<Leave> = match (&trial.after, reload) {
        (Err(why), _) => {
            lines.push(format!(
                "\nWith these changes, settings.toml would not read: {why}"
            ));
            vec![Leave::Edit, Leave::Discard]
        }
        (Ok(after), reload) => {
            lines.extend(sum(&tree, &on, &trial, after));
            if let Some(why) = reload.why_not() {
                lines.push(format!("\n{why}"));
            }
            match reload {
                Reload::Can => vec![Leave::Reload, Leave::Later, Leave::Edit, Leave::Discard],
                Reload::NoPlugin | Reload::NoLauncher => {
                    vec![Leave::Later, Leave::Edit, Leave::Discard]
                }
            }
        }
    };
    lines.push("Esc goes back to the list.".to_owned());
    let title = match tree.changed(&on) {
        1 => "1 plugin change".to_owned(),
        count => format!("{count} plugin changes"),
    };
    let mut picker = Picker::new(title).for_agent(agent).body(lines.join("\n"));
    picker.items = (answers.iter())
        .map(|answer| PickItem::new(answer.label(), ""))
        .collect();
    drop(trial);
    let question = LeaveQuestion {
        tree,
        on,
        agent,
        at,
        reload,
        answers,
    };
    commands.spawn((picker, question)).observe(answer);
}

/// Carries out the answer to the question: writes the changes to
/// settings.toml in one edit, and reloads for [`Leave::Reload`]; back to
/// the list for [`Leave::Edit`] or Esc. Closed with no choice where no
/// one sits at a terminal, no front showed it: nothing happens, so a
/// refused question never reopens the list.
fn answer(
    picked: On<Picked>,
    questions: Query<&LeaveQuestion>,
    invoked: Option<Res<Invoked>>,
    home: Option<Res<RigHome>>,
    status: Option<ResMut<ReloadStatus>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(question) = questions.get(picked.entity) else {
        return;
    };
    let leave = match &picked.chosen {
        Some(chosen) => chosen
            .items
            .first()
            .and_then(|index| question.answers.get(*index)),
        None if invoked.is_some_and(|invoked| invoked.interactive()) => Some(&Leave::Edit),
        None => None,
    };
    let Some(&leave) = leave else {
        return;
    };
    let LeaveQuestion {
        tree,
        on,
        agent,
        at,
        reload,
        ..
    } = question;
    match leave {
        Leave::Edit => {
            list(&mut commands, tree.clone(), on, *agent, *at);
            return;
        }
        Leave::Discard => return,
        Leave::Reload | Leave::Later => {}
    }
    let trial = tree.trial(on);
    let changes = &trial.changes;
    let path = home
        .map_or_else(Home::from_env, |home| home.0.clone())
        .settings();
    let edited = config::edit(&path, |file| {
        config::turn(file, changes).map(|changed| (changed, ()))
    });
    if let Err(why) = edited {
        notices.write(Notice::error(
            *agent,
            format!("{} is unchanged: {why}", path.display()),
        ));
        return;
    }
    let mut said = Vec::new();
    for (word, want) in [("on", true), ("off", false)] {
        let (plugins, branches) = turned((tree, trial.after()), changes, want);
        if !plugins.is_empty() {
            said.push(format!("turned {word} {}", plugins.join(", ")));
        }
        if !branches.is_empty() {
            let back = if want { "back on" } else { "off" };
            said.push(format!(
                "turned {back} the branches {}",
                branches.join(", ")
            ));
        }
    }
    let then = match (leave, status) {
        (Leave::Reload, Some(mut status)) => match status.ask(*agent, Rebuild::Build) {
            Ok(()) => "Reloading once no turn runs.",
            Err(why) => why,
        },
        _ => reload.later(),
    };
    notices.write(Notice::info(
        *agent,
        format!("{}: {}. {then}", path.display(), said.join("; ")),
    ));
}

#[cfg(test)]
mod tests;
