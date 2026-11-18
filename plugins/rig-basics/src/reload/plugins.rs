//! `/plugins`: rig's plugins and the extras of `RIG_HOME/settings.toml`,
//! read from that file in this process, each with whether it is on and
//! whether this build runs it (its `PluginSource` entity). In the terminal
//! view it is a picker of short names in the plugin tree's order (crate,
//! then area) with each plugin's own choice ticked, and the highlighted
//! plugin's detail beside it ([`PickPreview`]). A plugin a branch of
//! `[plugins] off` holds says it is off with its branch. Enter ticks and
//! unticks that plugin's own choice; each row that differs from the file
//! shows `+ on` or `- off`. Leaving with changes asks whether to reload now
//! ([`ReloadStatus::ask`]), at the next start, or to keep editing; either
//! of the first two writes every change in one edit of the file
//! ([`config::turn`]). `/plugins defaults` asks the same about going back to
//! rig's defaults, branches turned back on. Without a terminal it lists
//! them.
//!
//! `rig plugin enable` and `rig plugin disable` change the same file from a
//! shell, with the same rules.

use rig_harness::harness_protocol::config::{self, BUILT_IN, BuiltIn, Extra, Plugins};
use rig_harness::harness_protocol::{self, Home};
use rig_harness::pick::{Highlighted, PickPreview, Picked, Picker, Ticked};
use rig_harness::prelude::*;
use rig_harness::services::RigHome;

use crate::reload::{Rebuild, ReloadStatus};

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
    /// What the list shows: a built-in's short name, an extra's crate.
    name: String,
    /// Whether its own choice in settings.toml is on.
    on: bool,
    /// The branch of `[plugins] off` that keeps it off, if one does.
    branch_off: Option<String>,
    /// Whether rig's defaults turn it on here: an extra always.
    default: bool,
    /// Its state, short: `on`, `off`, or why it is off.
    state: String,
    /// What goes when it is off, for an essential plugin.
    essential: Option<&'static str>,
    /// Everything about it, for the preview.
    about: String,
}

/// The `/plugins` list: its rows, in its order, against settings.toml, and
/// the agent it is for.
#[derive(Component)]
struct PluginsPicker {
    rows: Vec<Row>,
    agent: Entity,
}

/// The question asked on leaving the list with changes: the list's rows,
/// which are ticked, the row highlighted, whether a reload can run and the
/// answers offered.
#[derive(Component)]
struct LeaveQuestion {
    rows: Vec<Row>,
    on: Vec<bool>,
    /// The branches of `[plugins] off` to turn back on.
    branches: Vec<String>,
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
}

impl Leave {
    /// Its label in the question.
    fn label(self) -> &'static str {
        match self {
            Leave::Reload => "Reload now",
            Leave::Later => "Later, at the next start",
            Leave::Edit => "Keep editing",
        }
    }
}

/// `/plugins`: the picker on an interactive terminal, else the list;
/// `/plugins defaults`: the question about going back to rig's defaults.
fn plugins(
    In(args): In<CommandArgs>,
    home: Option<Res<RigHome>>,
    invoked: Option<Res<Invoked>>,
    running: Query<&Name, With<PluginSource>>,
    reloads: Option<Res<ReloadStatus>>,
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
    let chosen = match config::chosen(&home) {
        Ok(chosen) => chosen,
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
    let running: Vec<String> = running.iter().map(Name::to_string).collect();
    let rows = rows(&chosen, &running);
    let interactive = invoked.is_some_and(|invoked| invoked.interactive());
    if defaults {
        let on: Vec<bool> = rows.iter().map(|row| row.default).collect();
        let changed = changed(&rows, &on);
        if changed.is_empty() && chosen.off.is_empty() {
            notices.write(Notice::info(
                args.agent,
                format!("{} already runs rig's defaults.", home.settings().display()),
            ));
            return;
        }
        // The question needs a terminal to show it.
        if !interactive {
            let commands: Vec<String> = (changed.iter())
                .map(|(row, on)| {
                    let verb = if *on { "enable" } else { "disable" };
                    format!("  rig plugin {verb} {}", row.plugin)
                })
                .chain((chosen.off.iter()).map(|branch| format!("  rig plugin enable {branch}")))
                .collect();
            notices.write(Notice::error(
                args.agent,
                format!(
                    "/plugins defaults asks before it writes, which needs rig on a terminal. \
                     In a shell, these go back to rig's defaults:\n{}",
                    commands.join("\n")
                ),
            ));
            return;
        }
        let reload = Reload::here(reloads.is_some());
        let question = (rows, on, chosen.off);
        ask(&mut commands, question, args.agent, 0, reload);
        return;
    }
    if !interactive {
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
    let on: Vec<bool> = rows.iter().map(|row| row.on).collect();
    list(&mut commands, rows, &on, args.agent, 0);
}

/// Opens the list of `rows` for `agent` with `on` ticked and row `at`
/// selected.
fn list(commands: &mut Commands, rows: Vec<Row>, on: &[bool], agent: Entity, at: usize) {
    let mut picker = Picker::new("Plugins")
        .for_agent(agent)
        .body(body(&rows, on))
        .multiple();
    picker.selected = at;
    picker.items = (rows.iter().zip(on))
        .map(|(row, on)| PickItem {
            checked: *on,
            // Typing a type path finds its plugin; → shows it with the whole state.
            more: Some(format!("{}: {}", row.plugin, row.state)),
            ..PickItem::new(&row.name, "").detail(detail(row, *on))
        })
        .collect();
    // The terminal view gives the preview its half of the screen from the
    // first frame only when the picker comes with one.
    let shown = rows
        .get(at)
        .map(|row| row.about.clone())
        .unwrap_or_default();
    let mut list = commands.spawn((
        picker,
        PickPreview::Text(shown),
        PluginsPicker { rows, agent },
    ));
    #[cfg(feature = "tui")]
    list.insert(rig_tui::ChoiceKeys(vec![(
        rig_tui::crossterm::event::KeyCode::Enter,
        rig_tui::ChoiceAction::Tick,
    )]));
    list.observe(leave);
}

/// The list's body: its keys and how many rows `on` changes.
fn body(rows: &[Row], on: &[bool]) -> String {
    let keys = "Enter ticks or unticks, Esc leaves, type to filter.";
    match changed(rows, on).len() {
        0 => format!("{keys} Nothing is written until you leave."),
        1 => format!("{keys} 1 change: leaving asks whether to reload now or later."),
        changes => {
            format!("{keys} {changes} changes: leaving asks whether to reload now or later.")
        }
    }
}

/// The rows that `on` turns on or off, each with whether it turns it on.
fn changed<'a>(rows: &'a [Row], on: &[bool]) -> Vec<(&'a Row, bool)> {
    (rows.iter().zip(on.iter().copied()))
        .filter(|(row, on)| row.on != *on)
        .collect()
}

/// Whether "Reload now" can run here, else why not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reload {
    /// `ReloadPlugin` is on and the launcher started rig.
    Can,
    /// `ReloadPlugin` is off.
    NoPlugin,
    /// The launcher did not start rig.
    NoLauncher,
}

impl Reload {
    /// Whether a reload can run, with `ReloadPlugin` on or not (`plugin`).
    fn here(plugin: bool) -> Self {
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

/// What the list shows beside `row` while `on` says whether it is ticked:
/// its state, or how ticking changes it.
fn detail(row: &Row, on: bool) -> String {
    match (row.on, on, row.essential) {
        (false, true, _) => match &row.branch_off {
            Some(branch) => format!("+ on, still off with its branch {branch}"),
            None => "+ on".to_owned(),
        },
        (true, false, Some(what)) => format!("- off, takes away {what}"),
        (true, false, None) => "- off".to_owned(),
        _ => row.state.clone(),
    }
}

/// Which of `count` rows `positions` ticks.
fn ticks(count: usize, positions: &[usize]) -> Vec<bool> {
    (0..count).map(|index| positions.contains(&index)).collect()
}

/// Every plugin, rig's in the tree's order ([`tree_order`]) and then the
/// extras, with their own choice in `chosen`, the branch keeping each off
/// and whether `running` holds each.
fn rows(chosen: &Plugins, running: &[String]) -> Vec<Row> {
    let mut built_ins: Vec<&BuiltIn> = BUILT_IN.iter().collect();
    tree_order(&mut built_ins);
    let on = |plugin: &str| chosen.own.iter().any(|on| on == plugin);
    let off = |plugin: &str| chosen.off_with(plugin).map(str::to_owned);
    let runs = |plugin: &str| running.iter().any(|runs| runs == plugin);
    let built_ins = (built_ins.into_iter()).map(|built_in| {
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
    built_ins.chain(extras).collect()
}

/// `built_ins` in the plugin tree's order: each crate where its first
/// plugin is, in it each area where its first plugin is, and the plugins of
/// one area in their order.
fn tree_order(built_ins: &mut Vec<&BuiltIn>) {
    let first = |held: &dyn Fn(&BuiltIn) -> bool| built_ins.iter().position(|it| held(it));
    let keys: Vec<(Option<usize>, Option<usize>)> = (built_ins.iter().enumerate())
        .map(|(at, built_in)| {
            let krate = first(&|other| other.krate == built_in.krate);
            let area = config::area(built_in.plugin).and_then(|area| {
                first(&|other| {
                    other.krate == built_in.krate && config::area(other.plugin) == Some(area)
                })
            });
            (krate, area.or(Some(at)))
        })
        .collect();
    let mut keyed: Vec<_> = keys.into_iter().zip(built_ins.drain(..)).collect();
    keyed.sort_by_key(|(key, _)| *key);
    built_ins.extend(keyed.into_iter().map(|(_, built_in)| built_in));
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

/// The row of rig's plugin `built_in`, its own choice `on` and the branch
/// keeping it off.
fn built_in_row(built_in: &BuiltIn, (on, branch_off): Choice, runs: bool) -> Row {
    let runs_on = on && branch_off.is_none();
    let mut state = if built_in.available() {
        state(on, branch_off.as_deref(), runs)
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
        states(Some(built_in.on), (on, branch_off.as_deref()), runs),
    );
    if !built_in.targets.is_empty() {
        about.push_str(&format!(
            "Builds on {} only.\n",
            built_in.targets.join(" and ")
        ));
    }
    about.push_str(&branch(branch_off.as_deref()));
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
        name: built_in.name.to_owned(),
        on,
        branch_off,
        default: built_in.on && built_in.available(),
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
fn extra_row(extra: &Extra, name: String, (on, branch_off): Choice, runs: bool) -> Row {
    let about = format!(
        "{name}\nYour plugins · {}\n{}\n\n{}\n{}{}\nFrom {}.",
        place(&extra.krate, &extra.plugin),
        extra.plugin,
        states(None, (on, branch_off.as_deref()), runs),
        branch(branch_off.as_deref()),
        reload(on && branch_off.is_none(), runs),
        extra.source,
    );
    Row {
        plugin: extra.plugin.clone(),
        name,
        on,
        default: true,
        state: state(on, branch_off.as_deref(), runs),
        branch_off,
        essential: None,
        about,
    }
}

/// A plugin's short state: on or off in settings.toml, off with the
/// branch `branch_off` despite its own choice `on`, and whether that waits
/// for the next reload because this build runs it or not.
fn state(on: bool, branch_off: Option<&str>, runs: bool) -> String {
    let word = match branch_off.filter(|_| on) {
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
             choice. Enter changes only its own choice; `rig plugin enable {branch}` turns the \
             branch back on.\n"
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
/// and the branch keeping it off, and whether this build runs it.
fn states(default: Option<bool>, (on, branch_off): (bool, Option<&str>), runs: bool) -> String {
    let word = |on: bool| if on { "on" } else { "off" };
    let mut parts = Vec::new();
    if let Some(default) = default {
        parts.push(format!("default {}", word(default)));
    }
    parts.push(format!("{} in settings", word(on)));
    if let Some(branch) = branch_off {
        parts.push(format!("off with its branch {branch}"));
    }
    parts.push(if runs { "running now" } else { "not running" }.to_owned());
    parts.join(" · ")
}

/// Shows the highlighted plugin's detail beside each `/plugins` picker.
fn preview(
    pickers: Query<
        (Entity, &PluginsPicker, &Highlighted, Option<&PickPreview>),
        Changed<Highlighted>,
    >,
    mut commands: Commands,
) {
    for (entity, picker, highlighted, shown) in &pickers {
        let Some(row) = picker.rows.get(highlighted.0) else {
            continue;
        };
        let preview = PickPreview::Text(row.about.clone());
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
/// user ticks.
fn mark(
    lists: Query<(Entity, &PluginsPicker, &Picker, &Ticked), Changed<Ticked>>,
    mut commands: Commands,
) {
    for (entity, list, shown, ticked) in &lists {
        let on = ticks(list.rows.len(), &ticked.0);
        let mut picker = shown.clone();
        picker.body = body(&list.rows, &on);
        for (item, (row, on)) in picker.items.iter_mut().zip(list.rows.iter().zip(&on)) {
            item.detail = Some(detail(row, *on));
        }
        if picker != *shown {
            commands.entity(entity).insert(picker);
        }
    }
}

/// On leaving the list: the question when what is ticked differs from
/// settings.toml, else nothing. A front that answers with the items picked
/// rather than ticks them gives those; one that closes it without ever
/// saying what is ticked never showed it.
fn leave(
    picked: On<Picked>,
    lists: Query<(
        &PluginsPicker,
        &Picker,
        Option<&Ticked>,
        Option<&Highlighted>,
    )>,
    reloads: Option<Res<ReloadStatus>>,
    mut commands: Commands,
) {
    let Ok((list, shown, ticked, highlighted)) = lists.get(picked.entity) else {
        return;
    };
    let count = list.rows.len();
    let on = match (&picked.chosen, ticked) {
        (Some(chosen), _) => ticks(count, &chosen.items),
        (None, Some(ticked)) => ticks(count, &ticked.0),
        (None, None) => return,
    };
    if changed(&list.rows, &on).is_empty() {
        return;
    }
    let at = highlighted.map_or(shown.selected, |highlighted| highlighted.0);
    let reload = Reload::here(reloads.is_some());
    let question = (list.rows.clone(), on, Vec::new());
    ask(&mut commands, question, list.agent, at, reload);
}

/// Asks whether to reload now with what `on` changes in `rows` and the
/// `branches` turned back on, at the next start, or to go back to the list
/// at row `at`. It offers to reload now only where `reload` says one can
/// run, and else says why.
fn ask(
    commands: &mut Commands,
    (rows, on, branches): (Vec<Row>, Vec<bool>, Vec<String>),
    agent: Entity,
    at: usize,
    reload: Reload,
) {
    let changed = changed(&rows, &on);
    let mut lines = Vec::new();
    for (word, want) in [("on", true), ("off", false)] {
        let names: Vec<&str> = (changed.iter())
            .filter(|(_, on)| *on == want)
            .map(|(row, _)| row.name.as_str())
            .collect();
        if !names.is_empty() {
            lines.push(format!("Turn {word}: {}", names.join(", ")));
        }
    }
    if !branches.is_empty() {
        lines.push(format!(
            "Turn back on the branches: {}",
            branches.join(", ")
        ));
    }
    for (row, _) in changed.iter().filter(|(_, on)| !on) {
        if let Some(what) = row.essential {
            lines.push(format!(
                "\nTurning off {} takes away {what}. In a shell, this brings it back:\n  \
                 rig plugin enable {}",
                row.name, row.plugin
            ));
        }
    }
    if let Some(why) = reload.why_not() {
        lines.push(format!("\n{why}"));
    }
    lines.push("Esc goes back to the list.".to_owned());
    let answers: Vec<Leave> = match reload {
        Reload::Can => vec![Leave::Reload, Leave::Later, Leave::Edit],
        Reload::NoPlugin | Reload::NoLauncher => vec![Leave::Later, Leave::Edit],
    };
    let title = match changed.len() + branches.len() {
        1 => "1 plugin change".to_owned(),
        count => format!("{count} plugin changes"),
    };
    let mut picker = Picker::new(title).for_agent(agent).body(lines.join("\n"));
    picker.items = (answers.iter())
        .map(|answer| PickItem::new(answer.label(), ""))
        .collect();
    let question = LeaveQuestion {
        rows,
        on,
        branches,
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
        rows,
        on,
        branches,
        agent,
        at,
        reload,
        ..
    } = question;
    if leave == Leave::Edit {
        list(&mut commands, rows.clone(), on, *agent, *at);
        return;
    }
    let changed = changed(rows, on);
    let path = home
        .map_or_else(Home::from_env, |home| home.0.clone())
        .settings();
    let changes: Vec<(&str, bool)> = (changed.iter())
        .map(|(row, on)| (row.plugin.as_str(), *on))
        .chain(branches.iter().map(|branch| (branch.as_str(), true)))
        .collect();
    let edited = config::edit(&path, |file| {
        config::turn(file, &changes).map(|changed| (changed, ()))
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
        let names: Vec<&str> = (changed.iter())
            .filter(|(_, on)| *on == want)
            .map(|(row, _)| row.name.as_str())
            .collect();
        if !names.is_empty() {
            said.push(format!("turned {word} {}", names.join(", ")));
        }
    }
    if !branches.is_empty() {
        said.push(format!(
            "turned back on the branches {}",
            branches.join(", ")
        ));
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
