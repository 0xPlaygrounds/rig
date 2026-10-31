//! Output styles: how an agent talks, switched with one command and kept
//! with the session.
//!
//! A style is a Markdown file in `RIG_HOME/styles/` or the project's
//! `.rig/styles/` (the project's wins a name clash), named after the file,
//! besides the built-in `default`, `terse` and `teaching`. Its front matter
//! says what it is for and whether subagents get it too:
//!
//! ```text
//! ---
//! description: Leave small exercises for me
//! subagents: false
//! ---
//! After each change, leave the user one small exercise about it.
//! ```
//!
//! `/style <name>` sets the style of the agent it is typed for, and
//! `/style` alone opens a picker of them. The agent's style is an
//! [`OutputStyle`] saved with the session, so a resume restores it.
//! A style other than `default` is one prompt section of that agent alone
//! (`<output_style>`), so it is in each of its requests once and edits no
//! message. A subagent gets its parent's style only when the style says
//! `subagents: true`. Both directories are watched: a style edited counts
//! from the next request, and an agent whose style file went is set back to
//! `default` with a notice.

use std::path::PathBuf;

use rig_harness::prelude::*;
use rig_harness::rig_tools::workspace::{Listing as Files, Workspace};
use rig_harness::watch::WatchPlugin;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The style that adds nothing.
pub const DEFAULT: &str = "default";

/// The tag of the prompt section a style's text goes in.
pub const SECTION: &str = "output_style";

/// Where the style's section goes: after the project's instructions,
/// before the environment, which changes more often.
const ORDER: i32 = 250;

/// Output styles: `/style`, the styles' prompt sections and, with the
/// `tui` feature, the `style: <name>` status item.
#[derive(Default)]
pub struct OutputStylesPlugin;

impl Plugin for OutputStylesPlugin {
    fn build(&self, app: &mut App) {
        let _ = app
            .register_type::<OutputStyle>()
            .keeps_saved::<OutputStyle>()
            .saved_as::<OutputStyle>("output-styles.style");
        if !app.is_plugin_added::<WatchPlugin>() {
            app.add_plugins(WatchPlugin);
        }
        app.init_resource::<Styles>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<ProjectRoot>()
            .init_resource::<RigHome>()
            .add_systems(Startup, (read_now, watch))
            .add_systems(Update, (reload, restyle).chain().before(AgentLoop))
            .add_observer(styled)
            .add_observer(unstyled)
            .add_observer(read)
            .add_observer(inherit)
            .add_command(
                "style",
                "Set how the agent talks: /style <name>, or /style to pick",
                style,
            );
        #[cfg(feature = "tui")]
        tui::build(app);
    }
}

/// On an agent: the name of its output style, saved with the session; an
/// agent without one talks as `default`.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct OutputStyle(pub String);

/// One style.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Style {
    /// Its name, which `/style` takes.
    pub name: String,
    /// What it is for, as the picker shows it.
    pub description: String,
    /// The text added to the system prompt; empty for `default`.
    pub body: String,
    /// Whether the subagents of an agent with this style get it too.
    pub subagents: bool,
}

/// The styles there are: the built-in ones, then the files' by name.
#[derive(Resource, Clone, Debug)]
pub struct Styles {
    /// Every style, in the order the picker lists them.
    pub styles: Vec<Style>,
    /// Whether the style directories were read: until then a style no
    /// file gives is not taken as gone.
    pub read: bool,
}

impl Default for Styles {
    fn default() -> Self {
        Self {
            styles: built_in(),
            read: false,
        }
    }
}

impl Styles {
    /// The style named `name`.
    pub fn get(&self, name: &str) -> Option<&Style> {
        self.styles.iter().find(|style| style.name == name)
    }

    /// Every style's name, in order, comma separated.
    pub fn names(&self) -> String {
        let names: Vec<&str> = self
            .styles
            .iter()
            .map(|style| style.name.as_str())
            .collect();
        names.join(", ")
    }
}

/// `default`, `terse` and `teaching`.
fn built_in() -> Vec<Style> {
    let style = |name: &str, description: &str, body: &str| Style {
        name: name.to_owned(),
        description: description.to_owned(),
        body: body.to_owned(),
        subagents: false,
    };
    vec![
        style(DEFAULT, "The agent's own way of talking", ""),
        style(
            "terse",
            "Just the change and a one-line summary",
            "Answer as briefly as you can. Make the change, then say what you did in one line. \
             No preamble, no walkthrough, no restating the request; explain only when asked.",
        ),
        style(
            "teaching",
            "Explains each step as if teaching a junior",
            "Work as if teaching a junior developer who watches. Before each change, say what \
             you are about to do and why it is needed; name the idea or technique behind it and \
             the trade-off it makes. After the work, sum up what someone should learn from it.",
        ),
    ]
}

/// A style file's front matter.
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct FrontMatter {
    description: Option<String>,
    subagents: bool,
}

/// The style `name` in `text`, or why it cannot be read.
fn parse(name: &str, text: &str) -> Result<Style, String> {
    let (front, body) = match text.strip_prefix("---\n") {
        Some(rest) => rest
            .split_once("\n---")
            .map(|(front, body)| (front, body.strip_prefix('\n').unwrap_or(body)))
            .ok_or("its front matter has no closing `---`")?,
        None => ("", text),
    };
    let front: FrontMatter = if front.trim().is_empty() {
        FrontMatter::default()
    } else {
        serde_yaml::from_str(front).map_err(|why| format!("its front matter: {why}"))?
    };
    Ok(Style {
        name: name.to_owned(),
        description: front.description.unwrap_or_default(),
        body: body.trim().to_owned(),
        subagents: front.subagents,
    })
}

/// The style directories, the user's first: a later one wins a clash.
fn dirs(home: &RigHome, root: &ProjectRoot) -> [PathBuf; 2] {
    [
        home.0.root().join("styles"),
        root.0.join(".rig").join("styles"),
    ]
}

/// The built-in styles and those of the files right in `dirs`, and why
/// any file was skipped.
fn read_styles(workspace: &dyn Workspace, dirs: &[PathBuf]) -> (Styles, Vec<String>) {
    let mut styles = built_in();
    let mut skipped = Vec::new();
    for dir in dirs {
        let mut files = workspace
            .files(dir, Some("*.md"), Files::All)
            .map(|listed| listed.files)
            .unwrap_or_default();
        files.retain(|path| path.parent() == Some(dir.as_path()));
        files.sort();
        for path in files {
            let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if name == DEFAULT || name.is_empty() || name.contains(char::is_whitespace) {
                skipped.push(format!(
                    "{}: `{name}` cannot be a style's name",
                    path.display()
                ));
                continue;
            }
            let text = workspace.read_text(&path).map_err(|why| why.to_string());
            match text.and_then(|text| parse(name, &text)) {
                Ok(style) => match styles.iter_mut().find(|old| old.name == style.name) {
                    Some(old) => *old = style,
                    None => styles.push(style),
                },
                Err(why) => skipped.push(format!("{}: {why}", path.display())),
            }
        }
    }
    (Styles { styles, read: true }, skipped)
}

/// On the watches of the style directories.
#[derive(Component)]
struct StyleWatch;

/// On the entity that reads the styles off the main thread.
#[derive(Component)]
struct Reading;

/// On the prompt section that holds an agent's style.
#[derive(Component)]
struct StyleSection;

/// Watches the style directories; each change reads them again.
fn watch(home: Res<RigHome>, root: Res<ProjectRoot>, mut commands: Commands) {
    for dir in dirs(&home, &root) {
        commands.spawn((StyleWatch, FileWatch::new(dir).listing(Files::All)));
    }
}

/// Reads the styles again when a style file changed.
fn reload(
    mut changes: MessageReader<FileChanges>,
    watches: Query<(), With<StyleWatch>>,
    mut commands: Commands,
) {
    let mine = changes
        .read()
        .fold(false, |mine, change| mine | watches.contains(change.watch));
    if mine {
        commands.queue(read_now);
    }
}

/// Reads the styles: at once on this machine, so before any request is
/// built, else off the main thread, so no frame waits on the network.
fn read_now(world: &mut World) {
    let workspace = world.get_resource::<ProjectWorkspace>().cloned();
    let workspace = workspace.unwrap_or_default().0;
    let home = world.get_resource::<RigHome>().cloned().unwrap_or_default();
    let root = world.get_resource::<ProjectRoot>().cloned();
    let dirs = dirs(&home, &root.unwrap_or_default());
    if workspace.host().is_none() {
        let (styles, skipped) = read_styles(&*workspace, &dirs);
        world.insert_resource(styles);
        for why in skipped {
            world.write_message(Notice::warn(None, format!("Style skipped: {why}")));
        }
        return;
    }
    let wake = world.get_resource::<Wake>().cloned().unwrap_or_default();
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let work = blocking(move || Ok(read_styles(&*workspace, &dirs)));
    let running = Running::spawn(pool, &wake, async move { work.await.ok() });
    world.spawn((Reading, running));
}

/// The styles a read off the main thread found.
fn read(
    done: On<Insert<Done<Option<(Styles, Vec<String>)>>>>,
    reads: Query<&Done<Option<(Styles, Vec<String>)>>, With<Reading>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok(Done(found)) = reads.get(done.entity) else {
        return;
    };
    if let Some((styles, skipped)) = found.clone() {
        commands.insert_resource(styles);
        for why in skipped {
            notices.write(Notice::warn(None, format!("Style skipped: {why}")));
        }
    }
    commands.entity(done.entity).despawn();
}

/// Brings `agent`'s style section in line with its style, made if it has
/// none, as one command, so two changes in a row never make two; a style
/// the read styles no longer have sets it back to `default`, with a
/// notice.
fn sync(agent: Entity) -> impl FnOnce(&mut World) {
    move |world: &mut World| {
        let Some(styles) = world.get_resource::<Styles>() else {
            return;
        };
        let name = world.get::<OutputStyle>(agent).map(|style| style.0.clone());
        let text = match (&name, name.as_deref().map(|name| styles.get(name))) {
            (None, _) => String::new(),
            (Some(_), Some(Some(style))) => style.body.clone(),
            (Some(_), _) if !styles.read => String::new(),
            (Some(name), _) => {
                let why = format!("The style `{name}` is gone; back to `{DEFAULT}`.");
                world.write_message(Notice::warn(agent, why));
                if let Ok(mut agent) = world.get_entity_mut(agent) {
                    agent.insert(OutputStyle(DEFAULT.to_owned()));
                }
                return;
            }
        };
        let mut sections = world.query_filtered::<(Entity, &SectionOf), With<StyleSection>>();
        let found = sections.iter(world).find(|(_, of)| of.0 == agent);
        match found.map(|(section, _)| section) {
            Some(section) => {
                if let Some(mut section) = world.get_mut::<PromptSection>(section)
                    && section.text != text
                {
                    section.text = text;
                }
            }
            None if !text.is_empty() && world.get_entity(agent).is_ok() => {
                let section = PromptSection::new(ORDER, SECTION, text);
                world.spawn((Name::new(SECTION), StyleSection, section, SectionOf(agent)));
            }
            None => {}
        }
    }
}

/// An agent's style set or restored: its section follows.
fn styled(inserted: On<Insert<OutputStyle>>, mut commands: Commands) {
    commands.queue(sync(inserted.entity));
}

/// An agent's style taken out, or the agent gone: its section is emptied
/// once the component is.
fn unstyled(removed: On<Remove<OutputStyle>>, mut commands: Commands) {
    commands.queue(sync(removed.entity));
}

/// The styles read again: every styled agent's section follows.
fn restyle(styles: Res<Styles>, agents: Query<Entity, With<OutputStyle>>, mut commands: Commands) {
    if styles.is_changed() {
        for agent in &agents {
            commands.queue(sync(agent));
        }
    }
}

/// A subagent gets its parent's style when the style says `subagents:
/// true`, unless it has one of its own.
fn inherit(
    spawned: On<Add<SpawnedBy>>,
    children: Query<&SpawnedBy>,
    agents: Query<&OutputStyle>,
    styles: Res<Styles>,
    mut commands: Commands,
) {
    let Ok(&SpawnedBy(parent)) = children.get(spawned.entity) else {
        return;
    };
    let Ok(OutputStyle(name)) = agents.get(parent) else {
        return;
    };
    if styles.get(name).is_some_and(|style| style.subagents) {
        let style = OutputStyle(name.clone());
        commands.entity(spawned.entity).insert_if_new(style);
    }
}

/// `/style <name>` sets the agent's style; `/style` opens a picker.
fn style(
    In(args): In<CommandArgs>,
    agents: Query<Option<&OutputStyle>, With<Agent>>,
    styles: Res<Styles>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent = args.agent;
    let Ok(current) = agents.get(agent) else {
        notices.write(Notice::error(agent, "/style: no such agent"));
        return;
    };
    let current = current.map_or(DEFAULT, |OutputStyle(name)| name.as_str());
    let name = args.args.trim();
    if name.is_empty() {
        commands.spawn(picker(&styles, agent, current));
        return;
    }
    if styles.get(name).is_none() {
        let why = format!("Unknown style `{name}`. Styles: {}.", styles.names());
        notices.write(Notice::error(agent, why));
        return;
    }
    commands.entity(agent).insert(OutputStyle(name.to_owned()));
    notices.write(Notice::info(agent, format!("Style: {name}.")));
}

/// The styles to pick from, `current` marked and selected.
fn picker(styles: &Styles, agent: Entity, current: &str) -> Picker {
    let mut picker = Picker::new("Output style").for_agent(agent);
    for (at, style) in styles.styles.iter().enumerate() {
        let mark = if style.name == current { "● " } else { "  " };
        picker.items.push(PickItem {
            label: format!("{mark}{}", style.name),
            command: format!("style {}", style.name),
            detail: Some(style.description.clone()).filter(|text| !text.is_empty()),
            more: None,
            checked: false,
        });
        if style.name == current {
            picker.selected = at;
        }
    }
    picker
}

#[cfg(feature = "tui")]
mod tui {
    //! The `style: <name>` status item.

    use rig_harness::prelude::*;
    use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

    use super::{DEFAULT, OutputStyle};

    /// After the reasoning setting; among the first to go.
    const STYLE: StatusItem = StatusItem::new("style", Side::Left, 45, 9);

    pub(super) fn build(app: &mut App) {
        app.add_systems(PostUpdate, show.in_set(StatusSystems::Show));
    }

    /// `style: <name>` for an agent whose style is not `default`.
    fn show(
        styled: Query<(Entity, &OutputStyle), Changed<OutputStyle>>,
        mut removed: RemovedComponents<OutputStyle>,
        mut items: Query<&mut StatusItems>,
    ) {
        for (agent, OutputStyle(name)) in &styled {
            let Ok(mut shown) = items.get_mut(agent) else {
                continue;
            };
            match name.as_str() {
                DEFAULT => shown.hide(STYLE.name),
                name => shown.show(STYLE.says(format!("style: {name}"), Tone::Plain)),
            }
        }
        for agent in removed.read() {
            if let Ok(mut shown) = items.get_mut(agent) {
                shown.hide(STYLE.name);
            }
        }
    }
}
