//! The Bevy UI window that draws the model: each part is drawn again only
//! when its resource changed, and the loop sleeps until the agents, the
//! terminal or the window wake it.

use std::time::Duration;

use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::picking::events::PointerClick;
use bevy::prelude::*;
use bevy::window::{ExitCondition, WindowPlugin};
use bevy::winit::{EventLoopProxyWrapper, UpdateMode, WinitSettings, WinitUserEvent};
use rig_harness::prelude::{Answerer, Invoked, ShownAgent, Wake};
use rig_tui::ratatui::style::{Color as TermColor, Modifier};
use rig_tui::ratatui::text::Line;

use super::{
    Block, Choose, Dismiss, FRONT, GuiAgents, GuiDialogs, GuiInput, GuiSystems, GuiTranscript,
    LONG_LINES, Show, Stop, Submit, Toggle, ToolBlock,
};

const BACKGROUND: Color = Color::srgb(0.08, 0.08, 0.1);
const PANEL: Color = Color::srgb(0.12, 0.12, 0.15);
const SELECTED: Color = Color::srgb(0.2, 0.22, 0.3);
const TEXT: Color = Color::srgb(0.88, 0.88, 0.9);
const DIM: Color = Color::srgb(0.55, 0.55, 0.6);
const USER: Color = Color::srgb(0.4, 0.8, 0.9);
const FROM: Color = Color::srgb(0.8, 0.5, 0.9);
const RUNNING: Color = Color::srgb(0.9, 0.8, 0.3);
const DONE: Color = Color::srgb(0.4, 0.8, 0.4);
const FAILED: Color = Color::srgb(0.9, 0.4, 0.4);
const FONT_SIZE: f32 = 15.0;
/// Pixels a wheel line scrolls.
const LINE_PIXELS: f32 = 20.0;
/// Notices shown under the transcript.
const SHOWN_NOTICES: usize = 5;

#[derive(Component)]
struct AgentsPanel;

#[derive(Component)]
struct TranscriptPanel;

#[derive(Component)]
struct InputText;

#[derive(Component)]
struct DialogPanel;

/// Opens the window as the front of a `--gui` run, or beside the terminal
/// view in an interactive one. Bevy's window and render plugins are added
/// in `build`, so the flag is read here, before `finish` could ask
/// `runs_front`.
pub(crate) fn build(app: &mut App) {
    let invoked = app.world().get_resource::<Invoked>();
    let front = invoked.is_some_and(|invoked| invoked.flag(FRONT));
    let beside = invoked.is_some_and(Invoked::interactive);
    if !beside && !front {
        return;
    }
    // Beside the terminal view, closing the window leaves the agent
    // running.
    let window = WindowPlugin {
        primary_window: Some(Window {
            title: "rig".to_owned(),
            ..default()
        }),
        exit_condition: if front {
            ExitCondition::OnPrimaryClosed
        } else {
            ExitCondition::DontExit
        },
        ..default()
    };
    // Frames when woken, and at least every second as without a window.
    let mode = UpdateMode::reactive_low_power(Duration::from_secs(1));
    app.add_plugins(rig_harness::windowed(DefaultPlugins.set(window)))
        .insert_resource(WinitSettings {
            focused_mode: mode,
            unfocused_mode: mode,
        })
        .add_systems(Startup, spawn_layout)
        .add_systems(Update, (type_keys, scroll))
        .add_systems(
            PostUpdate,
            (
                draw_agents
                    .run_if(resource_changed::<GuiAgents>.or_eager(resource_changed::<ShownAgent>)),
                draw_transcript.run_if(resource_changed::<GuiTranscript>),
                draw_input.run_if(resource_changed::<GuiInput>),
                draw_dialog.run_if(resource_changed::<GuiDialogs>),
            )
                .after(GuiSystems)
                .before(bevy::ui::UiSystems::Prepare),
        );
}

/// Once the window opens, it answers every agent's questions, and the
/// agent's [`Wake`] points at winit's loop, so agent activity and terminal
/// input wake the window instead of a poll.
pub(crate) fn finish(app: &mut App) {
    if let Some(proxy) = app.world().get_resource::<EventLoopProxyWrapper>() {
        let proxy = (**proxy).clone();
        app.insert_resource(Wake::new(move || {
            proxy.send_event(WinitUserEvent::WakeUp).ok();
        }));
        app.world_mut()
            .spawn((Name::new("answerer:gui"), Answerer::default()));
    }
}

fn spawn_layout(mut commands: Commands) {
    commands.spawn(Camera2d);
    let root = commands
        .spawn((
            Node {
                width: percent(100),
                height: percent(100),
                flex_direction: FlexDirection::Row,
                ..default()
            },
            BackgroundColor(BACKGROUND),
        ))
        .id();
    commands.spawn((
        AgentsPanel,
        Node {
            width: px(240),
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(px(8)),
            row_gap: px(2),
            overflow: Overflow::scroll_y(),
            ..default()
        },
        BackgroundColor(PANEL),
        ChildOf(root),
    ));
    let main = commands
        .spawn((
            Node {
                flex_grow: 1.0,
                flex_direction: FlexDirection::Column,
                min_width: px(0),
                ..default()
            },
            ChildOf(root),
        ))
        .id();
    commands.spawn((
        TranscriptPanel,
        Node {
            flex_grow: 1.0,
            min_height: px(0),
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(px(12)),
            row_gap: px(6),
            overflow: Overflow::scroll_y(),
            ..default()
        },
        ScrollPosition::default(),
        ChildOf(main),
    ));
    commands.spawn((
        Node {
            min_height: px(48),
            padding: UiRect::all(px(10)),
            border: UiRect::top(px(1)),
            ..default()
        },
        BorderColor::all(DIM),
        BackgroundColor(PANEL),
        ChildOf(main),
        children![(InputText, text("› ", TEXT))],
    ));
    commands.spawn((
        DialogPanel,
        Node {
            display: Display::None,
            position_type: PositionType::Absolute,
            left: percent(25),
            right: percent(25),
            top: percent(25),
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(px(16)),
            row_gap: px(8),
            border: UiRect::all(px(1)),
            ..default()
        },
        BorderColor::all(TEXT),
        BackgroundColor(PANEL),
        GlobalZIndex(10),
    ));
}

/// A text node in one color.
fn text(content: impl Into<String>, color: Color) -> impl Bundle {
    (
        Text::new(content),
        TextFont {
            font_size: FontSize::Px(FONT_SIZE),
            ..default()
        },
        TextColor(color),
    )
}

/// A span of a text node in one color.
fn span(content: impl Into<String>, color: Color) -> impl Bundle {
    (
        TextSpan::new(content),
        TextFont {
            font_size: FontSize::Px(FONT_SIZE),
            ..default()
        },
        TextColor(color),
    )
}

fn draw_agents(
    list: Res<GuiAgents>,
    shown: Res<ShownAgent>,
    panels: Query<Entity, With<AgentsPanel>>,
    mut commands: Commands,
) {
    for panel in &panels {
        commands.entity(panel).despawn_related::<Children>();
        for row in &list.rows {
            let agent = row.agent;
            let indent = "  ".repeat(row.depth);
            let background = if shown.0 == Some(agent) {
                SELECTED
            } else {
                PANEL
            };
            let status = if row.busy { RUNNING } else { DIM };
            commands
                .spawn((
                    Node {
                        padding: UiRect::axes(px(6), px(4)),
                        ..default()
                    },
                    BackgroundColor(background),
                    ChildOf(panel),
                    children![(
                        text(format!("{indent}{}", row.name), TEXT),
                        children![span(format!("  {}", row.status), status)],
                    )],
                ))
                .observe(move |mut click: On<PointerClick>, mut commands: Commands| {
                    click.propagate(false);
                    commands.trigger(Show { agent });
                });
        }
    }
}

fn draw_transcript(
    transcript: Res<GuiTranscript>,
    mut panels: Query<(Entity, &mut ScrollPosition), With<TranscriptPanel>>,
    mut commands: Commands,
) {
    for (panel, mut scroll) in &mut panels {
        commands.entity(panel).despawn_related::<Children>();
        for block in &transcript.blocks {
            draw_block(block, panel, &mut commands);
        }
        if !transcript.reasoning.is_empty() {
            commands.spawn((text(transcript.reasoning.clone(), DIM), ChildOf(panel)));
        }
        if !transcript.streaming.is_empty() {
            markdown(&transcript.streaming, panel, &mut commands);
        }
        let notices: Vec<_> = transcript.shown_notices().collect();
        for notice in notices
            .iter()
            .skip(notices.len().saturating_sub(SHOWN_NOTICES))
        {
            let color = if notice.error { FAILED } else { DIM };
            commands.spawn((text(notice.text.clone(), color), ChildOf(panel)));
        }
        // Layout clamps it to the bottom: new text is followed.
        scroll.0.y = f32::MAX;
    }
}

fn draw_block(block: &Block, panel: Entity, commands: &mut Commands) {
    match block {
        Block::User(said) => {
            commands.spawn((text(format!("› {said}"), USER), ChildOf(panel)));
        }
        Block::Delivered { from, text: said } => {
            commands.spawn((
                text(format!("⤶ {from}\n"), FROM),
                ChildOf(panel),
                children![span(said.clone(), DIM)],
            ));
        }
        Block::Answer(answer) => markdown(answer, panel, commands),
        Block::Reasoning {
            key,
            text: said,
            open,
        } => {
            let key = *key;
            let header = if *open {
                "▾ reasoning"
            } else {
                "▸ reasoning"
            };
            let shown = if *open { said.clone() } else { String::new() };
            commands
                .spawn((
                    text(header, DIM),
                    ChildOf(panel),
                    children![span(format!("\n{shown}"), DIM)],
                ))
                .observe(move |mut click: On<PointerClick>, mut commands: Commands| {
                    click.propagate(false);
                    commands.trigger(Toggle { key });
                });
        }
        Block::Tool(tool) => draw_tool(tool, panel, commands),
        Block::Stopped { replied } => {
            let said = if *replied {
                "[stopped]"
            } else {
                "[stopped before it was answered]"
            };
            commands.spawn((text(said, DIM), ChildOf(panel)));
        }
    }
}

fn draw_tool(tool: &ToolBlock, panel: Entity, commands: &mut Commands) {
    let color = match (&tool.result, tool.failed) {
        (None, _) => RUNNING,
        (Some(_), false) => DONE,
        (Some(_), true) => FAILED,
    };
    let mut body = tool.arguments.clone();
    if let Some(result) = &tool.result {
        body.push_str("\n→ ");
        body.push_str(result);
    }
    let marker = match (tool.long, tool.open) {
        (false, _) => "",
        (true, true) => " ▾",
        (true, false) => " ▸",
    };
    if tool.long && !tool.open {
        let head: Vec<&str> = body.lines().take(LONG_LINES).collect();
        body = format!("{}\n… (click to show all)", head.join("\n"));
    }
    let key = tool.key;
    let mut node = commands.spawn((
        text(format!("● {}{marker}\n", tool.name), color),
        ChildOf(panel),
        children![span(body, DIM)],
    ));
    if tool.long {
        node.observe(move |mut click: On<PointerClick>, mut commands: Commands| {
            click.propagate(false);
            commands.trigger(Toggle { key });
        });
    }
}

/// Markdown drawn as the terminal view draws it, a text node per line with
/// a span per styled run.
fn markdown(source: &str, panel: Entity, commands: &mut Commands) {
    for line in rig_tui::markdown::render(source, &rig_tui::Theme::default(), None) {
        draw_line(&line.colored(), panel, commands);
    }
}

fn draw_line(line: &Line<'_>, panel: Entity, commands: &mut Commands) {
    let root = commands.spawn((text("", TEXT), ChildOf(panel))).id();
    for styled in &line.spans {
        let style = line.style.patch(styled.style);
        let mut color = style.fg.map_or(TEXT, term_color);
        if style.add_modifier.contains(Modifier::DIM) {
            color = color.with_alpha(0.6);
        }
        commands.spawn((span(styled.content.to_string(), color), ChildOf(root)));
    }
}

/// A terminal color in the window.
fn term_color(color: TermColor) -> Color {
    match color {
        TermColor::Red | TermColor::LightRed => FAILED,
        TermColor::Green | TermColor::LightGreen => DONE,
        TermColor::Yellow | TermColor::LightYellow => RUNNING,
        TermColor::Blue | TermColor::LightBlue => Color::srgb(0.45, 0.6, 0.95),
        TermColor::Magenta | TermColor::LightMagenta => FROM,
        TermColor::Cyan | TermColor::LightCyan => USER,
        TermColor::Gray | TermColor::DarkGray => DIM,
        TermColor::Rgb(red, green, blue) => Color::srgb_u8(red, green, blue),
        _ => TEXT,
    }
}

fn draw_input(input: Res<GuiInput>, mut texts: Query<&mut Text, With<InputText>>) {
    for mut shown in &mut texts {
        shown.0 = format!("› {}▏", input.text);
    }
}

fn draw_dialog(
    dialogs: Res<GuiDialogs>,
    mut panels: Query<(Entity, &mut Node), With<DialogPanel>>,
    mut commands: Commands,
) {
    for (panel, mut node) in &mut panels {
        commands.entity(panel).despawn_related::<Children>();
        let Some(dialog) = dialogs.open.first() else {
            node.display = Display::None;
            continue;
        };
        node.display = Display::Flex;
        commands.spawn((text(dialog.title.clone(), TEXT), ChildOf(panel)));
        if !dialog.body.is_empty() {
            commands.spawn((text(dialog.body.clone(), DIM), ChildOf(panel)));
        }
        if dialog.free_text {
            commands.spawn((
                text("…or type the answer and press Enter", DIM),
                ChildOf(panel),
            ));
        }
        for (item, label) in dialog.choices.iter().enumerate() {
            commands.spawn((button(label), ChildOf(panel))).observe(
                move |mut click: On<PointerClick>, mut commands: Commands| {
                    click.propagate(false);
                    commands.trigger(Choose { item });
                },
            );
        }
        commands.spawn((button("Cancel"), ChildOf(panel))).observe(
            |mut click: On<PointerClick>, mut commands: Commands| {
                click.propagate(false);
                commands.trigger(Dismiss);
            },
        );
    }
}

/// A button showing `label`.
fn button(label: &str) -> impl Bundle {
    (
        Node {
            padding: UiRect::axes(px(10), px(6)),
            border: UiRect::all(px(1)),
            ..default()
        },
        BorderColor::all(DIM),
        BackgroundColor(SELECTED),
        children![text(label.to_owned(), TEXT)],
    )
}

/// Typing edits the input; Enter sends it (Shift+Enter is a new line), Esc
/// closes the dialog shown, else stops the agent.
fn type_keys(
    mut keys: MessageReader<KeyboardInput>,
    held: Res<ButtonInput<KeyCode>>,
    dialogs: Res<GuiDialogs>,
    mut input: ResMut<GuiInput>,
    mut commands: Commands,
) {
    let shift = held.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let control = held.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);
    for key in keys.read() {
        if key.state != ButtonState::Pressed {
            continue;
        }
        match &key.logical_key {
            Key::Escape if dialogs.open.is_empty() => commands.trigger(Stop),
            Key::Escape => commands.trigger(Dismiss),
            Key::Enter if shift => input.text.push('\n'),
            Key::Enter => commands.trigger(Submit),
            Key::Backspace => {
                input.text.pop();
            }
            _ if control => {}
            _ => {
                if let Some(typed) = &key.text
                    && typed.chars().all(|character| !character.is_control())
                {
                    input.text.push_str(typed);
                }
            }
        }
    }
}

/// The wheel scrolls the transcript.
fn scroll(
    mut wheel: MessageReader<MouseWheel>,
    mut panels: Query<(&mut ScrollPosition, &ComputedNode), With<TranscriptPanel>>,
) {
    let moved: f32 = wheel
        .read()
        .map(|wheel| match wheel.unit {
            MouseScrollUnit::Line => wheel.y * LINE_PIXELS,
            MouseScrollUnit::Pixel => wheel.y,
        })
        .sum();
    if moved == 0.0 {
        return;
    }
    // From where layout placed it: a followed bottom is past the end.
    for (mut position, node) in &mut panels {
        let at = node.scroll_position.y * node.inverse_scale_factor;
        position.0.y = (at - moved).max(0.0);
    }
}
