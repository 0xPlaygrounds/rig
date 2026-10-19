//! The queue in the terminal view: a panel under the transcript listing the
//! tasks while there are any, and a status-line item with how many run and
//! wait, and how the last one ended.

use bevy_ecs::schedule::SystemCondition;
use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    AppStatus, PanelCanvas, Placement, RequestRedraw, Side, StatusItem, StatusSystems, Theme, Tone,
    TuiPanel, TuiSystems,
};

use super::{QueueNote, QueuedTask, RunningTurns, TaskQueue, TaskState, line, live_cost};

/// After the other agents at work; gone early when the line is narrow.
const QUEUE_ITEM: StatusItem = StatusItem::new("sn-task-queue", Side::Left, 75, 9);

/// The most task rows the panel takes.
const ROWS: usize = 8;

/// On the queue's panel.
#[derive(Component)]
struct QueuePanel;

pub(super) fn add(app: &mut App) {
    app.init_resource::<AppStatus>()
        .init_resource::<Theme>()
        .add_message::<RequestRedraw>()
        .add_systems(
            PostUpdate,
            (
                place_panel.run_if(resource_changed::<TaskQueue>),
                draw.in_set(TuiSystems::Draw),
                status
                    .in_set(StatusSystems::Show)
                    .run_if(resource_changed::<TaskQueue>.or_eager(resource_changed::<QueueNote>)),
            ),
        );
}

/// Keeps the panel while there are tasks, as tall as they need, and asks
/// for a frame with the queue as it is now.
fn place_panel(
    queue: Res<TaskQueue>,
    mut panels: Query<(Entity, &mut TuiPanel), With<QueuePanel>>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let rows = queue.tasks.len().min(ROWS);
    let height = u16::try_from(rows + 2).unwrap_or(u16::MAX);
    let panel = TuiPanel::new(Placement::Bottom(Constraint::Length(height)));
    match (panels.single_mut(), rows) {
        (Ok((entity, _)), 0) => commands.entity(entity).despawn(),
        (Ok((_, mut placed)), _) => {
            placed.set_if_neq(panel);
        }
        (Err(_), 0) => {}
        (Err(_), _) => {
            commands.spawn((QueuePanel, panel));
        }
    }
    redraw.write(RequestRedraw);
}

/// Each task, newest last: its state, elapsed time, cost and answer.
fn draw(
    mut panels: Query<&mut PanelCanvas, With<QueuePanel>>,
    queue: Res<TaskQueue>,
    agents: Query<(Entity, &QueuedTask)>,
    turns: RunningTurns,
    theme: Res<Theme>,
) {
    let live = |number: u32| {
        let agent = agents.iter().find(|(_, task)| task.0 == number);
        live_cost(agent.map(|(agent, _)| agent), &turns)
    };
    let skip = queue.tasks.len().saturating_sub(ROWS);
    let lines: Vec<String> = queue
        .tasks
        .iter()
        .skip(skip)
        .map(|task| line(task, live(task.number)))
        .collect();
    for mut canvas in &mut panels {
        let block = theme.panel("tasks (/queue)");
        canvas.render(Paragraph::new(lines.join("\n")).block(block));
    }
}

/// `queue: 1 running, 2 pending · task 2 done`.
fn status(queue: Res<TaskQueue>, note: Res<QueueNote>, mut app_status: ResMut<AppStatus>) {
    let count = |state| {
        queue
            .tasks
            .iter()
            .filter(|task| task.state == state)
            .count()
    };
    let (running, pending) = (count(TaskState::Running), count(TaskState::Pending));
    let mut parts = Vec::new();
    if running + pending > 0 {
        parts.push(format!("queue: {running} running, {pending} pending"));
    }
    if !note.0.is_empty() {
        parts.push(note.0.clone());
    }
    app_status
        .0
        .show(QUEUE_ITEM.says(parts.join(" · "), Tone::Dim));
}
