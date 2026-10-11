//! Each agent's meter, and what its running turn spent, in the terminal
//! view's status line.

use rig_core::completion::{ContextUse, tokens_label};
use rig_harness::prelude::*;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

use super::{Spending, TurnSpending};

pub(super) fn add(app: &mut App) {
    app.add_systems(PostUpdate, (show_meter, show_turn).in_set(StatusSystems));
}

/// Where what the running turn spent is in the status line.
const TURN: StatusItem = StatusItem::at(Side::Left, 80, 11);
/// Where the meter's items are: the cached input goes first when the
/// line is too narrow, then the cost, the tokens and the context.
const TOKENS: StatusItem = StatusItem::at(Side::Right, 10, 4);
const CACHE: StatusItem = StatusItem::at(Side::Right, 20, 1);
const COST: StatusItem = StatusItem::at(Side::Right, 30, 3);
const CONTEXT: StatusItem = StatusItem::at(Side::Right, 40, 5);

/// Each agent's meter, once a model call of it finished: its uncached
/// input and output tokens, cache reads, cost, then the context against
/// the model's window, yellow past 70% and red past 90%. `/usage` details
/// the cache writes and reasoning.
fn show_meter(
    mut agents: Query<
        (&Spending, &LastUsage, Option<&Connection>, &mut StatusItems),
        Or<(Changed<Spending>, Changed<LastUsage>, Changed<Connection>)>,
    >,
) {
    for (Spending(spent), last, connection, mut items) in &mut agents {
        let shown = spent.calls > 0;
        let window = connection.and_then(|connection| connection.spec.context_window);
        let context = last.context().map(|tokens| ContextUse { tokens, window });
        let tokens = format!(
            "↑{} ↓{}",
            tokens_label(spent.uncached_input()),
            tokens_label(spent.tokens.output_tokens.unwrap_or(0))
        );
        let cache = (spent.tokens.cached_input_tokens)
            .filter(|read| *read > 0)
            .map(|read| format!("cache {}", tokens_label(read)));
        let tone = match context.and_then(|context| context.percent()) {
            Some(90..) => Tone::Red,
            Some(70..) => Tone::Yellow,
            _ => Tone::Dim,
        };
        let context = context.map(|context| format!("ctx {context}"));
        let text = |text: Option<String>| text.filter(|_| shown).unwrap_or_default();
        items.show(TOKENS.says(text(Some(tokens)), Tone::Dim));
        items.show(CACHE.says(text(cache), Tone::Dim));
        items.show(COST.says(text(spent.cost_label()), Tone::Dim));
        items.show(CONTEXT.says(text(context), tone));
    }
}

/// What each running turn spent so far, once a model call of it finished.
fn show_turn(mut turns: Query<(&TurnSpending, &mut StatusItems), Changed<TurnSpending>>) {
    for (TurnSpending(spent), mut items) in &mut turns {
        if spent.calls > 0 {
            let used = spent.cost_or_tokens();
            items.show(TURN.says(format!("this turn {used}"), Tone::Dim));
        }
    }
}
