//! Plugin corpus (held-out): H19: Name sessions automatically (spec `H19-auto-session-name.md`).
//!
//! Once the main agent's turn ends with an answer in a session with no
//! name, a model call of the plugin's own (on `[auto-name] model`, else the
//! agent's model) reads the first message the user typed and the start of
//! the answer, and its 3–6 word title, cleaned, becomes the session's
//! name: `rig_basics::sessions`' [`SessionTitle`], the one `/name` sets, shown in the
//! status line and `/resume`. The call is no turn's, so it never reaches
//! the conversation, and its usage is counted under the purpose
//! `auto-name`. A name the user set first, also while the call runs,
//! is kept; a failed call is tried once more after the next turn.

use bevy_log::{debug, warn};
use rig_basics::sessions::{SessionTitle, SessionsPlugin};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::CompletionRequest;
use rig_harness::rig_core::message::Message;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The settings' table, and the naming calls' purpose.
const NAME: &str = "auto-name";

/// How much of the answer the naming call reads.
const ANSWER_CHARS: usize = 500;

/// The longest title kept.
const TITLE_CHARS: usize = 50;

/// How many naming calls may fail before the plugin stops trying.
const ATTEMPTS: u8 = 2;

/// What the naming model is told.
const PREAMBLE: &str = "You title chat sessions. Given the user's first message and the \
    start of the assistant's answer, answer with a title of 3 to 6 words that says what the \
    user wants, such as \"Fix parser bug\". One line, no quotes, nothing else.";

#[derive(Default)]
pub struct AutoSessionNamePlugin;

impl Plugin for AutoSessionNamePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SessionsPlugin>() {
            app.add_plugins(SessionsPlugin);
        }
        app.add_setting::<AutoNameConfig>(NAME)
            .init_resource::<Failures>()
            .add_observer(ask_for_a_name)
            .add_observer(take_the_name)
            .add_observer(follow_successor::<NamingCall>);
    }
}

/// The `[auto-name]` table of the settings.
#[derive(Resource, Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AutoNameConfig {
    /// The catalog model (`vendor/model`) that writes titles; none uses
    /// the agent's own.
    pub model: Option<String>,
}

/// On the plugin's naming call, and on whichever call takes it over.
#[derive(Component, Debug)]
pub struct NamingCall;

/// How many naming calls failed in this run.
#[derive(Resource, Default)]
struct Failures(u8);

/// Asks for a title when the main agent's turn ended with an answer, the
/// session has no name, the user typed something, and no naming call
/// runs or failed too often.
fn ask_for_a_name(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    conversations: Query<&Conversation>,
    agents: MainAgentQuery,
    naming: Query<(), (With<NamingCall>, Without<Settled>)>,
    (title, failures, config, models): (
        Res<SessionTitle>,
        Res<Failures>,
        Res<AutoNameConfig>,
        Res<Models>,
    ),
    mut commands: Commands,
) {
    let Ok((TurnOutcome::Answered(message), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    if title.name.is_some()
        || !naming.is_empty()
        || failures.0 >= ATTEMPTS
        || main_agent(&agents) != Some(agent)
    {
        return;
    }
    let Some(asked) = conversations
        .get(agent)
        .ok()
        .and_then(|conversation| first_typed(conversation))
    else {
        return;
    };
    let answer: String = message
        .final_answer()
        .unwrap_or_default()
        .chars()
        .take(ANSWER_CHARS)
        .collect();
    let prompt = format!(
        "The user's first message:\n{asked}\n\nThe start of the assistant's answer:\n{answer}"
    );
    let mut request =
        ModelRequest::new(CompletionRequest::new(Message::user(prompt)).preamble(PREAMBLE));
    if let Some(reference) = config.model.as_deref() {
        match models.connect(reference) {
            Ok(connection) => request = request.on(connection),
            Err(why) => {
                warn!("auto-name: cannot use {reference}, so the agent's model names: {why}");
            }
        }
    }
    commands.spawn((
        Name::new("session name"),
        NamingCall,
        request,
        Effect::default().with_purpose(NAME),
        ForAgent(agent),
    ));
}

/// Names the session with the title the naming call answered, unless the
/// user named it meanwhile; a failure is logged and counted.
fn take_the_name(
    settled: On<Add<Settled>>,
    calls: Query<&Outcome, With<NamingCall>>,
    mut title: ResMut<SessionTitle>,
    mut failures: ResMut<Failures>,
) {
    let Ok(outcome) = calls.get(settled.entity) else {
        return;
    };
    let named = match (&outcome.0, outcome.completion()) {
        (Err(report), _) => Err(report.to_string()),
        (Ok(_), None) => Err("the model did not answer with a completion".to_owned()),
        (Ok(_), Some(response)) => Some(clean(&response.text()))
            .filter(|name| !name.is_empty())
            .ok_or_else(|| "the model answered no title".to_owned()),
    };
    match named {
        Ok(name) if title.name.is_none() => title.name = Some(name),
        Ok(_) => {}
        Err(why) => {
            failures.0 += 1;
            debug!("auto-name: the naming call failed: {why}");
        }
    }
}

/// The first line of `answer`, without quotes, markup or trailing
/// punctuation, cut to [`TITLE_CHARS`] at a word.
fn clean(answer: &str) -> String {
    const QUOTES: [char; 8] = ['"', '“', '”', '„', '«', '»', '`', '*'];
    const EDGES: [char; 13] = [
        '\'', '‘', '’', '#', '-', '.', ',', ';', ':', '!', '?', '…', ' ',
    ];
    let line = answer
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    let unquoted: String = line.chars().filter(|c| !QUOTES.contains(c)).collect();
    let mut title = String::new();
    for word in unquoted.split_whitespace() {
        let longer = if title.is_empty() {
            word.to_owned()
        } else {
            format!("{title} {word}")
        };
        if longer.chars().count() > TITLE_CHARS {
            if title.is_empty() {
                title = word.chars().take(TITLE_CHARS).collect();
            }
            break;
        }
        title = longer;
    }
    title.trim_matches(EDGES).to_owned()
}
