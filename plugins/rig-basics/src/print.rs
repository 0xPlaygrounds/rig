//! The agent over its pipes: `--print` answers one prompt and exits, and
//! `--json` writes the run as JSON lines on stdout and, without a prompt,
//! reads requests as JSON lines on stdin until stdin ends. Logs go to
//! stderr. `rig help-agents` is the guide a driver reads.
//!
//! A `--print` run ([`FRONT`]) is the front of every run that is not
//! interactive ([`Invoked::interactive`]), unless another front's option
//! chose one. It sends the prompt, with what was piped in on stdin after
//! it (`git diff | rig -p "review this"`), through the
//! [`headless`](crate::headless) loop, which picks the model, exits once
//! no agent works and sets the exit code: 0 answered, 1 failed, 2 no
//! prompt or a refused slash command. Without a prompt, a resumed
//! session's unfinished turn finishes (`rig -r <id> -p`). The answer goes
//! to stdout, then the [`PrintSection`]s; failures, notices and each
//! [`ToolProgress`] line go to stderr.
//!
//! `--json` ([`JSON`]) writes events in place of the answer. With
//! `-p` the run is the same one-shot run. Without, it is long-lived: each
//! line on stdin is a [`Request`], answered by a `reply` line, and the end
//! of stdin ends the agent at once, so an agent whose driver died never
//! lingers. Every line carries the protocol's [`VERSION`] as `v`, and a
//! line of another version is refused, naming the side to update.
//!
//! Nobody answers questions in a one-shot run: a question is refused, with
//! why, unless `--answer=ID[,ID…]` ([`ANSWER`]) gives choices to answer
//! with. A long-lived `--json` run is an [`Answerer`]: each question is an
//! `ask` event, and the driver's `answer` request answers it.

use std::io::{IsTerminal, Read, Write};

use async_channel::{Receiver, Sender};
use bevy_ecs::message::{MessageCursor, Messages};
use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::headless::{AppHeadlessExt, HeadlessEnded, HeadlessTurnEnded, KeepRunning};
use crate::telemetry::usage::{SessionUsage, TurnUsage};

/// `--print`'s name as a front.
pub const FRONT: &str = "print";

/// `--json`'s name as a front, and its option.
pub const JSON: &str = "json";

/// The name the print front decides effects as ([`Decider::plugin`]), in
/// effect logs: the module's.
pub const ORIGIN: &str = "print";

/// The option that answers a one-shot run's questions.
pub const ANSWER: &str = "answer";

/// The option that limits the agent to the tools it names.
pub const TOOLS: &str = "tools";

/// The version of the JSON lines, `v` on every line both ways.
pub const VERSION: u64 = 1;

/// What a print run writes to stdout after the answer, such as a todo
/// list: spawn one and keep its `text` current. Sections follow the
/// answer by `tag`, each after an empty line; an empty `text`, or Bevy's
/// `Disabled` on the entity, leaves one out.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct PrintSection {
    /// What sorts it, such as `todo`.
    pub tag: String,
    /// What is printed.
    pub text: String,
}

impl PrintSection {
    /// `text` printed after the answer, sorted by `tag`.
    pub fn new(tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// Where `--json` lines go: stdout unless the app inserted another.
#[derive(Resource)]
pub struct JsonOut(Box<dyn Write + Send + Sync>);

impl JsonOut {
    /// Lines to `out`.
    pub fn new(out: impl Write + Send + Sync + 'static) -> Self {
        Self(Box::new(out))
    }
}

/// The request lines of a long-lived `--json` run: stdin's unless the app
/// inserted another. They end when every sender dropped.
#[derive(Resource)]
pub struct JsonIn(Receiver<String>);

impl JsonIn {
    /// An input fed by the returned sender; it ends when the sender drops.
    pub fn channel() -> (Sender<String>, Self) {
        let (sender, receiver) = async_channel::unbounded();
        (sender, Self(receiver))
    }
}

/// A request a driver writes on stdin: one JSON line with `v`, `type`, an
/// optional `id` its `reply` repeats and, but for `state`, an optional
/// `agent` (the main agent without one). `send` delivers a message as
/// typed: it starts a turn, or joins the running one at its next model
/// call. `answer` decides an open question with choices by id or a text.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[allow(missing_docs)]
pub enum Request {
    Send {
        text: String,
        #[serde(default)]
        agent: Option<String>,
    },
    Stop {
        #[serde(default)]
        agent: Option<String>,
    },
    Answer {
        ask: u64,
        #[serde(default)]
        choices: Vec<String>,
        #[serde(default)]
        text: Option<String>,
    },
    State,
}

/// Answers `--print` and `--json` runs.
#[derive(Default)]
pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn build(&self, app: &mut App) {
        let json = "Write the run as JSON lines; without -p, read requests as JSON lines on \
                    stdin (rig help-agents)";
        app.add_front(Front::new(FRONT).headless())
            .add_front(Front::new(JSON).on_flag(JSON, json))
            .add_flag(
                Flag::new(
                    ANSWER,
                    "Answer the questions of a -p run with these choices",
                )
                .value("ID[,ID…]"),
            )
            .add_flag(Flag::new(TOOLS, "Give the agent only these tools").value("NAME[,NAME…]"));
    }

    fn finish(&self, app: &mut App) {
        let json = app.runs_front(JSON);
        if !json && !app.runs_front(FRONT) {
            return;
        }
        let invoked = app.world().get_resource::<Invoked>().cloned();
        let list = |name| -> Vec<String> {
            let list = invoked
                .as_ref()
                .and_then(|invoked| invoked.flag_value(name));
            let items = list.unwrap_or_default().split(',').map(str::trim);
            items
                .filter(|item| !item.is_empty())
                .map(str::to_owned)
                .collect()
        };
        let (answers, tools) = (list(ANSWER), list(TOOLS));
        if !tools.is_empty() {
            app.insert_resource(Tools(tools))
                .add_systems(First, limit_tools.after(AgentLoop));
        }
        let prompt = invoked.and_then(|invoked| invoked.args.print);
        if json {
            if !app.world().contains_resource::<JsonOut>() {
                app.insert_resource(JsonOut::new(std::io::stdout()));
            }
            app.add_observer(turn_started)
                .add_observer(started)
                .add_observer(settled)
                .add_systems(
                    Update,
                    (ready.run_if(resource_added::<SessionRestored>), notices),
                )
                .init_resource::<TextRead>()
                .add_systems(PostUpdate, stream_text);
        } else {
            let command = prompt
                .as_deref()
                .unwrap_or_default()
                .trim_start()
                .starts_with('/');
            app.init_resource::<Printed>()
                .add_systems(Update, (print_notices(command), print_progress))
                .add_observer(print_answer)
                .add_observer(print_output);
        }
        if json && prompt.is_none() {
            // A long-lived `--json` run: the driver answers questions, and
            // stdin keeps it going until it ends.
            app.world_mut()
                .spawn((Answerer::default(), Requests, KeepRunning));
            if !app.world().contains_resource::<JsonIn>() {
                let (sender, input) = JsonIn::channel();
                read_stdin(sender);
                app.insert_resource(input);
            }
            app.add_systems(Startup, listen)
                .add_observer(request)
                .add_observer(input_ended)
                .add_observer(headless_ended)
                .run_headless(None);
            return;
        }
        let prompt = prompt.unwrap_or_default();
        if !answers.is_empty() {
            app.world_mut().spawn(Answerer::default());
        }
        app.insert_resource(Answers(answers))
            .configure_sets(
                AgentPass,
                PrintSystems.in_set(Stage::Gate).after(AskSystems),
            )
            .add_systems(AgentPass, answer.in_set(PrintSystems))
            .add_observer(headless_ended)
            .run_headless(Some(full_prompt(&prompt)));
    }
}

/// The Gate system that answers each question of a one-shot run, after
/// [`AskSystems`], so a line the user typed wins.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrintSystems;

/// The choices `--answer` gives, in order; none without it.
#[derive(Resource)]
struct Answers(Vec<String>);

/// The tools `--tools` leaves the agent.
#[derive(Resource)]
struct Tools(Vec<String>);

/// Leaves each agent that has none of its own only the `--tools` tools.
fn limit_tools(
    agents: Query<Entity, (With<Agent>, Without<ToolAccess>)>,
    tools: Res<Tools>,
    mut commands: Commands,
) {
    for agent in &agents {
        commands
            .entity(agent)
            .insert(ToolAccess::Only(tools.0.clone()));
    }
}

/// Answers each question with the first of the `--answer` choices it
/// offers, or refuses it, saying why.
fn answer(
    mut asks: Query<(&Ask, &mut Decision), (ReadyToStart, PluginEffects)>,
    answers: Res<Answers>,
    mut notices: MessageWriter<Notice>,
) {
    for (ask, mut decision) in &mut asks {
        if decision.is_decided() {
            continue;
        }
        let offered = |id: &&String| ask.choices.iter().any(|choice| &choice.id == *id);
        let taken = ask.choices.iter().map(|choice| &choice.id);
        let taken: Vec<String> = taken.filter(|id| answers.0.contains(id)).cloned().collect();
        let answer = match ask.multiple {
            true => Some(taken)
                .filter(|taken| !taken.is_empty())
                .map(Answer::choices),
            false => answers.0.iter().find(offered).cloned().map(Answer::choice),
        };
        let outcome = match answer {
            Some(answer) => answer.outcome(),
            None => {
                let why = match answers.0.is_empty() {
                    true => format!(
                        "nobody answers questions in a -p run, such as “{}”: --answer=ID \
                         answers them",
                        ask.title
                    ),
                    false => format!("--answer offers none of the choices of “{}”", ask.title),
                };
                notices.write(Notice::warn(None, why.clone()));
                Outcome::refused(why)
            }
        };
        decision.decide(Decider::plugin(ORIGIN), outcome);
    }
}

/// How long a prompt waits for the first of what is piped in on stdin: a
/// pipe its driver never writes to nor closes, as an agent's shell may
/// leave stdin, must not hold the run.
const PIPED_WAIT: Duration = Duration::from_secs(1);

/// The prompt with what was piped in on stdin after it.
fn full_prompt(prompt: &str) -> String {
    match std::io::stdin().is_terminal() {
        true => prompt.to_owned(),
        false => with_piped(prompt, std::io::stdin(), PIPED_WAIT),
    }
}

/// `prompt` with what `piped` holds after it. Without a prompt, `piped` is
/// the prompt, read to its end; with one, it is read only when something
/// comes within `wait`.
fn with_piped(prompt: &str, mut piped: impl Read + Send + 'static, wait: Duration) -> String {
    let (sender, read) = std::sync::mpsc::channel();
    let (first, rest) = (sender.clone(), sender);
    std::thread::spawn(move || {
        let mut text = Vec::new();
        let mut byte = [0];
        let got = piped.read(&mut byte).unwrap_or_default();
        first.send(None).ok();
        text.extend(byte.get(..got).unwrap_or_default());
        if got > 0 {
            piped.read_to_end(&mut text).ok();
        }
        rest.send(Some(String::from_utf8_lossy(&text).into_owned()))
            .ok();
    });
    let wait = match prompt.trim().is_empty() {
        true => Duration::MAX,
        false => wait,
    };
    if read.recv_timeout(wait).is_err() {
        eprintln!(
            "rig: nothing came on stdin within {}s, so the prompt goes alone; give -p stdin \
             from /dev/null, or pipe in what goes with it",
            wait.as_secs_f32()
        );
        return prompt.to_owned();
    }
    match read.recv().ok().flatten().unwrap_or_default() {
        piped if piped.trim().is_empty() => prompt.to_owned(),
        piped if prompt.trim().is_empty() => piped,
        piped => format!("{prompt}\n\n{piped}"),
    }
}

/// Whether the run wrote anything to stdout yet.
#[derive(Resource, Default)]
struct Printed(bool);

impl Printed {
    /// Writes `block` to stdout, after an empty line when something came
    /// before it.
    fn print(&mut self, block: Option<String>) {
        if let Some(block) = block {
            match self.0 {
                true => println!("\n{block}"),
                false => println!("{block}"),
            }
            self.0 = true;
        }
    }
}

/// Writes each answer to stdout as it comes.
fn print_answer(ended: On<HeadlessTurnEnded>, mut printed: ResMut<Printed>) {
    printed.print(output(ended.answer.as_deref(), []));
}

/// Writes the [`PrintSection`]s to stdout once the run ends.
fn print_output(
    _: On<HeadlessEnded>,
    sections: Query<&PrintSection>,
    mut printed: ResMut<Printed>,
) {
    printed.print(output(None, sections));
}

/// Says how the run ended: why it could not send the prompt on stderr,
/// or the `exit` line.
fn headless_ended(
    ended: On<HeadlessEnded>,
    out: Option<ResMut<JsonOut>>,
    agents: Query<&AgentId>,
    (paths, usage): (Option<Res<SessionPaths>>, Option<Res<SessionUsage>>),
) {
    let Some(mut out) = out else {
        if let Some(why) = &ended.why {
            eprintln!("rig: {why}");
        }
        return;
    };
    let code = match ended.exit {
        ExitReason::Failed(code) => code,
        _ => 0,
    };
    let fields = json!({
        "code": code,
        "answer": ended.answer,
        "error": ended.why,
        "session": paths.and_then(|paths| Some(paths.0.id()?.as_str().to_owned())),
        "usage": usage.map(|usage| usage.0.total),
    });
    let agent = ended.agent.and_then(|agent| agents.get(agent).ok());
    write(&mut out, agent, "exit", fields);
}

/// What the run prints: the answer, then each [`PrintSection`] by tag,
/// each after an empty line; `None` when there is nothing.
fn output<'a>(
    answer: Option<&str>,
    sections: impl IntoIterator<Item = &'a PrintSection>,
) -> Option<String> {
    let mut sections: Vec<&PrintSection> = sections
        .into_iter()
        .filter(|section| !section.text.is_empty())
        .collect();
    sections.sort_by(|a, b| a.tag.cmp(&b.tag));
    let parts: Vec<&str> = answer
        .into_iter()
        .chain(sections.iter().map(|section| section.text.as_str()))
        .map(str::trim_end)
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// Writes each new line of a running call's [`ToolProgress`] to stderr,
/// after the tool's name.
fn print_progress(progress: Query<(&ToolProgress, Option<&ToolRequest>), Changed<ToolProgress>>) {
    for (ToolProgress(line), tool) in &progress {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match tool {
            Some(tool) => eprintln!("rig: {}: {line}", tool.call.function.name),
            None => eprintln!("rig: {line}"),
        }
    }
}

/// Writes warnings, errors and why a command was refused to stderr, with
/// every notice of a `/command` prompt.
fn print_notices(
    command: bool,
) -> impl FnMut(MessageReader<Notice>, MessageReader<Recalled>) + Send + Sync + 'static {
    move |mut notices, mut recalled| {
        for why in recalled.read().filter_map(|recalled| recalled.why.as_ref()) {
            eprintln!("rig: {why}");
        }
        for notice in notices.read() {
            match notice.level {
                NoticeLevel::Warn | NoticeLevel::Error => eprintln!("rig: {}", notice.text),
                _ if command => eprintln!("{}", notice.text),
                _ => {}
            }
        }
    }
}

/// Writes the event `kind` with `fields` about `agent` as one line.
fn write(out: &mut JsonOut, agent: Option<&AgentId>, kind: &str, fields: Value) {
    let mut line = Map::from_iter([
        ("v".to_owned(), VERSION.into()),
        ("type".to_owned(), kind.into()),
    ]);
    if let Value::Object(fields) = fields {
        line.extend(fields);
    }
    if let Some(agent) = agent {
        line.insert("agent".to_owned(), agent.0.clone().into());
    }
    writeln!(out.0, "{}", Value::Object(line)).ok();
    out.0.flush().ok();
}

/// How far the text events have read the model calls' [`StreamDelta`]s.
#[derive(Resource, Default)]
struct TextRead(MessageCursor<StreamDelta>);

/// Writes events, each naming its agent, after the text streamed before
/// them.
#[derive(SystemParam)]
struct Out<'w, 's> {
    out: ResMut<'w, JsonOut>,
    ids: Query<'w, 's, &'static AgentId>,
    deltas: Res<'w, Messages<StreamDelta>>,
    read: ResMut<'w, TextRead>,
    turn_calls: Query<'w, 's, (), With<TurnCall>>,
}

impl Out<'_, '_> {
    /// Writes the event `kind` with `fields` about `agent`.
    fn emit(&mut self, agent: Option<Entity>, kind: &str, fields: Value) {
        self.text();
        self.write(agent, kind, fields);
    }

    /// Writes the text each turn's model call streamed since the last
    /// event, one line per agent's run of it.
    fn text(&mut self) {
        let mut runs: Vec<(Option<Entity>, String)> = Vec::new();
        let deltas = self.read.0.read(&self.deltas);
        for delta in deltas.filter(|delta| self.turn_calls.contains(delta.effect)) {
            let StreamEvent::Text { text, .. } = &delta.event else {
                continue;
            };
            match runs.last_mut() {
                Some((agent, run)) if *agent == delta.agent => run.push_str(text),
                _ => runs.push((delta.agent, text.clone())),
            }
        }
        for (agent, text) in runs {
            self.write(agent, "text", json!({ "text": text }));
        }
    }

    fn write(&mut self, agent: Option<Entity>, kind: &str, fields: Value) {
        let id = agent.and_then(|agent| self.ids.get(agent).ok());
        write(&mut self.out, id, kind, fields);
    }
}

/// The first line of a long-lived run, once its session is restored.
fn ready(primary: MainAgentQuery, paths: Option<Res<SessionPaths>>, mut out: Out) {
    let session = paths
        .and_then(|paths| paths.0.id())
        .map(|id| id.as_str().to_owned());
    out.emit(main_agent(&primary), "ready", json!({ "session": session }));
}

fn turn_started(turn: On<Add<TurnOf>>, turns: Query<&TurnOf>, mut out: Out) {
    if let Ok(&TurnOf(agent)) = turns.get(turn.entity) {
        out.emit(Some(agent), "turn_start", json!({}));
    }
}

/// A tool call starts, or a question waits for its answer.
fn started(
    start: On<Add<Started>>,
    calls: Query<(&ToolRequest, &ForAgent)>,
    asks: Query<(&Ask, Option<&ForAgent>)>,
    mut out: Out,
) {
    if let Ok((request, &ForAgent(agent))) = calls.get(start.entity) {
        let call = &request.call;
        let fields = json!({
            "id": call.id.wire(),
            "name": call.function.name.as_str(),
            "args": call.function.arguments_value(),
        });
        out.emit(Some(agent), "tool_call", fields);
    } else if let Ok((ask, agent)) = asks.get(start.entity) {
        let choices = ask.choices.iter();
        let fields = json!({
            "ask": start.entity.to_bits(),
            "title": ask.title,
            "body": ask.body,
            "choices": choices.map(|choice| json!({ "id": choice.id, "label": choice.label })).collect::<Vec<_>>(),
            "text": ask.free_text,
            "multiple": ask.multiple,
        });
        out.emit(agent.map(|agent| agent.0), "ask", fields);
    }
}

/// A tool call or a turn ended.
fn settled(
    settled: On<Add<Settled>>,
    effects: Query<(
        &ForAgent,
        Option<&Outcome>,
        Option<&TurnOutcome>,
        Option<&TurnUsage>,
    )>,
    calls: Query<&ToolRequest>,
    mut out: Out,
) {
    let Ok((&ForAgent(agent), outcome, ended, used)) = effects.get(settled.entity) else {
        return;
    };
    if let (Ok(request), Some(outcome)) = (calls.get(settled.entity), outcome) {
        let result = outcome.tool_result();
        let fields = json!({
            "id": request.call.id.wire(),
            "name": request.call.function.name.as_str(),
            "ok": result.is_success(),
            "output": result.output().render(),
        });
        out.emit(Some(agent), "tool_result", fields);
    } else if let Some(ended) = ended {
        let (outcome, answer, error) = match ended {
            TurnOutcome::Answered(message) => ("answered", message.final_answer(), None),
            TurnOutcome::Failed(why) => ("failed", None, Some(why.clone())),
            TurnOutcome::Stopped(why) => ("stopped", None, why.clone()),
        };
        let usage = used.map(|used| used.0);
        let fields =
            json!({ "outcome": outcome, "answer": answer, "error": error, "usage": usage });
        out.emit(Some(agent), "turn_end", fields);
    }
}

/// Writes the text that streamed this frame.
fn stream_text(mut out: Out) {
    out.text();
}

fn notices(mut notices: MessageReader<Notice>, mut out: Out) {
    for notice in notices.read() {
        let level = match notice.level {
            NoticeLevel::Info => "info",
            NoticeLevel::Warn => "warn",
            NoticeLevel::Error => "error",
        };
        let fields = json!({ "level": level, "text": notice.text });
        out.emit(notice.agent, "notice", fields);
    }
}

/// The request lines of a long-lived run, a stream on this entity.
#[derive(Component)]
struct Requests;

/// Starts the thread that sends stdin's lines to `sender`.
fn read_stdin(sender: Sender<String>) {
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else {
                break;
            };
            if sender.send_blocking(line).is_err() {
                break;
            }
        }
    });
}

/// Reads the request lines as a stream on the [`Requests`] entity.
fn listen(
    input: Option<Res<JsonIn>>,
    requests: Query<Entity, With<Requests>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let (Some(input), Ok(entity)) = (input, requests.single()) else {
        return;
    };
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let stream = Running::stream(pool, &wake, input.0.clone());
    commands.entity(entity).insert(stream);
    commands.remove_resource::<JsonIn>();
}

/// The end of stdin ends the run.
fn input_ended(
    ended: On<Remove<Running>>,
    requests: Query<(), With<Requests>>,
    mut commands: Commands,
) {
    if requests.contains(ended.entity) {
        commands.trigger(Exit(ExitReason::Quit));
    }
}

/// What a request needs of the app.
#[derive(SystemParam)]
struct Requested<'w, 's> {
    agents: Query<
        'w,
        's,
        (
            Entity,
            &'static AgentId,
            Has<ActiveTurn>,
            Option<&'static ModelChoice>,
        ),
    >,
    primary: MainAgentQuery<'w, 's>,
    asks: Query<
        'w,
        's,
        (
            Entity,
            &'static Ask,
            Option<&'static ForAgent>,
            &'static mut Decision,
        ),
        (Without<Outcome>, Without<Settled>),
    >,
    commands: Commands<'w, 's>,
}

impl Requested<'_, '_> {
    /// The agent named `id`, else the main agent.
    fn agent(&self, id: Option<String>) -> Result<Entity, String> {
        let agent = match id {
            Some(id) => {
                let ids = self.agents.iter().map(|(entity, id, ..)| (entity, id));
                AgentId(id).find_in(ids)
            }
            None => main_agent(&self.primary),
        };
        agent.ok_or_else(|| "no such agent".to_owned())
    }

    /// The agents and the open questions.
    fn state(&self) -> Value {
        let id = |agent: Entity| self.agents.get(agent).ok().map(|(_, id, ..)| &id.0);
        let agents: Vec<Value> = self
            .agents
            .iter()
            .map(|(_, id, working, model)| {
                json!({ "agent": id.0, "working": working, "model": model.map(|model| &model.0) })
            })
            .collect();
        let asks: Vec<Value> = self
            .asks
            .iter()
            .map(|(ask, question, agent, _)| {
                let agent = agent.and_then(|agent| id(agent.0));
                json!({ "ask": ask.to_bits(), "agent": agent, "title": question.title })
            })
            .collect();
        json!({ "agents": agents, "asks": asks })
    }
}

/// Carries out a request line and answers it with a `reply`.
fn request(
    line: On<Insert<Done<String>>>,
    lines: Query<&Done<String>, With<Requests>>,
    mut requested: Requested,
    mut out: Out,
) {
    let Ok(Done(line)) = lines.get(line.entity) else {
        return;
    };
    if line.trim().is_empty() {
        return;
    }
    let (id, done) = match serde_json::from_str::<Value>(line) {
        Ok(value) => (
            value.get("id").cloned().unwrap_or(Value::Null),
            versioned(value).and_then(|request| carry_out(&mut requested, request)),
        ),
        Err(error) => (Value::Null, Err(format!("not a JSON line: {error}"))),
    };
    let fields = match done {
        Ok(None) => json!({ "id": id, "ok": true }),
        Ok(Some(state)) => json!({ "id": id, "ok": true, "state": state }),
        Err(error) => json!({ "id": id, "ok": false, "error": error }),
    };
    out.emit(None, "reply", fields);
}

/// The request a line of this version holds, or why it is refused.
fn versioned(mut value: Value) -> Result<Request, String> {
    let fields = value.as_object_mut().ok_or("a request is a JSON object")?;
    let version = fields.shift_remove("v");
    fields.shift_remove("id");
    match version.as_ref().and_then(Value::as_u64) {
        Some(VERSION) => {}
        Some(newer) if newer > VERSION => {
            return Err(format!(
                "this line is version {newer} and this rig speaks version {VERSION}: update rig"
            ));
        }
        _ => {
            return Err(format!(
                "this line is version {} and this rig speaks version {VERSION}: update the \
                 driver, whose every line carries \"v\":{VERSION}",
                version.map_or("none".to_owned(), |version| version.to_string())
            ));
        }
    }
    serde_json::from_value(value).map_err(|error| format!("not a request: {error}"))
}

/// Carries out `request`; what `state` asked for.
fn carry_out(requested: &mut Requested, request: Request) -> Result<Option<Value>, String> {
    match request {
        Request::Send { text, agent } => {
            let agent = requested.agent(agent)?;
            let message = Deliver::new(agent, text, DeliveryMode::Steer);
            requested.commands.trigger(message);
        }
        Request::Stop { agent } => {
            let agent = requested.agent(agent)?;
            requested.commands.trigger(StopTurn::new(agent));
        }
        Request::Answer { ask, choices, text } => {
            let entity = Entity::try_from_bits(ask).ok_or("no such question")?;
            let (_, question, _, mut decision) = requested
                .asks
                .get_mut(entity)
                .map_err(|_| "no open question with that id")?;
            if let Some(choice) = choices.iter().find(|choice| {
                !question
                    .choices
                    .iter()
                    .any(|offered| &offered.id == *choice)
            }) {
                return Err(format!("the question offers no choice `{choice}`"));
            }
            if choices.is_empty() && !(text.is_some() && question.free_text) {
                return Err("the answer needs a choice".to_owned());
            }
            let answer = Answer { choices, text };
            decision.decide(Decider::Person, answer.outcome());
        }
        Request::State => return Ok(Some(requested.state())),
    }
    Ok(None)
}

#[cfg(test)]
mod tests;
