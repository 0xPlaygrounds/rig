//! The words a driver reads: `rig help-agents` ([`FIRST_SCREEN`] and its
//! [`topic`]s) and the usage of each command-line verb ([`VERBS`]) and
//! option ([`OPTIONS`]). They live here, std-only, so the launcher answers
//! `rig help-agents` and `rig help --allow` before any agent is built, and
//! `rig-cli` prints the same text.

/// One command-line verb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verb {
    pub name: &'static str,
    pub usage: &'static str,
    pub help: &'static str,
}

/// One option of the verbs, `--name`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Opt {
    /// Without the dashes.
    pub name: &'static str,
    /// The short form, such as `p` for `-p`.
    pub short: Option<&'static str>,
    /// The value it takes, if any, such as `RULE`.
    pub value: Option<&'static str>,
    pub help: &'static str,
    /// The verbs that take it.
    pub verbs: &'static [&'static str],
}

/// The first screen of `rig help-agents`: enough for common use.
pub const FIRST_SCREEN: &str = "\
rig for agents: start rig sessions, give them work, wait for their answers.

  id=$(rig start --cwd . \"fix the failing test\")  # prints the session id
  rig wait \"$id\"                # blocks; prints the final answer
  rig send \"$id\" \"also update the docs\"   # more work or a correction
  rig wait \"$id\"
  rig answer \"$id\" --allow      # after wait exits 3; --text \"...\" after 4
  rig log \"$id\"                 # the transcript; --follow streams it
  rig stop \"$id\"                # stops it and every helper it started
  rig agents --tree             # sessions, their state and their helpers
  rig -p \"question\"             # one shot: start, wait, print the answer

Ids are the session ids `rig start` prints. Every verb takes --json: one
JSON object on stdout with \"rig\":1, \"kind\", the ids, \"status\", \"next\".
stdout carries only the answer or the JSON; build output never mixes in.

Exit codes (wait, -p, answer, stop):
  0 answered    1 failed      2 usage        3 needs approval
  4 needs an answer           5 stopped      6 not allowed (already answered)
  7 rig server unavailable    124 timed out (run rig wait again)
  130 interrupted

Every error ends with a `next:` line: the command that fixes it.
stdin is read only for a prompt given as `-` (or `rig -p` with no prompt).

More: rig help-agents TOPIC, TOPIC one of
  approvals children events remote json exit-codes protocol python
  troubleshooting
One verb or option: rig help VERB, rig help --OPTION.
";

/// The topics of `rig help-agents TOPIC`, in the order the first screen
/// lists them.
pub const TOPICS: &[(&str, &str)] = &[
    ("approvals", APPROVALS),
    ("children", CHILDREN),
    ("events", EVENTS),
    ("remote", REMOTE),
    ("json", JSON),
    ("exit-codes", EXIT_CODES),
    ("protocol", PROTOCOL),
    ("python", PYTHON),
    ("troubleshooting", TROUBLESHOOTING),
];

/// The text of `rig help-agents name`.
pub fn topic(name: &str) -> Option<&'static str> {
    TOPICS
        .iter()
        .find(|(topic, _)| *topic == name)
        .map(|(_, text)| *text)
}

const APPROVALS: &str = "\
Approvals and questions

A session that wants something approved, or asks a question, opens an ask
and waits. `rig wait` returns at once with exit 3 (approval) or 4
(question), printing the ask and the command that answers it:

  needs approval: shell `rm -rf target`  [ask a-3, agent main]
  next: rig answer s-1 a-3 --allow

  rig answer S [ASK] --allow        allow it once
  rig answer S [ASK] --deny         refuse it
  rig answer S [ASK] --option ID    pick one of the ask's options
  rig answer S [ASK] --text \"...\"   answer a question

ASK defaults to the session's only open ask. The first answer wins; a
later one exits 6 (already answered). Then `rig wait S` again.

Answer up front with --allow on rig start or rig -p, repeatable:
  --allow TOOL          every call of that tool, such as --allow edit
  --allow shell:PREFIX  shell commands starting with PREFIX
  --allow edits:DIR     edits to files under DIR
and refuse a tool outright with --deny TOOL.

Nobody answers: the ask stays open for 10 minutes, then is denied
(ask.resolved by \"timeout\") and the session carries on with the denial.
A helper's asks go to its parent first; what the parent may not allow
itself goes on up to you.
";

const CHILDREN: &str = "\
Helpers (child sessions)

A rig agent's helpers are child sessions, the same as `rig start` ones.
The model starts them with start_agent, then uses message_agent,
wait_agent, answer_agent, stop_agent and list_agents. From outside,
`rig agents --tree` shows them, and `rig attach ID` or `rig log ID`
watches one; every verb works on a helper's id.

A helper works in its parent's folder (or its own cwd or host), with the
parent's tools and folders or fewer, never more: a request for more is
refused (exit 6, -32015). A helper with narrower folders is sandboxed
(Landlock on Linux, sandbox-exec on macOS); a grant the platform can't
enforce is refused (exit 6, -32017), never widened.

A helper's asks go to its parent, which may approve only what its own
--allow rules allow; anything wider comes to you: `rig wait PARENT` exits 3
and `rig answer CHILD ASK --allow` answers it. You may answer any helper's
ask. An ask nobody answers is denied after 600 s.

At most 8 helpers of a tree run at once (more wait in a queue), at most 4
levels deep (exit 6, -32019). `rig start --budget-tokens N` caps the whole
tree: once spent, every running turn stops and new messages are refused.
`rig stop PARENT` stops the whole tree, leaves first; history is kept.
With `[agents] view` in settings.toml naming a command that opens a window
(a terminal, a tmux split, any command), start_agent {view:true} runs
`rig attach ID` in one, a live view of the helper.
";

const EVENTS: &str = "\
Events

Everything a session does is an event in its journal, numbered by seq
(1, 2, 3, ... per session, never reused). `rig log S --json` prints them,
one JSON object per line; --after SEQ starts after one, --follow keeps
streaming until the session stops.

  {\"rig\":1,\"session\":\"s-1\",\"seq\":7,\"at_ms\":...,\"kind\":\"message\",\"data\":{...}}

Kinds: session.started, command.accepted, turn.started, message.delta,
message (a message's final text), call.started, call.progress, call.ended,
ask.opened, ask.resolved, turn.ended (status, answer), usage, notice,
status, child.started, child.ended, worker.ready, worker.exited,
session.idle, session.stopped, session.faulted, and x/<plugin>/<name> for
a plugin's own. Skip kinds you don't know.

A reader that keeps the last seq it saw resumes with --after SEQ and
misses nothing; after worker.exited, drop the partial text of a message
that had not ended: its final `message` follows.
";

const REMOTE: &str = "\
Remote hosts

A host named in [remote.hosts] of $RIG_HOME/settings.toml is reached
through its transport (ssh, koh exec, ...), which runs `rig connect
--stdio` there:

  [remote.hosts.box]
  transport = [\"ssh\", \"box\"]

  rig start --host box --cwd /srv/app \"run the tests\"
  rig wait --host box S

The remote needs rig on its PATH and its own sign-in. Paths are the
remote's. Authentication is the transport's (ssh keys); no port is opened.
`rig connect --stdio` alone bridges stdin and stdout to the local server,
for any program that speaks the protocol.
";

const JSON: &str = "\
--json

Every verb takes --json and then prints one JSON object on stdout (rig log
--follow and rig -p --json: one per line). Each carries \"rig\":1 and a
\"kind\", and where they apply \"sessionId\", \"status\" and \"next\":

  rig start --json \"...\"  {\"rig\":1,\"kind\":\"started\",\"sessionId\":\"s-1\",\"seq\":1,
                           \"next\":\"rig wait s-1\"}
  rig send --json S \"...\" {\"rig\":1,\"kind\":\"sent\",\"sessionId\":\"s-1\",
                           \"commandSeq\":2,\"disposition\":\"queued\",...}
  rig wait --json S       {\"rig\":1,\"kind\":\"wait\",\"sessionId\":\"s-1\",
                           \"status\":\"answered\",\"answer\":\"...\",\"seq\":40,...}
  rig agents --json       {\"rig\":1,\"kind\":\"agents\",\"sessions\":[...]}
  rig -p --json \"...\"     the session's events, then {\"kind\":\"result\",...}

status is answered, needs_approval, needs_answer, failed, stopped or
timed_out, matching the exit code. An error is
  {\"rig\":1,\"kind\":\"error\",\"code\":-32012,\"message\":\"...\",\"next\":\"...\"}
";

const EXIT_CODES: &str = "\
Exit codes

  0    answered: the answer is on stdout
  1    failed: the turn or the command failed; rig log S says why
  2    usage: a command line rig refuses; rig help VERB
  3    needs approval: rig answer S --allow (or --deny), then rig wait S
  4    needs an answer: rig answer S --text \"...\", then rig wait S
  5    stopped: someone stopped the turn or the session
  6    not allowed: already answered, or wider than the parent's grant
  7    unavailable: the rig server could not be reached or started
  124  timed out: still running; rig wait S again (--timeout, default 10m)
  130  interrupted (Ctrl+C)

rig wait, rig -p, rig answer and rig stop use them all; rig start and rig
send exit 0 once the work is accepted, not done.
";

const PROTOCOL: &str = "\
The protocol

Every front of rig (the terminal view, these verbs, editors, scripts) is a
client of one rig server per user, which starts by itself and exits when
idle. It speaks JSON-RPC 2.0, one JSON object per line, each frame with
\"rig\":1, over $RIG_HOME/run/rig.sock (or stdio through `rig connect
--stdio`). The first request is rig/hello with the token in
$RIG_HOME/run/token. Then rig/session/start, send, wait, answer, stop,
list, read, rig/events/subscribe and read. Mutating requests carry a
\"key\": repeating one with the same key is safe and returns the first
result. The full spec is docs/protocol.md in the rig repository.

`rig acp` speaks Zed's Agent Client Protocol v1 on stdio instead, for
editors.
";

const PYTHON: &str = "\
Python

clients/python/rig_client.py in the rig repository is a stdlib-only
client:

  from rig_client import Client
  c = Client()                         # the local server, started if needed
  s = c.call(\"rig/session/start\", cwd=\".\", prompt=\"hi\", key=\"k1\")
  print(c.wait(s[\"sessionId\"]))        # the wait result: status, answer
  for ev in c.events(s[\"sessionId\"], after=0):
      print(ev[\"seq\"], ev[\"kind\"])

examples/drive.py drives a session end to end.
";

const TROUBLESHOOTING: &str = "\
Troubleshooting

exit 7, server unavailable  rig server status; RIG_VERBOSE=1 shows why.
                            The first command after an install builds
                            the agent (minutes); later ones start at once.
wait exits 124              Still working: rig wait S again, or rig log S.
wait exits 3 or 4           It is asking: the printed next: line answers.
answer exits 6              Someone answered first; rig wait S.
nothing on stderr           By design when stderr is not a terminal. Set
                            RIG_VERBOSE=1 for build and progress lines.
a stuck session             rig stop S; rig log S shows its last events.
session.faulted             Its worker crashed 3 times in a minute; the
                            event names its log. rig send S restarts it.
wrong build                 rig build, then rig server stop: the next
                            command starts the server on the new build.
";

/// Every verb of the command line, in the order `rig help` lists them.
pub const VERBS: &[Verb] = &[
    Verb {
        name: "start",
        usage: "rig start [-m MODEL] [--name N] [--cwd DIR] [--host H] [--plugins FILE] [--allow RULE]... [--deny TOOL]... [--budget-tokens N] [--key K] [--json] [PROMPT... | -]",
        help: "Starts a session and prints its id. With a prompt, the prompt is accepted, not waited on: `rig wait ID` waits. A prompt of `-` is read from stdin; without one the session waits for `rig send`.",
    },
    Verb {
        name: "send",
        usage: "rig send ID [--queue | --interrupt] [--key K] [--json] TEXT... | -",
        help: "Sends a session a message and prints its command number and what happened to it: started, steered (it reaches a running turn at its next step) or queued (--queue: after the turn). --interrupt stops the running step first. A line starting with / runs that slash command.",
    },
    Verb {
        name: "wait",
        usage: "rig wait ID [--timeout DURATION] [--after N] [--json]",
        help: "Waits for the outcome of the latest message sent to the session (its prompt, or rig send; a stop or an answer only changes how that turn ends), or of command N with --after, and prints the answer, or the open ask and the command that answers it. Exits 0 answered, 1 failed, 3 needs approval, 4 needs an answer, 5 stopped, 124 timed out (default --timeout 10m; 0 waits forever).",
    },
    Verb {
        name: "answer",
        usage: "rig answer ID [ASK] (--allow | --deny | --option OPTION | --text TEXT) [--key K] [--json]",
        help: "Answers an open approval or question. ASK defaults to the session's only open one. Exits 6 when it was answered already.",
    },
    Verb {
        name: "log",
        usage: "rig log ID [--after SEQ] [--follow] [--json]",
        help: "Prints the session's transcript, or with --json its events, one per line. --follow keeps printing until the session stops.",
    },
    Verb {
        name: "agents",
        usage: "rig agents [--tree] [--all] [--json]",
        help: "Lists the sessions with their state, name and folder; --tree shows each session's helpers under it, --all includes stopped ones.",
    },
    Verb {
        name: "attach",
        usage: "rig attach ID [--takeover]",
        help: "Opens the terminal view on a session (by id or name), as its driver; --takeover takes over from the view or client driving it. Several views can share a session, and leaving the view never stops it. `rig -r ID` does the same.",
    },
    Verb {
        name: "stop",
        usage: "rig stop ID [--turn] [--key K] [--json]",
        help: "Stops the session and every helper it started, helpers first, and waits until they have stopped. --turn stops only the running turn.",
    },
    Verb {
        name: "-p",
        usage: "rig -p [-m MODEL] [--json] [--allow RULE]... [--deny TOOL]... [--name N] [--cwd DIR] [--host H] [--plugins FILE] [--timeout DURATION] [-r ID] PROMPT... | -",
        help: "One shot: starts a session (or sends to session ID with -r), waits for its answer and prints it on stdout, nothing else. Without a prompt, or with `-`, the prompt is read from stdin; otherwise stdin is never read. Exits as rig wait does, or 130 on Ctrl+C, which stops the turn. --json prints the session's events, one per line, then the result.",
    },
    Verb {
        name: "connect",
        usage: "rig connect --stdio",
        help: "Bridges stdin and stdout to the local rig server, starting it if needed: the transport end of a [remote.hosts] entry.",
    },
    Verb {
        name: "acp",
        usage: "rig acp",
        help: "Runs an Agent Client Protocol (v1) agent on stdin and stdout, for editors such as Zed.",
    },
    Verb {
        name: "serve",
        usage: "rig serve",
        help: "Runs the rig server in the foreground, for debugging. Normally the first command starts it and it exits when idle.",
    },
    Verb {
        name: "server",
        usage: "rig server (status | stop) [--json]",
        help: "status: whether the local rig server runs (exit 0) or not (exit 7), and its pid. stop: stops it; its sessions stop and can be resumed.",
    },
    Verb {
        name: "help-agents",
        usage: "rig help-agents [TOPIC]",
        help: "The guide for agents and scripts driving rig, and its topics.",
    },
];

/// Every option of the verbs.
pub const OPTIONS: &[Opt] = &[
    Opt {
        name: "json",
        short: None,
        value: None,
        help: "Print one JSON object on stdout instead of text (one per line for log --follow and -p): \"rig\":1, \"kind\", the ids, \"status\" and \"next\". `rig help-agents json` shows the shapes.",
        verbs: &[
            "start", "send", "wait", "answer", "log", "agents", "stop", "-p", "server",
        ],
    },
    Opt {
        name: "allow",
        short: None,
        value: Some("RULE"),
        help: "Answer approvals up front: TOOL (every call of that tool), shell:PREFIX (shell commands starting with PREFIX) or edits:DIR (edits under DIR). Repeatable. On rig answer, without a value: allow the open ask once.",
        verbs: &["start", "-p", "answer"],
    },
    Opt {
        name: "deny",
        short: None,
        value: Some("TOOL"),
        help: "On rig start and -p: refuse every call of TOOL without asking. On rig answer, without a value: refuse the open ask.",
        verbs: &["start", "-p", "answer"],
    },
    Opt {
        name: "option",
        short: None,
        value: Some("OPTION"),
        help: "Answer the ask with one of its options, by the id rig wait printed.",
        verbs: &["answer"],
    },
    Opt {
        name: "text",
        short: None,
        value: Some("TEXT"),
        help: "Answer a question with this text.",
        verbs: &["answer"],
    },
    Opt {
        name: "name",
        short: None,
        value: Some("NAME"),
        help: "A name for the new session, shown by rig agents; -r takes it in place of the id.",
        verbs: &["start", "-p"],
    },
    Opt {
        name: "cwd",
        short: None,
        value: Some("DIR"),
        help: "The folder the session works in (default: the current one).",
        verbs: &["start", "-p"],
    },
    Opt {
        name: "host",
        short: None,
        value: Some("HOST"),
        help: "Run on the rig server of a [remote.hosts] entry, reached through its transport. `rig help-agents remote`.",
        verbs: &[
            "start", "send", "wait", "answer", "log", "agents", "stop", "-p",
        ],
    },
    Opt {
        name: "model",
        short: Some("m"),
        value: Some("MODEL"),
        help: "The catalog model (vendor/model) the session's agent uses, set before the prompt as /model MODEL would; without it, the last one chosen.",
        verbs: &["start", "-p"],
    },
    Opt {
        name: "plugins",
        short: None,
        value: Some("FILE"),
        help: "Run the session on a build of this plugin list (a plugins.toml) instead of the default one; it is built once, then reused while the file is unchanged. Its helpers run on the same build.",
        verbs: &["start", "-p"],
    },
    Opt {
        name: "budget-tokens",
        short: None,
        value: Some("N"),
        help: "A token budget for the session and all its helpers: once spent, every turn in the tree stops.",
        verbs: &["start", "-p"],
    },
    Opt {
        name: "key",
        short: None,
        value: Some("KEY"),
        help: "The idempotency key: a command repeated with the same key is done once and answers as the first time. Default: a new random key.",
        verbs: &["start", "send", "answer", "stop"],
    },
    Opt {
        name: "queue",
        short: None,
        value: None,
        help: "Deliver the message after the running turn ends, instead of at its next step.",
        verbs: &["send"],
    },
    Opt {
        name: "interrupt",
        short: None,
        value: None,
        help: "Stop the running step, then deliver the message.",
        verbs: &["send"],
    },
    Opt {
        name: "timeout",
        short: None,
        value: Some("DURATION"),
        help: "How long to wait, such as 30s, 10m or 1h; plain numbers are seconds, 0 waits forever. Then exit 124: the work goes on, and rig wait waits again. Default 10m for wait, none for -p.",
        verbs: &["wait", "-p"],
    },
    Opt {
        name: "after",
        short: None,
        value: Some("N"),
        help: "rig wait: the outcome of command N (the commandSeq rig send printed) rather than the latest message's. rig log: only events after seq N.",
        verbs: &["wait", "log"],
    },
    Opt {
        name: "follow",
        short: Some("f"),
        value: None,
        help: "Keep printing events as they come, until the session stops.",
        verbs: &["log"],
    },
    Opt {
        name: "tree",
        short: None,
        value: None,
        help: "Show each session's helpers under it.",
        verbs: &["agents"],
    },
    Opt {
        name: "all",
        short: None,
        value: None,
        help: "Include stopped sessions.",
        verbs: &["agents"],
    },
    Opt {
        name: "takeover",
        short: None,
        value: None,
        help: "Attach as the session's driver, taking over from the one driving it.",
        verbs: &["attach"],
    },
    Opt {
        name: "turn",
        short: None,
        value: None,
        help: "Stop only the running turn; the session stays.",
        verbs: &["stop"],
    },
    Opt {
        name: "resume",
        short: Some("r"),
        value: Some("ID"),
        help: "With -p: send the prompt to this session (id or name) instead of starting one. Without -p: open the terminal view on that saved session.",
        verbs: &["-p"],
    },
    Opt {
        name: "print",
        short: Some("p"),
        value: None,
        help: "One shot: start a session, wait for its answer, print it. `rig help -p`.",
        verbs: &["-p"],
    },
    Opt {
        name: "stdio",
        short: None,
        value: None,
        help: "Bridge stdin and stdout to the local rig server.",
        verbs: &["connect"],
    },
];

/// The verb `name`, `-p`/`--print` included.
pub fn verb(name: &str) -> Option<&'static Verb> {
    let name = match name {
        "--print" | "print" => "-p",
        name => name,
    };
    VERBS.iter().find(|verb| verb.name == name)
}

/// The option written `word`: `--name`, `-s`, `name`, or `--name=value`.
pub fn option(word: &str) -> Option<&'static Opt> {
    let word = word.split_once('=').map_or(word, |(name, _)| name);
    if let Some(name) = word.strip_prefix("--") {
        return OPTIONS.iter().find(|option| option.name == name);
    }
    if let Some(short) = word.strip_prefix('-') {
        return OPTIONS.iter().find(|option| option.short == Some(short));
    }
    OPTIONS.iter().find(|option| option.name == word)
}

/// `rig help VERB`: its usage, what it does, and its options.
pub fn verb_help(verb: &Verb) -> String {
    let mut text = format!("Usage: {}\n\n{}\n", verb.usage, wrap(verb.help, 76, ""));
    let options: Vec<&Opt> = OPTIONS
        .iter()
        .filter(|option| option.verbs.contains(&verb.name))
        .collect();
    if !options.is_empty() {
        text.push_str("\nOptions:\n");
        for option in options {
            text.push_str(&option_line(option));
        }
    }
    text
}

/// `rig help --OPTION`: what it does and which verbs take it.
pub fn option_help(option: &Opt) -> String {
    format!(
        "{}Taken by: {}\n",
        option_line(option),
        option
            .verbs
            .iter()
            .map(|verb| format!("rig {verb}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn option_line(option: &Opt) -> String {
    let short = option.short.map_or(String::new(), |s| format!("-{s}, "));
    let value = option.value.map_or(String::new(), |v| format!(" {v}"));
    format!(
        "  {short}--{}{value}\n{}\n",
        option.name,
        wrap(option.help, 72, "      ")
    )
}

/// `text` wrapped at `width` columns, each line after `indent`.
fn wrap(text: &str, width: usize, indent: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            lines.push(format!("{indent}{line}"));
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(format!("{indent}{line}"));
    }
    lines.join("\n")
}
