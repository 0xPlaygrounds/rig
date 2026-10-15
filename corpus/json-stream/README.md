# corpus-json-stream

`rig --json`: the agent's front for scripts and editors. Every event goes to
stdout as one JSON line, and commands come in as JSON lines on stdin.

- `rig --json -p "task"` is the harness's headless run: it sends the task,
  prints its events and exits once no agent works: 0 when the turn was
  answered, 1 otherwise (2 for an empty prompt or a refused slash command).
  Nobody answers questions in this run, so plugins apply their own policy,
  such as refusing.
- `rig --json --rpc` (or `rig --json` without `-p`) stays running and reads
  commands from stdin until it ends. With `-p`, the prompt is its first
  command, `{"type":"prompt","id":"print",…}`.

Nothing else goes to stdout; failures that end the run go to stderr.

## Schema, version 1

Every line is an object with `v` (1), `type`, `agent` (the agent's
`AgentId`, `null` for lines about the whole app) and, for a subagent,
`parent` (the `AgentId` of the agent that spawned it).

Events:

| `type` | fields |
|---|---|
| `session_start` | `agents: [{agent, parent, name}]` |
| `turn_start` | |
| `text_delta` | `text` |
| `reasoning_delta` | `text` |
| `tool_call_start` | `id`, `name`, `args` |
| `tool_call_end` | `id`, `name`, `result`, `is_error` |
| `notice` | `level` (`info` or `error`), `text` |
| `usage` | `usage` (the model call's token counts) |
| `turn_end` | `outcome` (`answered`, `failed` or `stopped`), `text` or `error` |
| `request` | `request_id`, `kind` (`ask` or `picker`), `title`, `body`, `choices: [label]`, `free_text` |
| `response` | `id` (the command's), `ok`, `error` when not ok |

A `request` is a question another plugin asks (an approval, a choice) or a
picker it opens, such as `/model`'s. It waits for an `answer`.

Commands, each answered by a `response` with its `id`; without `agent` a
command goes to the agent shown (the primary one unless a front shows
another):

```json
{"type":"prompt","id":1,"agent":"…","text":"fix the tests"}
{"type":"interrupt","id":2}
{"type":"set_model","id":3,"model":"deepseek/deepseek-flash"}
{"type":"answer","id":4,"request_id":4294967301,"choice":0}
{"type":"answer","id":5,"request_id":4294967302,"text":"a line of text"}
```

`choice` is the position of the chosen item; `text` answers a request whose
`free_text` is true.
