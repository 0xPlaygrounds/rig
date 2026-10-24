# rig-traffic

`rig_traffic::TrafficPlugin` records provider HTTP exchanges in the session
store's `traffic.jsonl`, beside `effects.jsonl`. The default disk store puts
it at `$RIG_HOME/sessions/<id>/files/traffic.jsonl`. It appends across restarts.

The launcher lists this plugin enabled, but recording is off. Turn it on for
one run with `rig --traffic-log`, or for every run in
`$RIG_HOME/settings.toml`:

```toml
[traffic]
log = true
```

An existing plugin list that predates this entry can add it with:

```sh
rig plugin add rig_traffic::TrafficPlugin
```

The plugin always declares its flag and settings. When off it installs no
transport wrapper, writer thread or systems. When on it registers a wrapper
of `Models` during plugin build and opens the writer in `finish`, after the
harness opens the session store. Without a store it writes nothing. Models
must connect after the store opens. Already connected models are unchanged.

The recorder and scrubber are
[`rig_cassette::traffic`](https://docs.rs/rig-cassette/latest/rig_cassette/traffic/index.html).
Each finished exchange becomes one versioned JSON line: the request's method,
URL, headers and body; the response's status, headers and body or timed
stream events; time to headers, first byte, last byte and end; any error; and
whether it completed, failed or was stopped when its reader dropped it.
The line also names its exchange, process and start time, and, when the
request carries `Serving`, its effect, handler and provider. The module docs
and `LINE_VERSION` define the format. Request-body deltas can refer to a
previous exchange with `request.after`. In `request.keep`, `true` retains a
whole field and a number retains that many leading array items; `body` holds
the changed fields and array tails. Reconstruct against the preceding body
of the same handler and URL, in file order, before treating it as a complete
request.

Credentials are scrubbed by the cassette engine before writing. Conversation
content remains in the log. The writer runs on a separate thread; write
failures warn once and do not fail or block model requests. On exit the
plugin flushes finished exchanges in `Last`, in `WriteLogs`, after turns
stop. The store's append operation does not promise an fsync.

The wrapper sees requests as they reach it. Wrappers registered earlier are
inside it, so their later transformations are not visible. Only clients
using the models' connector transport are recorded; independently created
HTTP clients and websocket connections are outside this plugin's coverage.

## Reconstruct request bodies

Save this as `rebuild.jq` and run `jq -nc -f rebuild.jq traffic.jsonl`.
It emits exchanges with full request bodies, removing the delta fields.
Read the file from its beginning so each base is available.

```jq
foreach inputs as $x ({};
  $x.request as $r
  | (if $r.after then .["\($r.after)"] else null end) as $base
  | (if $base then
       ($base | with_entries(select(.key as $k | ($r.keep | has($k)) or ($r.body | has($k))))) + $r.body
       | reduce ($r.keep | to_entries[]) as $e (.;
           .[$e.key] = if $e.value == true then $base[$e.key] else $base[$e.key][:$e.value] + $r.body[$e.key] end)
     else $r.body end) as $full
  | (if $r.after then del(.["\($r.after)"]) else . end)
  | .["\($x.exchange)"] = $full;
  .["\($x.exchange)"] as $full | $x | .request.body = $full | del(.request.after, .request.keep))
```

## Join exchanges to agents and turns

`effects.jsonl` is written by
`rig_basics::telemetry::effect_log::EffectLogPlugin`, which the default
`plugins.toml` lists disabled. Enable it before recording the session:

```sh
rig plugin enable rig_basics::telemetry::effect_log::EffectLogPlugin
```

Traffic's `effect` joins to an effect record's `id`; `handler` matches its
`key`. The effect's `scope` identifies the agent and `turn` is its zero-based
turn index. Some exchanges, such as authentication calls, have no effect.
Several exchanges can serve one effect. Keep effect logging enabled across
resumes so effect identifiers continue from the previous run.

From the session's `files/` directory, this reports the ten exchanges whose
agents waited longest for the first response byte, with their provider and
turn. First byte can be an SSE control event; it is not necessarily the first
token. Effect logs contain header lines, plain records and continuation
lines with a nested `record`. The query handles all three without needing to
rebuild conversation histories:

```sh
jq -n --slurpfile effects effects.jsonl '
  ($effects
   | map(select(.id != null) | . as $line
         | (.record // .) + {id: $line.id})
   | INDEX(.id | tostring)) as $by_id
  | [inputs
     | select(.effect != null and .timings.first_byte_ms != null)
     | . as $traffic
     | $by_id[(.effect | tostring)] as $effect
     | select($effect != null and $effect.key == $traffic.handler)
     | {provider, effect, agent: $effect.scope, turn: $effect.turn,
        first_byte_ms: .timings.first_byte_ms}]
  | sort_by(.first_byte_ms) | reverse | .[:10]
' traffic.jsonl
```
