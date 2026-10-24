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

Off, it installs nothing: no transport wrapper, thread or system. On, it
wraps the models' HTTP transport and opens the log once the session store is
open; a run without a store writes nothing. The writer runs on a thread of
its own, so a request never waits on the file, and a failed write is warned
about once. On exit the plugin flushes what has ended.

The recorder and scrubber are rig-cassette's
[`traffic`](https://docs.rs/rig-cassette/latest/rig_cassette/traffic/index.html)
module, whose docs define the line (version `LINE_VERSION`). A line holds the
request as sent (method, URL, headers, body), the reply's status, headers and
body or timed stream events, the time to headers, first byte, last byte and
end, the error, and how the exchange ended. A request made for a model call
names its `effect`, `handler` and `provider`. Credentials are scrubbed with
the cassette engine's scrubber; the conversation stays in the log.

It sees what goes through the models' connector transport, as the request
reaches it: a wrapper added by a plugin listed before it is inside it. Other
HTTP clients, such as the ChatGPT sign-in, and websockets are not recorded.

## Reconstruct request bodies

A model call repeats its conversation, so a JSON request body is written
against an earlier one of the same handler and URL in the same run:
`request.after` names that exchange, `request.keep` maps each field kept
whole to `true` and each array to the number of leading items kept, and
`request.body` holds the rest. This rebuilds every body, reading the file
from the start (`jq -nc -f rebuild.jq traffic.jsonl`):

```jq
foreach inputs as $x ({};
  $x.request as $r
  | (if $r.after then .["\($r.after)"] else null end) as $base
  | (if $base then
       ($base | with_entries(select(.key as $k | ($r.keep | has($k)) or ($r.body | has($k))))) + $r.body
       | reduce ($r.keep | to_entries[]) as $e (.;
           .[$e.key] = if $e.value == true then $base[$e.key] else $base[$e.key][:$e.value] + $r.body[$e.key] end)
     else $r.body end) as $full
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
