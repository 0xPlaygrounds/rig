# rig-typesafeai

Experimental TypeSafe Jev judgments for Rig. Enable the root `rig` feature
`typesafeai` to use `rig::typesafeai`, or depend on this crate and bind a Rig HTTP
transport explicitly. It uses Rig's `Operation`, `Wire`, `Bound`, and shared driver.
There are no Jev derive macros or schema-generation traits.

## One structure for questions and answers

Implement `Query` on a generic named struct. Its associated `Output` replaces
question fields with their answer types while preserving their names and layout:

```rust
use rig_typesafeai::{Error, Noul, NoulAnswer, Query};
use serde::{Serialize, Deserialize};

// The fields are declared once, for both questions and answers.
#[derive(Serialize, Deserialize)]
struct Assessment<R, V> {
    ready: R,
    needs_review: V,
}

impl<R: Query, V: Query> Query for Assessment<R, V> {
    type Response = Assessment<R::Response, V::Response>;
    type Output = Assessment<R::Output, V::Output>;

    fn decode(&self, response: Self::Response) -> Result<Self::Output, Error> {
        Ok(Assessment {
            ready: self.ready.decode(response.ready)?,
            needs_review: self.needs_review.decode(response.needs_review)?,
        })
    }
}

let query = Assessment {
    ready: Noul::new("Is this ready to ship?")?,
    needs_review: Noul::new("Does this need human review?")?,
};
# Ok::<(), Error>(())
```

With a bound Jev client, evaluate it directly:

```rust,ignore
let answers: Assessment<NoulAnswer, NoulAnswer> =
    jev.evaluate(&state, &query).await?.answers;
```

The query is reusable by reference. `.await?` handles construction-independent
transport and response errors once; the result contains ordinary typed fields.
Serde field names and `rename` attributes supply question IDs. The request is
serialized directly, and Serde reconstructs the matching response struct before
`decode` validates each answer against its question. No manual map construction
or lookup is needed. Use `#[serde(flatten)]` to compose named groups.
See [triage](../../examples/typesafeai_triage/src/main.rs) for a complete example
using Choice, Score, and Noul in the same generic struct. Its query and answer
forms are concrete instantiations of one type, with no type erasure.

## Question builders

- `Choice::new(instructions, [(variant, description), ...])` returns
  `ChoiceAnswer<T>` on evaluation. Values implement `Serialize`,
  `DeserializeOwned`, and `Ord`; their Serde representation must be a string.
- `Score::new(instructions, [(level, description), ...])` returns
  `ScoreAnswer<L>` with enum-keyed probabilities. **Array order** determines
  numeric positions, independently of enum discriminants and `Ord`.
  Evaluation requires `L: Ord + Clone` and the WASM-compatible transport bounds.
- `DynamicScore::new(instructions, descriptions)` returns `DynamicScoreAnswer`
  with numeric indices and a legend when rubric identities are runtime-defined.
- `Noul::new(instructions)` returns `NoulAnswer`, the probability of yes.
  `.criteria(yes, no)?` accepts structured descriptions for both outcomes.

Fixed arrays check choice counts (2–255) and score counts (2–10) at compile time.
`try_from_iter` constructors check runtime collections and reject duplicate
labels or levels. Builders accept serializable descriptions, including structs,
JSON objects, arrays, strings, and null. State accepts strings, objects, or arrays.

For runtime IDs, maps of questions implement `Query`, and `.named("ready")?`
gives a standalone question an ID. `.join(other)` flattens two query objects;
`.map(...)` projects validated answers. Empty or duplicate IDs and missing or
unexpected response IDs are rejected at the evaluation boundary.
For a fully runtime-defined question set, use `DynamicQuery::new(definitions)?`.
Its output is explicitly `BTreeMap<String, types::Answer>`; construction validates
the definitions and evaluation validates the provider's answers.

`Jev::from_env()` reads `JEV_TOKEN` without loading a secrets file. Its default
model is `jev-latest`; `.model(...)` selects another provider model identifier.
Configure timeouts and retry middleware on the Rig transport.

Questions share state but cannot consume each other's answers in one request.
Confidence measures concentration, not correctness. Keep distributions and put
threshold policy in application code. Validation preserves the provider's values,
including bounded hundredth-rounding error; it does not silently renormalize them.

## Examples and verification

Run the interactive console from the repository root with `JEV_TOKEN` already
exported:

```sh
cargo run -p typesafeai_chat
```

Add `-- --agent` for `gpt-5.6-sol` replies using `OPENAI_API_KEY`. The default
mode uses Jev alone and displays decisions plus deterministic next actions.
Both modes support `/reset`, `/quit`, and conversational history.

- [Interactive chat](../../examples/typesafeai_chat): named generic query structs and
  batched judgments over conversation state.
- [Mixed triage](../../examples/typesafeai_triage): all three primitives in one call.

Run `cargo test -p rig-typesafeai` to replay the synthetic, live-recorded
[plain](fixtures/triage.json), [structured](fixtures/structured.json), and
[rounded matrix](fixtures/rounded.json) fixtures
offline, check local validation invariants, and run compile-fail doctests.
The fixtures contain no authentication headers. They were recorded against
`jev-1.13.0`; replay checks request JSON and decoded answers, not future model
accuracy or calibration. Mutation tests exercise invalid responses separately.

Protocol sources: [API](https://docs.typesafe.ai/api),
[Choice](https://docs.typesafe.ai/primitives/choice),
[Score](https://docs.typesafe.ai/primitives/score),
[Noul](https://docs.typesafe.ai/primitives/noul).
