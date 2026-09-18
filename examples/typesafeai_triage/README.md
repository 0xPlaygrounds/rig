# Typed Jev triage

Requires `JEV_TOKEN` in the environment. From the workspace root:

```sh
cargo run -p typesafeai_triage
```

This sends a synthetic support ticket as a Serde struct and evaluates Choice,
Score, and Noul together. The selected route is a Rust enum. The example prints
distributions and confidence instead of hiding uncertainty behind that selection.
Score is an expected zero-based rubric index; Noul is a probability of yes.

The questions are independent: they share the ticket, not each other's answers.
See the [Jev primitives documentation](https://docs.typesafe.ai/primitives).

`Assessment<R, U, D>` declares its fields once. Its `Query` implementation maps
each field type to its associated output type. `AssessmentQuery::new()` supplies
explicit choice and score variants; no Jev derives are involved.

`client.evaluate(&ticket, AssessmentQuery::new()?).await?.answers` returns the
same named structure populated with typed answers.
