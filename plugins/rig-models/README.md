# rig-models

Choosing a model, in two of the [rig coding agent](../../crates/rig-harness)'s
default plugins:

- `ModelsPlugin`: `-m`, `/model` and `/effort`; with the default `tui`
  feature, their pickers and both in the status line. Plugins add choices to
  `/model` with `ModelItem`s, and say what an agent's model is with a
  `ModelLabel`.
- `DefaultsPlugin`: a new session starts with the model and reasoning setting
  last chosen.
