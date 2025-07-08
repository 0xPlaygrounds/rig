# rig-wasm: Rig, but it's WASM!
WASM bindings to Rig.

## Implementation checklist
Providers:
  - [ ] OpenAI
  - [ ] Everyone else

Agents:
  - [x] Prompting
  - [x] Tool usage (and related TS definitions)
  - [x] Multi-turn
  - [ ] Dynamic context
  - [ ] Documents

Embedding:
  - [ ] Embed
  - [ ] Embeddings builder
  - [ ] JS side plugin adapter

## Folder architecture
- `examples`: A list of examples (entirely in TS) that use `rig-wasm`.
- `pkg`: The JavaScript package.
  - `generated`: The generated WASM files.
  - `types`: Typescript definitions. These are manually hand-written, so if there are any changes in the `rig-wasm` library (particularly if there are any changes to the unsafe extern C blocks) you must ensure that these are kept up to date. Additionally, some translated types require sub-definitions which will be contained here.
  - `*`: The rest of the code. Generally, this will contain things that are unable to be replicated from the Rust side - such as code that is unable to be translated directly through WASM (ie traits, etc...).
- `src`: The Rust code.
  - `providers`: A folder of LLM providers (a WASM mirror of `rig-core/src/providers`).
  - `completion`: All items related to completions.
  - `lib`: The entrypoint file. All exported file definitions *must* come through here, so ensure there are no naming conflicts!
  - `tool`: All items related to tools.
