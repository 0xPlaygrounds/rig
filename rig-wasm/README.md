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
  - [x] Dynamic context
  - [x] Documents
  - [x] Options builder (for idiomatic DX)

Embedding:
  - [ ] Embed
  - [ ] Embeddings builder
  - [ ] JS side plugin adapter

## Folder architecture
- `examples`: A list of examples (entirely in TS) that use `rig-wasm`.
- `pkg`: The JavaScript package.
  - `types`: Typescript definitions. These are manually hand-written, so if there are any changes in the `rig-wasm` library (particularly if there are any changes to the unsafe extern C blocks) you must ensure that these are kept up to date. Additionally, some translated types require sub-definitions which will be contained here.
  - `src`: Some handwritten code. Primarily used for re-exporting and each file also serves as a module entrypoint for module hygiene.
- `src`: The Rust code.
- `build.rs`: A build script. Primarily used to regenerate all providers based on the contents of `rig-core/src/providers` as their `wasm-bindgen` counterparts.

## Development
- Use `just build-wasm` (which will activate the relevant command from the core justfile at the top level of the project).
- Go into `rig-wasm/pkg` and use `npm run build` which will run the Rollup pipeline as well as copying some WASM files over.
- Try some of the examples! Or do some development work.
