# rig-bevy

Experimental ECS-native orchestration for Rig. Run state is represented by
focused Bevy ECS components and advanced through ordered schedules. Async model,
tool, and memory work crosses the world boundary as owned, correlated effects.

Enable it from the root facade with `rig`'s opt-in `bevy` feature and use
`rig::bevy`. It is independent of `rig-agent`, native-only, and not yet a
supported or default runtime. Local handles preserve concrete provider finals;
hosted/erased observation uses non-persisted diagnostic envelopes.
