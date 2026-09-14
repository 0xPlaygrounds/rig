fn main() {
    let _contextual_parent = rig_runtime_core::telemetry::completion_parent_span!(
        target: "telemetry_macro_consumer",
        name: "chat",
        operation: "chat",
        system_instructions: Option::<&str>::None,
    );
    let _explicit_parent = rig_runtime_core::telemetry::completion_parent_span!(
        target: "telemetry_macro_consumer",
        parent: None,
        name: "chat",
        operation: rig_runtime_core::telemetry::Empty,
        system_instructions: Option::<&str>::None,
        gen_ai.agent.name = "assistant",
    );
}
