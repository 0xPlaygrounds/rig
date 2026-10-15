use super::*;

#[test]
fn the_prompt_names_each_tool_as_the_model_calls_it() {
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct Args {}
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, ReloadPlugin))
        .add_open_tool(
            "browser_tool:open",
            "Opens a page.",
            ToolOptions::default(),
            |_: On<ToolCalled<Args>>| {},
        );
    app.update();
    let mut sections = app.world_mut().query::<&PromptSection>();
    let text = sections
        .iter(app.world())
        .find(|section| section.tag == "rig_harness")
        .map(|section| section.text.clone())
        .unwrap_or_default();
    assert!(text.contains("Tools: browser_tool:open."), "{text}");
}
