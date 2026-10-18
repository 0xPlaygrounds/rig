use super::*;

#[test]
fn listed_plugins_show_their_crate_description_and_what_the_build_runs() {
    let output = "on  rig_tui::TuiPlugin            rig-tui        The terminal view.\n\
                  off rig_inspect::InspectPlugin    rig-inspect    The inspect tool.\n\
                  on  viz::VizPlugin                viz\n\
                  not a line\n";
    let listed = parse(output);
    let shown: Vec<(bool, &str, &str, &str)> = listed
        .iter()
        .map(|listed| {
            let Listed {
                enabled,
                type_path,
                krate,
                description,
            } = listed;
            (
                *enabled,
                type_path.as_str(),
                krate.as_str(),
                description.as_str(),
            )
        })
        .collect();
    assert_eq!(
        shown,
        [
            (true, "rig_tui::TuiPlugin", "rig-tui", "The terminal view."),
            (
                false,
                "rig_inspect::InspectPlugin",
                "rig-inspect",
                "The inspect tool."
            ),
            (true, "viz::VizPlugin", "viz", ""),
        ]
    );
    // This build runs the terminal view and inspect, not viz.
    let running = [
        "rig_tui::TuiPlugin".to_owned(),
        "rig_inspect::InspectPlugin".to_owned(),
    ];
    let details: Vec<String> = listed.iter().map(|l| detail(l, &running)).collect();
    assert_eq!(
        details,
        [
            "(rig-tui) The terminal view. [warning: disabling it takes away the terminal \
             view, leaving only -p and piped prompts]",
            "(rig-inspect) The inspect tool. [disabled, runs until the next reload]",
            "(viz) [enabled, takes effect at the next reload]",
        ]
    );
}
