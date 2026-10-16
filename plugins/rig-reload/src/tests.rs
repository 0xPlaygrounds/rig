use std::path::Path;

use super::tool::{SELF_EXTENSION, plugin_guide};

#[test]
fn the_self_extension_section_names_this_builds_plugin_guide() {
    assert!(Path::new(plugin_guide!()).is_file(), "{}", plugin_guide!());
    assert!(SELF_EXTENSION.contains(plugin_guide!()));
}
