use std::path::Path;

use super::tool::self_extension;

#[test]
fn the_self_extension_section_names_this_builds_plugin_guide() {
    let guide = rig_harness::PLUGIN_GUIDE;
    assert!(Path::new(guide).is_file(), "{guide}");
    assert!(self_extension().contains(guide));
    assert_eq!(self_extension(), self_extension());
}
