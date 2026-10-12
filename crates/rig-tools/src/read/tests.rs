use rig_core::tool::{FileRead, FilesRead, LinesShown, Tool, ToolContext, content_hash};

use super::{Read, ReadArgs};
use crate::fs::resolve;

/// Reading a file publishes it by its resolved path, with the hash of the
/// bytes read and the lines shown; a failed read publishes nothing.
#[test]
fn a_read_publishes_the_file_it_read() {
    let read_tool = Read::default();
    let path = std::env::temp_dir().join(format!("rig-tools-read-{}.txt", std::process::id()));
    std::fs::write(&path, "one\ntwo\n").ok();
    let path = path.to_string_lossy().into_owned();

    let mut context = ToolContext::new();
    let read = Tool::call(&read_tool, &mut context, ReadArgs::new(&path).lines(2, 1));
    let shown = futures::executor::block_on(read).unwrap_or_default();
    assert!(shown.starts_with("     2\ttwo\n"), "{shown}");
    let published = context.result::<FilesRead>().ok().flatten();
    assert_eq!(
        published,
        Some(FilesRead(vec![
            FileRead::new(
                resolve(None, &path).to_string_lossy(),
                content_hash(b"one\ntwo\n")
            )
            .with_lines(LinesShown {
                first: 2,
                count: 1,
                total: 2
            })
        ]))
    );

    let mut context = ToolContext::new();
    let missing = Tool::call(
        &read_tool,
        &mut context,
        ReadArgs::new(format!("{path}.gone")),
    );
    assert!(futures::executor::block_on(missing).is_err());
    assert_eq!(context.result::<FilesRead>().ok().flatten(), None);
    std::fs::remove_file(path).ok();
}
