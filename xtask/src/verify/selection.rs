//! The paths a change touches: the diff from the merge base with `--base`
//! (or `HEAD`), staged and unstaged, and untracked files.
use super::*;
use std::collections::BTreeSet;

pub(super) fn changes(root: &Path, opts: &Options) -> Result<BTreeSet<String>> {
    let revision = if let Some(base) = &opts.base {
        let merge = output(root, "git", &["merge-base", base, "HEAD"])?;
        println!("Base: {base}; merge base: {}", merge.trim());
        merge.trim().to_string()
    } else {
        "HEAD".into()
    };
    let mut paths = BTreeSet::new();
    // Union the index and working-tree deltas: a staged edit undone only in
    // the working tree still belongs in the plan. Include both sides of moves.
    for args in [
        vec!["diff", "--name-only", "--no-renames", "-z", &revision],
        vec![
            "diff",
            "--cached",
            "--name-only",
            "--no-renames",
            "-z",
            &revision,
        ],
    ] {
        paths.extend(execute::git_paths(root, &args)?);
    }
    paths.extend(execute::untracked_inputs(root)?);
    Ok(paths)
}
