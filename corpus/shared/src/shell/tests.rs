use super::reads_only;

#[test]
fn only_allowed_pipelines_pass() {
    for command in ["ls -a", "git status", "git diff | head", "cat file | wc -l"] {
        assert!(reads_only(command), "{command}");
    }
    for command in [
        "",
        " ",
        "git statusx",
        "git reset",
        "ls |",
        "ls; pwd",
        "cat < file",
        "ls > file",
        "ls & pwd",
        "echo $(pwd)",
        "cat $FILE",
        "ls\n pwd",
        "git diff --output=file",
        "rg --pre=command pattern",
    ] {
        assert!(!reads_only(command), "{command}");
    }
}
