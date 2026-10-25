use super::reads_only;

#[test]
fn only_allowed_pipelines_pass() {
    for command in [
        "ls -a",
        "git status",
        "git  status",
        "ls\t-a",
        "git diff | head",
        "cat file | wc -l",
        "git log --pretty=oneline",
        "cat notes--preview.md",
        "rg --pre-glob=gz pattern",
        "ls *.rs",
        "ls src/*-test.rs",
        "rg 'fn-.*main' src",
        "rg '[a-z]+' src",
        "rg -g '*-test.rs' pattern",
        "rg 'a b' src",
    ] {
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
        "git diff --output file",
        "rg --pre=command pattern",
        "rg --pre command pattern",
        "git diff '--output=file'",
        "git diff \"--output\" file",
        "git diff --output\\=file",
        "rg --p're'=command pattern",
        "git diff --output*",
        "rg --pre{,}=command pattern",
        "git diff *--output*",
        "git diff *-output=file",
        "git diff [-]-output=file",
        "git diff {--output=file,x}",
        "rg --pr{e,e}=command pattern",
        "rg {-,-}-pre=command pattern",
        "rg {\"a b\",--pre=command} file",
        "rg '-'-pre{,}=command pattern",
        "rg 'unclosed pattern",
    ] {
        assert!(!reads_only(command), "{command}");
    }
}
