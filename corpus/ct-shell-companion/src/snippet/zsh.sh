# rig shell companion (zsh): `?? [question]` asks the project's rig session
# about the last command, `?? --cmd <what>` puts the command it suggests in
# the edit buffer, Alt-? asks why the last command failed. Run a command as
# `rig-run <command>` to send its output's tail too.
__rig_state="${RIG_SHELL_STATE_DIR:-${XDG_RUNTIME_DIR:-$HOME/.cache}/rig-shell/$$}"
mkdir -p "$__rig_state" && chmod 700 "$__rig_state"
__rig_record() {
  print -r -- "$2" > "$__rig_state/command"
  print -r -- "$1" > "$__rig_state/exit"
  pwd > "$__rig_state/cwd"
}
__rig_preexec() { __rig_last=$1 }
__rig_precmd() {
  local status=$?
  case "$__rig_last" in
    '??'* | rig-run* | __rig* | '') ;;
    *) __rig_record $status "$__rig_last"; : > "$__rig_state/output" ;;
  esac
  __rig_last=
}
autoload -Uz add-zsh-hook
add-zsh-hook preexec __rig_preexec
add-zsh-hook precmd __rig_precmd
rig-run() {
  "$@" 2>&1 | tee "$__rig_state/output.full"
  local status=${pipestatus[1]}
  tail -n 200 "$__rig_state/output.full" > "$__rig_state/output"
  rm -f "$__rig_state/output.full"
  __rig_record $status "$*"
  return $status
}
__rig_ask() {
  rm -f "$__rig_state/suggested"
  "${RIG_BIN:-rig}" shell-ask --state "$__rig_state" "$@"
  local status=$?
  [[ -s "$__rig_state/suggested" ]] && print -z -- "$(< "$__rig_state/suggested")"
  return $status
}
alias '??'='noglob __rig_ask'
__rig_ask_widget() { zle -I; __rig_ask; zle reset-prompt }
zle -N __rig_ask_widget
bindkey '^[?' __rig_ask_widget
