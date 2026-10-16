# rig shell companion (bash): `?? [question]` asks the project's rig session
# about the last command, `?? --cmd <what>` asks it for a command, Alt-?
# asks why the last command failed. Run a command as `rig-run <command>` to
# send its output's tail too.
__rig_state="${RIG_SHELL_STATE_DIR:-${XDG_RUNTIME_DIR:-$HOME/.cache}/rig-shell/$$}"
mkdir -p "$__rig_state" && chmod 700 "$__rig_state"
__rig_record() {
  printf '%s\n' "$2" > "$__rig_state/command"
  printf '%s\n' "$1" > "$__rig_state/exit"
  pwd > "$__rig_state/cwd"
}
__rig_precmd() {
  local status=$? cmd
  cmd=$(HISTTIMEFORMAT= builtin history 1 | sed 's/^ *[0-9]*[* ]*//')
  case "$cmd" in
    '??'* | rig-run* | __rig* | '') ;;
    *) __rig_record "$status" "$cmd"; : > "$__rig_state/output" ;;
  esac
  return $status
}
PROMPT_COMMAND="__rig_precmd${PROMPT_COMMAND:+; $PROMPT_COMMAND}"
rig-run() {
  "$@" 2>&1 | tee "$__rig_state/output.full"
  local status=${PIPESTATUS[0]}
  tail -n 200 "$__rig_state/output.full" > "$__rig_state/output"
  rm -f "$__rig_state/output.full"
  __rig_record "$status" "$*"
  return "$status"
}
__rig_ask() {
  rm -f "$__rig_state/suggested"
  "${RIG_BIN:-rig}" shell-ask --state "$__rig_state" "$@"
  local status=$?
  if [ -s "$__rig_state/suggested" ]; then
    local suggested
    suggested=$(< "$__rig_state/suggested")
    if [ -n "${READLINE_POINT+x}" ]; then
      READLINE_LINE=$suggested
      READLINE_POINT=${#suggested}
    else
      builtin history -s "$suggested"
      echo "(press Up to edit it)"
    fi
  fi
  return $status
}
alias '??'='__rig_ask'
bind -x '"\e?": __rig_ask' 2> /dev/null
