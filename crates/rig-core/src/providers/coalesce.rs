//! Coalesce adjacent same-role messages in a provider's converted message list.
//!
//! Rig's agent loop can place an injected user turn (see [`crate::agent::inject`])
//! immediately after a tool-result user turn — and RAG documents and hoisted
//! system messages can likewise produce adjacent same-role turns. Some provider
//! APIs (notably AWS Bedrock's Converse API) require strictly alternating roles
//! and reject consecutive same-role messages; others combine or accept them. So
//! every provider runs the coalescing pass on its converted message list, where
//! tool-call role routing is already resolved, normalizing a run of same-role
//! turns into one valid turn regardless of the target API's leniency.
//!
//! Providers with a local message type implement [`CoalesceSameRole`] and call
//! [`coalesce_same_role`]; a provider crate whose message type is foreign (e.g.
//! `aws_bedrock::Message`, blocked by the orphan rule) uses the closure form
//! [`coalesce_same_role_with`].

/// A provider message that can absorb an adjacent same-role message. Implemented
/// on a provider's *native* message type (after rig→provider conversion), so the
/// role comparison reflects how that provider routes tool results.
pub trait CoalesceSameRole: Sized {
    /// Whether `next` is the same role as `self` and may be merged into it.
    fn can_coalesce(&self, next: &Self) -> bool;

    /// Merge `next`'s content onto the end of `self`'s content. Only called when
    /// [`can_coalesce`](Self::can_coalesce) returned `true`.
    fn coalesce(&mut self, next: Self);
}

/// Merge runs of adjacent same-role messages into single turns, for a message
/// type implementing [`CoalesceSameRole`]. A no-op when no two adjacent messages
/// share a role.
pub fn coalesce_same_role<T: CoalesceSameRole>(messages: Vec<T>) -> Vec<T> {
    coalesce_same_role_with(messages, T::can_coalesce, |mut acc, next| {
        acc.coalesce(next);
        acc
    })
}

/// Closure form of [`coalesce_same_role`] for a message type that cannot
/// implement [`CoalesceSameRole`] — e.g. a foreign SDK type (`aws_bedrock::Message`)
/// blocked by the orphan rule and its builder-only construction. `same_role`
/// decides whether two adjacent messages share a role; `merge(acc, next)` folds
/// `next` into the running turn and returns it.
pub fn coalesce_same_role_with<T>(
    messages: Vec<T>,
    same_role: impl Fn(&T, &T) -> bool,
    merge: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut out: Vec<T> = Vec::with_capacity(messages.len());
    for message in messages {
        match out.pop() {
            Some(last) if same_role(&last, &message) => out.push(merge(last, message)),
            Some(last) => {
                out.push(last);
                out.push(message);
            }
            None => out.push(message),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal role+content message for exercising the coalescing logic.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Msg {
        role: char,
        content: Vec<u32>,
    }

    fn msg(role: char, content: &[u32]) -> Msg {
        Msg {
            role,
            content: content.to_vec(),
        }
    }

    fn coalesce(messages: Vec<Msg>) -> Vec<Msg> {
        coalesce_same_role_with(
            messages,
            |a, b| a.role == b.role,
            |mut acc, mut next| {
                acc.content.append(&mut next.content);
                acc
            },
        )
    }

    #[test]
    fn alternating_history_is_unchanged() {
        let input = vec![msg('u', &[1]), msg('a', &[2]), msg('u', &[3])];
        assert_eq!(coalesce(input.clone()), input);
    }

    #[test]
    fn adjacent_same_role_turns_merge_in_order() {
        // The headline shape: assistant, then two consecutive user turns.
        let input = vec![msg('a', &[1]), msg('u', &[2]), msg('u', &[3])];
        assert_eq!(coalesce(input), vec![msg('a', &[1]), msg('u', &[2, 3])]);
    }

    #[test]
    fn a_run_of_three_same_role_collapses_to_one() {
        let input = vec![msg('u', &[1]), msg('u', &[2]), msg('u', &[3])];
        assert_eq!(coalesce(input), vec![msg('u', &[1, 2, 3])]);
    }

    #[test]
    fn empty_is_empty() {
        assert_eq!(coalesce(Vec::new()), Vec::new());
    }
}
