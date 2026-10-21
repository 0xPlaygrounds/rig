use super::*;

#[test]
fn editing_thinking_drops_its_signature_without_changing_signed_siblings() {
    for signature in ["sig_original", "reasoning_content"] {
        let unchanged = json!({"type": "thinking", "thinking": "first", "signature": "sig_first"});
        let mut edited = message::AssistantContent::reasoning("before")
            .with_native(json!({"type": "thinking", "thinking": "before", "signature": signature}));
        let message::AssistantContent::Reasoning(reasoning) = &mut edited else {
            panic!("reasoning constructor must produce reasoning");
        };
        reasoning.text = "after".into();
        let sent = assistant_wire(vec![
            message::AssistantContent::reasoning("first").with_native(unchanged.clone()),
            edited,
            message::AssistantContent::text("done"),
        ]);
        assert_eq!(
            sent,
            vec![
                unchanged,
                json!({"type": "text", "text": "after"}),
                json!({"type": "text", "text": "done"}),
            ]
        );
    }
}
