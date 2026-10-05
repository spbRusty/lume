//! Qwen2.5 ChatML template rendering.

use lume_core::types::{Message, Role, ToolCall, ToolSpec};

/// Render messages in Qwen2.5 ChatML format.
pub fn render_chatml(system: Option<&str>, messages: &[Message], tools: &[ToolSpec]) -> String {
    let mut out = String::new();

    // System turn
    out.push_str("<|im_start|>system\n");
    if let Some(s) = system {
        out.push_str(s);
        if !s.ends_with('\n') {
            out.push('\n');
        }
    }
    if !tools.is_empty() {
        out.push_str("Available tools:\n");
        out.push_str(&serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_string()));
        out.push('\n');
    }
    out.push_str("<|im_end|>\n");

    for msg in messages {
        match msg.role {
            Role::System => {
                out.push_str("<|im_start|>system\n");
                out.push_str(&msg.content);
                if !msg.content.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("<|im_end|>\n");
            }
            Role::User => {
                out.push_str("<|im_start|>user\n");
                out.push_str(&msg.content);
                if !msg.content.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("<|im_end|>\n");
            }
            Role::Assistant => {
                out.push_str("<|im_start|>assistant\n");
                if !msg.tool_calls.is_empty() {
                    // Render tool calls as JSON
                    let calls: Vec<ToolCall> = msg.tool_calls.clone();
                    out.push_str(
                        &serde_json::to_string(&calls).unwrap_or_else(|_| "[]".to_string()),
                    );
                } else {
                    out.push_str(&msg.content);
                }
                if !msg.content.ends_with('\n') && msg.tool_calls.is_empty() {
                    out.push('\n');
                }
                if !msg.tool_calls.is_empty() && !out.ends_with('\n') {
                    // ensure newline before end marker context not required; but follow style
                }
                out.push_str("<|im_end|>\n");
            }
            Role::Tool => {
                out.push_str("<|im_start|>user\n");
                out.push_str("<tool_response>\n");
                out.push_str(&msg.content);
                if !msg.content.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("</tool_response>\n");
                out.push_str("<|im_end|>\n");
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lume_core::types::{Message, Role, ToolCall};
    use serde_json::json;

    #[test]
    fn system_only() {
        let rendered = render_chatml(Some("You are helpful."), &[], &[]);
        assert!(rendered.contains("<|im_start|>system\nYou are helpful.\n<|im_end|>"));
    }

    #[test]
    fn single_user_turn() {
        let msg = Message {
            role: Role::User,
            content: "Hello".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        };
        let rendered = render_chatml(None, &[msg], &[]);
        assert!(rendered.contains("<|im_start|>user\nHello\n<|im_end|>"));
    }

    #[test]
    fn assistant_with_one_tool_call() {
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "src/main.rs"}),
        };
        let msg = Message {
            role: Role::Assistant,
            content: "".to_string(),
            tool_calls: vec![call],
            tool_call_id: None,
            name: None,
        };
        let rendered = render_chatml(None, &[msg], &[]);
        assert!(rendered.contains("<|im_start|>assistant\n"));
        assert!(rendered.contains("read_file"));
    }

    #[test]
    fn one_tool_response() {
        let msg = Message {
            role: Role::Tool,
            content: "{\"ok\":true}".to_string(),
            tool_calls: vec![],
            tool_call_id: Some("call_1".to_string()),
            name: None,
        };
        let rendered = render_chatml(None, &[msg], &[]);
        assert!(rendered.contains("<tool_response>"));
        assert!(rendered.contains("</tool_response>"));
    }
}
