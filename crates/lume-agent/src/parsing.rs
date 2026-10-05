//! Tool call parsing utilities.

use lume_core::types::ToolCall;
use serde_json::Value;

/// Parse tool calls from a model reply.
///
/// Strategy:
/// 1. If the whole trimmed reply parses as a JSON array, read each element's `name`/`arguments`
///    (accept `arguments` as either an object or a JSON string that itself parses).
/// 2. Otherwise scan for balanced `{...}` regions - respecting string literals and escapes
///    so a `}` inside a quoted string does not end the region early - and try to parse each.
/// 3. Give up and return empty.
pub fn parse_tool_calls(reply: &str) -> Vec<ToolCall> {
    let trimmed = reply.trim();
    if trimmed.is_empty() {
        return vec![];
    }

    // Try strategy 1: whole reply is a JSON array
    if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(trimmed) {
        let mut calls = Vec::new();
        for (i, elem) in arr.iter().enumerate() {
            if let Some(call) = parse_tool_call_elem(elem, i + 1) {
                calls.push(call);
            }
        }
        return calls;
    }

    // Try strategy 2: scan for balanced {...} regions
    let mut calls = Vec::new();
    let mut i = 0;
    let bytes = trimmed.as_bytes();
    let len = bytes.len();
    let mut call_num = 0;
    while i < len {
        let c = bytes[i] as char;
        if c == '{' {
            // Try to extract balanced brace region
            if let Some((start, end)) = extract_balanced_braces(trimmed, i) {
                let region = &trimmed[start..end];
                if let Ok(v) = serde_json::from_str::<Value>(region) {
                    call_num += 1;
                    if let Some(call) = parse_tool_call_elem(&v, call_num) {
                        calls.push(call);
                    }
                }
                i = end;
                continue;
            }
        }
        i += c.len_utf8();
    }

    calls
}

/// Strip tool calls from reply, returning prose content.
pub fn strip_tool_calls(reply: &str) -> String {
    let trimmed = reply;
    let mut result = String::with_capacity(trimmed.len());
    let mut i = 0;
    let bytes = trimmed.as_bytes();
    let len = bytes.len();
    let mut last_pos = 0;
    while i < len {
        let c = bytes[i] as char;
        if c == '{' {
            if let Some((start, end)) = extract_balanced_braces(trimmed, i) {
                // Check if this looks like a tool call by trying to parse
                let region = &trimmed[start..end];
                if let Ok(v) = serde_json::from_str::<Value>(region) {
                    // Looks like a JSON object - if it has name field, treat as tool call to strip
                    if v.get("name").is_some()
                        || v.get("tool").is_some()
                        || v.get("function").is_some()
                    {
                        // Append content before this region
                        result.push_str(&trimmed[last_pos..start]);
                        last_pos = end;
                        i = end;
                        continue;
                    }
                }
            }
        }
        i += c.len_utf8();
    }
    result.push_str(&trimmed[last_pos..]);
    result.trim().to_string()
}

/// True when the reply is nothing but a tool-call-shaped object that was refused, such
/// as the `{"name": "", "arguments": {}}` a 7b model emits when it loses the thread.
///
/// The tool-call keys are required so that a model answering with plain JSON, e.g.
/// `{"a": 1}`, is still treated as a real answer rather than a failed call.
pub fn is_rejected_tool_call_blob(reply: &str) -> bool {
    let unfenced = reply.replace("```json", "").replace("```", "");
    let trimmed = unfenced.trim();
    if !trimmed.starts_with('{') || !parse_tool_calls(trimmed).is_empty() {
        return false;
    }
    serde_json::from_str::<Value>(trimmed)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .is_some_and(|map| {
            map.contains_key("name") || map.contains_key("arguments") || map.contains_key("args")
        })
}

/// Parse a single tool call element from JSON value.
fn parse_tool_call_elem(elem: &Value, default_num: usize) -> Option<ToolCall> {
    // Try to get name
    let name = elem
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| elem.get("tool").and_then(|v| v.as_str()))
        .or_else(|| {
            elem.get("function")
                .and_then(|v| v.get("name").and_then(|n| n.as_str()))
        })
        .map(|s| s.to_string());

    let name = name?;
    // A blank name would reach dispatch and fail there with an opaque
    // "tool not found: ", hiding the fact that the model emitted nothing usable.
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    let name = name.to_string();

    // Get arguments
    let arguments = elem
        .get("arguments")
        .cloned()
        .or_else(|| elem.get("args").cloned())
        .or_else(|| {
            elem.get("function")
                .and_then(|v| v.get("arguments").cloned())
        })
        .unwrap_or(Value::Object(serde_json::Map::new()));

    // If arguments is a string, try to parse it as JSON
    let arguments = if let Value::String(s) = &arguments {
        serde_json::from_str::<Value>(s).unwrap_or(arguments.clone())
    } else {
        arguments
    };

    // Get id
    let id = elem
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("call_{default_num}"));

    Some(ToolCall {
        id,
        name,
        arguments,
    })
}

/// Extract balanced braces starting from position i (which should be '{').
/// Respects string literals and escapes.
fn extract_balanced_braces(s: &str, start: usize) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    if start >= bytes.len() || bytes[start] as char != '{' {
        return None;
    }
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;
    let mut i = start;
    let len = bytes.len();
    while i < len {
        let b = bytes[i];
        let c = b as char;
        if escape_next {
            escape_next = false;
            i += c.len_utf8();
            continue;
        }
        if in_string {
            if c == '\\' {
                escape_next = true;
            } else if c == '"' {
                in_string = false;
            }
            i += c.len_utf8();
            continue;
        }
        if c == '"' {
            in_string = true;
            i += c.len_utf8();
            continue;
        }
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                return Some((start, i + c.len_utf8()));
            }
        }
        i += c.len_utf8();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn clean_single_call() {
        let reply = r#"{"name": "read_file", "arguments": {"path": "src/main.rs"}}"#;
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, json!({"path": "src/main.rs"}));
    }

    #[test]
    fn array_of_two_calls() {
        let reply = r#"[
            {"name": "read_file", "arguments": {"path": "src/main.rs"}},
            {"name": "list_dir", "arguments": {"path": "."}}
        ]"#;
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[1].name, "list_dir");
    }

    #[test]
    fn call_wrapped_in_prose() {
        let reply = r#"Let me read the file.
{"name": "read_file", "arguments": {"path": "src/main.rs"}}
Done."#;
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
    }

    #[test]
    fn brace_inside_quoted_argument() {
        let reply = r#"{"name": "grep", "arguments": {"pattern": "fn foo() { ... }"}}"#;
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
    }

    #[test]
    fn malformed_json_yields_empty() {
        let reply = r#"{"name": "read_file", "arguments": {"path": "src/main.rs""#;
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn element_without_name_dropped() {
        let reply = r#"[
            {"arguments": {"path": "src/main.rs"}},
            {"name": "list_dir", "arguments": {"path": "."}}
        ]"#;
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "list_dir");
    }

    #[test]
    fn strip_tool_calls_removes_json_blocks() {
        let reply = r#"Let me look up the file.
{"name": "read_file", "arguments": {"path": "src/main.rs"}}
Here is what I found."#;
        let stripped = strip_tool_calls(reply);
        assert!(stripped.contains("Let me look up the file"));
        assert!(stripped.contains("Here is what I found"));
        assert!(!stripped.contains("read_file"));
    }

    #[test]
    fn blank_name_is_dropped() {
        let reply = r#"[
            {"name": "", "arguments": {"path": "a"}},
            {"name": "   ", "arguments": {"path": "b"}},
            {"name": "list_dir", "arguments": {"path": "."}}
        ]"#;
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "list_dir");
    }

    #[test]
    fn fenced_json_block_from_a_real_qwen_reply_is_parsed() {
        let reply = "```json\n{\n  \"name\": \"write_file\",\n  \"arguments\": {\n    \"path\": \"hello.txt\",\n    \"content\": \"hi\"\n  }\n}\n```";
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(
            calls[0].arguments,
            json!({"path": "hello.txt", "content": "hi"})
        );
        assert_eq!(calls[0].id, "call_1");
    }

    #[test]
    fn a_reply_that_is_only_prose_yields_no_calls() {
        let reply = "I could not do that. Here is a JSON sketch instead: {\"a\": 1}";
        assert!(parse_tool_calls(reply).is_empty());
    }

    #[test]
    fn arguments_encoded_as_a_json_string_are_decoded() {
        let reply = r#"{"name": "write_file", "arguments": "{\"file\": \"hello.txt\", \"content\": \"hi\"}"}"#;
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1, "parsed {calls:?}");
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(
            calls[0].arguments,
            json!({"file": "hello.txt", "content": "hi"})
        );
    }

    #[test]
    fn a_rejected_tool_call_blob_is_recognised() {
        // Exactly what qwen2.5-coder:7b emitted after a successful write.
        assert!(is_rejected_tool_call_blob(
            r#"{"name": "", "arguments": {}}"#
        ));
    }

    #[test]
    fn plain_json_is_still_a_real_answer() {
        assert!(!is_rejected_tool_call_blob(r#"{"a": 1}"#));
    }

    #[test]
    fn a_dispatchable_call_is_not_a_blob() {
        assert!(!is_rejected_tool_call_blob(
            r#"{"name": "list_dir", "arguments": {"path": "."}}"#
        ));
    }

    #[test]
    fn prose_is_not_a_blob() {
        assert!(!is_rejected_tool_call_blob("here is what I found"));
    }

    #[test]
    fn a_fenced_rejected_blob_is_recognised_too() {
        assert!(is_rejected_tool_call_blob(
            "```json\n{\"name\": \"\", \"arguments\": {}}\n```"
        ));
    }

    #[test]
    fn a_fenced_real_answer_survives() {
        assert!(!is_rejected_tool_call_blob("```json\n{\"a\": 1}\n```"));
    }
}
