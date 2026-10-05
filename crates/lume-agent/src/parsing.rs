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
}
