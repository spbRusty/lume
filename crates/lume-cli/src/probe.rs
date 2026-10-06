//! `lume mcp probe`: connect to configured MCP servers and really use them.
//!
//! A probe reads the server entries out of the configuration, spawns each one,
//! negotiates, lists the tools, calls one of them with arguments that cannot
//! change anything, and closes the connection. Every server is probed on its own,
//! so one that is missing, slow, or broken is reported and the others still run.

use std::time::Duration;

use lume_core::config::McpServerConfig;
use lume_core::error::{LumeError, Result};
use lume_core::types::ToolSpec;
use lume_mcp::{ContentType, McpServerConnection};
use serde_json::{Value, json};

/// Arguments for read-only tools that do take one, by tool name.
///
/// A tool whose schema declares no arguments needs no table entry; these are the
/// ones that need a path, where the only way to probe without touching anything
/// is to recognise the tool by name.
const KNOWN_READ_ONLY_ARGS: &[(&str, &str)] = &[
    ("list_directory", r#"{"path": "."}"#),
    ("list_directory_with_sizes", r#"{"path": "."}"#),
    ("list_files", r#"{"path": "."}"#),
    ("directory_tree", r#"{"path": "."}"#),
    ("git_status", "{}"),
];

/// What one tool call returned, as text.
pub struct ToolCallReport {
    /// Tool that was called.
    pub tool: String,
    /// Arguments it was called with.
    pub arguments: Value,
    /// The text content blocks of the result, joined by newlines.
    pub text: String,
}

/// What one server did when probed.
pub enum ServerProbe {
    /// Connected, listed tools, and called one.
    Called {
        /// Name the server is configured under.
        server: String,
        /// Revision negotiated with it.
        revision: String,
        /// Identity it reported.
        server_info: Option<String>,
        /// Tool names, in the server's order.
        tools: Vec<String>,
        /// The one call that was made.
        call: ToolCallReport,
    },
    /// Connected and listed tools, but had nothing safe to call.
    Uncalled {
        /// Name the server is configured under.
        server: String,
        /// Revision negotiated with it.
        revision: String,
        /// Identity it reported.
        server_info: Option<String>,
        /// Tool names, in the server's order.
        tools: Vec<String>,
        /// Why no call was made.
        reason: String,
    },
    /// Could not be used.
    Failed {
        /// Name the server is configured under.
        server: String,
        /// Why.
        reason: String,
    },
}

impl ServerProbe {
    /// Whether this server answered at all.
    pub fn connected(&self) -> bool {
        !matches!(self, Self::Failed { .. })
    }
}

/// Whether a tool takes no arguments, judged by the schema the server declared.
///
/// Both halves matter: `required` alone misses a tool whose schema lists
/// optional properties, and `properties` alone misses a tool that lists
/// `required` without declaring the properties. A tool that declares no schema at
/// all has nothing to fill in.
fn takes_no_arguments(spec: &ToolSpec) -> bool {
    let Some(schema) = spec.input_schema.as_object() else {
        return true;
    };
    let no_properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .is_none_or(|properties| properties.is_empty());
    let no_required = schema
        .get("required")
        .and_then(Value::as_array)
        .is_none_or(|required| required.is_empty());
    no_properties && no_required
}

/// Arguments that let a tool be called without changing anything.
///
/// `None` means the tool needs arguments a probe cannot invent: guessing a write
/// path, or a filter that could walk a whole tree, is worse than not calling it.
/// Both example servers offer a tool that takes nothing, so probing a working
/// configuration always makes a real call.
fn benign_arguments(spec: &ToolSpec) -> Option<Value> {
    if takes_no_arguments(spec) {
        return Some(json!({}));
    }
    KNOWN_READ_ONLY_ARGS
        .iter()
        .find(|(name, _)| *name == spec.name)
        .and_then(|(_, arguments)| serde_json::from_str(arguments).ok())
}

/// The first tool that can be called safely, and the arguments for it.
fn call_target(tools: &[ToolSpec]) -> Option<(String, Value)> {
    tools
        .iter()
        .find_map(|spec| benign_arguments(spec).map(|args| (spec.name.clone(), args)))
}

fn tool_names(tools: &[ToolSpec]) -> Vec<String> {
    tools.iter().map(|spec| spec.name.clone()).collect()
}

fn identity(connection: &McpServerConnection) -> Option<String> {
    connection
        .server_info
        .as_ref()
        .map(|info| format!("{} {}", info.name, info.version))
}

/// Call one tool on an open connection.
///
/// `Ok(None)` means the server offered nothing a probe may call.
async fn call_one(
    connection: &McpServerConnection,
    timeout: Duration,
) -> Result<Option<ToolCallReport>> {
    let Some((tool, arguments)) = call_target(&connection.tools) else {
        return Ok(None);
    };
    let client = connection.client();
    let mut client = client.lock().await;
    let result = within(timeout, client.call_tool(&tool, arguments.clone())).await?;
    let text = result
        .content
        .iter()
        .filter(|block| block.type_ == ContentType::Text)
        .filter_map(|block| block.text.clone())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Some(ToolCallReport {
        tool,
        arguments,
        text,
    }))
}

/// Probe one already-connected server: list its tools, then call one of them.
///
/// The connection is closed before returning, whatever the outcome, so a probe
/// never leaves a server process behind.
pub async fn probe_connected(connection: McpServerConnection, timeout: Duration) -> ServerProbe {
    let server = connection.server.clone();
    let revision = connection.revision.to_string();
    let server_info = identity(&connection);
    let tools = tool_names(&connection.tools);

    let called = call_one(&connection, timeout).await;
    let _ = connection.close().await;

    match called {
        Ok(Some(call)) => ServerProbe::Called {
            server,
            revision,
            server_info,
            tools,
            call,
        },
        Ok(None) => ServerProbe::Uncalled {
            server,
            revision,
            server_info,
            tools,
            reason: "it offers no tool that takes no arguments, and none of the \
                     read-only tools with known-safe arguments"
                .to_string(),
        },
        Err(err) => ServerProbe::Failed {
            server,
            reason: err.to_string(),
        },
    }
}

/// Probe every server in `servers`, or only the one named `only`.
///
/// Failures are reported per server, so a sweep is never aborted by one entry
/// that cannot be used. A `only` that matches nothing yields no reports, which
/// the caller reports as an unknown name.
pub async fn probe(
    servers: &[McpServerConfig],
    only: Option<&str>,
    timeout: Duration,
) -> Vec<ServerProbe> {
    let mut reports = Vec::new();
    for config in servers {
        if only.is_some_and(|only| only != config.name) {
            continue;
        }
        let server = config.name.clone();
        match lume_mcp::connect_server(config, timeout).await {
            Ok(connection) => reports.push(probe_connected(connection, timeout).await),
            Err(err) => reports.push(ServerProbe::Failed {
                server,
                reason: err.to_string(),
            }),
        }
    }
    reports
}

async fn within<T>(timeout: Duration, future: impl Future<Output = Result<T>>) -> Result<T> {
    tokio::time::timeout(timeout, future)
        .await
        .map_err(|_| LumeError::Protocol(format!("no answer within {timeout:?}")))?
}

#[cfg(test)]
mod tests {
    use lume_core::types::ToolSpec;
    use serde_json::{Value, json};

    use super::{KNOWN_READ_ONLY_ARGS, benign_arguments, call_target, takes_no_arguments};

    fn spec(name: &str, schema: Value) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: String::new(),
            input_schema: schema,
        }
    }

    #[test]
    fn an_empty_schema_declares_no_arguments() {
        assert!(takes_no_arguments(&spec("t", json!({"type": "object"}))));
        assert!(takes_no_arguments(&spec(
            "t",
            json!({"type": "object", "properties": {}})
        )));
        assert!(takes_no_arguments(&spec(
            "t",
            json!({"type": "object", "properties": {}, "required": []})
        )));
        assert!(takes_no_arguments(&spec("t", json!({}))));
    }

    #[test]
    fn an_optional_or_required_property_defeats_the_zero_argument_test() {
        assert!(!takes_no_arguments(&spec(
            "t",
            json!({"type": "object", "properties": {"path": {"type": "string"}}})
        )));
        assert!(!takes_no_arguments(&spec(
            "t",
            json!({"type": "object", "required": ["path"]})
        )));
    }

    #[test]
    fn a_tool_that_takes_nothing_is_called_with_an_empty_object() {
        let target = spec("list_allowed_directories", json!({"type": "object"}));
        assert_eq!(benign_arguments(&target), Some(json!({})));
    }

    #[test]
    fn a_read_only_tool_with_a_known_path_is_given_one() {
        for (name, arguments) in KNOWN_READ_ONLY_ARGS {
            let target = spec(
                name,
                json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
            );
            assert_eq!(
                benign_arguments(&target),
                Some(serde_json::from_str::<Value>(arguments).expect("preset is json")),
                "{name} must be callable"
            );
        }
    }

    #[test]
    fn a_write_tool_is_never_called_by_a_probe() {
        let write = spec(
            "write_file",
            json!({
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]
            }),
        );
        let search = spec(
            "search_files",
            json!({
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"]
            }),
        );
        assert_eq!(benign_arguments(&write), None);
        assert_eq!(benign_arguments(&search), None);
        assert_eq!(call_target(&[write, search]), None);
    }

    #[test]
    fn the_first_safe_tool_in_the_servers_order_is_the_one_called() {
        let tools = vec![
            spec(
                "write_file",
                json!({"type": "object", "required": ["path"]}),
            ),
            spec("read_file", json!({"type": "object", "required": ["path"]})),
            spec("list_allowed_directories", json!({"type": "object"})),
        ];
        assert_eq!(
            call_target(&tools),
            Some(("list_allowed_directories".to_string(), json!({})))
        );
    }
}
