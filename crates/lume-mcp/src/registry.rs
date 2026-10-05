//! Tool registry and MCP tool collection.
//!
//! Tools are stored in the order the server returned them. Servers are told to
//! answer `tools/list` deterministically so LLM prompt caches hit, and that only
//! pays off if the order survives all the way into the chat request: sorting
//! would rewrite the tool block on every turn.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use lume_core::config::McpServerConfig;
use lume_core::error::{LumeError, Result};
use lume_core::tool::Tool;
use lume_core::types::{ToolCall, ToolSpec};

use crate::capabilities::ClientInfo;
use crate::client::McpClient;
use crate::results::{ContentType, ResultType};
use crate::transport::{HttpTransport, StdioTransport, Transport};

/// Registry of callable tools, kept in registration order.
pub struct ToolRegistry {
    tools: Vec<(String, Box<dyn Tool>)>,
}

impl ToolRegistry {
    /// Create new registry.
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Register a tool, replacing any tool of the same name in place.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.spec().name.clone();
        match self.tools.iter_mut().find(|(held, _)| *held == name) {
            Some(slot) => slot.1 = tool,
            None => self.tools.push((name, tool)),
        }
    }

    /// Register tools in the given order.
    pub fn register_many(&mut self, tools: impl IntoIterator<Item = Box<dyn Tool>>) {
        for tool in tools {
            self.register(tool);
        }
    }

    /// Get tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|(held, _)| held == name)
            .map(|(_, tool)| tool.as_ref())
    }

    /// Get tool names, in registration order.
    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|(name, _)| name.clone()).collect()
    }

    /// Get tool specs, in registration order.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|(_, tool)| tool.spec()).collect()
    }

    /// Get number of tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Dispatch a tool call.
    pub async fn dispatch(&self, call: &ToolCall) -> Result<String> {
        let tool = self
            .get(&call.name)
            .ok_or_else(|| LumeError::ToolNotFound(call.name.clone()))?;
        tool.call(call.arguments.clone()).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// MCP tool adapter.
pub struct McpToolAdapter {
    client: Arc<Mutex<McpClient>>,
    spec: ToolSpec,
}

impl McpToolAdapter {
    /// Create new adapter.
    pub fn new(client: Arc<Mutex<McpClient>>, spec: ToolSpec) -> Self {
        Self { client, spec }
    }

    /// The client behind this tool, needed to answer Multi Round-Trip Requests.
    pub fn client(&self) -> Arc<Mutex<McpClient>> {
        Arc::clone(&self.client)
    }
}

#[async_trait]
impl Tool for McpToolAdapter {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, args: serde_json::Value) -> Result<String> {
        let mut client = self.client.lock().await;
        let result = client.call_tool(&self.spec.name, args).await?;
        match result.result_type {
            ResultType::Complete => Ok(result
                .content
                .iter()
                .filter(|block| block.type_ == ContentType::Text)
                .filter_map(|block| block.text.clone())
                .collect::<Vec<_>>()
                .join("\n")),
            ResultType::InputRequired => Err(LumeError::Protocol(format!(
                "tool '{}' is waiting for input; answer it through McpClient::resume on the adapter client",
                self.spec.name
            ))),
        }
    }
}

/// Collect MCP tools from server configs.
pub async fn collect_mcp_tools(servers: &[McpServerConfig]) -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();
    for server in servers {
        let transport: Box<dyn Transport> = match server.transport {
            lume_core::config::McpTransport::Stdio => {
                let command = server
                    .command
                    .as_deref()
                    .ok_or_else(|| LumeError::Protocol("missing command".to_string()))?;
                Box::new(StdioTransport::spawn(command, &server.args, &server.env).await?)
            }
            lume_core::config::McpTransport::Http => {
                let url = server
                    .url
                    .as_deref()
                    .ok_or_else(|| LumeError::Protocol("missing url".to_string()))?;
                Box::new(HttpTransport::new(url))
            }
        };
        let mut client = McpClient::connect(transport);
        client.initialize(ClientInfo::lume()).await?;
        let specs = client.list_tools().await?;
        let client = Arc::new(Mutex::new(client));
        registry.register_many(
            specs.into_iter().map(|spec| {
                Box::new(McpToolAdapter::new(Arc::clone(&client), spec)) as Box<dyn Tool>
            }),
        );
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use lume_core::tool::Tool;
    use lume_core::types::{ToolCall, ToolSpec};

    use super::ToolRegistry;

    struct EchoTool {
        name: String,
    }

    impl EchoTool {
        fn boxed(name: &str) -> Box<dyn Tool> {
            Box::new(Self {
                name: name.to_string(),
            })
        }
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.clone(),
                description: format!("echo {}", self.name),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        async fn call(&self, args: serde_json::Value) -> lume_core::error::Result<String> {
            Ok(args.to_string())
        }
    }

    #[test]
    fn register_many_preserves_the_servers_order() {
        let mut registry = ToolRegistry::new();
        registry.register_many([
            EchoTool::boxed("zeta"),
            EchoTool::boxed("alpha"),
            EchoTool::boxed("mu"),
        ]);
        assert_eq!(registry.names(), ["zeta", "alpha", "mu"]);
        assert_eq!(
            registry
                .specs()
                .iter()
                .map(|spec| spec.name.clone())
                .collect::<Vec<_>>(),
            ["zeta", "alpha", "mu"]
        );
        assert_eq!(registry.len(), 3);
        assert!(!registry.is_empty());
    }

    #[test]
    fn reregistering_a_name_keeps_its_position() {
        let mut registry = ToolRegistry::new();
        registry.register_many([EchoTool::boxed("a"), EchoTool::boxed("b")]);
        registry.register(EchoTool::boxed("a"));
        assert_eq!(registry.names(), ["a", "b"]);
        assert_eq!(registry.len(), 2);
    }

    #[tokio::test]
    async fn dispatch_resolves_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register_many([EchoTool::boxed("b"), EchoTool::boxed("a")]);
        let call = ToolCall {
            id: "1".to_string(),
            name: "a".to_string(),
            arguments: serde_json::json!({"x": 1}),
        };
        assert_eq!(
            registry.dispatch(&call).await.expect("dispatch"),
            r#"{"x":1}"#
        );
        let missing = ToolCall {
            id: "2".to_string(),
            name: "nope".to_string(),
            arguments: serde_json::json!({}),
        };
        assert!(registry.dispatch(&missing).await.is_err());
    }
}
