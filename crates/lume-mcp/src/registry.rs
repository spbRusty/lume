//! Tool registry and MCP tool collection.
//!
//! Tools are stored in the order the server returned them. Servers are told to
//! answer `tools/list` deterministically so LLM prompt caches hit, and that only
//! pays off if the order survives all the way into the chat request: sorting
//! would rewrite the tool block on every turn.
//!
//! A tool name belongs to exactly one provider. Builtins win: an MCP tool whose
//! name a builtin already holds is skipped and reported through
//! [`ToolRegistry::register_server`] rather than silently shadowing, because a
//! silently shadowed builtin hands a model a tool whose behaviour it was never
//! told about.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;

use lume_core::config::{McpServerConfig, McpTransport};
use lume_core::error::{LumeError, Result};
use lume_core::tool::Tool;
use lume_core::types::{ToolCall, ToolSpec};

use crate::capabilities::{ClientInfo, ServerInfo};
use crate::client::McpClient;
use crate::negotiation::ProtocolRevision;
use crate::results::{ContentType, ResultType};
use crate::transport::{HttpTransport, StdioTransport, Transport};

/// How long one phase of a server's handshake may take before it is given up on.
///
/// Generous, because the first `tools/list` of an `npx`-spawned server includes
/// downloading the package.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(120);

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

    /// Register one server's tools, skipping every name the registry already holds.
    ///
    /// The tool that got there first keeps the name, so registering the builtins
    /// first is what makes them win a collision with an MCP server. Skipped names
    /// come back in [`ServerRegistration::collisions`] to be reported: the caller
    /// has to say which tool a model will actually get, and silently dropping the
    /// MCP tool would hide that the server offers something unreachable.
    pub fn register_server(&mut self, connection: &McpServerConnection) -> ServerRegistration {
        let mut registration = ServerRegistration::default();
        for spec in connection.tools.clone() {
            if self.contains(&spec.name) {
                registration.collisions.push(NameCollision {
                    name: spec.name.clone(),
                    server: connection.server.clone(),
                });
                continue;
            }
            registration.registered.push(spec.name.clone());
            self.register(Box::new(McpToolAdapter::new(connection.client(), spec)));
        }
        registration
    }

    /// Whether a tool is registered under `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.iter().any(|(held, _)| held == name)
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

/// A live connection to one MCP server, together with its tool list.
///
/// Dropping it does not close the connection, because the registry holds the same
/// client behind an `Arc`; call [`McpServerConnection::close`] at the end of a run
/// so the server's process is killed rather than left orphaned.
pub struct McpServerConnection {
    /// Name the server is configured under.
    pub server: String,
    /// Revision negotiated with it.
    pub revision: ProtocolRevision,
    /// Identity the server reported, if it reported one.
    pub server_info: Option<ServerInfo>,
    /// Tools it exposes, in its own order.
    pub tools: Vec<ToolSpec>,
    client: Arc<Mutex<McpClient>>,
}

impl McpServerConnection {
    /// The client, for calls the registry does not make.
    pub fn client(&self) -> Arc<Mutex<McpClient>> {
        Arc::clone(&self.client)
    }

    /// Close the connection and kill the server process.
    pub async fn close(&self) -> Result<()> {
        self.client.lock().await.close().await
    }
}

/// A configured server that could not be connected to.
///
/// Collected instead of returned, because one broken server must not cost the
/// model every other server's tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerFailure {
    /// Name the server is configured under.
    pub server: String,
    /// Why connecting failed.
    pub reason: String,
}

impl ServerFailure {
    fn new(server: &McpServerConfig, reason: impl std::fmt::Display) -> Self {
        Self {
            server: server.name.clone(),
            reason: reason.to_string(),
        }
    }
}

/// A tool name an MCP server offers that the registry already holds.
///
/// The incumbent is not named: a [`ToolRegistry`] keys tools by name alone, so it
/// cannot say whether a builtin or an earlier MCP server got there first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameCollision {
    /// The contested name.
    pub name: String,
    /// The server that offered it.
    pub server: String,
}

impl std::fmt::Display for NameCollision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MCP server {:?} offers {:?}, which is already registered; \
             the registered tool keeps the name",
            self.server, self.name
        )
    }
}

/// What registering one server's tools did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerRegistration {
    /// Names registered, in the server's order.
    pub registered: Vec<String>,
    /// Names skipped because something already held them.
    pub collisions: Vec<NameCollision>,
}

/// The transport a server config asks for, spawned but not yet handshaken.
pub async fn transport_for(server: &McpServerConfig) -> Result<Box<dyn Transport>> {
    match server.transport {
        McpTransport::Stdio => {
            let command = server
                .command
                .as_deref()
                .ok_or_else(|| LumeError::Protocol("missing command".to_string()))?;
            Ok(Box::new(
                StdioTransport::spawn(command, &server.args, &server.env).await?,
            ))
        }
        McpTransport::Http => {
            let url = server
                .url
                .as_deref()
                .ok_or_else(|| LumeError::Protocol("missing url".to_string()))?;
            Ok(Box::new(HttpTransport::new(url)))
        }
    }
}

/// Connect to one server, negotiate, and list its tools.
///
/// `timeout` bounds each phase separately, so a server that answers the handshake
/// and then goes quiet costs `timeout`, not several times it. On failure the
/// client is closed before the error is returned, so a spawned server cannot
/// outlive the attempt to reach it.
pub async fn connect_server(
    server: &McpServerConfig,
    timeout: Duration,
) -> Result<McpServerConnection> {
    let transport = within(timeout, transport_for(server)).await?;
    let mut client = McpClient::connect(transport);
    let connected = async {
        let session = within(timeout, client.initialize(ClientInfo::lume())).await?;
        let tools = within(timeout, client.list_tools()).await?;
        Ok::<_, LumeError>((session, tools))
    }
    .await;
    match connected {
        Ok((session, tools)) => Ok(McpServerConnection {
            server: server.name.clone(),
            revision: client.revision(),
            server_info: Some(session.server_info),
            tools,
            client: Arc::new(Mutex::new(client)),
        }),
        Err(err) => {
            client.close().await?;
            Err(err)
        }
    }
}

/// Connect to every configured server, keeping per-server failures aside.
///
/// Returns the connections that came up and the failures that did not, so the
/// caller can use every reachable server and report the rest.
pub async fn connect_servers(
    servers: &[McpServerConfig],
    timeout: Duration,
) -> (Vec<McpServerConnection>, Vec<ServerFailure>) {
    let mut connections = Vec::new();
    let mut failures = Vec::new();
    for server in servers {
        match connect_server(server, timeout).await {
            Ok(connection) => connections.push(connection),
            Err(err) => failures.push(ServerFailure::new(server, err)),
        }
    }
    (connections, failures)
}

/// Run `future`, failing if it outlives `timeout`.
async fn within<T>(timeout: Duration, future: impl Future<Output = Result<T>>) -> Result<T> {
    tokio::time::timeout(timeout, future)
        .await
        .map_err(|_| LumeError::Protocol(format!("no answer within {timeout:?}")))?
}

/// Collect MCP tools from server configs, failing on the first server that does.
///
/// Prefer [`connect_servers`], which survives a broken server.
pub async fn collect_mcp_tools(servers: &[McpServerConfig]) -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::new();
    for server in servers {
        let connection = connect_server(server, DEFAULT_CONNECT_TIMEOUT).await?;
        registry.register_server(&connection);
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use serde_json::{Value, json};
    use tokio::sync::Mutex;

    use lume_core::config::{McpServerConfig, McpTransport};
    use lume_core::error::{LumeError, Result};
    use lume_core::tool::Tool;
    use lume_core::types::{ToolCall, ToolSpec};

    use super::{
        McpServerConnection, NameCollision, ToolRegistry, connect_servers, transport_for, within,
    };
    use crate::client::{CALL_TOOL_METHOD, McpClient};
    use crate::negotiation::{CURRENT_PROTOCOL_VERSION, ProtocolRevision, SERVER_DISCOVER_METHOD};
    use crate::protocol::JsonRpcRequest;
    use crate::transport::Transport;

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

    /// A transport that answers from a script and records what it was sent, so
    /// these tests exercise the adapters without spawning a server.
    struct FakeServer {
        answers: Vec<(String, Value)>,
        sent: Arc<StdMutex<Vec<JsonRpcRequest>>>,
    }

    #[async_trait]
    impl Transport for FakeServer {
        async fn send(&mut self, req: JsonRpcRequest) -> Result<Value> {
            self.sent.lock().expect("sent log").push(req.clone());
            let index = self
                .answers
                .iter()
                .position(|(method, _)| *method == req.method)
                .ok_or_else(|| LumeError::Protocol(format!("unscripted method {}", req.method)))?;
            Ok(self.answers.remove(index).1)
        }

        async fn send_notification(&mut self, _req: JsonRpcRequest) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn discover_answer() -> (String, Value) {
        (
            SERVER_DISCOVER_METHOD.to_string(),
            json!({"jsonrpc": "2.0", "id": 1, "result": {
                "protocolVersions": [CURRENT_PROTOCOL_VERSION],
                "capabilities": {"tools": {"listChanged": true}},
                "serverInfo": {"name": "files", "version": "2.1"},
            }}),
        )
    }

    fn call_answer(text: &str) -> (String, Value) {
        (
            CALL_TOOL_METHOD.to_string(),
            json!({"jsonrpc": "2.0", "id": 1, "result": {
                "resultType": "complete",
                "content": [{"type": "text", "text": text}],
            }}),
        )
    }

    fn connection(
        server: &str,
        tools: Vec<ToolSpec>,
        answers: Vec<(String, Value)>,
    ) -> (McpServerConnection, Arc<StdMutex<Vec<JsonRpcRequest>>>) {
        let sent = Arc::new(StdMutex::new(Vec::new()));
        let mut script = vec![discover_answer()];
        script.extend(answers);
        let client = McpClient::connect(Box::new(FakeServer {
            answers: script,
            sent: Arc::clone(&sent),
        }));
        (
            McpServerConnection {
                server: server.to_string(),
                revision: ProtocolRevision::Stateless2026_07_28,
                server_info: None,
                tools,
                client: Arc::new(Mutex::new(client)),
            },
            sent,
        )
    }

    fn spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: format!("mcp {name}"),
            input_schema: json!({"type": "object"}),
        }
    }

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: "call-1".to_string(),
            name: name.to_string(),
            arguments,
        }
    }

    #[test]
    fn a_registered_tool_keeps_its_name_against_an_mcp_server() {
        let mut registry = ToolRegistry::new();
        registry.register_many([EchoTool::boxed("read_file"), EchoTool::boxed("list_dir")]);
        let (files, _sent) = connection(
            "files",
            vec![spec("read_file"), spec("list_directory")],
            Vec::new(),
        );

        let registration = registry.register_server(&files);

        assert_eq!(registration.registered, ["list_directory"]);
        assert_eq!(
            registration.collisions,
            [NameCollision {
                name: "read_file".to_string(),
                server: "files".to_string()
            }]
        );
        assert!(
            registration.collisions[0]
                .to_string()
                .contains("\"read_file\"")
                && registration.collisions[0].to_string().contains("\"files\""),
            "the warning must name both sides: {}",
            registration.collisions[0]
        );
        assert_eq!(
            registry.names(),
            ["read_file", "list_dir", "list_directory"]
        );
        assert!(
            registry
                .get("read_file")
                .expect("read_file")
                .spec()
                .description
                .starts_with("echo"),
            "the builtin must still be the tool behind a contested name"
        );
    }

    #[test]
    fn the_first_server_to_offer_a_name_keeps_it() {
        let mut registry = ToolRegistry::new();
        let (files, _sent) = connection("files", vec![spec("search")], Vec::new());
        let (index, _sent) = connection("index", vec![spec("search")], Vec::new());

        let first = registry.register_server(&files);
        let second = registry.register_server(&index);

        assert_eq!(first.registered, ["search"]);
        assert!(second.registered.is_empty());
        assert_eq!(second.collisions[0].server, "index");
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn a_registered_mcp_tool_forwards_its_arguments_to_the_server() {
        let mut registry = ToolRegistry::new();
        let (files, sent) = connection(
            "files",
            vec![spec("read_text_file")],
            vec![call_answer("file contents")],
        );
        registry.register_server(&files);

        let answer = registry
            .dispatch(&call(
                "read_text_file",
                json!({"path": "/etc/hostname", "head": 2}),
            ))
            .await
            .expect("mcp dispatch");

        assert_eq!(answer, "file contents");
        let sent = sent.lock().expect("sent log").clone();
        let forwarded = sent
            .iter()
            .find(|req| req.method == CALL_TOOL_METHOD)
            .expect("the adapter must call the server");
        let params = forwarded.params.as_ref().expect("params");
        assert_eq!(params["name"], json!("read_text_file"));
        assert_eq!(
            params["arguments"],
            json!({"path": "/etc/hostname", "head": 2}),
            "the model's arguments must reach the wire unaltered"
        );
    }

    #[tokio::test]
    async fn dispatching_an_mcp_tool_the_server_never_listed_fails() {
        let mut registry = ToolRegistry::new();
        let (files, _sent) = connection("files", vec![spec("read_text_file")], Vec::new());
        registry.register_server(&files);

        let err = registry
            .dispatch(&call("not_a_tool", json!({})))
            .await
            .expect_err("unregistered tool");

        assert!(matches!(err, LumeError::ToolNotFound(name) if name == "not_a_tool"));
    }

    #[tokio::test]
    async fn transport_for_rejects_an_entry_that_names_no_endpoint() {
        let stdio = McpServerConfig {
            name: "no-command".to_string(),
            transport: McpTransport::Stdio,
            command: None,
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
        };
        let http = McpServerConfig {
            name: "no-url".to_string(),
            transport: McpTransport::Http,
            command: None,
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
        };

        let Err(error) = transport_for(&stdio).await else {
            panic!("a stdio entry without a command must not open a transport");
        };
        assert!(matches!(error, LumeError::Protocol(ref m) if m == "missing command"));
        let Err(error) = transport_for(&http).await else {
            panic!("an http entry without a url must not open a transport");
        };
        assert!(matches!(error, LumeError::Protocol(ref m) if m == "missing url"));
    }

    #[tokio::test]
    async fn an_unreachable_server_becomes_a_failure_not_an_error() {
        let servers = vec![McpServerConfig {
            name: "ghost".to_string(),
            transport: McpTransport::Stdio,
            command: Some("/nonexistent/lume-mcp-nonexistent-server".to_string()),
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
        }];

        let (connections, failures) = connect_servers(&servers, Duration::from_secs(5)).await;

        assert!(connections.is_empty());
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].server, "ghost");
        assert!(!failures[0].reason.is_empty());
    }

    #[tokio::test]
    async fn a_server_that_never_answers_is_given_up_on() {
        let error = within(
            Duration::from_millis(20),
            std::future::pending::<Result<()>>(),
        )
        .await
        .expect_err("no answer");

        assert!(
            matches!(error, LumeError::Protocol(ref m) if m.contains("no answer within")),
            "{error}"
        );
    }
}
