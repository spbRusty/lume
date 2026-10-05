//! MCP client.
//!
//! The client speaks the stateless `2026-07-28` revision and negotiates down to
//! the legacy `2025-06-18` handshake when a server rejects `server/discover`.
//! Both paths exist because nearly every MCP server deployed today predates
//! `server/discover`; a client that implemented only the new revision could not
//! talk to anything actually running.

use std::sync::atomic::{AtomicI64, Ordering};

use lume_core::error::{LumeError, Result};
use lume_core::types::ToolSpec;
use serde_json::{Value, json};

use crate::capabilities::{ClientCapabilities, ClientInfo};
use crate::meta::RequestMeta;
use crate::negotiation::{
    DiscoverProbe, DiscoverResult, INITIALIZE_METHOD, INITIALIZED_NOTIFICATION, InitializeParams,
    InitializeResult, LEGACY_PROTOCOL_VERSION, NegotiatedProtocol, ProtocolRevision,
    SERVER_DISCOVER_METHOD,
};
use crate::protocol::{INTERNAL, JSONRPC_VERSION, JsonRpcRequest, RpcError, result_payload};
use crate::results::{CallToolParams, CallToolResult, ListToolsResult, PendingInput, ResultType};
use crate::transport::Transport;

/// Method that lists the tools a server exposes.
pub const LIST_TOOLS_METHOD: &str = "tools/list";

/// Method that invokes a tool.
pub const CALL_TOOL_METHOD: &str = "tools/call";

/// MCP client for one server connection.
pub struct McpClient {
    transport: Box<dyn Transport>,
    next_id: AtomicI64,
    initialized: bool,
    client_info: ClientInfo,
    capabilities: ClientCapabilities,
    protocol: Option<NegotiatedProtocol>,
    handshake_revision: Option<ProtocolRevision>,
    pending_input: Option<PendingInput>,
}

impl McpClient {
    /// Connect over `transport`, advertising the lume identity.
    pub fn connect(transport: Box<dyn Transport>) -> Self {
        Self::with_client_info(transport, ClientInfo::lume())
    }

    /// Connect over `transport`, advertising `client_info` in every `_meta`.
    pub fn with_client_info(transport: Box<dyn Transport>, client_info: ClientInfo) -> Self {
        Self {
            transport,
            next_id: AtomicI64::new(1),
            initialized: false,
            client_info,
            capabilities: ClientCapabilities::default(),
            protocol: None,
            handshake_revision: None,
            pending_input: None,
        }
    }

    /// Whether a protocol revision has been negotiated.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// The revision in use: the negotiated one, the revision declared by an
    /// in-flight legacy handshake, or the current revision before either.
    pub fn revision(&self) -> ProtocolRevision {
        self.handshake_revision.unwrap_or_else(|| {
            self.protocol.as_ref().map_or(
                ProtocolRevision::Stateless2026_07_28,
                NegotiatedProtocol::revision,
            )
        })
    }

    /// The negotiation result, once [`McpClient::negotiate`] has run.
    pub fn negotiated(&self) -> Option<&NegotiatedProtocol> {
        self.protocol.as_ref()
    }

    /// The Multi Round-Trip Request awaiting answers, if any.
    pub fn pending_input(&self) -> Option<&PendingInput> {
        self.pending_input.as_ref()
    }

    /// Negotiate the protocol revision, once per connection.
    ///
    /// `server/discover` is tried first; a method-not-found answer means the
    /// server predates the stateless revision, so the legacy `initialize` plus
    /// `notifications/initialized` handshake runs instead.
    pub async fn negotiate(&mut self) -> Result<NegotiatedProtocol> {
        if let Some(protocol) = &self.protocol {
            return Ok(protocol.clone());
        }
        let probe = match self.request_result(SERVER_DISCOVER_METHOD, None).await {
            Ok(payload) => {
                DiscoverProbe::Answered(serde_json::from_value::<DiscoverResult>(payload)?)
            }
            Err(err) if err.is_method_not_found() => DiscoverProbe::MethodNotFound,
            Err(err) => return Err(err.into()),
        };
        let legacy_handshake = match probe {
            DiscoverProbe::MethodNotFound => {
                self.handshake_revision = Some(ProtocolRevision::LegacyHandshake2025_06_18);
                let result = self.legacy_handshake().await;
                self.handshake_revision = None;
                Some(result?)
            }
            DiscoverProbe::Answered(_) => None,
        };
        let negotiated = NegotiatedProtocol::from_probe(probe, legacy_handshake);
        self.protocol = Some(negotiated.clone());
        Ok(negotiated)
    }

    /// Establish a session and report what the server declared.
    ///
    /// Kept as a wrapper over [`McpClient::negotiate`] because other crates
    /// call it. The legacy handshake inside it runs only when the negotiated
    /// revision is [`ProtocolRevision::LegacyHandshake2025_06_18`]; on the
    /// stateless path the returned result is synthesised from `server/discover`.
    pub async fn initialize(&mut self, client_info: ClientInfo) -> Result<InitializeResult> {
        self.client_info = client_info;
        let negotiated = self.negotiate().await?;
        self.initialized = true;
        negotiated.initialize_result().ok_or_else(|| {
            LumeError::Protocol("server has not reported its identity yet".to_string())
        })
    }

    /// List the tools a server exposes, in the server's own order.
    pub async fn list_tools(&mut self) -> Result<Vec<ToolSpec>> {
        let payload = self
            .request_result(LIST_TOOLS_METHOD, None)
            .await
            .map_err(LumeError::from)?;
        Ok(serde_json::from_value::<ListToolsResult>(payload)?.into_specs())
    }

    /// Call a tool.
    ///
    /// An interim `resultType: "input_required"` answer is not an error: its
    /// `inputRequests` payload is stashed and the interim result is returned, so
    /// the caller can inspect what the server asked for. Finish the round trip
    /// with [`McpClient::resume`].
    pub async fn call_tool(&mut self, name: &str, args: Value) -> Result<CallToolResult> {
        let params = CallToolParams {
            name: name.to_string(),
            arguments: args,
        };
        let request = self.build_request(CALL_TOOL_METHOD, Some(serde_json::to_value(params)?))?;
        let payload = self
            .dispatch(request.clone())
            .await
            .map_err(LumeError::from)?;
        let result = serde_json::from_value::<CallToolResult>(payload)?;
        match result.result_type {
            ResultType::Complete => {
                if let Some(true) = result.is_error {
                    return Err(LumeError::ToolFailed {
                        name: name.to_string(),
                        message: "tool returned error".to_string(),
                    });
                }
            }
            ResultType::InputRequired => {
                self.pending_input = Some(PendingInput {
                    input_requests: result.input_requests.clone().unwrap_or(Value::Null),
                    original: request,
                });
            }
        }
        Ok(result)
    }

    /// Answer a stashed Multi Round-Trip Request and finish the original call.
    ///
    /// `input_responses` is attached to the original request as `inputResponses`,
    /// which is then re-issued under a fresh request id. An automated harness has
    /// no human to answer an elicitation prompt, so this is the only way past a
    /// `resultType: "input_required"` answer: the caller must produce the answers
    /// from the model, a policy, or a scripted fixture.
    pub async fn resume(&mut self, input_responses: Value) -> Result<Value> {
        let pending = self.pending_input.take().ok_or_else(|| {
            LumeError::Protocol("no multi round-trip request is awaiting input".to_string())
        })?;
        let mut request = pending.original;
        request.id = None;
        request.params = Some(with_input_responses(request.params.take(), input_responses));
        let payload = self
            .dispatch(request.clone())
            .await
            .map_err(LumeError::from)?;
        let result = serde_json::from_value::<CallToolResult>(payload)?;
        match result.result_type {
            ResultType::Complete => {}
            ResultType::InputRequired => {
                self.pending_input = Some(PendingInput {
                    input_requests: result.input_requests.clone().unwrap_or(Value::Null),
                    original: request,
                });
            }
        }
        Ok(serde_json::to_value(result)?)
    }

    /// Close the underlying transport.
    pub async fn close(&mut self) -> Result<()> {
        self.transport.close().await
    }

    async fn legacy_handshake(&mut self) -> Result<InitializeResult> {
        let params = InitializeParams {
            protocol_version: LEGACY_PROTOCOL_VERSION.to_string(),
            client_info: self.client_info.clone(),
            capabilities: self.capabilities.clone(),
        };
        let payload = self
            .request_result(INITIALIZE_METHOD, Some(serde_json::to_value(params)?))
            .await
            .map_err(LumeError::from)?;
        let result = serde_json::from_value::<InitializeResult>(payload)?;
        let mut notification = self.build_request(INITIALIZED_NOTIFICATION, None)?;
        notification.id = None;
        self.transport.send_notification(notification).await?;
        Ok(result)
    }

    fn build_request(&self, method: &str, params: Option<Value>) -> Result<JsonRpcRequest> {
        Ok(JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(serde_json::to_value(
                self.next_id.fetch_add(1, Ordering::SeqCst),
            )?),
            method: method.to_string(),
            params,
            meta: Some(
                RequestMeta::new(self.revision(), &self.client_info, &self.capabilities)
                    .to_value()?,
            ),
        })
    }

    async fn request_result(
        &mut self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, RpcError> {
        self.dispatch(self.build_request(method, params).map_err(internal)?)
            .await
    }

    async fn dispatch(&mut self, request: JsonRpcRequest) -> Result<Value, RpcError> {
        let envelope = self.transport.send(request).await.map_err(internal)?;
        result_payload(&envelope)
    }
}

fn with_input_responses(params: Option<Value>, input_responses: Value) -> Value {
    match params {
        Some(Value::Object(mut fields)) => {
            fields.insert("inputResponses".to_string(), input_responses);
            Value::Object(fields)
        }
        _ => json!({ "inputResponses": input_responses }),
    }
}

fn internal(err: impl std::fmt::Display) -> RpcError {
    RpcError {
        code: INTERNAL,
        message: err.to_string(),
        data: None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex as StdMutex};

    use async_trait::async_trait;
    use serde_json::{Value, json};

    use super::{CALL_TOOL_METHOD, LIST_TOOLS_METHOD, McpClient};
    use crate::capabilities::ClientInfo;
    use crate::meta::{META_CLIENT_CAPABILITIES, META_CLIENT_INFO, META_PROTOCOL_VERSION};
    use crate::negotiation::{
        INITIALIZE_METHOD, INITIALIZED_NOTIFICATION, ProtocolRevision, SERVER_DISCOVER_METHOD,
    };
    use crate::protocol::{JsonRpcRequest, METHOD_NOT_FOUND};
    use crate::results::ResultType;
    use crate::transport::Transport;
    use lume_core::error::{LumeError, Result};

    struct ScriptedServer {
        scripted: Vec<(String, Value)>,
        sent: Arc<StdMutex<Vec<JsonRpcRequest>>>,
    }

    impl ScriptedServer {
        fn new(scripted: Vec<(&str, Value)>) -> (Self, Arc<StdMutex<Vec<JsonRpcRequest>>>) {
            let sent = Arc::new(StdMutex::new(Vec::new()));
            (
                Self {
                    scripted: scripted
                        .into_iter()
                        .map(|(method, result)| (method.to_string(), result))
                        .collect(),
                    sent: Arc::clone(&sent),
                },
                sent,
            )
        }

        fn record(&self, req: &JsonRpcRequest) {
            self.sent.lock().expect("sent log").push(req.clone());
        }
    }

    #[async_trait]
    impl Transport for ScriptedServer {
        async fn send(&mut self, req: JsonRpcRequest) -> Result<Value> {
            self.record(&req);
            let index = self
                .scripted
                .iter()
                .position(|(method, _)| *method == req.method)
                .ok_or_else(|| LumeError::Protocol(format!("unscripted method {}", req.method)))?;
            Ok(self.scripted.remove(index).1)
        }

        async fn send_notification(&mut self, req: JsonRpcRequest) -> Result<()> {
            self.record(&req);
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn result_envelope(result: Value) -> Value {
        json!({"jsonrpc": "2.0", "id": 1, "result": result})
    }

    fn error_envelope(code: i64, message: &str) -> Value {
        json!({"jsonrpc": "2.0", "id": 1, "error": {"code": code, "message": message}})
    }

    fn discover_result() -> Value {
        result_envelope(json!({
            "resultType": "complete",
            "protocolVersions": ["2026-07-28"],
            "capabilities": {"tools": {"listChanged": true}},
            "serverInfo": {"name": "files", "version": "2.1"},
        }))
    }

    fn tools_result(names: &[&str]) -> Value {
        result_envelope(json!({
            "resultType": "complete",
            "tools": names
                .iter()
                .map(|name| json!({
                    "name": name,
                    "description": format!("tool {name}"),
                    "inputSchema": {"type": "object"},
                }))
                .collect::<Vec<_>>(),
            "ttlMs": 30000,
            "cacheScope": "public",
        }))
    }

    fn meta_protocol_versions(sent: &[JsonRpcRequest]) -> Vec<String> {
        sent.iter()
            .map(|req| {
                req.meta
                    .as_ref()
                    .and_then(|meta| meta.get(META_PROTOCOL_VERSION))
                    .and_then(Value::as_str)
                    .expect("every request declares its protocol version")
                    .to_string()
            })
            .collect()
    }

    #[tokio::test]
    async fn stateless_server_is_reached_through_server_discover() {
        let (transport, sent) = ScriptedServer::new(vec![
            (SERVER_DISCOVER_METHOD, discover_result()),
            (LIST_TOOLS_METHOD, tools_result(&["zeta", "alpha"])),
        ]);
        let mut client = McpClient::connect(Box::new(transport));

        let negotiated = client.negotiate().await.expect("stateless negotiation");

        assert_eq!(negotiated.revision(), ProtocolRevision::Stateless2026_07_28);
        assert_eq!(negotiated.advertised_versions(), ["2026-07-28"]);
        assert!(!client.is_initialized());
        let tools = client.list_tools().await.expect("tools");
        assert_eq!(
            tools
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            ["zeta", "alpha"]
        );

        let sent = sent.lock().expect("sent log").clone();
        assert_eq!(
            sent.iter()
                .map(|req| req.method.as_str())
                .collect::<Vec<_>>(),
            [SERVER_DISCOVER_METHOD, LIST_TOOLS_METHOD]
        );
        assert!(sent.iter().all(|req| {
            req.meta.as_ref().is_some_and(|meta| {
                meta.get(META_CLIENT_INFO).is_some() && meta.get(META_CLIENT_CAPABILITIES).is_some()
            })
        }));
        assert_eq!(meta_protocol_versions(&sent), ["2026-07-28", "2026-07-28"]);
    }

    #[tokio::test]
    async fn legacy_server_falls_back_to_the_initialize_handshake() {
        let (transport, sent) = ScriptedServer::new(vec![
            (
                SERVER_DISCOVER_METHOD,
                error_envelope(METHOD_NOT_FOUND, "Method not found"),
            ),
            (
                INITIALIZE_METHOD,
                result_envelope(json!({
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": true}},
                    "serverInfo": {"name": "legacy", "version": "0.9"},
                })),
            ),
            (LIST_TOOLS_METHOD, tools_result(&["only"])),
        ]);
        let mut client = McpClient::connect(Box::new(transport));

        let session = client
            .initialize(ClientInfo::lume())
            .await
            .expect("legacy negotiation");

        assert!(client.is_initialized());
        assert_eq!(
            client.negotiated().map(super::NegotiatedProtocol::revision),
            Some(ProtocolRevision::LegacyHandshake2025_06_18)
        );
        assert_eq!(session.protocol_version, "2025-06-18");
        assert_eq!(session.server_info.name, "legacy");
        assert!(client.list_tools().await.expect("tools").len() == 1);

        let sent = sent.lock().expect("sent log").clone();
        assert_eq!(
            sent.iter()
                .map(|req| req.method.as_str())
                .collect::<Vec<_>>(),
            [
                SERVER_DISCOVER_METHOD,
                INITIALIZE_METHOD,
                INITIALIZED_NOTIFICATION,
                LIST_TOOLS_METHOD
            ]
        );
        assert_eq!(
            sent[1].params.as_ref().expect("params")["protocolVersion"],
            json!("2025-06-18")
        );
        assert_eq!(sent[2].id, None, "the handshake notification has no id");
        // The probe advertises the new revision, since asking whether the server
        // supports it is the probe's whole purpose; everything from the legacy
        // handshake onward must declare 2025-06-18 in both `_meta` and params.
        assert_eq!(
            meta_protocol_versions(&sent),
            ["2026-07-28", "2025-06-18", "2025-06-18", "2025-06-18"]
        );
    }

    #[tokio::test]
    async fn interim_result_stashes_a_request_that_resume_can_reissue() {
        let (transport, sent) = ScriptedServer::new(vec![
            (SERVER_DISCOVER_METHOD, discover_result()),
            (
                CALL_TOOL_METHOD,
                result_envelope(json!({
                    "resultType": "input_required",
                    "inputRequests": [{"name": "confirm", "prompt": "overwrite?"}],
                })),
            ),
            (
                CALL_TOOL_METHOD,
                result_envelope(json!({
                    "resultType": "complete",
                    "content": [{"type": "text", "text": "done"}],
                })),
            ),
        ]);
        let mut client = McpClient::connect(Box::new(transport));
        client.negotiate().await.expect("stateless negotiation");

        let interim = client
            .call_tool("write_file", json!({"path": "/tmp/x"}))
            .await
            .expect("interim result");
        assert_eq!(interim.result_type, ResultType::InputRequired);
        let pending = client.pending_input().expect("stashed input request");
        assert_eq!(pending.original.method, CALL_TOOL_METHOD);
        assert_eq!(pending.input_requests[0]["name"], json!("confirm"));

        let finished = client
            .resume(json!([{"name": "confirm", "value": true}]))
            .await
            .expect("resumed result");
        assert_eq!(finished["resultType"], json!("complete"));
        assert_eq!(finished["content"][0]["text"], json!("done"));
        assert!(client.pending_input().is_none());

        let sent = sent.lock().expect("sent log").clone();
        let reissued = sent.last().expect("re-issued request");
        assert_eq!(reissued.method, CALL_TOOL_METHOD);
        let params = reissued.params.as_ref().expect("params");
        assert_eq!(params["name"], json!("write_file"));
        assert_eq!(params["arguments"]["path"], json!("/tmp/x"));
        assert_eq!(params["inputResponses"][0]["value"], json!(true));
        assert_ne!(
            reissued.id, sent[1].id,
            "a resumed call is a new request, not a replay of the old id"
        );
    }

    #[tokio::test]
    async fn resume_without_a_pending_request_fails() {
        let (transport, _sent) = ScriptedServer::new(Vec::new());
        let mut client = McpClient::connect(Box::new(transport));
        assert!(client.resume(json!([])).await.is_err());
    }

    #[tokio::test]
    async fn json_rpc_errors_surface_with_their_code() {
        let (transport, _sent) = ScriptedServer::new(vec![(
            SERVER_DISCOVER_METHOD,
            error_envelope(-32022, "unsupported protocol version"),
        )]);
        let mut client = McpClient::connect(Box::new(transport));
        let err = client.negotiate().await.expect_err("probe must fail");
        assert!(err.to_string().contains("-32022"), "{err}");
    }
}
