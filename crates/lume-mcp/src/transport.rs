//! MCP transports: stdio and Streamable HTTP.
//!
//! Streamable HTTP is the only HTTP transport MCP supports; the older HTTP+SSE
//! split transport is deprecated. A Streamable HTTP POST must carry `Mcp-Method`
//! and `Mcp-Name`, and its reply is either a JSON body or an SSE-framed stream of
//! JSON-RPC messages. There are no sessions: `Mcp-Session-Id` was removed in
//! `2026-07-28`, and a broken stream is recovered by re-issuing the request with a
//! new id rather than by resuming with `Last-Event-ID`.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderValue};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use lume_core::error::{LumeError, Result};

use crate::protocol::JsonRpcRequest;

/// Header naming the JSON-RPC method of a Streamable HTTP POST.
pub const MCP_METHOD_HEADER: &str = "Mcp-Method";

/// Header naming the tool or operation a Streamable HTTP POST addresses.
pub const MCP_NAME_HEADER: &str = "Mcp-Name";

/// Content type Streamable HTTP clients must accept.
pub const MCP_ACCEPT: &str = "application/json, text/event-stream";

/// Transport trait for MCP communication.
#[async_trait]
pub trait Transport: Send {
    /// Send a request and return the JSON-RPC response envelope.
    async fn send(&mut self, req: JsonRpcRequest) -> Result<Value>;

    /// Send a notification: no id, no response, no waiting for one.
    async fn send_notification(&mut self, req: JsonRpcRequest) -> Result<()>;

    /// Close transport.
    async fn close(&mut self) -> Result<()>;
}

/// Stdio transport: newline-delimited JSON-RPC over a child process.
///
/// The child is killed when the transport is dropped, not only when
/// [`Transport::close`] is called: an abandoned `Box<dyn Transport>` from a
/// cancelled task or a panic would otherwise leave the server process running
/// for the lifetime of the machine. A stdio server is useless once its pipe is
/// gone, so there is nothing worth preserving.
pub struct StdioTransport {
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout_reader: BufReader<tokio::process::ChildStdout>,
    next_id: i64,
    mutex: Arc<Mutex<()>>,
}

impl StdioTransport {
    /// Spawn a new stdio transport.
    pub async fn spawn(
        command: &str,
        args: &[String],
        env: &BTreeMap<String, String>,
    ) -> Result<Self> {
        let mut cmd = Command::new(command);
        cmd.args(args);
        for (key, value) in env {
            cmd.env(key, value);
        }
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::inherit());
        cmd.kill_on_drop(true);
        let mut child = cmd.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| LumeError::Protocol("no stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LumeError::Protocol("no stdout".to_string()))?;
        Ok(Self {
            child,
            stdin,
            stdout_reader: BufReader::new(stdout),
            next_id: 1,
            mutex: Arc::new(Mutex::new(())),
        })
    }
}

async fn write_line(stdin: &mut tokio::process::ChildStdin, req: &JsonRpcRequest) -> Result<()> {
    let line = serde_json::to_string(req)?;
    stdin.write_all(line.as_bytes()).await?;
    stdin.write_all(b"\n").await?;
    Ok(stdin.flush().await?)
}

#[async_trait]
impl Transport for StdioTransport {
    async fn send(&mut self, mut req: JsonRpcRequest) -> Result<Value> {
        let mutex = Arc::clone(&self.mutex);
        let _guard = mutex.lock().await;
        let id = self.next_id;
        self.next_id += 1;
        req.id = Some(serde_json::to_value(id)?);
        write_line(&mut self.stdin, &req).await?;
        loop {
            let mut buf = String::new();
            let read = self.stdout_reader.read_line(&mut buf).await?;
            if read == 0 {
                return Err(LumeError::Protocol(
                    "server closed stdout before answering".to_string(),
                ));
            }
            let line = buf.trim();
            if line.is_empty() {
                continue;
            }
            let envelope: Value = serde_json::from_str(line)?;
            if envelope.get("id").and_then(Value::as_i64) == Some(id) {
                return Ok(envelope);
            }
        }
    }

    async fn send_notification(&mut self, req: JsonRpcRequest) -> Result<()> {
        let mutex = Arc::clone(&self.mutex);
        let _guard = mutex.lock().await;
        write_line(&mut self.stdin, &req).await
    }

    async fn close(&mut self) -> Result<()> {
        let _ = self.child.kill().await;
        Ok(())
    }
}

/// Streamable HTTP transport.
#[derive(Clone)]
pub struct HttpTransport {
    url: String,
    client: Client,
}

impl HttpTransport {
    /// Create a new Streamable HTTP transport.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: Client::new(),
        }
    }

    async fn post(&self, req: &JsonRpcRequest) -> Result<reqwest::Response> {
        let name = header_value(&operation_name(&req.method, req.params.as_ref()))?;
        Ok(self
            .client
            .post(&self.url)
            .header(MCP_METHOD_HEADER, header_value(&req.method)?)
            .header(MCP_NAME_HEADER, name)
            .header(reqwest::header::ACCEPT, MCP_ACCEPT)
            .json(req)
            .send()
            .await?)
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn send(&mut self, req: JsonRpcRequest) -> Result<Value> {
        let response = self.post(&req).await?;
        let framed_as_sse = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|content_type| content_type.starts_with("text/event-stream"));
        let body = response.text().await?;
        if framed_as_sse {
            first_event_message(&body)
                .ok_or_else(|| LumeError::Protocol("no json-rpc message in sse body".to_string()))
        } else {
            Ok(serde_json::from_str(&body)?)
        }
    }

    async fn send_notification(&mut self, req: JsonRpcRequest) -> Result<()> {
        self.post(&req).await?;
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// The operation a Streamable HTTP POST addresses: the tool name for
/// `tools/call`, otherwise the trailing segment of the method name.
pub fn operation_name(method: &str, params: Option<&Value>) -> String {
    if let Some(name) = params
        .and_then(|params| params.get("name"))
        .and_then(Value::as_str)
    {
        return name.to_string();
    }
    method.rsplit('/').next().unwrap_or(method).to_string()
}

/// First JSON-RPC message in an SSE-framed body.
fn first_event_message(body: &str) -> Option<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .find_map(|payload| serde_json::from_str(payload.trim()).ok())
}

fn header_value(value: &str) -> Result<HeaderValue> {
    HeaderValue::from_str(value)
        .map_err(|_| LumeError::Protocol(format!("invalid http header value: {value}")))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{MCP_METHOD_HEADER, MCP_NAME_HEADER, first_event_message, operation_name};

    #[test]
    fn operation_name_uses_the_tool_name_when_present() {
        assert_eq!(
            operation_name(
                "tools/call",
                Some(&json!({"name": "read_file", "arguments": {}}))
            ),
            "read_file"
        );
        assert_eq!(operation_name("tools/list", None), "list");
        assert_eq!(
            operation_name("server/discover", Some(&json!({}))),
            "discover"
        );
    }

    #[test]
    fn header_names_match_the_spec() {
        assert_eq!(MCP_METHOD_HEADER, "Mcp-Method");
        assert_eq!(MCP_NAME_HEADER, "Mcp-Name");
    }

    #[test]
    fn sse_framed_reply_yields_the_first_json_rpc_message() {
        let body =
            "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}\n\n";
        let message = first_event_message(body).expect("json-rpc message");
        assert_eq!(message["result"]["tools"], json!([]));
    }

    #[test]
    fn sse_framed_reply_skips_keepalive_comments() {
        let body = ": keep-alive\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{}}\n";
        let message = first_event_message(body).expect("json-rpc message");
        assert_eq!(message["id"], json!(2));
        assert!(first_event_message(": only a comment\n").is_none());
    }
}
