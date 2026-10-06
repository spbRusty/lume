//! MCP client for the current protocol revision.
//!
//! The crate speaks the stateless `2026-07-28` revision and negotiates down to
//! the legacy `2025-06-18` handshake for servers that reject `server/discover`.
//! `initialize`, `ping`, `logging/setLevel`, `notifications/roots/list_changed`,
//! `Mcp-Session-Id` and SSE stream resumability were removed in `2026-07-28`;
//! roots, sampling and logging are deprecated until at least `2027-07-28`.
//! None of them are implemented here.

#![warn(missing_docs)]

pub mod capabilities;
pub mod client;
pub mod meta;
pub mod negotiation;
pub mod protocol;
pub mod registry;
pub mod results;
pub mod transport;

pub use capabilities::*;
pub use client::McpClient;
pub use meta::*;
pub use negotiation::*;
pub use protocol::*;
pub use registry::{
    DEFAULT_CONNECT_TIMEOUT, McpServerConnection, McpToolAdapter, NameCollision, ServerFailure,
    ServerRegistration, ToolRegistry, collect_mcp_tools, connect_server, connect_servers,
    transport_for,
};
pub use results::*;
pub use transport::{HttpTransport, StdioTransport, Transport};
