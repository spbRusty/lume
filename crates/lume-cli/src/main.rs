//! Lume CLI.

mod chat;
mod probe;

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

use lume_agent::{
    AgentConfig, EditFileTool, GrepTool, ListDirTool, ReadFileTool, ShellTool, ToolRegistry,
    WriteFileTool,
};
use lume_core::Tool;
use lume_core::config::LumeConfig;
use lume_core::model::{Model, ModelTier};
use lume_core::types::{ChatRequest, Message, Role, SamplingParams};
use lume_llm::OllamaBackend;
use lume_mcp::{DEFAULT_CONNECT_TIMEOUT, McpServerConnection};
use lume_orchestrator::{Orchestrator, RetryPolicy, TokenBudget, classify, tier_for};

use crate::chat::chat;
use crate::probe::{ServerProbe, probe};

/// CLI.
#[derive(Parser)]
#[command(name = "lume", about = "Lume agent harness")]
struct Cli {
    /// Log level
    #[arg(long, global = true, default_value = "info", env = "LUME_LOG")]
    log: LevelFilter,

    /// Subcommand
    #[command(subcommand)]
    command: Command,
}

/// Commands.
#[derive(Subcommand)]
enum Command {
    /// Run a task
    Run {
        /// Task description
        task: String,
        /// Model override, applied to both tiers
        #[arg(long)]
        model: Option<String>,
        /// Config path
        #[arg(long)]
        config: Option<PathBuf>,
        /// Seconds one phase of an MCP connection may take
        #[arg(long, default_value_t = DEFAULT_CONNECT_TIMEOUT.as_secs())]
        timeout: u64,
        /// Stream the model's reply to stdout as it is generated
        ///
        /// Sends the task to the chosen model as a single completion — no
        /// orchestrator, no tools — and prints every chunk as it arrives.
        #[arg(long)]
        stream: bool,
    },
    /// Chat interactively
    Chat {
        /// Model override
        #[arg(long)]
        model: Option<String>,
        /// Config path
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// List models
    Models {
        /// Config path
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Doctor check
    Doctor {
        /// Config path
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// MCP commands
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
}

/// MCP actions.
#[derive(Subcommand)]
enum McpAction {
    /// List configured MCP servers
    List {
        /// Config path
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Connect to configured MCP servers, list their tools, and call one
    Probe {
        /// Server name; every configured server when omitted
        name: Option<String>,
        /// Config path
        #[arg(long)]
        config: Option<PathBuf>,
        /// Seconds one phase of a connection may take
        #[arg(long, default_value_t = DEFAULT_CONNECT_TIMEOUT.as_secs())]
        timeout: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(cli.log.into()))
        .init();

    match cli.command {
        Command::Run {
            task,
            model,
            config,
            timeout,
            stream,
        } => {
            let cfg = LumeConfig::resolve(config.as_deref())?;
            let (small_model, large_model) = match &model {
                Some(only) => (only.clone(), only.clone()),
                None => (cfg.small_model.clone(), cfg.large_model.clone()),
            };
            if stream {
                run_streamed(&task, &cfg, small_model, large_model).await?;
                return Ok(());
            }
            let root = workspace_root(&cfg)?;
            let (mut tools, connections) = with_mcp_tools(registry(root), &cfg, timeout).await;
            let orchestrator = Orchestrator::new(
                Arc::new(OllamaBackend::new(&cfg.ollama_url, &large_model)),
                Arc::new(OllamaBackend::new(&cfg.ollama_url, &small_model)),
                Arc::new(std::mem::take(&mut tools)),
                RetryPolicy::default(),
                TokenBudget {
                    limit: cfg.context_window.saturating_mul(cfg.max_iterations),
                    spent: 0,
                },
            )
            .with_agent_config(AgentConfig {
                params: SamplingParams {
                    temperature: cfg.temperature,
                    seed: cfg.seed,
                    ..SamplingParams::default()
                },
                ..AgentConfig::default()
            });
            let outcome = orchestrator.execute(&task).await;
            close_all(&connections).await;
            let outcome = outcome?;
            println!("{}", outcome.final_text);
            if !outcome.tool_calls.is_empty() {
                println!(
                    "\n--- {} tool call(s) over {} iteration(s) ---",
                    outcome.tool_calls.len(),
                    outcome.iterations
                );
            }
        }
        Command::Chat { model, config } => {
            let cfg = LumeConfig::resolve(config.as_deref())?;
            let chosen = model.unwrap_or_else(|| cfg.small_model.clone());
            let root = workspace_root(&cfg)?;
            let (tools, connections) =
                with_mcp_tools(registry(root), &cfg, DEFAULT_CONNECT_TIMEOUT.as_secs()).await;
            chat(chosen, &cfg, tools, connections).await?;
        }
        Command::Models { config } => {
            let cfg = LumeConfig::resolve(config.as_deref())?;
            let backend = OllamaBackend::new(&cfg.ollama_url, &cfg.small_model);
            match backend.list_models().await {
                Ok(models) => {
                    for m in models {
                        println!("{m}");
                    }
                }
                Err(e) => eprintln!("error: {e}"),
            }
        }
        Command::Doctor { config } => {
            let cfg = LumeConfig::resolve(config.as_deref())?;
            println!("ollama_url:     {}", cfg.ollama_url);
            println!("small_model:    {}", cfg.small_model);
            println!("large_model:    {}", cfg.large_model);
            println!("max_iterations: {}", cfg.max_iterations);
            println!("context_window: {}", cfg.context_window);
            println!("workspace_root: {}", workspace_root(&cfg)?.display());
            println!("mcp_servers:    {}", cfg.mcp_servers.len());
        }
        Command::Mcp { action } => match action {
            McpAction::List { config } => {
                let cfg = LumeConfig::resolve(config.as_deref())?;
                if cfg.mcp_servers.is_empty() {
                    println!("no MCP servers configured");
                }
                for server in &cfg.mcp_servers {
                    let detail = match (&server.command, &server.url) {
                        (Some(command), _) => command.clone(),
                        (None, Some(url)) => url.clone(),
                        (None, None) => "-".to_string(),
                    };
                    println!("{:<16} {:?} {}", server.name, server.transport, detail);
                }
            }
            McpAction::Probe {
                name,
                config,
                timeout,
            } => {
                let cfg = LumeConfig::resolve(config.as_deref())?;
                if cfg.mcp_servers.is_empty() {
                    println!("no MCP servers configured");
                    return Ok(());
                }
                let timeout = Duration::from_secs(timeout);
                let reports = probe(&cfg.mcp_servers, name.as_deref(), timeout).await;
                if let Some(name) = &name
                    && !cfg.mcp_servers.iter().any(|server| &server.name == name)
                {
                    eprintln!("no MCP server named {name:?} in the configuration");
                    return Ok(());
                }
                for report in &reports {
                    print_report(report);
                }
                let broken = reports.iter().filter(|report| !report.connected()).count();
                if broken == 0 {
                    println!("\nall {} configured server(s) usable", reports.len());
                } else {
                    anyhow::bail!(
                        "{broken} of {} configured MCP server(s) could not be used",
                        reports.len()
                    );
                }
            }
        },
    }
    Ok(())
}

fn workspace_root(cfg: &LumeConfig) -> Result<PathBuf> {
    match &cfg.workspace_root {
        Some(root) => Ok(root.clone()),
        None => std::env::current_dir().context("cannot read the current directory"),
    }
}

/// Stream one completion for `task` straight to stdout as it arrives.
///
/// `--stream` is a direct line to the model: the task is sent as a single user
/// message with no orchestrator and no tools, and every chunk is written and
/// flushed the moment it lands, so output appears while the model is still
/// generating. The tier the router would have picked for the task chooses the
/// model, so a streamed run and a normal run route the same way.
async fn run_streamed(
    task: &str,
    cfg: &LumeConfig,
    small_model: String,
    large_model: String,
) -> Result<()> {
    let model = match tier_for(classify(task)) {
        ModelTier::Small => small_model,
        ModelTier::Large => large_model,
    };
    let backend = OllamaBackend::new(&cfg.ollama_url, &model);
    let request = ChatRequest {
        model,
        messages: vec![Message {
            role: Role::User,
            content: task.to_string(),
            ..Default::default()
        }],
        tools: Vec::new(),
        params: SamplingParams {
            temperature: cfg.temperature,
            seed: cfg.seed,
            ..SamplingParams::default()
        },
    };

    let mut chunks = backend.chat_stream(request).await?;
    let mut stdout = std::io::stdout();
    let mut usage = None;
    while let Some(chunk) = chunks.recv().await {
        if let Some(delta) = chunk.delta {
            stdout.write_all(delta.as_bytes())?;
            stdout.flush()?;
        }
        usage = chunk.usage.or(usage);
    }
    writeln!(stdout)?;
    if let Some(usage) = usage {
        println!(
            "--- {} prompt + {} completion tokens ---",
            usage.prompt_tokens, usage.completion_tokens
        );
    }
    Ok(())
}

fn registry(root: PathBuf) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register_many([
        Box::new(ReadFileTool::new(root.clone())) as Box<dyn Tool>,
        Box::new(WriteFileTool::new(root.clone())),
        Box::new(EditFileTool::new(root.clone())),
        Box::new(ListDirTool::new(root.clone())),
        Box::new(GrepTool::new(root.clone())),
        Box::new(ShellTool::new(root)),
    ]);
    tools
}

/// Add every configured MCP server's tools to the builtins, and report what was
/// skipped.
///
/// Returns the connections so the caller can close them: the registry holds the
/// same clients, so nothing else would kill the server processes.
async fn with_mcp_tools(
    mut tools: ToolRegistry,
    cfg: &LumeConfig,
    timeout: u64,
) -> (ToolRegistry, Vec<McpServerConnection>) {
    let (connections, failures) =
        lume_mcp::connect_servers(&cfg.mcp_servers, Duration::from_secs(timeout)).await;
    for failure in &failures {
        eprintln!(
            "warning: MCP server {:?} is unavailable, skipping it: {}",
            failure.server, failure.reason
        );
    }
    for connection in &connections {
        let registration = tools.register_server(connection);
        for collision in &registration.collisions {
            eprintln!("warning: {collision}");
        }
        eprintln!(
            "mcp: {} contributes {} tool(s) over {}",
            connection.server,
            registration.registered.len(),
            connection.revision
        );
    }
    (tools, connections)
}

async fn close_all(connections: &[McpServerConnection]) {
    for connection in connections {
        let _ = connection.close().await;
    }
}

fn print_report(report: &ServerProbe) {
    match report {
        ServerProbe::Called {
            server,
            revision,
            server_info,
            tools,
            call,
        } => {
            println!(
                "{}: connected over {}{}, {} tool(s)",
                server,
                revision,
                server_info
                    .as_ref()
                    .map(|info| format!(" ({info})"))
                    .unwrap_or_default(),
                tools.len()
            );
            for tool in tools {
                println!("    {tool}");
            }
            println!("    call {} {}", call.tool, call.arguments);
            println!("    -> {}", indent(&call.text));
        }
        ServerProbe::Uncalled {
            server,
            revision,
            server_info,
            tools,
            reason,
        } => {
            println!(
                "{}: connected over {}{}, {} tool(s), none called: {reason}",
                server,
                revision,
                server_info
                    .as_ref()
                    .map(|info| format!(" ({info})"))
                    .unwrap_or_default(),
                tools.len()
            );
            for tool in tools {
                println!("    {tool}");
            }
        }
        ServerProbe::Failed { server, reason } => {
            println!("{server}: FAILED: {reason}");
        }
    }
}

fn indent(text: &str) -> String {
    if text.is_empty() {
        return "(no text content)".to_string();
    }
    text.lines()
        .map(|line| format!("       {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command, indent};

    #[test]
    fn indentation_pads_every_line_of_a_result() {
        assert_eq!(
            indent("Allowed directories:\n/home/vlad"),
            "       Allowed directories:\n       /home/vlad"
        );
    }

    #[test]
    fn an_empty_result_says_so_instead_of_printing_nothing() {
        assert_eq!(indent(""), "(no text content)");
    }

    #[test]
    fn run_accepts_the_stream_flag() {
        let cli = Cli::try_parse_from(["lume", "run", "--stream", "fix the typo"])
            .expect("the flag parses");

        match cli.command {
            Command::Run { task, stream, .. } => {
                assert_eq!(task, "fix the typo");
                assert!(stream);
            }
            _ => panic!("expected the run subcommand"),
        }
    }

    #[test]
    fn run_streams_only_when_asked() {
        let cli = Cli::try_parse_from(["lume", "run", "fix the typo"]).expect("parses");

        match cli.command {
            Command::Run { stream, .. } => assert!(!stream),
            _ => panic!("expected the run subcommand"),
        }
    }
}
