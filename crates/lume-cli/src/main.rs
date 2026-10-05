//! Lume CLI.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;

use lume_agent::{
    EditFileTool, GrepTool, ListDirTool, ReadFileTool, ShellTool, ToolRegistry, WriteFileTool,
};
use lume_core::Tool;
use lume_core::config::LumeConfig;
use lume_llm::OllamaBackend;
use lume_orchestrator::{Orchestrator, RetryPolicy, TokenBudget};

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
    /// Probe MCP server
    Probe {
        /// Server name
        name: String,
        /// Config path
        #[arg(long)]
        config: Option<PathBuf>,
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
        } => {
            let cfg = LumeConfig::resolve(config.as_deref())?;
            let root = workspace_root(&cfg)?;
            let (small_model, large_model) = match &model {
                Some(only) => (only.clone(), only.clone()),
                None => (cfg.small_model.clone(), cfg.large_model.clone()),
            };
            let orchestrator = Orchestrator::new(
                Arc::new(OllamaBackend::new(&cfg.ollama_url, &large_model)),
                Arc::new(OllamaBackend::new(&cfg.ollama_url, &small_model)),
                Arc::new(registry(root)),
                RetryPolicy::default(),
                TokenBudget {
                    limit: cfg.context_window.saturating_mul(cfg.max_iterations),
                    spent: 0,
                },
            );
            let outcome = orchestrator.execute(&task).await?;
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
            eprintln!(
                "chat mode is not wired up yet; it would run against {chosen}.\n\
                 Use `lume run \"<task>\"` for a one-shot task instead."
            );
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
            McpAction::Probe { name, config } => {
                let cfg = LumeConfig::resolve(config.as_deref())?;
                match cfg.mcp_servers.iter().find(|s| s.name == name) {
                    Some(server) => eprintln!(
                        "probing {:?} is not wired up yet ({}); MCP servers are listed but not yet connected",
                        server.transport, name
                    ),
                    None => eprintln!("no MCP server named {name:?} in the configuration"),
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
