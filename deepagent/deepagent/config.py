"""Configuration loader for DeepAgent."""

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

log = logging.getLogger(__name__)

@dataclass
class LlamaCppConfig:
    model_path: str = "./models/qwen3-1.5b-instruct-q4_k_m.gguf"
    n_ctx: int = 2048
    n_threads: int = 4
    n_gpu_layers: int = 0
    roles_to_models: Dict[str, str] = field(
        default_factory=lambda: {
            "worker": "./models/qwen3-1.5b-instruct-q4_k_m.gguf",
            "manager_small": "./models/qwen3-1.5b-instruct-q4_k_m.gguf",
            "manager_medium": "./models/qwen3.5-3b-instruct-q4_k_m.gguf"
        }
    )

@dataclass
class LLMConfig:
    backend_type: str = "llama_cpp"
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    model_id: str = "google/gemma-3-4b-it"
    quantization: str = "4bit"  # "4bit", "8bit", "none"
    context_length: int = 8192
    temperature: float = 0.7
    max_new_tokens: int = 1024
    device: str = "cpu"
    roles_to_models: Dict[str, str] = field(
        default_factory=lambda: {
            "judge_primary_1": "google/gemma-3-4b-it",
            "judge_primary_2": "unsloth/Qwen3.5-0.8B-GGUF",
            "judge_backup": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "worker_tier_1": "unsloth/Qwen3.5-0.8B-GGUF",
            "worker_tier_2": "unsloth/Qwen3-1.7B-GGUF",
            "worker_tier_3": "unsloth/Qwen3.5-2B-GGUF"
        }
    )


@dataclass
class AgentConfig:
    max_iterations: int = 50
    max_time_minutes: int = 180
    enable_critic: bool = True
    enable_breakthrough_detection: bool = True
    enable_tool_creation: bool = True
    max_research_mins: int = 60
    max_gathering_mins: int = 30
    max_conclusion_mins: int = 20


@dataclass
class MemoryConfig:
    short_term_tokens: int = 4096
    long_term_db_path: str = "./memory_store"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class ToolsConfig:
    workspace_dirs: list = field(default_factory=lambda: ["./workspace"])
    allowed_commands: list = field(
        default_factory=lambda: ["python", "pip", "node", "npm", "npx", "git", "curl", "wget"]
    )
    blocked_patterns: list = field(
        default_factory=lambda: ["rm -rf /", "format c:", "del /s /q c:"]
    )
    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"
    max_command_timeout: int = 300
    custom_tools_dir: str = "./custom_tools"


@dataclass
class SafetyConfig:
    confirm_destructive: bool = True
    auto_cleanup: bool = True
    max_file_size_mb: int = 100


@dataclass
class DeepAgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


def resolve_model_path(path_or_repo: str) -> str:
    """
    Resolves a model path. If it's a HuggingFace repo ID, checks local cache first.
    If not cached entirely, dynamically downloads it then uses the local variant.
    """
    if os.path.exists(path_or_repo):
        return str(Path(path_or_repo).absolute())

    # Check if it looks like a HF repo (user/repo)
    if "/" in path_or_repo and not path_or_repo.startswith((".", "/")):
        try:
            from huggingface_hub import hf_hub_download, list_repo_files, HfApi
            import glob

            repo_id = path_or_repo
            filename = None

            if repo_id.endswith(".gguf"):
                parts = repo_id.split("/")
                filename = parts[-1]
                repo_id = "/".join(parts[:-1])
            else:
                # 1. Search local cache first (OFFLINE FIRST / local models only)
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                repo_folder = f"models--{repo_id.replace('/', '--')}"
                repo_path = os.path.join(cache_dir, repo_folder)
                
                if os.path.exists(repo_path):
                    # Cache hit! Find any .gguf file locally
                    gguf_files = glob.glob(os.path.join(repo_path, "snapshots", "**", "*.gguf"), recursive=True)
                    if gguf_files:
                        q4 = [f for f in gguf_files if "q4_k_m" in f.lower()]
                        best_local = q4[0] if q4 else gguf_files[0]
                        best_local = os.path.normpath(best_local)
                        log.info(f"Offline cache hit for {repo_id}: using {best_local}")
                        return best_local

                # 2. Dynamic download from huggingface when not cached
                api = HfApi()
                try:
                    files = list_repo_files(repo_id)
                    web_gguf = [f for f in files if f.endswith(".gguf")]
                    if web_gguf:
                        q4 = [f for f in web_gguf if "q4_k_m" in f.lower()]
                        filename = q4[0] if q4 else web_gguf[0]
                except Exception:
                    pass

                # If no file found in repo, try finding a quant repo
                if not filename:
                    model_basename = repo_id.split("/")[-1]
                    log.info(f"Searching for GGUF of {model_basename} on Hugging Face Hub...")
                    models = api.list_models(
                        search=model_basename,
                        tags="gguf",
                        sort="downloads",
                        direction=-1,
                        limit=5
                    )
                    for m in list(models):
                        try:
                            m_files = list_repo_files(m.id)
                            m_gguf_files = [f for f in m_files if f.endswith(".gguf")]
                            if m_gguf_files:
                                q4 = [f for f in m_gguf_files if "q4_k_m" in f.lower()]
                                filename = q4[0] if q4 else m_gguf_files[0]
                                repo_id = m.id
                                log.info(f"Auto-resolved to GGUF repo: {repo_id} / {filename}")
                                break
                        except Exception:
                            pass

            if filename:
                log.info(f"Downloading {filename} from {repo_id} dynamically...")
                return hf_hub_download(repo_id=repo_id, filename=filename)

            return repo_id
        except ImportError:
            log.warning("huggingface_hub not installed. Cannot resolve dynamically.")
        except Exception as e:
            log.error(f"Failed to resolve model path {path_or_repo}: {e}")

    return path_or_repo


def load_config(config_path: Optional[str] = None) -> DeepAgentConfig:
    """Load configuration from YAML file, falling back to defaults."""
    if config_path is None:
        # Search upward from CWD for config.yaml
        search = Path.cwd()
        for parent in [search] + list(search.parents):
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break

    config = DeepAgentConfig()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Recursive update
        def update_dataclass(instance, data):
            for key, value in data.items():
                if hasattr(instance, key):
                    current_val = getattr(instance, key)
                    if isinstance(value, dict) and not isinstance(current_val, dict):
                        update_dataclass(current_val, value)
                    else:
                        setattr(instance, key, value)

        update_dataclass(config, raw)

    return config
