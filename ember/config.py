"""Non-secret configuration defaults. Secrets live in .env, which is never committed."""

DOTENV_PATH = ".env"

DEFAULT_API_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Thinking"
DEFAULT_WORKSPACE = "./workspace"
DEFAULT_MAX_STEPS = 100
DEFAULT_CONTEXT_WINDOW = 32_000
DEFAULT_TOOL_TIMEOUT = 30.0
DEFAULT_COMPACT_THRESHOLD = 0.80
DEFAULT_KEEP_FRESH = 6
DEFAULT_LLM_MAX_RETRIES = 3
DEFAULT_LLM_RETRY_BASE_DELAY = 1.0

MEMORY_MAX_FILE_BYTES = 5_000_000
MAX_FILE_READ_BYTES = 400_000
EXEC_STDOUT_CAP = 50_000
EXEC_STDERR_CAP = 12_000
GREP_RESULTS_CAP = 100
GREP_OUTPUT_CAP = 30_000

# --- TinyFish web tools (https://docs.tinyfish.ai) ---------------------------
# The API key itself is read from the TINYFISH_API_KEY environment variable at
# call time and is never stored here.
TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"
TINYFISH_FETCH_URL = "https://api.fetch.tinyfish.ai"
TINYFISH_TIMEOUT = 45.0
TINYFISH_MAX_URLS = 10          # hard limit imposed by the Fetch API
WEB_SEARCH_RESULTS_CAP = 10
WEB_SNIPPET_CAP = 350
WEB_FETCH_CHAR_CAP = 20_000
