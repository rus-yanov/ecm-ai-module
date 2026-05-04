from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_timeout_sec: int = 60

    ocr_confidence_threshold: float = 0.6
    default_review_threshold: float = 0.7

    max_file_size_mb: int = 20

    ecm_base_url: str = "http://localhost:8001"
    ecm_api_key: str = "mock-key"
    ecm_retry_count: int = 3
    ecm_retry_delays: list[float] = [1.0, 3.0, 9.0]

    log_level: str = "INFO"
    thresholds_config_path: str = "config/thresholds.yaml"


settings = Settings()
