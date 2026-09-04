import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Centralized configuration for RecoveryOS backend.
    
    All values can be overridden via environment variables or a .env file.
    Defaults match the original hardcoded values to preserve existing behavior.
    """

    # ---------------------------------------------------------
    # SERVER
    # ---------------------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # ---------------------------------------------------------
    # DATABASE (Neon PostgreSQL via Prisma)
    # ---------------------------------------------------------
    DATA_DIR: str = Field(
        default_factory=lambda: os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        )
    )
    DATABASE_URL: str = ""  # Must be set to Neon PostgreSQL connection string

    # ---------------------------------------------------------
    # CORS
    # ---------------------------------------------------------
    CORS_ORIGINS: str = "*"  # Comma-separated origins, or "*" for all

    # ---------------------------------------------------------
    # RAZORPAY TEST MODE CREDENTIALS
    # ---------------------------------------------------------
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""

    # ---------------------------------------------------------
    # GUARDRAIL POLICY THRESHOLDS
    # ---------------------------------------------------------
    HIGH_VALUE_THRESHOLD: float = 50000.0
    MAX_RECOVERY_ATTEMPTS: int = 2
    MAX_DISCOUNT_PCT: float = 5.0
    MAX_DISCOUNT_CAP: float = 500.0
    MIN_CONFIDENCE_THRESHOLD: float = 0.60

    # ---------------------------------------------------------
    # SCORING ENGINE
    # ---------------------------------------------------------
    DEFAULT_MARGIN_PCT: float = 0.40

    # ---------------------------------------------------------
    # DATA SEEDING
    # ---------------------------------------------------------
    SEED_EVENT_COUNT: int = 1000

    # ---------------------------------------------------------
    # GROQ LLM (Recovery Reasoning Agent)
    # ---------------------------------------------------------
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    model_config = {
        "env_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }

    def model_post_init(self, __context) -> None:
        # Ensure DATA_DIR exists
        os.makedirs(self.DATA_DIR, exist_ok=True)

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse CORS_ORIGINS string into a list."""
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
    
    @property
    def has_razorpay_credentials(self) -> bool:
        """Check if Razorpay test credentials are configured."""
        return bool(self.RAZORPAY_KEY_ID) and bool(self.RAZORPAY_KEY_SECRET) and \
               self.RAZORPAY_KEY_ID != "rzp_test_XXXXXXXXXXXXXX"
    
    @property
    def has_groq_key(self) -> bool:
        """Check if Groq API key is configured."""
        return bool(self.GROQ_API_KEY)


settings = Settings()
