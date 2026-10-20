from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

TZ = "Europe/Bucharest"
ZONES = ["RO", "HU", "BG", "RS", "SK"]
NEIGHBORS = ["HU", "BG", "RS", "SK"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    entsoe_token: str = ""
    database_url: str = "sqlite:///pzu.db"
    opcom_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
