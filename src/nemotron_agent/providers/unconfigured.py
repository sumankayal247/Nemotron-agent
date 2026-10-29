"""Placeholder used until interactive first-run setup has completed."""

from nemotron_agent.logging_utils import AuthenticationError
from nemotron_agent.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ConnectionTestResult,
    ProviderClient,
)


class UnconfiguredProviderClient(ProviderClient):
    def __init__(self, provider: str) -> None:
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise AuthenticationError(
            "Provider is not configured. Run /setup or /auth set-key."
        )

    def validate_connection(self, model_id: str) -> ConnectionTestResult:
        raise AuthenticationError(
            "Provider is not configured. Run /setup or /auth set-key."
        )
