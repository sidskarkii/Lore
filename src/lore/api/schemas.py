"""API schemas — Pydantic models for all request/response payloads.

These define the contract between the frontend and backend. Every API
endpoint uses these for validation and serialization. The frontend
TypeScript types should mirror these exactly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Search ────────────────────────────────────────────────────────────────
"""
SearchRequest takes a query string, optional result
count, and optional topic/subtopic filters. SearchResult is what comes back
for each match: the chunk text, which collection/episode it's from,
timestamps, and a URL to link back to the source.
"""
#--------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Search the knowledge base."""
    query: str = Field(..., description="Natural language search query")
    n_results: int = Field(5, ge=1, le=20, description="Number of results to return")
    topic: str | None = Field(None, description="Filter by topic (e.g. '3d', 'ai', 'code')")
    subtopic: str | None = Field(None, description="Filter by subtopic (e.g. 'blender', 'houdini')")


class SearchResult(BaseModel):
    """A single search result with source metadata."""
    text: str = Field(..., description="The chunk text (may be parent-expanded)")
    collection: str = Field(..., description="Collection ID")
    collection_display: str = Field(..., description="Human-readable collection name")
    episode_num: int = Field(..., description="Episode number within collection")
    episode_title: str = Field(..., description="Episode title")
    timestamp: str = Field(..., description="Start timestamp as MM:SS")
    start_sec: int = Field(..., description="Start time in seconds")
    end_sec: int = Field(..., description="End time in seconds")
    url: str = Field("", description="Source URL (video link, doc page)")
    topic: str = Field("")
    subtopic: str = Field("")


class SearchResponse(BaseModel):
    """Search results."""
    results: list[SearchResult]
    query: str
    total: int


# ── Chat ──────────────────────────────────────────────────────────────────
"""
ChatRequest carries the conversation history as a list of
messages, plus options like how many sources to retrieve, whether to use
multi-hop search, and optional provider/model overrides. ChatResponse bundles
the LLM's answer with the sources it used.
"""
#---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    """Send a message and get an LLM response grounded in search results."""
    session_id: str | None = Field(None, description="Session ID to continue (creates new if null)")
    messages: list[ChatMessage] = Field(..., description="Conversation history")
    n_sources: int = Field(5, ge=1, le=10, description="Number of sources to retrieve")
    topic: str | None = Field(None, description="Filter search by topic")
    subtopic: str | None = Field(None, description="Filter search by subtopic")
    model: str | None = Field(None, description="Override the active provider's model")
    provider: str | None = Field(None, description="Override the active provider")
    multi_hop: bool = Field(False, description="Use multi-hop search for complex cross-tutorial queries")


class ChatResponse(BaseModel):
    """Non-streaming chat response."""
    session_id: str = Field(..., description="Session ID (new or continued)")
    answer: str = Field(..., description="The LLM's response")
    sources: list[SearchResult] = Field(default_factory=list, description="Retrieved sources")
    provider: str = Field(..., description="Which provider generated the answer")
    model: str = Field(..., description="Which model was used")


# ── Providers ─────────────────────────────────────────────────────────────

class ProviderModelInfo(BaseModel):
    """A model available through a provider."""
    id: str
    name: str
    free: bool = False
    context_window: int = 0


class ProviderInfo(BaseModel):
    """Status of a provider."""
    name: str = Field(..., description="Provider ID")
    display_name: str = Field(..., description="Human-readable name")
    installed: bool = False
    authenticated: bool = False
    version: str | None = None
    error: str | None = None
    models: list[ProviderModelInfo] = Field(default_factory=list)
    free_model_count: int = 0
    is_active: bool = False


class ProvidersResponse(BaseModel):
    """Status of all providers."""
    providers: list[ProviderInfo]
    active: str | None = Field(None, description="Currently active provider name")


class SetActiveRequest(BaseModel):
    """Switch the active provider."""
    provider: str = Field(..., description="Provider name to activate")
    model: str | None = Field(None, description="Default model for this provider")


class TestConnectionRequest(BaseModel):
    """Test a provider's connection."""
    provider: str | None = Field(None, description="Provider to test (uses active if null)")
    model: str | None = Field(None, description="Model to test with")


class TestConnectionResponse(BaseModel):
    """Result of a connection test."""
    success: bool
    provider: str
    model: str
    latency_ms: float = 0
    response_preview: str = ""
    error: str | None = None


# ── Collections ───────────────────────────────────────────────────────────
"""
For browsing and managing indexed content. A collection is a group of related content (like a YouTube playlist), and each has episodes within it.
"""
# ----------------------------------------------------------------------------


class EpisodeInfo(BaseModel):
    """An episode within a collection."""
    episode_num: int
    episode_title: str


class CollectionInfo(BaseModel):
    """A collection of indexed content."""
    collection: str = Field(..., description="Collection ID")
    collection_display: str = Field(..., description="Human-readable name")
    topic: str
    subtopic: str
    episode_count: int
    episodes: list[EpisodeInfo]


class CollectionsResponse(BaseModel):
    """All indexed collections."""
    collections: list[CollectionInfo]
    total_chunks: int


class DeleteCollectionRequest(BaseModel):
    """Delete a collection."""
    collection: str = Field(..., description="Collection ID to delete")


# ── Sessions ──────────────────────────────────────────────────────────────
"""
Chat session persistence. SessionDetail extends
SessionInfo (line 192) to add the full message list, so the list view can be
lightweight while the detail view has everything.
"""
# ----------------------------------------------------------------------------
class MessageInfo(BaseModel):
    """A message in a chat session."""
    id: str
    role: str
    content: str
    sources: list[dict] = Field(default_factory=list)
    created_at: str
    session_id: str | None = None
    session_title: str | None = None


class SessionInfo(BaseModel):
    """Summary of a chat session (for list views)."""
    id: str
    title: str
    created_at: str
    updated_at: str
    provider: str | None = None
    model: str | None = None
    message_count: int = 0


class SessionDetail(SessionInfo):
    """Full session with all messages."""
    messages: list[MessageInfo] = Field(default_factory=list)


class SessionsResponse(BaseModel):
    """List of sessions."""
    sessions: list[SessionInfo]


class RenameSessionRequest(BaseModel):
    """Rename a session."""
    title: str = Field(..., description="New title")


class SearchMessagesRequest(BaseModel):
    """Search across all message history."""
    query: str = Field(..., description="Search query")
    limit: int = Field(20, ge=1, le=100)


class SearchMessagesResponse(BaseModel):
    """Search results from message history."""
    query: str
    results: list[MessageInfo]


# ── Health ────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Server health check."""
    status: str = "ok"
    version: str = ""
    embedding_model: str = ""
    reranker_model: str = ""
    total_chunks: int = 0
    active_provider: str | None = None
