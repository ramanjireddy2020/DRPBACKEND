"""Shared constants for the InnoDD platform."""

# Session settings
SESSION_TTL_SECONDS = 3600  # 1 hour

# Agent names
AGENT_SUPERVISOR = "supervisor"
AGENT_TXKG = "txkg"
AGENT_LITMINEX = "litminex"
AGENT_CURATEX = "curatex"
AGENT_SCREENSUITE = "screensuite"
AGENT_NOVSEARCH = "novsearch"
AGENT_GENIE = "genie"

# Tool names
TOOL_GRAPH_RESOLUTION = "graph_resolution_tool"
TOOL_TARGET_SCORING = "target_scoring_tool"
TOOL_EVIDENCE_INTERPRETATION = "evidence_interpretation_tool"
TOOL_QUERY_EXPANSION = "query_expansion_tool"
TOOL_RETRIEVAL = "retrieval_tool"
TOOL_SCORING_SUMMARY = "scoring_summary_tool"
TOOL_TARGET_PROFILE_BUILDER = "target_profile_builder"
TOOL_CANDIDATE_MATCHING = "candidate_matching_tool"
TOOL_EVIDENCE_VALIDATION = "evidence_validation_tool"
TOOL_DOCKING_PIPELINE = "docking_pipeline_tool"
TOOL_PATENT_RETRIEVAL = "patent_retrieval_tool"
TOOL_DOCUMENT_PROCESSING = "document_processing_tool"
TOOL_REASONING = "reasoning_tool"
TOOL_GENIE_SQL = "genie_sql_tool"
TOOL_SCHEMA_DISCOVERY = "schema_discovery_tool"
TOOL_CONVERSATIONAL_API = "conversational_api_tool"

# API endpoints
API_BASE_PATH = "/api/v1"
AGENT_ENDPOINT = f"{API_BASE_PATH}/agents"
SUPERVISOR_ENDPOINT = f"{AGENT_ENDPOINT}/supervisor"