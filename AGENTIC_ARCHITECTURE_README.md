# Drug Repurposing Platform - Agentic Architecture

## Overview

This is the backend implementation of the agentic user journey for the Drug Repurposing Platform. The system uses a supervisor-orchestrated multi-agent setup where one supervisor agent owns the conversation, session state, intent routing, and next-step decisions.

## Architecture Components

### 1. Base Framework (`src/DRP_Main/app/agents/base/`)
- **base_tool.py**: Abstract base class for agent tools with `ToolResult` dataclass
- **base_agent.py**: Abstract base agent with tool execution framework

### 2. Session Management (`src/DRP_Main/session/`)
- **models.py**: Session state dataclasses including `SessionTypes` enum
- **memory.py**: In-memory session store with optional file persistence
- **serializers.py**: JSON serialization utilities for export

### 3. Supervisor Agent (`src/DRP_Main/app/supervisor/`)
- **supervisor.py**: Main orchestrator with agent registration and routing
- **router.py**: Intent classification and agent routing logic
- **state_manager.py**: Session state management helpers
- **report_compiler.py**: Consolidated report generation
- **conversation_manager.py**: Conversation history and step validation

### 4. Specialist Agents

#### TxKG Agent (`src/DRP_Main/app/agents/txkg/`)
- **Tools**: `GraphResolutionTool`, `TargetScoringTool`, `EvidenceInterpretationTool`
- **Input**: Disease name
- **Output**: Ranked targets, evidence, recommendation, subgraph artifact

#### LitMineX Agent (`src/DRP_Main/app/agents/litminex/`)
- **Service**: Wraps existing `LiteratureService`
- **Input**: Protein names, search keywords
- **Output**: Ranked articles, summaries, recommendation

#### CurateX Agent (`src/DRP_Main/app/agents/curatex/`)
- **Service**: Wraps existing `DrugCurationService`
- **Input**: Target/disease criteria
- **Output**: Candidate compounds, evidence, recommendation

#### ScreenSuite Agent (`src/DRP_Main/app/agents/screensuite/`)
- **Pipeline**: Uses existing `pipeline_service`
- **Input**: Protein and drug names
- **Output**: Docking results, top hits, interaction artifact

#### NovSearch Agent (`src/DRP_Main/app/agents/novsearch/`)
- **Input**: Drug/target query
- **Output**: Patent findings, novelty verdict, recommendation

#### Genie Agent (`src/DRP_Main/app/agents/genie/`)
- **Fallback agent** for general SQL/lakehouse queries
- Wraps Databricks Genie conversational API

### 5. Core Services (`src/DRP_Main/core/`)
- **llm.py**: Unified LLM client (Groq, Gemini)
- **databricks_client.py**: Databricks SDK wrapper
- **vector_search.py**: Qdrant vector search client
- **config.py**: Settings (existing)

### 6. Artifacts (`src/DRP_Main/artifacts/`)
- **subgraph_renderer.py**: Knowledge graph subgraph JSON
- **binding_pose_renderer.py**: Interaction visualization JSON

### 7. API Layer (`src/DRP_Main/api/`)
- **routes.py**: Agent interface endpoints
- **Endpoints**:
  - `POST /api/v1/agents/message`: Send message to supervisor
  - `GET /api/v1/agents/session/{id}`: Get session state
  - `GET /api/v1/agents/report/{id}`: Generate report
  - `DELETE /api/v1/agents/session/{id}`: Delete session

## Usage

### Starting a new session
```python
from DRP_Main.app.supervisor.supervisor import supervisor

result = supervisor.execute({
    "message": "Find targets for Alzheimer's disease",
    "session_id": None  # Creates new session
})
```

### Continuing a session
```python
result = supervisor.execute({
    "message": "Show me drugs for the top target",
    "session_id": "<session_id from previous result>"
})
```

### Running full pipeline
```python
result = supervisor.execute({
    "message": "Run full analysis for Alzheimer's disease",
    "session_id": None
})
```

## Session State Structure

```python
SessionState:
  - session_id: UUID
  - disease: Optional[str]
  - targets: List[Dict]
  - selected_drugs: List[Dict]
  - docking_results: List[Dict]
  - top_hits: List[Dict]
  - articles: List[Dict]
  - patent_findings: List[Dict]
  - last_agent: Optional[str]
```

## Intent Types

- `FULL_PIPELINE`: Complete workflow from target identification to novelty search
- `STEP_SPECIFIC`: Single agent step (e.g., "show drugs", "dock these")
- `FRESH_QUERY`: New session query
- `FOLLOW_UP`: Continuation of previous agent's output
- `GENERAL_QUERY`: Non-pipeline query (routes to Genie agent)

## Next Steps

1. Complete tool implementations for each agent
2. Add WebSocket support for streaming responses
3. Integrate Genie conversational API
4. Add comprehensive error handling
5. Write unit tests for all components