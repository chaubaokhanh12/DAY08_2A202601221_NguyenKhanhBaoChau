# UniAssist Streamlit UI Recreation

## Goal

Replace the current user interface in `app.py` with the supplied UniAssist AI design while preserving the existing Streamlit chat behavior and RAG integration.

## Scope

The change is limited to the presentation and interaction layer in `app.py`. The existing `generate_with_citation(query, top_k=top_k)` call, chat history, suggested queries, source data, and error handling remain functional. Retrieval and generation modules under `src/` are not redesigned.

## Implementation Approach

Use native Streamlit widgets for all stateful interactions and apply a custom CSS theme plus small HTML fragments for decorative structure. This keeps Streamlit reruns and session state reliable while matching the supplied HTML closely. The design will not embed a separate JavaScript application or reproduce the supplied page as a non-interactive iframe.

## Visual System

- Use Inter as the primary typeface with system sans-serif fallback.
- Use an off-white `#f9f9f9` page surface, white elevated inputs and cards, deep navy `#002b5a` for institutional accents, and teal `#006a65` for AI/status accents.
- Constrain the conversation column to 800px and center it within the available viewport.
- Use 16px radii for content containers, pill shapes for suggestion controls, subtle neutral borders, and low-opacity ambient shadows.
- Suppress Streamlit's default header, menu, footer, sidebar, and excess block padding so the custom interface controls the visual hierarchy.

## Page Structure

### Header

A fixed 64px header contains the UniAssist AI mark and wordmark on the left, a minimal Home navigation label, and compact theme/profile affordances on the right. On narrow screens, secondary navigation is hidden while the brand remains visible.

### Conversation Area

The scrollable conversation region begins below the fixed header and leaves sufficient bottom space for the composer. User messages are right-aligned in a pale blue-gray rounded bubble. Assistant responses are left-aligned beside a compact bot avatar and rendered as readable, unboxed content with generous line height.

When no messages exist, a compact welcome state introduces UniAssist and directs the user toward the suggestion chips or composer.

### Sources

Each assistant response with retrieved sources includes a Sources section. Sources appear as responsive cards containing the source name, document type, score, and an expandable excerpt. The cards use a primary or teal accent edge and preserve every source returned by the pipeline.

### Composer

The bottom composer is visually anchored beneath the conversation. Suggested-query pills appear above the native `st.chat_input`. Clicking a suggestion sets the pending query and follows the same processing path as typed input. A brief institutional-information disclaimer appears below the input.

### Settings

The default Streamlit sidebar is removed. The `top_k` control remains available in a compact settings popover-like expander near the top of the content area. Its allowed range remains 3 through 10 with a default of 5.

## Data and Interaction Flow

1. Initialize `messages`, `pending_query`, and `top_k` in Streamlit session state.
2. Render prior messages and their source collections.
3. Accept either a suggestion click or a value from `st.chat_input`.
4. Append and render the user message.
5. Call `generate_with_citation` inside the assistant status/spinner state.
6. Render the returned answer and source cards, then append both to chat history.
7. Preserve the current friendly messages for unimplemented generation and unexpected pipeline failures.

## Responsive Behavior

At desktop widths, the header spans the page and the chat remains centered at 800px. At tablet and mobile widths, horizontal padding decreases, user bubbles may occupy up to roughly 88% of the width, header utilities are reduced, source cards become full-width, and suggestion pills scroll horizontally without a visible scrollbar.

## Error Handling

- `NotImplementedError` produces the existing task-completion guidance without breaking the chat thread.
- Other exceptions produce a concise pipeline error inside the assistant response.
- Missing or malformed source fields use safe defaults for source name, type, score, and content.
- Empty source lists omit the Sources section.

## Verification

- Confirm `app.py` parses and imports successfully in the available Python environment.
- Run focused automated tests that do not require unavailable external services.
- Launch Streamlit and inspect the initial, active-chat, source-card, error, desktop, and narrow-screen states when the local environment supports it.
- Confirm typed questions and suggestion pills both reach `generate_with_citation` with the selected `top_k`.

## Non-Goals

- Replacing Streamlit with React, Tailwind, or another standalone frontend.
- Changing retrieval, ranking, generation, or evaluation behavior.
- Implementing authentication, real profile management, or persistent dark-mode state.
- Depending on externally hosted reference images from the supplied HTML.
