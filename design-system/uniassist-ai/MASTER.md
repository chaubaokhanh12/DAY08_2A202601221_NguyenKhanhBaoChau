# UniAssist AI — Academic Research Copilot

## Design Intent

UniAssist is a trustworthy university-services research assistant. The interface combines AI-native conversational patterns with an editorial bento layout, visible knowledge metrics, and the authority of an institutional service. It should feel energetic and contemporary while keeping long answers calm, evidence-led, and readable.

## Style

- Primary: Bento Grid + Soft UI Evolution.
- Supporting: AI-Native UI + Trust & Authority + Accessible & Ethical.
- Visual variance: 7/10 — asymmetric welcome composition with disciplined card geometry.
- Motion: 3/10 — short state transitions and one subtle status pulse.
- Density: 5/10 — rich overview, spacious answer reading area.
- Avoid: generic AI purple gradients, glass-heavy surfaces, ornamental motion, emoji as structural icons, and dense dashboard chrome.

## Color Tokens

| Token | Value | Role |
|---|---:|---|
| `--ink-950` | `#0F172A` | Highest-emphasis text |
| `--navy-800` | `#1E3A5F` | Institutional identity and deep surfaces |
| `--cobalt-600` | `#2563EB` | Primary actions and active states |
| `--teal-700` | `#0F766E` | Retrieval, evidence, and system status |
| `--orange-600` | `#EA580C` | Warm emphasis and knowledge highlights |
| `--blue-100` | `#DBEAFE` | Cobalt supporting surface |
| `--teal-100` | `#CCFBF1` | Status backgrounds |
| `--orange-100` | `#FFEDD5` | Warm supporting surface |
| `--canvas` | `#F3F6FB` | App background |
| `--surface` | `#FFFFFF` | Cards, input, and sidebar |
| `--surface-muted` | `#EDF2F6` | User messages and quiet controls |
| `--text` | `#172033` | Body text |
| `--text-muted` | `#526173` | Secondary text |
| `--border` | `#D9E2EC` | Dividers and card outlines |
| `--danger` | `#B42318` | Error state |

## Typography

- Family: Inter when available, followed by Segoe UI and system sans-serif.
- Base size: 16px with 1.6 line height for long answers.
- Headings: 600–700 weight, tight tracking only above 24px.
- Labels: minimum 12px only for tertiary metadata; interactive and body text remain 14–16px.
- No blocking web-font import.

## Layout

- Fixed 72px application bar.
- Collapsible left retrieval sidebar, 288px desktop width and up to 86vw mobile width.
- Main conversation measure: maximum 800px.
- Spacing uses a 4/8px rhythm with 16, 24, 32, and 48px section tiers.
- Composer remains visible and content has sufficient bottom inset.

## Components

- App bar: vector brand mark, product name, evidence status, and initials avatar.
- Bento hero: asymmetric deep-blue statement card paired with three knowledge metric cards.
- Insight rail: document count, policy/news composition, coverage labels, and knowledge freshness.
- Empty state: bento overview followed by four dataset-grounded topic cards.
- Retrieval sidebar: `top_k` slider plus semantic, BM25, rerank, and fallback status.
- Assistant response: clear role label, readable answer, then an evidence panel.
- Topic cards: distinct cobalt, teal, orange, and navy accents with a short content promise.
- Evidence cards: source title, document type, score, and an optional excerpt.
- User prompt: right-aligned navy bubble with high-contrast white text.

## Interaction and Accessibility

- Every interactive control has a minimum 44px height and at least 8px separation.
- Visible 3px focus ring uses teal with an outer white offset.
- Hover and pressed transitions run for 150–220ms without layout shifts.
- Loading state explicitly says the knowledge base is being searched.
- `prefers-reduced-motion` disables non-essential animation and transitions.
- Color never acts as the only status indicator; each state also has text.
- Native Streamlit inputs and buttons remain in use for keyboard and screen-reader behavior.

## Responsive Rules

- Below 720px, hide secondary app-bar copy and reduce gutters to 16px.
- The sidebar becomes an overlay no wider than 320px or 86vw.
- Question starters stack naturally and message width expands to at most 90%.
- Source cards remain full-width and avoid horizontal scrolling.
