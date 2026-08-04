# UniAssist AI — Academic Research Copilot

## Design Intent

UniAssist is a trustworthy university-services research assistant. The interface combines AI-native conversational patterns with the clarity of a knowledge base and the authority of an institutional service. It should feel calm, evidence-led, and fast rather than playful or futuristic.

## Style

- Primary: AI-Native UI + Minimalism.
- Supporting: Accessible & Ethical + Trust & Authority.
- Avoid: AI purple gradients, glass-heavy surfaces, ornamental motion, emoji as structural icons, and dense dashboard chrome.

## Color Tokens

| Token | Value | Role |
|---|---:|---|
| `--navy-950` | `#001B3A` | Highest-emphasis text and header |
| `--navy-800` | `#002B5A` | Primary actions and institutional identity |
| `--navy-700` | `#0B4778` | Hover and selected states |
| `--teal-700` | `#006A65` | Retrieval, evidence, and system status |
| `--teal-100` | `#D9F2EF` | Status backgrounds |
| `--canvas` | `#F4F7FA` | App background |
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
- Knowledge strip: document count, coverage labels, and knowledge freshness.
- Empty state: concise evidence-led message and four dataset-grounded question starters.
- Retrieval sidebar: `top_k` slider plus semantic, BM25, rerank, and fallback status.
- Assistant response: clear role label, readable answer, then an evidence panel.
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
