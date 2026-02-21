# Experimental Protocol

## Recruitment
Platform: Prolific Academic | Session: ~90 min | Pay: ~$27 (~$18/hr)
Target N: 200 | Max N: 250

## Session Flow
[Consent] → [Screening Survey] → [Practice Task (unscored)]
→ x9: [Condition Instructions] → [Task ≤20min] → [Submit] → [NASA-TLX]
→ [Exit Survey] → [Debrief]

## Condition Scripts (verbatim — show to participant)
CONTROL: "Complete this task using only your knowledge and a web browser.
Do NOT use any AI tool (ChatGPT, Copilot, Gemini, etc.).
AI use will disqualify your submission. You have 20 minutes."

T1: "Complete this task using the provided AI assistant.
It uses only its training knowledge — no external documents.
Ask it as many questions as you like. You have 20 minutes."

T2: "Complete this task using the provided AI assistant.
It retrieves passages from a curated document collection and cites them.
Review cited sources as part of your answer. You have 20 minutes."

Condition labels T1/T2 are NEVER shown to participants.

## Time Measurement
- Start: server timestamp on 'Start Task' click
- End: server timestamp on 'Submit' click
- Pauses: window blur > 60s subtracted; logged as total_pause_s
- Timeout: 20-min cap; forced submit flagged status:timeout

## NASA-TLX (0-100 slider, administered per-task before any score feedback)
1. Mental Demand  2. Physical Demand  3. Temporal Demand
4. Performance    5. Effort           6. Frustration
Scoring: Raw TLX = unweighted mean (Hart & Staveland 1988)
