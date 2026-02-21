# Task Bank — 30 Tasks (10 per category)
Difficulty pre-calibrated from pilot data. Top-tercile median time = hard; bottom = easy.

---
## Category A — Information Synthesis

| ID  | Diff   | Task |
|-----|--------|------|
| A01 | Medium | Summarize the three strongest empirical arguments for and against carbon pricing as climate policy. Cite the causal mechanism for each. |
| A02 | Easy   | What are the key components of HIPAA's Safe Harbor de-identification standard? Provide a structured summary a non-lawyer could act on. |
| A03 | Medium | Explain precision, recall, and F1. Give a concrete medical diagnosis example where optimizing each leads to a different clinical decision. |
| A04 | Hard   | Synthesize empirical evidence on whether raising the minimum wage causes unemployment. Summarize two competing findings and the methodological reason they differ. |
| A05 | Easy   | What is the difference between Type I and Type II errors? Describe a real-world scenario where each type of error would be more costly. |
| A06 | Medium | Describe the principal-agent problem in corporate governance. Explain two mechanisms firms use to reduce it and their known limitations. |
| A07 | Hard   | What are the main regulatory differences between a chartered US bank and a fintech lender? Focus on capital requirements and consumer protection. |
| A08 | Medium | Explain how HTTPS protects data in transit. Cover the TLS handshake and identify what it does and does not protect against. |
| A09 | Hard   | Summarize the ACA's individual coverage mandate, its legal history after the 2017 TCJA, and its current legal status. |
| A10 | Medium | What are the primary factors cited in economics literature for the gender wage gap? Which are empirically contested and why? |

---
## Category B — Structured Writing

| ID  | Diff   | Task |
|-----|--------|------|
| B01 | Medium | Write a 200-word executive summary for a board presentation recommending migration of a legacy CRM to Salesforce. Include cost, key risks, and projected benefit. |
| B02 | Hard   | Draft a policy memo recommending a 3-days-in-office hybrid schedule for a 500-person consulting firm. Address productivity evidence, culture, and real estate cost. |
| B03 | Easy   | Write a professional email to a client informing them a deliverable is delayed 10 business days due to an external dependency. |
| B04 | Medium | Compose a job description for a Senior ML Engineer at a climate tech startup. Include responsibilities, requirements, and a realistic compensation range. |
| B05 | Easy   | Write a 150-word abstract for a paper on the effect of sleep deprivation on decision-making in financial traders. |
| B06 | Medium | Draft a change management email to all staff announcing retirement of a legacy expense system in favor of a new automated tool in 30 days. |
| B07 | Hard   | Write a risk register entry for a company deploying a customer-facing financial advice chatbot. Address compliance risk, likelihood, impact, and mitigations. |
| B08 | Medium | Compose a cold proposal email to a potential academic co-author pitching a joint study on algorithmic hiring bias in large firms. |
| B09 | Hard   | Write a vendor comparison table and recommendation for AWS vs GCP vs Azure for a HIPAA-covered healthcare analytics workload. |
| B10 | Easy   | Draft an internal FAQ (5 Q&As) for employees about a new company AI acceptable-use policy. |

---
## Category C — Coding / Debugging

| ID  | Diff   | Task |
|-----|--------|------|
| C01 | Easy   | Write Python `deduplicate_df(df, key)` that removes duplicate rows by key keeping first. Include docstring. |
| C02 | Medium | Fix IndexError for len<=1: `def second_largest(lst): return sorted(set(lst))[-2]`. Add docstring and tests. |
| C03 | Medium | Write 3 pytest tests for `merge_sorted_arrays(a, b)`. Cover: empty input, equal elements, negative numbers. |
| C04 | Medium | SQL on orders(order_id, customer_id, amount, order_date) + customers(customer_id, name): top 5 customers by total revenue last 30 days. |
| C05 | Hard   | Vectorize with NumPy: `result=[]; [result.append(A[i]*B[j]) for i in range(len(A)) for j in range(len(B))]` |
| C06 | Hard   | Python decorator `@log_calls` logging function name, inputs, return value, and wall-clock time to calls.log as JSON per line. |
| C07 | Hard   | JS memory leak — identify and fix: `document.getElementById('btn').addEventListener('click', () => { fetch('/api/data').then(r=>r.json()).then(data=>render(data)); });` |
| C08 | Medium | Bash script: check disk usage on / every 5 min; append alert to /var/log/disk_alert.log if >80%. |
| C09 | Easy   | Python regex validating an email address. Explain each component of the pattern. |
| C10 | Hard   | Fix race condition: `async def update(key, val): shared_dict[key] = val`. Add asyncio locking. |
