# Degree-Portal Submission Checklist (recorded 2026-07-19)

Pipeline as shown in the university system, with statuses and deadlines.
Hard external deadlines: **blind-review submission 2026-07-25**, **final
submission 2026-09-30** (portal 实际到期时间 confirms 09-30).

| # | Stage | Status | Notes / gate |
|---|-------|--------|--------------|
| 1 | 开题报告 (proposal) | ⚠️ submitted 05-31, **学院待审核 (college review pending)** | ACTION: ask graduate secretary to process — may gate everything downstream. Registered title differs slightly from final thesis title (see below). |
| 2 | 预答辩 (pre-defense) | ❌ not submitted | ACTION: confirm whether it gates thesis upload; if yes, schedule with supervisor THIS WEEK. |
| 3 | 科研成果 (research outputs) | empty — OK | Path A (all A/B reviews) needs nothing here. Add paper/patent later via 科研模块 if they materialize before 09-30 (tag 学位相关 + 相关章节). |
| 4 | 学位申请 (degree application) | to initiate | May double as defense-eligibility application that unlocks thesis upload. |
| 5 | 学位上报信息 (degree data report) | ✅ done | Verify name spelling "IAKSHIBAEV TIMUR" + discipline 081200 once — diploma text source. |
| 6 | 学位论文 (thesis upload / 送审稿) | ← July target | **Gate: CNKI 查重 (similarity check) mandatory before review chain advances. Only 3 attempts per student.** After-pass chain: 导师审核 → 学院审核 → 送审 (blind review). |
| 7 | 论文答辩 (defense) | later | After reviews return (≈ mid-Aug per long-term plan). |

## Internal schedule (working backwards from Jul 25)

- **Jul 21 (Mon):** message graduate secretary: (a) process 开题 approval;
  (b) does 预答辩 gate upload? (c) does own unpublished 开题报告 text appear
  in the CNKI comparison corpus (self-reuse question)? (d) any PDF
  naming/format requirements for the 送审稿 (blind copy?).
- **Jul 21–22:** user read-through of the PDF; fixes applied.
- **Jul 23:** consistency pass → **upload thesis + run 查重 attempt #1**.
- **Jul 24:** blind-review build finalisation (BlindReview=true, neutralise
  acknowledgement, PDF-text identity grep); buffer for one edit + 查重
  attempt #2 if the rate is high.
- **Jul 25:** supervisor approval nudge + submission complete.

## Title-drift note

Registered at proposal: "Long-Range Video Consistency Evaluation for
Long-Form Video Super-Resolution".
Final thesis: "Long-Range Consistency Evaluation for Long-Video
Super-Resolution: Metric Design and Positional-Encoding Analysis".
Minor drift is normally acceptable; watch for a title field / 题目变更 step
in the 学位论文 form. If the system enforces exact match, aligning the
thesis to the registered title is a 5-minute change (zjuthesis.tex +
covers).

## 查重 (similarity check) facts

- Mandatory before 导师审核; blocks pipeline otherwise.
- 3 attempts total; results advisory — thresholds per department (ask).
- If rate too high: [编辑] → revised PDF → re-check (uses an attempt).
- Thesis reuses own proposal text by design — confirm the proposal is not
  in the comparison corpus before spending attempt #1.
