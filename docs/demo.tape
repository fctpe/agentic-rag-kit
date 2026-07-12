# Regenerate with: vhs docs/demo.tape  (from the repo root)
Output docs/demo.gif
Set FontSize 13
Set Width 1120
Set Height 700
Set Padding 14
Set TypingSpeed 30ms

Type "cd backend && uv run pytest -q"
Enter
Sleep 5s
Type "cd .."
Enter
Type "clear"
Enter
Type `echo "── Retrieval ablation (38 golden questions, hit@6 / MRR) ──"`
Enter
Sleep 500ms
Type `for m in hybrid vector text_only; do jq -r --arg m "$m" '.summary | "\($m)\t hit@6=\(.hit_rate_at_k)  MRR=\(.mrr)  recall=\(.mean_article_recall)"' evals/results/retrieval_$m.json 2>/dev/null; done`
Enter
Sleep 3s
Type `echo "── RAGAS (38 questions, gpt-4o-mini judge) ──" && jq -c '.summary' evals/results/ragas.json`
Enter
Sleep 3s
Type `echo "── Red team (injection / PII / out-of-scope) ──" && jq -r '.summary | "\(.passed)/\(.total) passed"' evals/results/redteam_final_14of14.json`
Enter
Sleep 4s
