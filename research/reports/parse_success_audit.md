# Parse Success Audit

## Ground Truth From The Parser

`research/real_trace_experiments.py` marks `parse_success = 1` only when the generation matches the exact expected output mode:

- `structured_four_line`: all four labeled lines must be present.
- `minimal_json`: a valid JSON object must be found and parsed.

If exact parsing fails, the code still tries fallback answer extraction from freeform text. That means a trace can have:

- a correct recovered `answer`
- valid hidden states
- `parse_success = 0`
- default `confidence = 50`
- default `stop = False`

So `parse_success` is a strict exact-format metric, not a synonym for answer correctness.

## DeepSeek 1.5B Legacy Runs

### Observed behavior

- Coverage report parse-success rate: `0.010888888888888889` (98 exact parses out of 9000 steps).
- Representative rows show:
  - `output_format_type = fallback_freeform`
  - `answer_extraction_source = full_text_cue_segment` or `after_think_cue_segment`
  - `hit_max_new_tokens = 1`
  - `truncated_output_suspected = 1`

### Diagnosis

DeepSeek 1.5B usually emits long freeform reasoning instead of the requested exact structured format, then hits the token cap before a clean final structured answer is produced. The fallback extractor often recovers a numeric answer from the prose, but that recovery is noisy:

- some steps recover the correct final answer, like `320`
- some steps recover intermediate numbers, like `160` or `2`
- stop and confidence fields are not recovered, so they remain defaults

### Conclusion

This is primarily a prompt-format plus truncation failure, not a trace-schema failure. The traces are usable for some answer-level auditing, but the exact parse-success metric is correctly low.

## Qwen 3.5 9B Frontier Smoke Runs

### Observed behavior

- Parse-success rate: `0.0` across all 4 smoke steps.
- All 4 steps still have:
  - `correct = 1`
  - recovered answers (`tuesday`, `1/4`)
  - valid hidden states
  - `hit_max_new_tokens = 1`
  - `truncated_output_suspected = 1`
  - `output_format_type = fallback_freeform`
  - `stop_extraction_source = default`
  - `confidence_extraction_source = default`

### Raw-output pattern

The raw text shows that Qwen 3.5 9B understands the task and often restates the protocol, but it does not emit the requested final JSON object before truncation. Examples include:

- a prose explanation ending with “I should do one short verification step and return”
- a second-step output beginning with `Thinking Process:` and then truncating before JSON keys are emitted

### Conclusion

This is a strict exact-format parse failure caused by prompt-mode mismatch plus token truncation. It is not an answer-extraction failure for the answer field itself, and it is not a hidden-state failure.

## Classification Summary

| trace set | parse_success_rate | answer_recovery | primary failure mode | secondary issue |
| --- | --- | --- | --- | --- |
| DeepSeek 1.5B legacy full run | 0.0109 | partial, noisy | freeform reasoning instead of exact structured output | heavy truncation causes cue-based intermediate-number extraction |
| Qwen 3.5 9B frontier smoke | 0.0000 | yes, clean answers recovered on all smoke steps | freeform / protocol-restating output instead of exact JSON | truncation prevents stop/confidence fields from appearing |

## Bottom Line

- The parser is behaving consistently with its strict definition of `parse_success`.
- DeepSeek 1.5B is a genuine exact-format failure with truncation and noisy fallback recovery.
- Qwen 3.5 9B smoke is a format/truncation failure, but the answer field is still usable on the observed smoke tasks.
- No evidence suggests a trace schema bug or hidden-state corruption.