# Pinned evaluation gold

`minerva_mathai.jsonl` is the unmodified test JSONL from
[`math-ai/minervamath`](https://huggingface.co/datasets/math-ai/minervamath),
revision `ee46ddc498933b1977577953250ca5c66be64f96`.

- Source: <https://huggingface.co/datasets/math-ai/minervamath/resolve/ee46ddc498933b1977577953250ca5c66be64f96/test.jsonl>
- SHA256: `e04ec2bc1d8f80a0091b4b1989f1ffb366f3583f5bae381546a066d4ccc26e3d`
- 272 unique questions with the original short `answer` field.

PeRL release evaluation uses this dataset. The rescorer requires an exact question
match and preserves the input `gold_answer` separately from `judge_gold_answer`.
The snapshot is included so saved-generation repair does not require network access.
