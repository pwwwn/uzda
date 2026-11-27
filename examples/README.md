# Uzda Examples

All manifests expect an OpenAI-compatible streaming endpoint with tool call support. Set your own `model.url` and `model.name`; no model is included. Run from the project root, for example:

```sh
cargo run -- serve examples/multistage.yaml
```

Open `http://localhost:8080/sedlo.html`. Set `UZDA_API_KEY` for the endpoint if needed. Configurations:

| Manifest | Purpose |
| --- | --- |
| `chat.yaml` | Simple chat without external commands. |
| `terminal.yaml` | Terminal assistant with arbitrary `bash` access: use only in a trusted, isolated environment. |
| `document-review.yaml` | Read and search text files. |
| `document-drafting.yaml` | The same, plus creation of a new text file. |
| `multistage.yaml` | Analyze a request and review the answer in two stages without external commands. |

`read-document.yaml`, `search-documents.yaml`, and `write-document.yaml` are tool argument schemas, not standalone manifests. Files in `prompts/` and `skills/` are instructions, not executable programs.

## Text documents

Reading and searching require `cat` and `grep` with support for `-r`, `-I`, `-n`, `-F`, `-i`, and `--` (for example, GNU grep); writing requires Python 3. Searches are recursive, line-based, and case-insensitive according to the locale; binary files are skipped. If there are no matches, `grep` returns exit code 1, which the runtime may report as a tool error. Access errors also require separate investigation. An empty pattern matches every line.

Paths are relative to `examples` or absolute; access is not restricted to this directory. Commands run with the permissions of the Uzda process. Writing through `write-document.py` creates a new UTF-8 file and does not overwrite an existing file. PDF, DOCX, XLSX, PPTX, and OCR are not supported: first extract the text using a trusted tool. Tool output may be truncated; do not assume results from large files or searches are complete without further verification.

Example requests (create your own UTF-8 files first):

- “Read `contract.txt` and list the obligations and deadlines with quotations.”
- “Find the word ‘approval’ in the text files in the `reports` directory.”
- “Draft a letter based on `brief.txt` and save it as a new `draft.txt`.”

## Testing the multistage workflow

Run `multistage.yaml` and try these in separate sessions:

1. “A box contained 17 pencils, 9 were handed out, and 4 were added. How many are left? Show the calculation.” Expect 12 and the calculation `17 - 9 + 4`.
2. “Write a short letter about rescheduling a meeting, but the date and recipient are not yet known.” Expect a draft without an invented date or name, or a clarifying question.
3. “What was the company's revenue last year?” Without supplied data, expect a request for clarification rather than a fabricated number.

Check the events or trace for the transition from `analyze` to `review` (`transfer_to_review`) and the final answer. The transition and review quality depend on the model: the manifest does not force it to call the tool, and preliminary text may be visible to the user. This workflow does not connect external tools or check facts online.

## Security and data

The tools are not a sandbox; untrusted requests and documents require external isolation and least-privilege access. Instructions inside documents may influence the model despite the prompt. Requests, document contents, and tool results are sent to the model endpoint and may be stored in traces; do not use sensitive data without appropriate access and retention policies.
