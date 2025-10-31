# gemini-grounding

Grounded Gemini queries with citation extraction and redirect resolution.

## Install (linux)

```bash
git clone https://github.com/darshxm/google-search-with-gemini.git
cd google-search-with-gemini
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
## Install (Windows)

```bash
git clone https://github.com/darshxm/google-search-with-gemini.git
cd google-search-with-gemini
python -m venv .venv
.venv/Scripts/activate
pip install -e .
```

## Usage (library)

```python
from gemini_grounding import query_with_grounded_citations

res = query_with_grounded_citations(
    prompt="Who won the Euro 2024?",
    model="gemini-2.5-flash",
    follow_redirects=True,
    include_raw_resolution=True
)
print(res["enriched_text"])
for c in res["citations"]:
    print(c["redirect_url"], "->", c.get("final_url"))
```

## CLI

```bash
gemini-ground -p "Who won the Euro 2024?" -o out/result.json
```

## Auth

Set `GEMINI_API_KEY` **or** provide an `api.json`:

```json
{"gemini_key": "YOUR_KEY"}
```
