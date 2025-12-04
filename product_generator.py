import os
import re
import json
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior ecommerce copywriter for an Australian horse-gear retailer. Write unique, warm, and practical storefront descriptions for each input item, following all rules below.

Your output must always follow Australian consumer law requirements and ACCC guidance for truthful, clear, and non-misleading representations. Treat all product details and any supplied rulebook data as authoritative, and resolve any ambiguity conservatively in favour of safety, clarity, and accuracy. You must also cross-check a provided Excel sheet containing the Australian Horse Guide Association Rule of 2025, ensuring your description does not contradict, breach, or imply non-compliant usage according to those rules. Never explicitly mention the rules, the association, the law, or the sheet; simply avoid statements that would conflict with them.

The input may include a “description” column containing existing marketing text. Extract factual content only; all wording must be fully rewritten. The input may also include an “image” column containing a product image.

────────────────────────────────────────
MANDATORY REWRITE RULE (STRICT)
────────────────────────────────────────

The “description” column may contain existing marketing copy. You MUST fully rewrite any such text into a new, original, non-derivative narrative.

• Do NOT copy or reuse original sentences, structure, sequencing, adjectives, or marketing phrasing.  
• You may use factual elements only (e.g., materials, closures, weights), but ALL wording must be newly written.  
• The final description must read as independently authored, not as a paraphrase.  
• Do NOT mirror the flow or order of the input text.  
• Avoid promotional tone or brand-style flourishes found in the input.

────────────────────────────────────────
IMAGE-ACCURACY RULES (STRICT & MANDATORY)
────────────────────────────────────────

When an image is provided, you MUST visually inspect it and include **exactly ONE** (1) image-derived visual trait in the final paragraph.

Allowed visual traits include:
• an unambiguous visible colour,  
• a visible pattern,  
• tidy stitching,  
• a smooth or clean finish,  
• a straight or streamlined profile,  
• neat fastening placement.

You MUST NOT:
• guess or invent colours,  
• assume brand-typical colours unless clearly visible,  
• infer performance, durability, safety, or fit from the image,  
• describe anything not plainly visible,  
• describe the horse itself.

If colour appears ambiguous, washed out, distorted, shadowed, or unclear, you MUST choose a **neutral, non-colour visual trait** such as:
• “a tidy straight profile”,  
• “a smooth outer finish”,  
• “clean stitching”,  
• “a neat streamlined outline”.

***You may NOT omit the visual trait under any circumstance.  
If a neutral, factual visual trait is visible, you MUST include it.***  
This rule overrides model uncertainty. You must include EXACTLY one (1) image-derived trait—never zero, never more.

You MUST integrate the visual trait naturally without mentioning or implying image inspection.  
Forbidden: “image”, “photo”, “picture”, “as seen”, “shown here”, “in the image”.

────────────────────────────────────────
OVERALL OUTPUT RULES
────────────────────────────────────────

• Output ONLY the final description paragraph.  
• One paragraph, smooth narrative (no headings, lists, or markdown).  
• Start directly with the description.

Word count rules:
• ≤5 attributes → max 100 words  
• 6–10 attributes → 100–150 words  
• 10+ attributes → 150–250 words

If `manufacturer_part_number` is provided, include it naturally near the end.

BANNED WORDS:
premium, immersive, elevate/transform, revolutionary/breakthrough, ultimate, optimise, leverage

Forbidden:
• mention of ACCC, compliance, regulations, Australia, or rulebooks,  
• unverifiable performance claims,  
• exaggerated benefit claims,  
• comparative superiority (“best”, “superior”),  
• any implication of image analysis.

Tone: friendly, natural, conversational; avoid list-like writing.

────────────────────────────────────────
ACCC-SAFE CLAIMS
────────────────────────────────────────

You must:
• Avoid absolute claims (“guaranteed”, “perfect fit”).  
• Avoid performance or durability promises unless explicitly supported.  
• Avoid veterinary, corrective, or therapeutic implications.  
• Present benefits as factual and reasonable.  
• When unsure, choose the safest, least extreme phrasing.

────────────────────────────────────────
RULEBOOK ALIGNMENT
────────────────────────────────────────

Ensure the description is consistent with the Australian Horse Guide Association Rule of 2025:

• Do not imply unsafe, discouraged, or improper use.  
• Emphasise appropriate handling or fit only when relevant.  
• Never reference the rulebook or regulations.  
• Avoid therapeutic or corrective implications.

────────────────────────────────────────
STRUCTURE & FLOW
────────────────────────────────────────

STEP 1 — Parse inputs  
• Identify product purpose, category, and key attributes.  
• Extract factual details from the description column and rewrite fully.  
• Extract exactly one compliant visual trait from the image.  
• Ensure consistency with ACCC expectations and rulebook alignment.

STEP 2 — Opening  
• Introduce the product’s purpose or defining role.  
• Optionally weave in the chosen visual trait.

STEP 3 — Body  
• Maintain fluid, connected narrative.  
• Describe materials, intended use, comfort, and practical details conservatively.  
• Integrate exactly one image-derived trait.  
• Avoid exaggeration or invented features.

STEP 4 — Closing  
• Finish with a grounded benefit or simple use note.  
• Include model number when available.  
• Avoid hype or urgency language.

────────────────────────────────────────
FINAL CHECKLIST (STRICT ENFORCEMENT)
────────────────────────────────────────

Before outputting the description, silently confirm:

□ All language is newly written (not paraphrased).  
□ EXACTLY ONE image-derived visual trait is included.  
□ The trait is factual, visible, and not invented.  
□ If colour is unclear, a neutral visual trait was used.  
□ No references to images, photos, or inspection.  
□ No banned words.  
□ No mentions of law, ACCC, Australia, or rulebooks.  
□ No unverifiable claims or exaggerations.  
□ Word count fits rules.  
□ Model number included (if provided).  
□ Exactly one paragraph, no lists or headings.

Output ONLY the final rewritten description paragraph.
"""

# Rotate styles to keep copy varied across a batch.
# (Avoid "premium" here since it's banned in the prompt)
STYLE_ROTATIONS = [
    "friendly & concise",
    "down-to-earth & practical",
    "sporty & performance-aware",
    "minimalist & factual",
    "rugged & everyday use",
]


@dataclass
class BatchConfig:
    model: str = "gpt-5.1"
    temperature: float = 0.8
    batch_size: int = 12
    concurrency: int = 4
    api_key_env: str = "OPENAI_API_KEY"
    rules_text: Optional[str] = None


def _load_rules_text(rules_path: Optional[str]) -> Optional[str]:
    """Load the rulebook Excel/CSV into a compact CSV string for prompt injection.

    If the path is not provided or the file is missing, this is treated as NON-FATAL
    (we log a warning and return None) so that generation can still proceed.
    """
    if not rules_path:
        logger.info("No rules_path provided; running without rulebook injection.")
        return None

    rules_path_str = str(rules_path)
    if not os.path.exists(rules_path_str):
        logger.warning("Rules file not found at %s; running without rulebook.", rules_path_str)
        return None

    ext = rules_path_str.lower().split(".")[-1]
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(rules_path_str)
    else:
        df = pd.read_csv(rules_path_str)

    csv_text = df.to_csv(index=False)
    max_chars = 12000
    if len(csv_text) > max_chars:
        csv_text = csv_text[:max_chars] + "\n...[truncated for prompt length]..."
    return csv_text


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map flexible input headings to normalized keys."""
    norm = {c: re.sub(r"\s+", " ", c.strip()).lower() for c in df.columns}
    rev = {v: k for k, v in norm.items()}

    def pick(*cands):
        for c in cands:
            if c in rev:
                return rev[c]
        return None

    cols = {
        "name": pick("product name", "name", "product", "title"),
        "desc": pick("description", "product description"),
        "brand": pick("brand", "manufacturer"),
        "price": pick("price", "rrp"),
        "url": pick("product url", "url", "link"),
        "image": pick("image", "image url", "img"),
        # Optional, but helpful for prompt:
        "manufacturer_part_number": pick(
            "manufacturer part number",
            "mpn",
            "model number",
            "sku",
        ),
    }

    missing = [k for k, v in cols.items() if v is None and k in ("name", "desc", "brand")]
    if missing:
        raise ValueError(
            f"Missing required columns (could not find logical match for): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    rename_map = {v: k for k, v in cols.items() if v is not None}
    return df.rename(columns=rename_map)


def _build_item_payload(row: Dict[str, Any], row_id: int, style_hint: str) -> Dict[str, Any]:
    """Create one item payload to pack into a batch call."""
    return {
        "id": row_id,
        "name": str(row.get("name", "") or "").strip(),
        "brand": str(row.get("brand", "") or "").strip(),
        "desc": str(row.get("desc", "") or "").strip(),
        "price": str(row.get("price", "") or "").strip(),
        "url": str(row.get("url", "") or "").strip(),
        "image": str(row.get("image", "") or "").strip(),
        "manufacturer_part_number": str(row.get("manufacturer_part_number", "") or "").strip(),
        "style": style_hint,
    }


def _user_content_for_batch(
    items: List[Dict[str, Any]],
    rules_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build a multimodal 'user' message for a whole batch.
    We provide a JSON array of item objects. Where 'image' is a public URL, we attach it
    as an image_url content block.

    The model must return a JSON array of objects:
        [{ "id": int, "description": str }, ...]
    """
    text_header = (
        "You will receive a JSON array 'items'. For each item, write one paragraph following the system instructions.\n"
        "NEVER mention images/photos/pictures or that you looked at one.\n"
        "Return ONLY JSON (no markdown) as an array of objects: "
        '[{"id": <int>, "description": <string>}, ...]. The order must match the input "id"s.\n\n'
    )

    items_for_text = []
    image_blocks: List[Dict[str, Any]] = []
    for it in items:
        t = {
            "id": it["id"],
            "name": it["name"],
            "brand": it["brand"],
            "price": it["price"],
            "url": it["url"],
            "desc": it["desc"],
            "manufacturer_part_number": it.get("manufacturer_part_number", ""),
            "style_hint": it["style"],
        }
        items_for_text.append(t)

        img = it.get("image", "") or ""
        if isinstance(img, str) and img.lower().startswith(("http://", "https://")):
            image_blocks.append({"type": "image_url", "image_url": {"url": img}})

    text_body = json.dumps({"items": items_for_text}, ensure_ascii=False)
    content: List[Dict[str, Any]] = []

    if rules_text:
        content.append(
            {
                "type": "text",
                "text": "Rulebook table parsed from Australian_Horse_Guide_Association_Rules_2025.xlsx (CSV):\n"
                + rules_text,
            }
        )

    content.append({"type": "text", "text": text_header + text_body})
    content.extend(image_blocks)
    return content


class RateLimitError(Exception):
    """Marker exception for Tenacity to retry on."""
    pass


def _parse_batch_json(s: str) -> List[Dict[str, Any]]:
    """Parse JSON from a model response that may be wrapped in fences or objects."""
    s = s.strip()

    # Strip simple ```json fences if present
    if s.startswith("```"):
        # Remove first and last fence line crudely
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()

    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "items" in obj:
                return obj["items"]
        except Exception:
            pass

    return json.loads(s)


def _extract_output_text(resp) -> str:
    """
    Extract text from either the Responses API shape or Chat Completions shape.
    """
    # Newer "Responses" API
    if hasattr(resp, "output_text"):
        return (resp.output_text or "").strip()

    output = getattr(resp, "output", None)
    if output:
        chunks: List[str] = []
        for item in output:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []):
                    if getattr(c, "type", "") == "output_text":
                        chunks.append(getattr(c, "text", "") or "")
        if chunks:
            return "".join(chunks).strip()

    # Chat Completions shape
    choices = getattr(resp, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        if msg:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if t:
                            parts.append(t)
                if parts:
                    return "".join(parts).strip()

    return ""


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.2, min=2, max=20),
    retry=retry_if_exception_type((RateLimitError, TimeoutError, ConnectionError)),
)
async def _call_batch(
    client: AsyncOpenAI,
    cfg: BatchConfig,
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    One API call for a batch of items; returns list of {id, description}.
    Tries Responses API first, then falls back to Chat Completions.
    """
    user_content = _user_content_for_batch(items, cfg.rules_text)

    # 1) Try Responses API (where available)
    try:
        resp = await client.responses.create(
            model=cfg.model,
            temperature=cfg.temperature,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except TypeError as e:
        # Likely an older SDK complaining about arguments/signature; fall back
        if "unexpected keyword" not in str(e).lower() and "positional" not in str(e).lower():
            raise
        logger.info("Falling back to Chat Completions due to TypeError: %s", e)
        resp = None
    except Exception as e:
        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
            raise RateLimitError(e)
        logger.info("Responses API failed; falling back to Chat Completions: %s", e)
        resp = None

    if resp is None:
        # 2) Fallback: Chat Completions
        try:
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as e:
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                raise RateLimitError(e)
            raise

    text = _extract_output_text(resp).strip()
    if not text:
        raise ValueError("Empty response from model for a batch.")

    # Parse JSON (accept either {"items": [...]} or a bare list)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "items" in parsed:
            return parsed["items"]
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    try:
        return _parse_batch_json(text)
    except Exception as pe:
        raise ValueError(f"Failed to parse model JSON. Raw text:\n{text}") from pe


async def _run_all_batches(
    client: AsyncOpenAI,
    cfg: BatchConfig,
    df_norm: pd.DataFrame,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[int, str]:
    """Schedule all batches concurrently and return {row_index: description}."""
    sem = asyncio.Semaphore(cfg.concurrency)
    results: Dict[int, str] = {}

    total_rows = len(df_norm)

    async def worker(batch_items: List[Tuple[int, Dict[str, Any]]]):
        async with sem:
            payload = []
            for _, (row_idx, row_dict) in enumerate(batch_items):
                style = STYLE_ROTATIONS[(row_idx) % len(STYLE_ROTATIONS)]
                payload.append(_build_item_payload(row_dict, row_idx, style))
            out_list = await _call_batch(client, cfg, payload)
            for obj in out_list:
                rid = int(obj["id"])
                desc = obj.get("description", "") or ""
                desc = " ".join(desc.split())
                words = desc.split()
                if len(words) > 220:
                    desc = " ".join(words[:220])
                results[rid] = desc

                # NEW: report progress for this row
                if progress_cb:
                    try:
                        progress_cb(rid, total_rows)
                    except Exception:
                        # Never let a callback crash the worker
                        pass

    indices = list(df_norm.index)
    batches = [indices[i:i+cfg.batch_size] for i in range(0, len(indices), cfg.batch_size)]
    tasks = []
    for b in batches:
        batch_items = [(i, df_norm.loc[i].to_dict()) for i in b]
        tasks.append(asyncio.create_task(worker(batch_items)))
    await asyncio.gather(*tasks)
    return results



def generate_contextualized_descriptions_batched(
    input_path: str,
    output_path: str,
    model: str = "gpt-5.1",
    temperature: float = 0.8,
    batch_size: int = 12,
    concurrency: int = 4,
    api_key_env: str = "OPENAI_API_KEY",
    preview_rows: Optional[int] = None,
    rules_path: Optional[str] = "Australian_Horse_Guide_Association_Rules_2025.xlsx",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:

    """
    Efficiently generate unique product descriptions by batching rows per API call
    and running multiple calls concurrently.

    Columns (flexible): (Product Name|Name|Product), Description, Brand, Price,
                        (Product URL|URL), Image, (optional) Manufacturer Part Number/SKU.

    Returns the resulting DataFrame with a new column 'Contextualized Description'
    and writes it to `output_path`.
    """
    input_path_str = str(input_path)
    logger.info("Reading input file from %s", input_path_str)

    ext = input_path_str.lower().split(".")[-1]
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(input_path_str)
    else:
        df = pd.read_csv(input_path_str)

    # Normalize columns
    df = _normalize_columns(df)

    # Optional preview subset
    if preview_rows and preview_rows > 0:
        logger.info("Using preview_rows=%s (first N rows only).", preview_rows)
        df = df.head(preview_rows)

    # Load rulebook (if present)
    rules_text = _load_rules_text(rules_path)

    # OpenAI client
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Please set your API key in env var {api_key_env}")
    client = AsyncOpenAI(api_key=api_key)

    cfg = BatchConfig(
        model=model,
        temperature=temperature,
        batch_size=max(1, batch_size),
        concurrency=max(1, concurrency),
        api_key_env=api_key_env,
        rules_text=rules_text,
    )

    logger.info(
        "Starting generation: model=%s, batch_size=%s, concurrency=%s, rows=%s",
        cfg.model,
        cfg.batch_size,
        cfg.concurrency,
        len(df),
    )

    # Run all batches
    # Run with optional progress reporting
    results = asyncio.run(_run_all_batches(client, cfg, df, progress_cb=progress_cb))


    # Attach results in the same row order
    out_col = [results.get(idx, "") for idx in df.index]
    df["Contextualized Description"] = out_col

    # Ensure output directory exists
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_ext = out_path.suffix.lower().lstrip(".")
    logger.info("Writing output to %s (ext=%s)", out_path, out_ext)

    if out_ext in ("xlsx", "xls"):
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    logger.info("Wrote %s rows to %s", len(df), out_path)
    return df
