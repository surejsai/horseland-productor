import threading
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


from product_generator import generate_contextualized_descriptions_batched

ROOT = Path(__file__).parent.resolve()
OUTPUT_NAME = "horseland_products_described.xlsx"
OUTPUT_PATH = ROOT / OUTPUT_NAME
WORK_DIR = ROOT / "_work"
WORK_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Horseland Generator API (with progress)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Progress state ---------
progress_lock = threading.Lock()
progress_state: Dict[str, Any] = {
    "status": "idle",          # idle | running | done | error
    "total_rows": 0,
    "processed_rows": 0,
    "rows": [],                # list of {"row": int, "status": str}
    "error": None,
}

def _set_progress(status: Optional[str] = None,
                  total_rows: Optional[int] = None,
                  processed_rows: Optional[int] = None,
                  row_done: Optional[int] = None,
                  error: Optional[str] = None):
    with progress_lock:
        if status is not None:
            progress_state["status"] = status
        if total_rows is not None:
            progress_state["total_rows"] = total_rows
        if processed_rows is not None:
            progress_state["processed_rows"] = processed_rows
        if error is not None:
            progress_state["error"] = error
        if row_done is not None:
            progress_state["processed_rows"] += 1
            progress_state["rows"].append({
                "row": row_done + 1,  # 1-based row index for UI
                "status": "done",
            })


def _reset_progress():
    _set_progress(status="idle", total_rows=0, processed_rows=0, error=None)
    with progress_lock:
        progress_state["rows"] = []


def _background_job(products_path: Path, rules_path: Optional[Path], model: str = "gpt-5.1"):
    try:
        # progress callback from generator
        def cb(row_id: int, total_rows: int):
            # We only care about counting; total_rows should be consistent
            _set_progress(total_rows=total_rows, row_done=row_id)

        generate_contextualized_descriptions_batched(
            input_path=str(products_path),
            output_path=str(OUTPUT_PATH),
            model=model,
            temperature=0.8,
            batch_size=12,
            concurrency=4,
            preview_rows=None,
            rules_path=str(rules_path) if rules_path else None,
            progress_cb=cb,
        )
        _set_progress(status="done")
    except Exception as exc:
        _set_progress(status="error", error=str(exc))


@app.post("/run-generator")
async def run_generator(
    products: UploadFile = File(...),
    rules: Optional[UploadFile] = File(None),
):
    # Prevent starting a new job if one is running
    with progress_lock:
        if progress_state["status"] == "running":
            raise HTTPException(status_code=409, detail="A generation job is already running")

    _reset_progress()
    _set_progress(status="running")

    # Save uploaded files into WORK_DIR so the background thread can use them
    products_path = WORK_DIR / "products_upload.xlsx"
    with products_path.open("wb") as f:
        f.write(await products.read())

    rules_path = None
    if rules is not None:
        rules_path = WORK_DIR / "rules_upload.xlsx"
        with rules_path.open("wb") as f:
            f.write(await rules.read())

    # Compute total rows once for nicer UX
    try:
        df_head = pd.read_excel(products_path) if products_path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(products_path)
        total_rows = len(df_head)
    except Exception:
        total_rows = 0

    if total_rows:
        _set_progress(total_rows=total_rows)

    # Start background thread
    thread = threading.Thread(
        target=_background_job,
        args=(products_path, rules_path),
        daemon=True,
    )
    thread.start()

    # Return immediately; frontend will poll /progress
    return JSONResponse({"ok": True, "message": "Job started", "total_rows": total_rows})


@app.get("/progress")
def get_progress():
    with progress_lock:
        return JSONResponse(progress_state.copy())


@app.get(f"/{OUTPUT_NAME}")
def download_output():
    if not OUTPUT_PATH.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(
        OUTPUT_PATH,
        filename=OUTPUT_NAME,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

@app.head(f"/{OUTPUT_NAME}")
def head_output():
    if not OUTPUT_PATH.exists():
        return Response(status_code=404)
    return Response(status_code=200)
