from __future__ import annotations

import json
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _approx_token_eq(a: str, b: str) -> bool:
    """Approximate equality for tokens to handle simple inflections.

    Rules kept minimal and deterministic:
    - Exact match -> True
    - Case-insensitive already handled by normalization in _tokens
    - Strip possessive: trailing "'s" or "’s"
    - Singularize basic English plurals:
      * trailing 's' (length > 3)
      * 'ies' -> 'y'
      * 'es' for words ending with s, x, z, ch, sh
    """

    if a == b:
        return True

    def singularize(t: str) -> str:
        # strip possessive
        if t.endswith("'s") or t.endswith("’s"):
            t = t[:-2]
        # common plural -> singular
        if len(t) > 3 and t.endswith("ies"):
            return t[:-3] + "y"
        if len(t) > 3 and t.endswith("s"):
            # handle -es for s, x, z, ch, sh
            if t.endswith("es") and (
                t[:-2].endswith("s")
                or t[:-2].endswith("x")
                or t[:-2].endswith("z")
                or t[:-2].endswith("ch")
                or t[:-2].endswith("sh")
            ):
                return t[:-2]
            return t[:-1]
        return t

    return singularize(a) == singularize(b)


def _normalize_text(s: str) -> str:
    """Normalize text to match tokenization used in tests.

    - Lowercase
    - Replace hyphen-like characters with space
    - Collapse multiple spaces
    """
    s2 = re.sub(r"[\-\u2010\u2011\u2012\u2013\u2014]", " ", s.lower())
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _tokens(s: str) -> List[str]:
    s2 = _normalize_text(s)
    return re.findall(r"\b\w+\b", s2)


def _is_finite_number(x: Any) -> bool:
    try:
        fx = float(x)
    except Exception:
        return False
    return fx == fx and fx not in (float("inf"), float("-inf"))


def _extract_success_words(word_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for w in word_items:
        if not isinstance(w, dict):
            continue
        if w.get("case") != "success":
            continue
        if not ("start" in w and "end" in w):
            continue
        if not (_is_finite_number(w["start"]) and _is_finite_number(w["end"])):
            continue
        out.append(w)
    return out


def _build_word_token_list(success_words: List[Dict[str, Any]]) -> List[str]:
    return [t for t in (_tokens(w.get("word", "")) for w in success_words) for t in t]


def _find_subsequence_span(
    tokens_line: List[str], tokens_words: List[str]
) -> Tuple[int, int]:
    """Find indices in tokens_words corresponding to the first and last token of tokens_line.

    Greedy in-order matching: for each token in tokens_line, find the next occurrence
    in tokens_words strictly after the previous match. Returns (first_index, last_index).
    Raises ValueError if any token is not found in order.
    """
    if not tokens_line:
        raise ValueError("alignment_required: empty line tokens")

    pos = 0
    first_idx: Optional[int] = None
    last_idx: Optional[int] = None
    n = len(tokens_words)

    for tok in tokens_line:
        found = False
        while pos < n:
            if tokens_words[pos] == tok:
                if first_idx is None:
                    first_idx = pos
                last_idx = pos
                pos += 1
                found = True
                break
            pos += 1
        if not found:
            raise ValueError(
                "alignment_required: failed to create text_over item (token not found)"
            )

    assert first_idx is not None and last_idx is not None
    return first_idx, last_idx


class TextOverBuilder3:
    """Xây dựng text_over bằng thuật toán con trỏ đơn giản (không fallback), phù hợp
    với cách Gentle tạo words.json.

    Nguồn token (theo Gentle):
    - Dùng trường `word` của mọi phần tử để tạo chuỗi token toàn cục (word_token_stream)
      bằng tokenizer nhẹ `_tokens()`; giữ nguyên thứ tự như words.json.
    - Vẫn giữ thứ tự và số lượng phần tử như words.json; có thể gặp:
      * case == "success": có `start`/`end`.
      * case == "not-found-in-audio": không có thời gian.
    - Lưu ánh xạ: token_index -> chỉ số phần tử gốc trong words.json.

    Cho mỗi dòng transcript:
    - Tokenize nhẹ theo cùng tinh thần Gentle: lowercase, bỏ punctuation ngoại biên,
      giữ "'" và "-" khi nằm giữa chữ/số. Nếu dòng rỗng sau chuẩn hoá thì BỎ QUA (không coi là fail).
    - Tìm start (t0) bằng con trỏ:
      * base = cursor; thử lần lượt base+0, +1, +2, +3 với token đầu dòng (x = offset khớp).
      * Nếu không khớp, dịch base theo lô step_k (= 4) và lặp; dừng khi base >= len(stream).
    - Tìm end (t1):
      * L = len(line_toks); est_t1 = t0 + (L - x) - 1.
      * Snap theo thứ tự cố định: est, est-1, est+1, est-2, est+2 (giới hạn trong [t0, n-1]).
      * Nếu vẫn không có, thử đúng 1 bước lùi (est_t1 - 1). Không fallback subsequence.
    - Nếu có (t0, t1): tính thời gian và emit item; sau đó cursor = t1 + 1.

    Chính sách thời gian (success-only):
    - Chỉ lấy thời gian từ phần tử `case == "success"`.
    - Ánh xạ (t0..t1) -> (w0..w1) theo chỉ số word gốc, rồi tìm success bao-quanh:
      * w0_success: success đầu tiên tại/ sau w0
      * w1_success: success cuối cùng tại/ trước w1
      Nếu thiếu 1 trong 2, coi dòng thất bại.

    Hành vi khi thất bại:
    - Không dùng fallback subsequence. Không tìm được start/end hoặc không suy được thời gian -> raise.
    - Thông điệp lỗi khuyến nghị:
      * "alignment_failed: start_not_found"
      * "alignment_failed: end_not_found"
      * "alignment_failed: timing_not_resolvable"
    - Duration = end(w1_success) - start(w0_success), chặn không âm. Không mở rộng/padding.
    """

    class SequentialMatchConfig:
        """Config for pointer-based sequential matching.

        - step_k: primary stride for probing start token positions (e.g., +3)
        - relax_x: relaxation for expected end index computation (len(line)-x)
        - end_tolerance: try to snap the last token within +/- this window
        - max_linear_scan: fallback linear scan distance when step probes miss
        """

        def __init__(
            self,
            *,
            step_k: int = 4,
            relax_x: int = 0,
            end_tolerance: int = 2,
            max_linear_scan: int = 256,
        ) -> None:
            self.step_k = max(1, step_k)
            self.relax_x = max(0, relax_x)
            self.end_tolerance = max(0, end_tolerance)
            self.max_linear_scan = max(1, max_linear_scan)

    def __init__(
        self,
        *,
        temp_dir: str,
        config: Optional["TextOverBuilder3.SequentialMatchConfig"] = None,
    ) -> None:
        self._temp_dir = temp_dir
        self._cfg = config or TextOverBuilder3.SequentialMatchConfig()

    def build(
        self,
        *,
        word_items: List[Dict[str, Any]],
        chunks: List[str],
        text_over_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(word_items, list) or not word_items:
            raise ValueError("alignment_required: word_items are required")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("alignment_required: chunks are required")

        success_words = _extract_success_words(word_items)
        if not success_words:
            raise ValueError("alignment_required: no successful words with timings")

        # Build global token stream from ALL word_items using tokens of 'word'.
        # Keep token -> original word index map.
        word_token_stream: List[str] = []
        token_to_word_index: List[int] = []
        for wi, w in enumerate(word_items):
            toks = _tokens(w.get("word", ""))
            for tok in toks:
                word_token_stream.append(tok)
                token_to_word_index.append(wi)

        if not word_token_stream:
            raise ValueError("alignment_required: no tokens in success words")

        results: List[Dict[str, Any]] = []

        # Maintain a moving pointer over the token stream across lines
        cursor_token_idx = 0
        n_tokens_words = len(word_token_stream)

        def _safe_emit(w0_idx: int, w1_idx: int, line_text: str, wc: int) -> None:
            nonlocal results
            if w1_idx < w0_idx:
                w0_idx, w1_idx = w1_idx, w0_idx

            # Find surrounding success items to resolve timing
            n_words = len(word_items)
            if not (0 <= w0_idx < n_words and 0 <= w1_idx < n_words):
                raise ValueError("alignment_failed: timing_not_resolvable")

            # w0_success: first success at or after w0_idx
            w0_succ: Optional[int] = None
            for j in range(w0_idx, n_words):
                wj = word_items[j]
                if (
                    wj.get("case") == "success"
                    and _is_finite_number(wj.get("start"))
                    and _is_finite_number(wj.get("end"))
                ):
                    w0_succ = j
                    break

            # w1_success: last success at or before w1_idx
            w1_succ: Optional[int] = None
            for j in range(min(w1_idx, n_words - 1), -1, -1):
                wj = word_items[j]
                if (
                    wj.get("case") == "success"
                    and _is_finite_number(wj.get("start"))
                    and _is_finite_number(wj.get("end"))
                ):
                    w1_succ = j
                    break

            if w0_succ is None or w1_succ is None:
                raise ValueError("alignment_failed: timing_not_resolvable")

            start = float(word_items[w0_succ]["start"])  # type: ignore[arg-type]
            end = float(word_items[w1_succ]["end"])  # type: ignore[arg-type]
            if end < start:
                end = start
            results.append(
                {
                    "text": line_text,
                    "start_time": start,
                    "duration": max(0.0, end - start),
                    "word_count": wc,
                }
            )

        for line in chunks:
            try:
                line_toks = _tokens(line)
                if not line_toks:
                    continue

                # 1) Tìm start theo base+0..+3; nếu không thấy, base += step_k và lặp
                start_tok = line_toks[0]
                found_start_tok_idx: Optional[int] = None
                matched_offset: Optional[int] = None  # x in {0,1,2,3}
                base = cursor_token_idx
                while base < n_tokens_words and found_start_tok_idx is None:
                    for o in range(0, 4):
                        idx = base + o
                        if idx >= n_tokens_words:
                            break
                        wt = word_token_stream[idx]
                        if _approx_token_eq(wt, start_tok):
                            found_start_tok_idx = idx
                            matched_offset = o
                            break
                    if found_start_tok_idx is None:
                        base += self._cfg.step_k

                if found_start_tok_idx is None or matched_offset is None:
                    raise ValueError("alignment_failed: start_not_found")

                t0_idx = found_start_tok_idx

                # 2) Ước lượng end theo L - x, rồi snap theo thứ tự cố định
                L = len(line_toks)
                x = matched_offset
                est_t1 = t0_idx + (L - x) - 1
                est_t1 = min(est_t1, n_tokens_words - 1)

                last_tok = line_toks[-1]
                t1_idx: Optional[int] = None
                # Snap order based on configurable end_tolerance within bounds [t0_idx, n-1]
                tol = self._cfg.end_tolerance
                snap_cands: List[int] = []
                # Always try est first, then ±1..±tol interleaved
                snap_cands.append(est_t1)
                for d in range(1, max(1, tol) + 1):
                    snap_cands.append(est_t1 - d)
                    snap_cands.append(est_t1 + d)
                for cand in snap_cands:
                    if cand < t0_idx or cand >= n_tokens_words:
                        continue
                    wt = word_token_stream[cand]
                    if _approx_token_eq(wt, last_tok):
                        t1_idx = cand
                        break

                # 2b) Nếu vẫn chưa thấy, thử quét tuyến tính ngắn quanh est_t1 trong giới hạn max_linear_scan
                if t1_idx is None:
                    max_scan = self._cfg.max_linear_scan
                    # Forward scan from est_t1+1
                    i = est_t1 + 1
                    steps = 0
                    while i < n_tokens_words and steps < max_scan and t1_idx is None:
                        if i >= t0_idx and _approx_token_eq(word_token_stream[i], last_tok):
                            t1_idx = i
                            break
                        i += 1
                        steps += 1
                    # Backward scan from est_t1-1 if still not found
                    if t1_idx is None:
                        i = max(t0_idx, est_t1 - 1)
                        steps = 0
                        while i >= t0_idx and steps < max_scan:
                            if _approx_token_eq(word_token_stream[i], last_tok):
                                t1_idx = i
                                break
                            i -= 1
                            steps += 1

                if t1_idx is None:
                    raise ValueError("alignment_failed: end_not_found")

                # 3) Map token indices to word indices and emit
                w0_abs = token_to_word_index[t0_idx]
                w1_abs = token_to_word_index[t1_idx]
                _safe_emit(w0_abs, w1_abs, line, len(line_toks))

                # 4) Advance cursor to token after t1
                cursor_token_idx = min(n_tokens_words, t1_idx + 1)
            except ValueError as ve:
                msg = str(ve)
                if msg.startswith("alignment_failed"):
                    logger.warning("TextOverBuilder3: %s for line: %s", msg, line)
                    continue
                raise

        # Persist if requested
        if text_over_id:
            out_dir = Path(self._temp_dir) / text_over_id
            os.makedirs(out_dir, exist_ok=True)
            out_path = out_dir / "text_over.json"
            out_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                f"TextOverBuilder3:Persisted {len(results)} text overlay items to {out_path}"
            )

        return results
