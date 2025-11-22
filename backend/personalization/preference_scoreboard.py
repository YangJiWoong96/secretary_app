from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple


def _get_redis(url: str):
    try:
        import redis  # type: ignore

        return redis.Redis.from_url(url, decode_responses=True)
    except Exception:
        return None


class PreferenceScoreboard:
    """
    행동 기반 선호 점수기(norm_key 단위)
    - explicit/choose/positive/negative 이벤트 가중치 반영
    - 일자 단위 감쇠
    - 점수→status 전이(pending/active/retired)
    """

    # 기본 가중치(설정으로 오버라이드 가능)
    W = {"explicit": 0.5, "choose": 0.2, "positive": 0.2, "negative": -0.5}

    def __init__(self, redis_url: str):
        self._r = _get_redis(redis_url)

    def available(self) -> bool:
        return self._r is not None

    def _k(self, user_id: str, norm_key: str) -> str:
        return f"pref:{user_id}:{norm_key}"

    def _k_idx(self, user_id: str) -> str:
        return f"pref:idx:{user_id}"

    def _k_stab(self, user_id: str) -> str:
        return f"pref:stab:{user_id}"

    def update(
        self,
        user_id: str,
        norm_key: str,
        events: Dict[str, int],
        intensity: float = 0.0,
        now: Optional[int] = None,
    ) -> Dict:
        if not self._r or not user_id or not norm_key:
            return {}
        now = now or int(time.time())
        cur = {}
        try:
            cur = json.loads(self._r.get(self._k(user_id, norm_key)) or "{}")
        except Exception:
            cur = {}

        # --- 시간 감쇠(일일) ---
        s_prev = float(cur.get("score", 0.5))
        last = int(cur.get("ts", now))
        days = max(0, (now - last) // 86400)
        decay_d = float(cur.get("decay_d", 0.05))  # 사용자별 보정 가능
        s_t = max(0.0, s_prev - decay_d * float(days))

        # --- 감정 강도 부스트 ---
        boost = 1.0 + 0.3 * max(0.0, min(1.0, float(intensity or 0.0)))

        # --- 이벤트 가중 합산 ---
        pos = int(cur.get("pos", 0))
        neg = int(cur.get("neg", 0))
        alpha = float(cur.get("alpha", self.W["explicit"]))
        beta = float(cur.get("beta", self.W["choose"]))
        gamma = float(cur.get("gamma", self.W["positive"]))
        delta = float(cur.get("delta", self.W["negative"]))

        for name, cnt in (events or {}).items():
            w = {
                "explicit": alpha,
                "choose": beta,
                "positive": gamma,
                "negative": delta,
            }.get(name, 0.0)
            s_t += boost * float(w) * int(cnt)
            if name == "positive":
                pos += int(cnt)
            if name == "negative":
                neg += int(cnt)

        # --- 안정성(Stability) 하한/상한 ---
        # 외부 Big Five 안정성/분산(EWVar)로부터 변환된 stability 입력(0~1)을 사용할 수 있음
        stability = float(cur.get("stability", 0.5))
        import math as _math

        # f(var) 대체: stability가 낮을수록 하한이 낮아지도록
        lower_bound = max(0.0, min(0.8, _math.exp(-2.0 * max(0.0, (1.0 - stability)))))
        s_t = min(1.0, max(lower_bound, s_t))

        # --- 증거 수 누적 및 상태 결정 ---
        ev_cnt = int(cur.get("evidence_count", 0)) + sum(
            int(v) for v in (events or {}).values()
        )
        tau0 = float(cur.get("tau0", 0.4))
        tau1 = float(cur.get("tau1", 0.75))
        m = int(cur.get("min_evidence", 3))

        if (s_t >= tau1) and (ev_cnt >= m):
            status = "active"
        elif s_t >= tau0:
            status = "pending"
        else:
            status = "retired"

        out = {
            "score": round(s_t, 3),
            "status": status,
            "pos": pos,
            "neg": neg,
            "evidence_count": ev_cnt,
            "stability": stability,  # 외부에서 업데이트 가능
            "decay_d": decay_d,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "tau0": tau0,
            "tau1": tau1,
            "min_evidence": m,
            "ts": now,
        }
        try:
            self._r.set(self._k(user_id, norm_key), json.dumps(out, ensure_ascii=False))
            # 사용자 인덱스에 norm_key 등록
            self._r.sadd(self._k_idx(user_id), norm_key)
        except Exception:
            pass
        # 선택: profile_chunks 상태/신뢰도 동기화(있을 때만)
        try:
            from backend.rag.milvus import ensure_profile_collection

            coll = ensure_profile_collection()
            # 가장 최신 1개만 업데이트(동일 norm_key에 여러 버전이 있을 수 있음)
            rows = coll.query(
                expr=f"user_id == '{user_id}' and norm_key == '{norm_key}'",
                output_fields=["id", "status", "confidence"],
                limit=1,
            )
            if rows:
                row_id = rows[0].get("id")
                if row_id:
                    coll.upsert(
                        [
                            {
                                "id": row_id,
                                "status": out["status"],
                                "confidence": float(out["score"]),
                            }
                        ]
                    )
        except Exception:
            pass
        return out

    def get(self, user_id: str, norm_key: str) -> Dict:
        """단일 항목 조회"""
        if not self._r or not user_id or not norm_key:
            return {}
        try:
            cur = json.loads(self._r.get(self._k(user_id, norm_key)) or "{}")
            return cur if isinstance(cur, dict) else {}
        except Exception:
            return {}

    def list_keys(self, user_id: str) -> List[str]:
        """사용자 인덱스에 등록된 norm_key 목록 반환(없으면 빈 리스트)"""
        if not self._r or not user_id:
            return []
        try:
            return list(self._r.smembers(self._k_idx(user_id)) or [])
        except Exception:
            return []

    def top(
        self, user_id: str, top_n: int = 5, include_pending: bool = True
    ) -> List[Tuple[str, Dict]]:
        """점수 상위 N개 반환: [(norm_key, entry), ...]"""
        keys = self.list_keys(user_id)
        rows: List[Tuple[str, Dict]] = []
        for k in keys:
            try:
                d = self.get(user_id, k)
                st = str(d.get("status") or "")
                if (st == "active") or (include_pending and st == "pending"):
                    rows.append((k, d))
            except Exception:
                continue
        rows.sort(key=lambda x: float((x[1] or {}).get("score", 0.0)), reverse=True)
        return rows[: max(1, int(top_n))]

    def set_stability(self, user_id: str, stability: float) -> None:
        """사용자 전체 안정성(0~1)을 별도 키에 기록 (외부 TraitAggregator 연동)"""
        if not self._r or not user_id:
            return
        try:
            self._r.set(
                self._k_stab(user_id), f"{max(0.0, min(1.0, float(stability))):.4f}"
            )
        except Exception:
            return

    def get_top_n(self, user_id: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        사용자의 상위 N개 선호도 조회 (점수 순)

        Args:
            user_id: 사용자 ID
            n: 반환 개수

        Returns:
            List[Dict]: 상위 선호도 리스트
                [
                    {
                        "norm_key": str,
                        "score": float,
                        "status": str,
                        "stability": float,
                        "evidence_count": int
                    },
                    ...
                ]
        """
        if not self._r or not user_id:
            return []

        try:
            idx_key = self._k_idx(user_id)
            all_keys = self._r.smembers(idx_key) or []

            if not all_keys:
                return []

            # 모든 키의 점수 조회
            items = []
            for norm_key in all_keys:
                data = self._r.get(self._k(user_id, norm_key))
                if not data:
                    continue

                try:
                    entry = json.loads(data)
                    items.append(
                        {
                            "norm_key": norm_key,
                            "score": float(entry.get("score", 0.0)),
                            "status": entry.get("status", "pending"),
                            "stability": float(entry.get("stability", 0.5)),
                            "evidence_count": int(entry.get("evidence_count", 0)),
                        }
                    )
                except Exception:
                    continue

            # 점수 순 정렬
            items.sort(key=lambda x: -x["score"])

            return items[:n]
        except Exception:
            return []
