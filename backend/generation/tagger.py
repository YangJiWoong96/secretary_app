import json
from typing import Any, Dict


async def extract_tags(
    question: str,
    router_meta: Dict[str, Any],
    evidence_meta: Dict[str, Any],
    stwm_meta: Dict[str, Any],
    profile_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    gpt-4o-mini 태깅 체인. 본문 TEXT는 절대 주지 않는다.
    입력: question, router_meta(p_rag/p_web/EMA 등), evidence_meta(후보 ID/도메인/유사도/recency),
         stwm_meta(개별 엔터티/토픽), profile_meta(선호/제약 라벨)
    출력(JSON): {
      "route": "web|rag|conv",
      "freshness": "low|medium|high",
      "domain_tags": [str],
      "safety_flags": [str],
      "selected_web_ids": [str],  # meta 기반 선택(유사도/recency/도메인)
      "selected_rag_ids": [str],
      "citations_policy": "strict|lenient"
    }
    """
    from backend.utils.retry import openai_chat_with_retry

    sys = (
        "너는 태깅기다. 본문 텍스트를 보지 않고 메타 정보만으로 의도/도메인/신선도/안전 태그와 "
        "채널(web/rag/conv)을 선택한다. 후보 문서는 id, domain, sim, age_s만 전달된다. "
        "선정 기준: (1) router 확률/EMA, (2) sim 상위, (3) recency 우선, (4) 도메인 적합성. "
        "dialogue_features.co_ref=true이면 route=conv를 우선 선택하라. conv 최소 윈도우는 항상 보존된다고 가정하고, "
        "co_ref=true일 때 selected_web_ids/selected_rag_ids는 최대 1개 이내로 제한하라. "
        "출력은 JSON 하나로, 스키마를 준수하라."
    )
    user = json.dumps(
        {
            "question": question,
            "router_meta": router_meta,
            "evidence_meta": evidence_meta,
            "stwm_meta": stwm_meta,
            "profile_meta": profile_meta,
            # 추가: 대화 특징량(메타 전용)
            "dialogue_features": router_meta.get("dialogue_features", {}),
        },
        ensure_ascii=False,
    )
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    resp = await openai_chat_with_retry(
        model="gpt-4o-mini",
        messages=msgs,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=300,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        # 정규화: route/ids/citations_policy 기본값 보정 및 Top-K 폴백
        rf = (router_meta.get("route_fallback") or "conv").strip()

        route_eff = str(data.get("route") or rf or "conv").strip()

        def _norm_ids(kind: str) -> list[str]:
            xs = data.get(f"selected_{kind}_ids") or []
            out: list[str] = []
            for sid in xs:
                s = str(sid).strip()
                if not s:
                    continue
                out.append(s if ":" in s else f"{kind}:{s}")
            # 폴백: evidence_meta 상위 Top-K id 사용
            if not out:
                try:
                    cand = (evidence_meta.get(kind) or [])[:3]
                    out = [str(c.get("id")) for c in cand if c.get("id")]
                except Exception:
                    out = []
            return out

        out_data = {
            "route": route_eff,
            "freshness": data.get("freshness") or "low",
            "domain_tags": data.get("domain_tags") or [],
            "safety_flags": data.get("safety_flags") or [],
            "selected_web_ids": _norm_ids("web"),
            "selected_rag_ids": _norm_ids("rag"),
            "citations_policy": data.get("citations_policy") or "strict",
        }
        return out_data
    except Exception:
        return {
            "route": router_meta.get("route_fallback", "conv"),
            "freshness": "low",
            "domain_tags": [],
            "safety_flags": [],
            "selected_web_ids": [],
            "selected_rag_ids": [],
            "citations_policy": "strict",
        }
