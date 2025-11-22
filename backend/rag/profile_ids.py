def bot_user_id_for(user_id: str) -> str:
    """
    주어진 사용자에 대한 개인화된 봇 user_id를 생성한다.
    - 입력이 'bot:'로 시작하면 그대로 반환
    - 입력이 비어있으면 'bot:global' 반환
    - 그 외에는 'bot:{user_id}' 반환
    """
    uid = (user_id or "").strip()
    if not uid:
        return "bot:global"
    if uid.startswith("bot:"):
        return uid
    return f"bot:{uid}"
