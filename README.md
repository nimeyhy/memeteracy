# 밈터러시

밈이 실제로 어떤 맥락에서 쓰이는지 네이버 검색 결과로 분석해 보여 주는 리터러시 도구입니다.

- `app.py` — 화면 (Streamlit)
- `pipeline.py` — 수집·전처리·임베딩·군집·유의 군집 판정
- `prepare_examples.py` — 팀 분석 결과 → `data/examples.json` 변환
- `data/` — 연구 결과 예시 데이터와 그림

Streamlit Community Cloud 앱 설정의 Secrets에 다음을 넣어야 실시간 검색이 동작합니다.

```toml
NAVER_CLIENT_ID = "..."
NAVER_CLIENT_SECRET = "..."
```
