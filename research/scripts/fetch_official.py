"""Fetch competition pages and all discussion topics/messages via the Kaggle SDK.

Usage: python -I fetch_official.py <out_dir>
Auth comes from ~/.kaggle (the token is never printed).
"""
import json
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiListCompetitionPagesRequest,
    ApiListCompetitionTopicsRequest,
    ApiListTopicMessagesRequest,
)

C = "dataverse-detecting-emotions-from-hindi-speech"
out = sys.argv[1]
api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as k:
    svc = k.competitions.competition_api_client
    r = ApiListCompetitionPagesRequest()
    r.competition_name = C
    for p in svc.list_competition_pages(r).pages:
        d = p.to_dict()
        open(f"{out}/page_{d['name']}.md", "w").write(d.get("content", ""))
        print("page", d["name"], len(d.get("content", "")))

    topics = []
    for page in range(1, 10):
        r = ApiListCompetitionTopicsRequest()
        r.competition_name = C
        r.page = page
        ts = [t.to_dict() for t in svc.list_competition_topics(r).topics]
        if not ts:
            break
        topics += ts
    for t in topics:
        r = ApiListTopicMessagesRequest()
        r.competition_name = C
        r.topic_id = t["id"]
        t["messages"] = [m.to_dict() for m in svc.list_topic_messages(r).messages]
    json.dump(topics, open(f"{out}/discussions.json", "w"), indent=1, default=str)
    print("topics", len(topics))
