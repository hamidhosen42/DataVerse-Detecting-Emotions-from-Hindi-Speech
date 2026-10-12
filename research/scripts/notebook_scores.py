"""List every public notebook of the competition and fetch its best public score via the Kaggle search API.

Usage: python -I notebook_scores.py <out_csv>
Writes ref,title,author,last_run,votes,best_public_score,has_linked_submission.
"""
import sys

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.search.types.search_api_service import ListEntitiesFilters, ListEntitiesRequest
from kagglesdk.search.types.search_enums import DocumentType

C = "dataverse-detecting-emotions-from-hindi-speech"
api = KaggleApi()
api.authenticate()

kernels = []
page = 1
while True:
    batch = api.kernels_list(competition=C, page=page, page_size=100)
    if not batch:
        break
    kernels += batch
    if len(batch) < 100:
        break
    page += 1

rows = []
with api.build_kaggle_client() as k:
    for kr in kernels:
        ref = kr.ref
        owner, slug = ref.split("/")
        req = ListEntitiesRequest()
        f = ListEntitiesFilters()
        f.query = kr.title
        f.document_types = [DocumentType.KERNEL]
        req.filters = f
        req.page_size = 50
        score, linked = None, None
        for d in k.search.search_api_client.list_entities(req).documents:
            same_owner = d.owner_user is not None and d.owner_user.user_name == owner
            if same_owner and d.slug.rstrip("/").split("/")[-1] == slug and d.kernel_document is not None:
                score = d.kernel_document.best_public_score
                linked = d.kernel_document.has_linked_submission
                break
        rows.append(dict(ref=ref, title=kr.title, author=getattr(kr, "author", ""), last_run=getattr(kr, "last_run_time", ""),
                         votes=getattr(kr, "total_votes", ""), best_public_score=score, has_linked_submission=linked))
df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], index=False)
print(df.to_string(index=False))
