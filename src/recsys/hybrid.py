"""Hybrid recommender: collaborative signal first, content-based backfill.

Collaborative (co-purchase) recommendations are used first because they
reflect actual buying behavior. But Day 5 showed only ~15% of products have
any co-purchase signal at all, so whenever collaborative filtering can't
fill k slots, the remaining slots are backfilled with content-based
similarity (category/price/review) instead of just returning fewer results.
"""

from __future__ import annotations


class HybridRecommender:
    def __init__(self, collaborative, content) -> None:
        self.collaborative = collaborative
        self.content = content

    def recommend(self, product_id: str, k: int = 10) -> list[str]:
        collab_results = self.collaborative.recommend(product_id, k=k)
        if len(collab_results) >= k:
            return collab_results

        needed = k - len(collab_results)
        # ask content for extras beyond what we need, since some may
        # overlap with collab_results and get filtered out below
        content_results = self.content.recommend(product_id, k=k + len(collab_results))
        seen = set(collab_results)
        backfill = []
        for pid in content_results:
            if pid not in seen:
                backfill.append(pid)
                seen.add(pid)
            if len(backfill) == needed:
                break

        return collab_results + backfill
