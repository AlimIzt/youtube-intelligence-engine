"""Collect top-level YouTube comments via the Data API v3.

Refactored from the Lab 1 CommentsExtractor: the API key now comes from the
environment (config/.env) instead of being hardcoded.
"""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import settings


def _progress(current: int, total: int, bar_len: int = 40) -> None:
    pct = min(current / total, 1.0) if total else 0.0
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r[{bar}] {current:,} / {total:,} ({pct*100:.1f}%)", end="", flush=True)


def scrape_comments(
    video_ids: Iterable[str] | None = None,
    target: int | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Scrape up to `target` top-level comments across `video_ids`."""
    api_key = api_key or settings.youtube_api_key
    if not api_key:
        raise RuntimeError(
            "No YouTube API key. Set YOUTUBE_API_KEY in your .env file."
        )
    video_ids = list(video_ids or settings.video_ids)
    target = target or settings.scrape_target

    youtube = build("youtube", "v3", developerKey=api_key)
    rows: list[dict] = []

    print(f"Starting scraper — target: {target:,} comments\n")
    for vid in video_ids:
        if len(rows) >= target:
            break
        print(f"📹 Video: {vid}")
        token = None
        while len(rows) < target:
            try:
                res = (
                    youtube.commentThreads()
                    .list(
                        part="snippet",
                        videoId=vid,
                        maxResults=100,
                        pageToken=token,
                        textFormat="plainText",
                    )
                    .execute()
                )
            except HttpError as e:
                print(f"\nError on {vid}: {e}")
                break

            for item in res["items"]:
                s = item["snippet"]["topLevelComment"]["snippet"]
                rows.append(
                    {
                        "video_id": vid,
                        "author": s["authorDisplayName"],
                        "text": s["textDisplay"],
                        "likes": s["likeCount"],
                        "published_at": s["publishedAt"],
                    }
                )
            _progress(len(rows), target)
            token = res.get("nextPageToken")
            if not token:
                print("\nNo more comments available for this video.")
                break
            time.sleep(0.3)

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"\n\nScraped {len(df):,} unique comments.")
    return df


def main() -> None:
    df = scrape_comments()
    settings.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.raw_csv, index=False, encoding="utf-8-sig")
    print(f"Saved → {settings.raw_csv}")


if __name__ == "__main__":
    main()
