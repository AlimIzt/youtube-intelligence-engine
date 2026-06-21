"""Lab 5 Part 2: Aspect-Based Sentiment Analysis with pyABSA (ATEPC).

Faithful port of the Lab 5 pyabsa_ATEPC.py + task9 wordcloud workflow:
extract aspect terms and their polarity, then build positive/negative aspect
word clouds.

NOTE: pyABSA pins transformers==4.29, which conflicts with the transformers 5.x
used elsewhere in this project. As in the lab, run this in a SEPARATE virtual
env (the lab used `pyabsa_env`). The import is therefore lazy so the rest of the
project still works without pyABSA installed.

    python -m venv pyabsa_env
    pyabsa_env/Scripts/pip install transformers==4.29.0 pyabsa wordcloud matplotlib
    pyabsa_env/Scripts/python -m src.analysis.absa
"""
from __future__ import annotations

from collections import Counter


def get_extractor():
    """Load the FAST_LCF_ATEPC English aspect extractor (lazy import)."""
    from pyabsa import AspectTermExtraction as ATEPC

    return ATEPC.AspectExtractor("english")


def extract_aspects(reviews: list[str], extractor=None) -> tuple[list[str], list[str]]:
    """Return (positive_aspects, negative_aspects) across the reviews."""
    extractor = extractor or get_extractor()
    results = extractor.predict(reviews)

    positive, negative = [], []
    for r in results:
        aspects = r.get("aspect", [])
        sentiments = r.get("sentiment", [])
        for asp, sent in zip(aspects, sentiments):
            asp_clean = asp.strip().lower()
            if sent == "Positive":
                positive.append(asp_clean)
            elif sent == "Negative":
                negative.append(asp_clean)
    return positive, negative


def aspect_wordclouds(
    positive: list[str], negative: list[str], save_path: str = "absa_wordclouds.png"
) -> None:
    """Side-by-side positive/negative aspect word clouds (Lab 5 task 9)."""
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    pos_text = " ".join(positive) if positive else "none"
    neg_text = " ".join(negative) if negative else "none"

    wc_pos = WordCloud(width=800, height=400, background_color="white",
                       colormap="Greens", max_words=80).generate(pos_text)
    wc_neg = WordCloud(width=800, height=400, background_color="white",
                       colormap="Reds", max_words=80).generate(neg_text)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(wc_pos, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Positive Aspects", fontsize=16, fontweight="bold", color="green")
    axes[1].imshow(wc_neg, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Negative Aspects", fontsize=16, fontweight="bold", color="red")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Word cloud saved to {save_path}")


def main() -> None:
    import pandas as pd

    from config import settings

    df = pd.read_csv(settings.clean_csv)
    reviews = df["text"].dropna().sample(200, random_state=42).tolist()
    print(f"Running ABSA on {len(reviews)} comments...")

    positive, negative = extract_aspects(reviews)
    print(f"Positive aspects: {len(positive)}  |  Negative aspects: {len(negative)}")
    aspect_wordclouds(positive, negative)

    print("\nTop 10 POSITIVE aspects:", Counter(positive).most_common(10))
    print("Top 10 NEGATIVE aspects:", Counter(negative).most_common(10))


if __name__ == "__main__":
    main()
