import pandas as pd

RAW_SOCIAL = "data/raw/customer_social_profiles.csv"
RAW_TRANS = "data/raw/customer_transactions.csv"
OUT = "data/processed/merged_dataset.csv"


def load_social(path=RAW_SOCIAL):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    # ids are stored as "A178" here but as plain 178 in the transactions file,
    # so the prefix has to come off before the two sides can be matched at all
    df["customer_id"] = df["customer_id_new"].str.extract(r"(\d+)").astype(int)
    return df.drop(columns=["customer_id_new"])


def load_transactions(path=RAW_TRANS):
    df = pd.read_csv(path)
    df = df.rename(columns={"customer_id_legacy": "customer_id"})
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["rating_missing"] = df["customer_rating"].isna().astype(int)
    df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].median())
    return df


def aggregate_social(df):
    # a customer can appear on several platforms, and transactions repeat per customer
    # too, so joining the raw tables many-to-many duplicates real transactions.
    # collapsing to one row per customer first keeps the transaction grain intact.
    g = df.groupby("customer_id")
    out = g.agg(
        engagement_score=("engagement_score", "mean"),
        purchase_interest_score=("purchase_interest_score", "mean"),
        platform_count=("social_media_platform", "nunique"),
    )
    out["social_media_platform"] = g["social_media_platform"].agg(lambda x: x.mode()[0])
    out["review_sentiment"] = g["review_sentiment"].agg(lambda x: x.mode()[0])
    return out.reset_index()


def merge_data(trans, social):
    merged = trans.merge(social, on="customer_id", how="left")
    merged["has_social_profile"] = merged["engagement_score"].notna().astype(int)
    return merged


def add_features(df):
    df = df.sort_values("purchase_date").copy()
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.dayofweek

    # spend history per customer. deliberately no per-customer product_category
    # feature here, that would hand the model its own target
    cust = df.groupby("customer_id")["purchase_amount"]
    df["customer_txn_count"] = cust.transform("size")
    df["customer_avg_amount"] = cust.transform("mean")
    df["amount_vs_customer_avg"] = df["purchase_amount"] / df["customer_avg_amount"]
    return df


def fill_unmatched(df):
    # the 33 unmatched rows are customers with no social profile at all, not lost
    # values. has_social_profile keeps that distinction so the model can use it
    # instead of reading an imputed median as a real engagement score.
    for col in ["engagement_score", "purchase_interest_score", "platform_count"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["social_media_platform", "review_sentiment"]:
        df[col] = df[col].fillna("unknown")
    return df


def validate(merged, trans):
    assert len(merged) == len(trans), f"row count changed: {len(trans)} -> {len(merged)}"
    assert merged["transaction_id"].is_unique, "merge duplicated transactions"
    matched = merged["has_social_profile"].sum()
    print(f"rows: {len(merged)} (transactions in: {len(trans)})")
    print(f"matched to a social profile: {matched} ({matched / len(merged):.1%})")
    print(f"unmatched: {len(merged) - matched}")
    print(f"nulls remaining:\n{merged.isna().sum()[merged.isna().sum() > 0]}")


def build():
    social = aggregate_social(load_social())
    trans = load_transactions()
    merged = fill_unmatched(add_features(merge_data(trans, social)))
    validate(merged, trans)
    merged.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return merged


if __name__ == "__main__":
    build()
