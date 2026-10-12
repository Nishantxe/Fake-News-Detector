"""
Fake News Detector  -  TF-IDF + Passive Aggressive Classifier (main model)
                       TF-IDF + Logistic Regression            (baseline)

Labels:  1 = FAKE,  0 = REAL

Run examples
  python fake_news_detector_pa.py                                   # tiny built-in demo data
  python fake_news_detector_pa.py --fake Fake.csv --real True.csv   # real Kaggle data
  python fake_news_detector_pa.py --fake Fake.csv --real True.csv --no-tune   # skip tuning (faster)

Install
  pip install scikit-learn pandas joblib matplotlib
"""
import argparse
import re
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")                      # lets us save plots even without a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------
# Passive Aggressive model.
# Newer scikit-learn versions (1.8+) are retiring PassiveAggressiveClassifier, so we try it
# first and, if it is gone, use the identical SGDClassifier recipe instead.
# ----------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.linear_model import PassiveAggressiveClassifier

    def make_pa():
        return PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    PA_C = "clf__C"                      # the 'aggressiveness' setting is called C here
except ImportError:
    from sklearn.linear_model import SGDClassifier

    def make_pa():
        return SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1",
                             eta0=1.0, max_iter=1000, random_state=42)
    PA_C = "clf__eta0"                   # ...and eta0 in the SGD version


# ----------------------------------------------------------------------------
# Tiny demo data (ONLY to prove the code runs - use the Kaggle files for real results)
# ----------------------------------------------------------------------------
DEMO_FAKE = [
    "SHOCKING: Scientists confirm drinking hot water cures all cancer in 3 days",
    "You won't believe what this celebrity is hiding, the media won't tell you",
    "Government secretly putting microchips in vaccines, leaked documents reveal",
    "Miracle pill melts belly fat overnight, doctors hate this trick",
    "BREAKING: Aliens landed in Delhi, army covering up the truth",
    "Forward this message to 10 people or your WhatsApp will be deleted tomorrow",
    "Eating one banana before bed will make you rich, astrologers reveal",
    "NASA admits the moon is hollow and made by an ancient civilization",
    "Secret herb cures diabetes permanently, pharma companies are furious",
    "Free iPhone for everyone who clicks this link today, government scheme",
    "Famous actor dead? Shocking truth they don't want you to know",
    "5G towers spread disease, hidden study proves it",
    "Drink this juice every morning and never fall sick again, doctors stunned",
    "Secret document proves election was stolen, media silent",
    "Hidden camera shows what really happens inside the laboratory, share before deleted",
    "Doctors hate this one weird trick to grow hair in a week",
    "Government will cancel all bank accounts tomorrow, forward to save your money",
    "Shocking video proves the earth is flat, scientists panic",
    "Ancient remedy cures every disease, big pharma hiding it from you",
    "This one trick will double your salary overnight, you won't believe it",
    "Leaked: celebrity secretly arrested, mainstream media refuses to report",
    "Scientists admit vaccines change your DNA, the shocking truth revealed",
    "Click here to claim your free prize before the government blocks this link",
    "Mysterious light in the sky proves aliens are among us, army hides footage",
    "Mobile phone will explode if you charge it at night, forward to everyone",
    "Banned miracle fruit cures cancer, they don't want you to know",
    "Shocking truth about your tap water that authorities are hiding",
    "Urgent: new law will ban all cash tomorrow, share this immediately",
    "Famous cricketer exposes secret deal, media refuses to show this video",
    "Drinking lemon water with salt makes you immune to every virus, experts shocked",
]
DEMO_REAL = [
    "Reserve Bank of India keeps repo rate unchanged at its latest policy meeting",
    "Parliament passes bill on data protection after two days of debate",
    "Indian cricket team wins the third test match by six wickets",
    "Health ministry reports decline in dengue cases compared to last year",
    "ISRO successfully launches communication satellite from Sriharikota",
    "Local university announces new admission schedule for the upcoming session",
    "Heavy rainfall expected in northern states this week, says weather department",
    "Stock markets close higher as banking shares gain on strong quarterly results",
    "Government announces new scholarship scheme for rural students",
    "Researchers publish study on air pollution levels in major cities",
    "Election commission releases voter turnout figures for the first phase",
    "Railway ministry to introduce additional trains during the festive season",
    "State government allocates funds for repair of rural roads",
    "Finance ministry extends deadline for filing income tax returns",
    "City council approves new budget for public transport improvements",
    "Supreme Court hears petition on environmental clearance for highway project",
    "Central bank governor says inflation is expected to ease in coming months",
    "Agriculture department distributes subsidised seeds to farmers in the district",
    "Metro rail project completes testing of new section ahead of opening",
    "Education board announces date sheet for upcoming class twelve examinations",
    "Hospital launches free health check-up camp for senior citizens",
    "Power utility says electricity demand reached a record high during summer",
    "Ministry of external affairs holds talks with visiting delegation",
    "Scientists at national laboratory develop low cost water purification method",
    "Municipal corporation begins cleanliness drive across the city",
    "Sensex and Nifty end the day marginally lower amid global cues",
    "Police register case after complaint about road accident near the bridge",
    "Airport authority adds new flights to domestic routes from next month",
    "Telecom regulator releases report on mobile internet speeds across states",
    "Local team wins district level football tournament after penalty shootout",
]


# ----------------------------------------------------------------------------
# STEP 2: clean text
# ----------------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()                       # "Shocking" and "shocking" become the same word
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove web links
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    return re.sub(r"\s+", " ", text).strip()       # remove extra spaces


# ----------------------------------------------------------------------------
# STEP 1: load data
# ----------------------------------------------------------------------------
def load_data(fake_path, real_path):
    if fake_path and real_path:
        fake, real = pd.read_csv(fake_path), pd.read_csv(real_path)
        fake["label"], real["label"] = 1, 0                    # 1 = fake, 0 = real
        df = pd.concat([fake, real], ignore_index=True)
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
        df = df[["content", "label"]].sample(frac=1, random_state=42)   # shuffle
        print(f"[info] Loaded {len(df)} articles from CSV files.\n")
        return df
    print("[info] No CSV given -> using the tiny built-in demo data (only to test the code).\n")
    return pd.DataFrame({"content": DEMO_FAKE + DEMO_REAL,
                         "label": [1] * len(DEMO_FAKE) + [0] * len(DEMO_REAL)})


# ----------------------------------------------------------------------------
# STEP 4 + 5: TF-IDF + model, wrapped in a Pipeline
# ----------------------------------------------------------------------------
def make_pipeline(clf):
    tfidf = TfidfVectorizer(stop_words="english",   # drop useless words: is, the, and ...
                            ngram_range=(1, 2),     # single words AND word pairs
                            max_df=0.9)             # ignore words that appear in >90% of texts
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def fake_confidence(model, text):
    """Passive Aggressive has no predict_proba, so we squash its decision score into 0-1.
    Treat it as a confidence-like score, NOT a true probability."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba([text])[0][1])
    score = float(model.decision_function([text])[0])
    return 1.0 / (1.0 + np.exp(-2.0 * score))


def top_words(model, n=10):
    names = model.named_steps["tfidf"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    return [names[i] for i in order[-n:][::-1]], [names[i] for i in order[:n]]


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred REAL", "Pred FAKE"]); ax.set_yticklabels(["Actual REAL", "Actual FAKE"])
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[saved] {path}")


def main():
    ap = argparse.ArgumentParser(description="Fake news detector: TF-IDF + Passive Aggressive")
    ap.add_argument("--fake", help="path to Fake.csv")
    ap.add_argument("--real", help="path to True.csv")
    ap.add_argument("--no-tune", action="store_true", help="skip the (slower) parameter search")
    ap.add_argument("--no-interactive", action="store_true", help="skip the type-a-headline loop")
    args = ap.parse_args()

    df = load_data(args.fake, args.real)
    df["content"] = df["content"].apply(clean_text)

    # STEP 3: split data  (75% learn, 25% hidden exam)
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.25, random_state=42, stratify=df["label"])

    # STEP 5: train both models
    pa = make_pipeline(make_pa())
    if args.no_tune:
        print("[info] Skipping tuning, using default C=1.0, max_iter=1000 for Passive Aggressive.\n")
    else:
        grid = {PA_C: [0.1, 0.5, 1.0], "clf__max_iter": [500, 1000]}
        search = GridSearchCV(pa, grid, cv=3, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        pa = search.best_estimator_
        print(f"[tuning] best Passive Aggressive settings: {search.best_params_}\n")
    pa.fit(X_train, y_train)

    lr = make_pipeline(LogisticRegression(max_iter=1000))
    lr.fit(X_train, y_train)

    # STEP 6: test on the hidden 25%
    rows = {}
    for name, model in [("Passive Aggressive", pa), ("Logistic Regression", lr)]:
        pred = model.predict(X_test)
        rows[name] = {"Accuracy": accuracy_score(y_test, pred),
                      "Precision": precision_score(y_test, pred, zero_division=0),
                      "Recall": recall_score(y_test, pred, zero_division=0),
                      "F1-score": f1_score(y_test, pred, zero_division=0)}
    print("=== RESULTS ON HIDDEN TEST DATA (1 = fake) ===")
    print(pd.DataFrame(rows).T.round(3).to_string(), "\n")

    pa_pred = pa.predict(X_test)
    print("Classification report - Passive Aggressive (main model):")
    print(classification_report(y_test, pa_pred, target_names=["real", "fake"], zero_division=0))

    cm = confusion_matrix(y_test, pa_pred)
    print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
    print("  TN = real kept as real | FP = real wrongly called fake (false alarm)")
    print("  FN = fake missed       | TP = fake correctly caught\n")
    plot_confusion(cm, "Passive Aggressive", "confusion_matrix.png")

    fake_words, real_words = top_words(pa, 10)
    print("\nTop 10 words pushing toward FAKE:", fake_words)
    print("Top 10 words pushing toward REAL:", real_words)

    print("\nNOTE: on the Kaggle dataset, accuracy can look inflated because source clues "
          "(like 'Reuters') leak the answer.\n      Test on a second dataset such as LIAR to "
          "check how well the model really generalises.\n")

    joblib.dump(pa, "fake_news_pa_model.joblib")       # pipeline = vectorizer + classifier together
    joblib.dump(lr, "fake_news_lr_model.joblib")
    print("[saved] fake_news_pa_model.joblib and fake_news_lr_model.joblib")
    print("        reload later with: model = joblib.load('fake_news_pa_model.joblib')\n")

    samples = ["Shocking secret cure doctors don't want you to know, share now",
               "Finance ministry announces new tax filing deadline for the year"]
    for s in samples:
        c = fake_confidence(pa, clean_text(s))
        print(f"[{'FAKE' if c >= 0.5 else 'REAL'}] fake-confidence = {c:.2f}  |  {s}")

    if args.no_interactive or not sys.stdin.isatty():
        return
    print("\nType a headline to test (just press Enter to quit):")
    while True:
        s = input("> ").strip()
        if not s:
            break
        c = fake_confidence(pa, clean_text(s))
        print(f"   -> {'FAKE' if c >= 0.5 else 'REAL'}  (fake-confidence = {c:.2f})")


if __name__ == "__main__":
    main()
