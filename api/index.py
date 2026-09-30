"""
Fake Healthcare Review Detection — serverless API.
Trains TF-IDF + LogisticRegression at cold start on the embedded
labelled sample, then classifies incoming review text.
POST/GET body: {"text": "..."} -> {"label", "confidence"}
"""
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Labelled sample (1 = fake/suspicious, 0 = genuine) — representative of the
# Kaggle hospital-reviews dataset the full pipeline trains on.
SAMPLES = [
    ("This hospital is the BEST hospital ever!! Amazing doctors, changed my life!!! Everyone should go here now!!!", 1),
    ("Five stars!!! Incredible service, best doctors in the world, no complaints at all, perfect perfect perfect", 1),
    ("I love this place so much! The staff are angels! Best experience of my entire life! Go here immediately!", 1),
    ("Wonderful wonderful wonderful. Best hospital. Visit today. You will not regret. Trust me!!!", 1),
    ("Amazing clinic, zero wait time, doctors are gods, recommend to literally everyone on earth!!", 1),
    ("PERFECT hospital!! My surgery was flawless, nurses were heavenly, 10/10 would get sick again!!", 1),
    ("The doctor changed my life forever!! Best decision ever!! Everyone must visit this hospital now!!", 1),
    ("Outstanding!!! Marvelous!!! The greatest medical facility in history!! Book your appointment TODAY!!!", 1),
    ("I visited for a routine checkup. Waited about 25 minutes past my appointment. The doctor was thorough and answered my questions.", 0),
    ("The front desk staff were polite. My blood test results took 3 days to arrive, which was a bit slow, but the GP explained them well.", 0),
    ("Decent experience overall. Parking was difficult and the waiting room was crowded, but the consultation itself was good.", 0),
    ("Had knee surgery here last month. Recovery instructions were clear. The billing department made an error but corrected it quickly.", 0),
    ("My appointment started 40 minutes late. Doctor seemed rushed but was competent. Pharmacy on-site was convenient.", 0),
    ("Referred here for an MRI. Staff explained the procedure beforehand. Some noise in the waiting area but clean facilities.", 0),
    ("Took my father for a cardiology consult. The specialist spent good time with us. Follow-up scheduling was a bit confusing.", 0),
    ("Emergency room visit for a sprained ankle. Triage was fast, X-ray took a while. Overall satisfied with the care.", 0),
]

_model = None


def get_model():
    global _model
    if _model is None:
        texts = [t for t, _ in SAMPLES]
        labels = [l for _, l in SAMPLES]
        vec = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        X = vec.fit_transform(texts)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, labels)
        _model = (vec, clf)
    return _model


def classify(text: str):
    vec, clf = get_model()
    X = vec.transform([text])
    proba = clf.predict_proba(X)[0]
    label = int(clf.predict(X)[0])
    return label, round(float(max(proba)) * 100, 1)


class handler(BaseHTTPRequestHandler):
    def _respond(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._respond(200, {})

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        text = (q.get("text") or [""])[0]
        self._handle(text)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            data = {}
        self._handle(data.get("text", ""))

    def _handle(self, text: str):
        text = (text or "").strip()
        if len(text) < 10:
            self._respond(400, {"error": "text must be at least 10 characters"})
            return
        label, confidence = classify(text)
        self._respond(
            200,
            {
                "label": "FAKE" if label == 1 else "GENUINE",
                "confidence": confidence,
                "model": "tfidf + logistic-regression (embedded sample)",
            },
        )
