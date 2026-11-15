import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

try:
    import jieba
except Exception:
    jieba = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "Data", "scenarios.jsonl")
DEFAULT_OUTPUT = os.path.join(HERE, "outputs", "retrieval")
PUNCTUATION = re.compile(r"[！？。，、；：“”‘’（）《》【】\s,.;:!?\"'()\[\]{}<>|]+")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")
SHINGLE_SIZE = 3
SIMHASH_BITS = 64
MASK64 = (1 << 64) - 1


def fail(message):
    sys.stderr.write("[error] " + str(message) + "\n")
    sys.exit(1)


def info(message):
    sys.stdout.write("[info] " + str(message) + "\n")
    sys.stdout.flush()


def warn(message):
    sys.stderr.write("[warn] " + str(message) + "\n")
    sys.stderr.flush()


def normalize_text(text):
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = value.replace("　", " ")
    value = PUNCTUATION.sub(" ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(text):
    normalized = normalize_text(text)
    if len(normalized) == 0:
        return []
    if jieba is not None:
        tokens = []
        for token in jieba.lcut(normalized):
            token = token.strip()
            if len(token) == 0:
                continue
            tokens.append(token)
        if len(tokens) > 0:
            return tokens
    return TOKEN_PATTERN.findall(normalized)


def char_ngrams(text, size=SHINGLE_SIZE):
    normalized = normalize_text(text)
    if len(normalized) == 0:
        return set()
    if len(normalized) < size:
        return {normalized}
    grams = set()
    for index in range(len(normalized) - size + 1):
        grams.add(normalized[index:index + size])
    return grams


def jaccard(left, right):
    a = char_ngrams(left)
    b = char_ngrams(right)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return float(intersection) / float(union)


def cosine_counter(left_tokens, right_tokens):
    left = Counter(left_tokens)
    right = Counter(right_tokens)
    if len(left) == 0 or len(right) == 0:
        return 0.0
    common = set(left.keys()) & set(right.keys())
    numerator = 0.0
    for key in common:
        numerator += float(left[key]) * float(right[key])
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def hash_token(token):
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class SimHash:
    def __init__(self, bits=SIMHASH_BITS):
        self.bits = bits

    def compute(self, text):
        tokens = tokenize(text)
        if len(tokens) == 0:
            return 0
        weights = Counter(tokens)
        vector = [0] * self.bits
        for token, weight in weights.items():
            value = hash_token(token)
            for index in range(self.bits):
                if value & (1 << index):
                    vector[index] += weight
                else:
                    vector[index] -= weight
        result = 0
        for index in range(self.bits):
            if vector[index] > 0:
                result |= (1 << index)
        return result

    def distance(self, left, right):
        value = (left ^ right) & MASK64
        return bin(value).count("1")

    def similarity(self, left, right):
        return 1.0 - float(self.distance(left, right)) / float(self.bits)


class MinHash:
    def __init__(self, permutations=64, seed=42):
        self.permutations = permutations
        self.coefficients = []
        prime = (1 << 61) - 1
        state = seed
        for _ in range(permutations):
            state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
            a = (state % prime) + 1
            state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
            b = state % prime
            self.coefficients.append((a, b, prime))

    def signature(self, text):
        grams = char_ngrams(text)
        if len(grams) == 0:
            return [0] * self.permutations
        hashes = []
        for gram in grams:
            hashes.append(hash_token(gram))
        signature = []
        for a, b, prime in self.coefficients:
            minimum = None
            for value in hashes:
                candidate = (a * value + b) % prime
                if minimum is None or candidate < minimum:
                    minimum = candidate
            signature.append(minimum if minimum is not None else 0)
        return signature

    def similarity(self, left_signature, right_signature):
        if len(left_signature) == 0 or len(right_signature) == 0:
            return 0.0
        matches = 0
        for left, right in zip(left_signature, right_signature):
            if left == right:
                matches += 1
        return float(matches) / float(len(left_signature))


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left, right):
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        return True

    def groups(self):
        buckets = defaultdict(list)
        for item in self.parent:
            buckets[self.find(item)].append(item)
        return {root: sorted(members) for root, members in buckets.items()}


def load_records(path):
    if not os.path.isfile(path):
        fail("文件不存在：%s" % path)
    records = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception as err:
                    fail("%s:%d JSON 解析失败：%s" % (path, line_no, err))
                records.append(item)
        return records
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]
    if not isinstance(payload, list):
        fail("文件格式不正确：%s" % path)
    return payload


def record_text(record):
    if isinstance(record, str):
        return record
    for key in ("scenario", "text", "content", "description", "response"):
        if key in record and isinstance(record[key], str):
            return record[key]
    if "messages" in record and isinstance(record["messages"], list):
        for item in reversed(record["messages"]):
            if isinstance(item, dict) and str(item.get("role", "")) == "assistant":
                return str(item.get("content", ""))
    return json.dumps(record, ensure_ascii=False)


class TFIDFIndex:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.texts = []

    def fit(self, texts):
        self.texts = list(texts)
        if TfidfVectorizer is None or len(texts) == 0:
            return self
        tokenized = [" ".join(tokenize(text)) for text in texts]
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)
        self.matrix = self.vectorizer.fit_transform(tokenized)
        return self

    def query(self, text, top_k=10):
        if self.vectorizer is None or self.matrix is None:
            return []
        tokenized = " ".join(tokenize(text))
        vector = self.vectorizer.transform([tokenized])
        scores = cosine_similarity(vector, self.matrix)[0]
        order = scores.argsort()[::-1]
        results = []
        for index in order[:top_k]:
            results.append({"index": int(index), "score": float(scores[index]), "text": self.texts[int(index)]})
        return results


class BM25Index:
    def __init__(self):
        self.index = None
        self.texts = []

    def fit(self, texts):
        self.texts = list(texts)
        if BM25Okapi is None or len(texts) == 0:
            return self
        tokenized = [tokenize(text) for text in texts]
        self.index = BM25Okapi(tokenized)
        return self

    def query(self, text, top_k=10):
        if self.index is None:
            return []
        scores = self.index.get_scores(tokenize(text))
        order = scores.argsort()[::-1]
        results = []
        for index in order[:top_k]:
            results.append({"index": int(index), "score": float(scores[index]), "text": self.texts[int(index)]})
        return results


class EmbeddingIndex:
    def __init__(self, model_name, cache_folder=None):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.texts = []
        self.cache_folder = cache_folder

    def available(self):
        return SentenceTransformer is not None

    def load(self):
        if SentenceTransformer is None:
            fail("未安装 sentence-transformers，无法使用向量检索")
        kwargs = {}
        if self.cache_folder:
            kwargs["cache_folder"] = self.cache_folder
        self.model = SentenceTransformer(self.model_name, **kwargs)
        return self.model

    def fit(self, texts):
        self.texts = list(texts)
        if self.model is None:
            self.load()
        if len(texts) == 0:
            return self
        self.embeddings = self.model.encode(
            self.texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self

    def query(self, text, top_k=10):
        if self.model is None or self.embeddings is None:
            return []
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        scores = []
        for index, embedding in enumerate(self.embeddings):
            score = float(sum(float(a) * float(b) for a, b in zip(vector, embedding)))
            scores.append((score, index))
        scores.sort(reverse=True)
        results = []
        for score, index in scores[:top_k]:
            results.append({"index": index, "score": score, "text": self.texts[index]})
        return results


class DuplicateDetector:
    def __init__(self, method="simhash", threshold=0.9):
        self.method = method
        self.threshold = threshold
        self.simhash = SimHash()
        self.minhash = MinHash()

    def pairwise(self, texts):
        results = []
        signatures = []
        for text in texts:
            signatures.append(self.simhash.compute(text))
        for left in range(len(texts)):
            for right in range(left + 1, len(texts)):
                score = self.similarity(texts[left], texts[right], signatures[left], signatures[right])
                if score >= self.threshold:
                    results.append({"left": left, "right": right, "score": score})
        results.sort(key=lambda item: -item["score"])
        return results

    def similarity(self, left, right, left_signature=None, right_signature=None):
        if self.method == "exact":
            return 1.0 if normalize_text(left) == normalize_text(right) else 0.0
        if self.method == "jaccard":
            return jaccard(left, right)
        if self.method == "cosine":
            return cosine_counter(tokenize(left), tokenize(right))
        if self.method == "minhash":
            if left_signature is None:
                left_signature = self.minhash.signature(left)
            if right_signature is None:
                right_signature = self.minhash.signature(right)
            return self.minhash.similarity(left_signature, right_signature)
        if left_signature is None:
            left_signature = self.simhash.compute(left)
        if right_signature is None:
            right_signature = self.simhash.compute(right)
        return self.simhash.similarity(left_signature, right_signature)

    def clusters(self, texts):
        finder = UnionFind()
        for index in range(len(texts)):
            finder.add(index)
        for item in self.pairwise(texts):
            finder.union(item["left"], item["right"])
        groups = finder.groups()
        ordered = []
        for members in groups.values():
            ordered.append(sorted(members))
        ordered.sort(key=lambda members: (-len(members), members[0]))
        return ordered


class NoveltyScorer:
    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.cosine_index = TFIDFIndex().fit(self.corpus)
        self.embeddings = None
        self.simhash = SimHash()

    def lexical_novelty(self, text):
        if len(self.corpus) == 0:
            return 1.0
        results = self.cosine_index.query(text, top_k=1)
        if results:
            return 1.0 - float(results[0]["score"])
        best = 0.0
        for item in self.corpus:
            score = jaccard(text, item)
            if score > best:
                best = score
        return 1.0 - best

    def hash_distance(self, text):
        signature = self.simhash.compute(text)
        distances = []
        for item in self.corpus:
            distances.append(self.simhash.distance(signature, self.simhash.compute(item)))
        if len(distances) == 0:
            return float(SIMHASH_BITS)
        return float(min(distances))

    def score(self, text):
        lexical = self.lexical_novelty(text)
        distance = self.hash_distance(text)
        combined = 0.7 * lexical + 0.3 * (distance / float(SIMHASH_BITS))
        return {
            "novelty": combined,
            "lexical_novelty": lexical,
            "min_hash_distance": distance,
        }


def build_parser():
    parser = argparse.ArgumentParser(description="语义检索、去重与文本处理")
    parser.add_argument("--input", type=str, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", type=str, default="dedup")
    parser.add_argument("--method", type=str, default="simhash")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--no_embedding", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.task == "similar":
        if len(args.text.strip()) == 0:
            fail("similar 任务需要 --text")
        texts = [record_text(item) for item in load_records(args.input)]
        index = TFIDFIndex().fit(texts)
        results = index.query(args.text, top_k=args.top_k)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    records = load_records(args.input)
    texts = [record_text(item) for item in records]
    info("载入 %d 条文本，方法 %s，阈值 %.2f" % (len(texts), args.method, args.threshold))
    if args.task == "dedup":
        detector = DuplicateDetector(method=args.method, threshold=args.threshold)
        pairs = detector.pairwise(texts)
        clusters = detector.clusters(texts)
        duplicate_clusters = [members for members in clusters if len(members) > 1]
        payload = {
            "records": len(texts),
            "method": args.method,
            "threshold": args.threshold,
            "duplicate_pairs": len(pairs),
            "duplicate_clusters": duplicate_clusters,
            "cluster_count": len(duplicate_clusters),
            "pairs": pairs[:100],
        }
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "dedup_result.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        info("重复簇 %d 个，近重复对 %d 个，结果写入 %s"
             % (len(duplicate_clusters), len(pairs), path))
        return
    if args.task == "novelty":
        scorer = NoveltyScorer(texts)
        results = []
        for index, text in enumerate(texts):
            item = scorer.score(text)
            item["index"] = index
            item["text"] = text
            results.append(item)
        results.sort(key=lambda item: item["novelty"])
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "novelty.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        info("新颖性结果写入 %s，最低 %.3f，最高 %.3f"
             % (path, results[0]["novelty"], results[-1]["novelty"]))
        return
    if args.task == "index":
        texts = [record_text(item) for item in load_records(args.input)]
        payload = {"count": len(texts), "methods": []}
        if TfidfVectorizer is not None:
            payload["methods"].append("tfidf")
        if BM25Okapi is not None:
            payload["methods"].append("bm25")
        if SentenceTransformer is not None and not args.no_embedding:
            payload["methods"].append("embedding")
        payload["tokenizer"] = "jieba" if jieba is not None else "regex"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    fail("未知任务：%s" % args.task)


if __name__ == "__main__":
    main()
