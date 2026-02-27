#!/usr/bin/env python3
"""
Phase 1: Parse LoCoMo dataset and build corpus/queries/ground truth.

Each conversation sample in locomo10.json contains:
  - conversation: dict with speaker_a, speaker_b, session_N_date_time, session_N
    Each session_N is a list of dialog turns [{dia_id, speaker, text}, ...]
  - qa: list of QA items [{question, answer, evidence, category}, ...]
    evidence is a string repr of a list of dia_id strings, e.g. "['D1:3']"

Document granularity: each dialog turn becomes one document.
Query: each QA pair becomes one query.
Ground truth: QA evidence (list of dia_id strings) maps to doc_ids.

Input: locomo10.json
Output:
  - corpus.jsonl: one document per dialog turn
  - queries.jsonl: one query per QA pair
  - ground_truth.jsonl: query_id -> relevant doc_ids
  - stats.json: dataset statistics
"""

import argparse
import ast
import json
import os
import re
from collections import defaultdict
from typing import Any


def _extract_turns(conversation: dict) -> list[dict]:
    """Extract all dialog turns from the conversation dict, ordered by session."""
    turns = []
    # Find all session keys like session_1, session_2, ...
    session_keys = sorted(
        [k for k in conversation if re.match(r"^session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )
    for sk in session_keys:
        session_turns = conversation[sk]
        if isinstance(session_turns, list):
            turns.extend(session_turns)
    return turns


def _parse_evidence(evidence_raw) -> list[str]:
    """Parse evidence field which may be a string repr of a list or an actual list."""
    if isinstance(evidence_raw, list):
        return [str(e) for e in evidence_raw]
    if isinstance(evidence_raw, str):
        try:
            parsed = ast.literal_eval(evidence_raw)
            if isinstance(parsed, list):
                return [str(e) for e in parsed]
        except (ValueError, SyntaxError):
            pass
        # Might be a single dia_id string
        stripped = evidence_raw.strip()
        if stripped:
            return [stripped]
    return []


def parse_locomo(input_path: str, output_dir: str) -> dict[str, Any]:
    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversation samples")

    corpus: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []

    # (sample_idx, dia_id_str) -> doc_id
    dia_key_to_docid: dict[tuple[int, str], int] = {}

    category_counts: defaultdict[str, int] = defaultdict(int)
    evidence_counts: list[int] = []

    for sample_idx, sample in enumerate(data):
        sample_id = sample.get("sample_id", sample_idx)
        conversation = sample.get("conversation", {})

        # Build corpus from dialog turns across all sessions
        turns = _extract_turns(conversation)
        for turn in turns:
            dia_id = turn["dia_id"]  # e.g. "D1:1"
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            doc_text = f"{speaker}: {text}"

            doc_id = len(corpus)
            dia_key_to_docid[(sample_idx, dia_id)] = doc_id
            corpus.append({
                "doc_id": doc_id,
                "sample_id": sample_id,
                "dia_id": dia_id,
                "speaker": speaker,
                "text": doc_text,
            })

        # Build queries and ground truth from QA pairs
        for qa in sample.get("qa", []):
            query_id = len(queries)
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            category = qa.get("category", "unknown")
            evidence_raw = qa.get("evidence", [])

            category_counts[category] += 1

            # Parse evidence dia_ids and map to doc_ids
            evidence_dia_ids = _parse_evidence(evidence_raw)
            relevant_doc_ids = []
            for evi_dia_id in evidence_dia_ids:
                key = (sample_idx, evi_dia_id)
                if key in dia_key_to_docid:
                    relevant_doc_ids.append(dia_key_to_docid[key])

            relevant_doc_ids = list(set(relevant_doc_ids))
            evidence_counts.append(len(relevant_doc_ids))

            queries.append({
                "query_id": query_id,
                "question": question,
                "answer": answer,
                "category": category,
                "sample_id": sample_id,
            })

            ground_truth.append({
                "query_id": query_id,
                "relevant_doc_ids": relevant_doc_ids,
            })

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)

    corpus_path = os.path.join(output_dir, "corpus.jsonl")
    print(f"Writing {len(corpus)} documents to {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    queries_path = os.path.join(output_dir, "queries.jsonl")
    print(f"Writing {len(queries)} queries to {queries_path}...")
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    gt_path = os.path.join(output_dir, "ground_truth.jsonl")
    print(f"Writing {len(ground_truth)} ground truth entries to {gt_path}...")
    with open(gt_path, "w", encoding="utf-8") as f:
        for gt in ground_truth:
            f.write(json.dumps(gt, ensure_ascii=False) + "\n")

    stats = {
        "total_samples": len(data),
        "total_documents": len(corpus),
        "total_queries": len(queries),
        "category_distribution": dict(category_counts),
        "avg_evidence_per_query": (
            sum(evidence_counts) / len(evidence_counts)
            if evidence_counts else 0
        ),
        "min_evidence": min(evidence_counts) if evidence_counts else 0,
        "max_evidence": max(evidence_counts) if evidence_counts else 0,
        "queries_with_no_evidence": sum(1 for c in evidence_counts if c == 0),
    }

    stats_path = os.path.join(output_dir, "stats.json")
    print(f"Writing statistics to {stats_path}...")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Statistics ===")
    print(f"Total conversation samples: {stats['total_samples']}")
    print(f"Total documents (dialog turns): {stats['total_documents']}")
    print(f"Total queries (QA pairs): {stats['total_queries']}")
    print(f"Category distribution: {stats['category_distribution']}")
    print(f"Avg evidence docs per query: {stats['avg_evidence_per_query']:.2f}")
    print(f"Min/Max evidence: {stats['min_evidence']}/{stats['max_evidence']}")
    print(f"Queries with no evidence: {stats['queries_with_no_evidence']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Parse LoCoMo dataset")
    parser.add_argument(
        "--input", required=True, help="Path to locomo10.json"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for generated files"
    )
    args = parser.parse_args()
    parse_locomo(args.input, args.output_dir)


if __name__ == "__main__":
    main()
