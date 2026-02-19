#!/usr/bin/env python3
"""
Phase 1: Parse HotpotQA Training Set and build corpus/queries/ground truth.

Input: hotpot_train_v1.1.json
Output:
  - corpus.jsonl: deduplicated documents (title -> text)
  - queries.jsonl: all questions with metadata
  - ground_truth.jsonl: query_id -> relevant doc_ids
  - title_to_docid.json: title -> doc_id mapping
  - stats.json: data statistics
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


def parse_hotpotqa(input_path: str, output_dir: str) -> dict[str, Any]:
    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} questions")
    
    title_to_docid: dict[str, int] = {}
    corpus: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    
    type_counts: dict[str, int] = defaultdict(int)
    level_counts: dict[str, int] = defaultdict(int)
    supporting_facts_per_query: list[int] = []
    
    print("Building corpus and queries...")
    for item in tqdm(data, desc="Parsing"):
        query_id = len(queries)
        question = item["question"]
        answer = item["answer"]
        q_type = item.get("type", "unknown")
        level = item.get("level", "unknown")
        supporting_facts = item.get("supporting_facts", [])
        context = item.get("context", [])
        
        type_counts[q_type] += 1
        level_counts[level] += 1
        
        supporting_titles = set()
        for sf in supporting_facts:
            supporting_titles.add(sf[0])
        
        supporting_facts_per_query.append(len(supporting_titles))
        
        supporting_doc_ids = []
        for title, sentences in context:
            if title not in title_to_docid:
                doc_id = len(corpus)
                title_to_docid[title] = doc_id
                text = " ".join(sentences)
                corpus.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": text
                })
            else:
                doc_id = title_to_docid[title]
            
            if title in supporting_titles:
                supporting_doc_ids.append(doc_id)
        
        supporting_doc_ids = list(set(supporting_doc_ids))
        
        queries.append({
            "query_id": query_id,
            "question": question,
            "answer": answer,
            "type": q_type,
            "level": level,
            "supporting_titles": list(supporting_titles),
            "supporting_facts": supporting_facts
        })
        
        ground_truth.append({
            "query_id": query_id,
            "relevant_doc_ids": supporting_doc_ids
        })
    
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
    
    title_map_path = os.path.join(output_dir, "title_to_docid.json")
    print(f"Writing title->docid mapping ({len(title_to_docid)} entries) to {title_map_path}...")
    with open(title_map_path, "w", encoding="utf-8") as f:
        json.dump(title_to_docid, f, ensure_ascii=False, indent=2)
    
    stats = {
        "total_queries": len(queries),
        "total_documents": len(corpus),
        "type_distribution": dict(type_counts),
        "level_distribution": dict(level_counts),
        "avg_supporting_docs_per_query": (
            sum(supporting_facts_per_query) / len(supporting_facts_per_query)
            if supporting_facts_per_query else 0
        ),
        "min_supporting_docs": min(supporting_facts_per_query) if supporting_facts_per_query else 0,
        "max_supporting_docs": max(supporting_facts_per_query) if supporting_facts_per_query else 0,
    }
    
    stats_path = os.path.join(output_dir, "stats.json")
    print(f"Writing statistics to {stats_path}...")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print("\n=== Statistics ===")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Total documents (deduplicated): {stats['total_documents']}")
    print(f"Type distribution: {stats['type_distribution']}")
    print(f"Level distribution: {stats['level_distribution']}")
    print(f"Avg supporting docs per query: {stats['avg_supporting_docs_per_query']:.2f}")
    print(f"Min/Max supporting docs: {stats['min_supporting_docs']}/{stats['max_supporting_docs']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Parse HotpotQA training set")
    parser.add_argument("--input", required=True, help="Path to hotpot_train_v1.1.json")
    parser.add_argument("--output_dir", required=True, help="Output directory for generated files")
    args = parser.parse_args()
    
    parse_hotpotqa(args.input, args.output_dir)


if __name__ == "__main__":
    main()
