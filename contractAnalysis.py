"""
Legal Contract Analysis Pipeline
"""

import os
import csv
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import pandas as pd


PDF_DIR = r"D:\Contract_Analysis\contracts"
OUTPUT_DIR = "output"

MAX_CONTRACTS = 50

RESUME_FROM_CHECKPOINT = True 

DEBUG_MODE = False 

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 50
MAX_CONTEXT_CHARS = 25000

Path(OUTPUT_DIR).mkdir(exist_ok=True)


def get_latest_checkpoint_file():
    checkpoint_files = list(Path(OUTPUT_DIR).glob("checkpoint_*.json"))
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getctime)

def load_checkpoint():
    checkpoint_file = get_latest_checkpoint_file()
    
    if not checkpoint_file:
        print("  No checkpoint found. Starting from scratch.")
        return None, []
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        results = checkpoint_data if isinstance(checkpoint_data, list) else checkpoint_data.get('results', checkpoint_data)
        
        print("\n" + "=" * 80)
        print(" CHECKPOINT FOUND - RESUMING PIPELINE")
        print("=" * 80)
        print(f"Checkpoint file: {checkpoint_file.name}")
        print(f"Contracts already processed: {len(results)}")
        
        processed_ids = {r['contract_id'] for r in results}
        return processed_ids, results
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, []

def save_checkpoint(results: List[Dict], index: int):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_file = Path(OUTPUT_DIR) / f"checkpoint_{index}_{timestamp}.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n Checkpoint saved: {index} contracts processed")


def load_contracts(pdf_dir: str, limit: int = None) -> Tuple[List[Document], Dict[str, str]]:
    print("=" * 80)
    print("STEP 1: LOADING CONTRACTS")
    print("=" * 80)
    
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
    if limit:
        pdf_files = pdf_files[:limit]
    
    documents = []
    full_texts = {}
    
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            reader = PdfReader(pdf_path)
            text = " ".join([page.extract_text() for page in reader.pages])
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            text = text.strip()
            
            if len(text) > 100:
                documents.append(Document(page_content=text, metadata={'source': pdf_file}))
                full_texts[pdf_file] = text
        except Exception as e:
            print(f"Failed to load {pdf_file}: {e}")
    
    print(f"\n Loaded {len(documents)} contracts successfully")
    return documents, full_texts



def create_vector_store(documents: List[Document]) -> FAISS:
    print("\n" + "=" * 80)
    print("STEP 2: CREATING VECTOR STORE (Semantic Search - BONUS)")
    print("=" * 80)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} contracts")
    
    print("Building embeddings (this may take a few minutes)...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(" Vector store created - semantic search enabled")
    return vector_store


def get_enriched_context(vector_store, full_text: str, contract_id: str, keywords: List[str], max_chars: int) -> str:
    if DEBUG_MODE:
        print(f"\n[DEBUG] Searching contract: {contract_id} for keywords: {keywords[:3]}")
    
    all_relevant_sections = []
    seen_content = set()
    
    # STRATEGY 1: RAG Semantic Search
    for keyword in keywords:
        results = vector_store.similarity_search(keyword, k=TOP_K)
        for chunk in results:
            if chunk.metadata.get('source') == contract_id:
                content = chunk.page_content
                if content not in seen_content and len(content) > 100:
                    all_relevant_sections.append(('rag', content))
                    seen_content.add(content)
    
    # STRATEGY 2: Exact Keyword Matching
    paragraphs = re.split(r'\n\s*\n+', full_text)
    for para in paragraphs:
        if len(para.strip()) < 100:
            continue
        para_lower = para.lower()
        for keyword in keywords:
            if keyword.lower() in para_lower:
                para_start = full_text.find(para)
                if para_start != -1:
                    context_start = max(0, para_start - 500)
                    context_end = min(len(full_text), para_start + len(para) + 500)
                    expanded = full_text[context_start:context_end]
                    if expanded not in seen_content:
                        all_relevant_sections.append(('keyword', expanded))
                        seen_content.add(expanded)
                break
    
    # STRATEGY 3: Section Header Detection
    section_patterns = [
        r'(?:^|\n)\s*(?:ARTICLE|SECTION|§)\s*\d+[\.\):]?\s*([^\n]{5,80})\s*\n',
        r'(?:^|\n)\s*\d+\.\d*\s*([^\n]{5,80})\s*\n',
        r'(?:^|\n)\s*([A-Z][A-Z\s]{10,60})\s*\n',
    ]
    
    for pattern in section_patterns:
        for match in re.finditer(pattern, full_text):
            header = match.group(1).lower()
            if any(kw.lower() in header for kw in keywords):
                section_start = match.start()
                section_end = min(len(full_text), section_start + 2500)
                section_text = full_text[section_start:section_end]
                if section_text not in seen_content:
                    all_relevant_sections.append(('header', section_text))
                    seen_content.add(section_text)
    
    if DEBUG_MODE:
        print(f"[DEBUG] Found {len(all_relevant_sections)} relevant sections")
        print(f"[DEBUG]    RAG: {sum(1 for s in all_relevant_sections if s[0] == 'rag')}")
        print(f"[DEBUG]    Keyword: {sum(1 for s in all_relevant_sections if s[0] == 'keyword')}")
        print(f"[DEBUG]    Header: {sum(1 for s in all_relevant_sections if s[0] == 'header')}")
    
    # Combine: Headers > Keywords > RAG
    prioritized = []
    prioritized.extend([s[1] for s in all_relevant_sections if s[0] == 'header'])
    prioritized.extend([s[1] for s in all_relevant_sections if s[0] == 'keyword'])
    prioritized.extend([s[1] for s in all_relevant_sections if s[0] == 'rag'])
    
    beginning = full_text[:8000]
    combined = beginning + "\n\n=== RELEVANT CLAUSES ===\n\n" + "\n\n".join(prioritized)
    final_context = combined[:max_chars]
    
    if DEBUG_MODE:
        print(f"[DEBUG] Final context length: {len(final_context)} chars")
    
    return final_context



def extract_party_names(text: str) -> List[str]: # (No Changes)
    patterns = [
        r'between\s+([A-Z][A-Za-z\s&,\.]+(?:Inc|LLC|Ltd|Corp|Corporation|Company))',
        r'by and between\s+([A-Z][A-Za-z\s&,\.]+(?:Inc|LLC|Ltd|Corp|Corporation|Company))',
        r'"([A-Z][A-Za-z]+)"[,\s]+(?:a|an)\s+(?:Delaware|California|Florida|Texas|Nevada|New York)',
    ]
    
    party_names = []
    for pattern in patterns:
        matches = re.findall(pattern, text[:5000])
        party_names.extend(matches)
    
    party_names = [name.strip() for name in party_names]
    party_names = list(set([name for name in party_names if len(name) > 3]))
    return party_names[:5]

def extract_financial_terms(text: str) -> Dict[str, str]:
   
    financial_info = {}
    # Regex improved 
    amounts = re.findall(r'\$\s*[\d,]+(?:\.\d{2})?(?:[.-])?(?:\s*(?:million|M|thousand|K))?', text[:20000])
    if amounts:
        
        cleaned_amounts = [re.sub(r'[\s.-]', '', amt) for amt in amounts]
        financial_info['amounts'] = list(set(cleaned_amounts))[:5]
        
    percentages = re.findall(r'\d+(?:\.\d+)?%', text[:20000])
    if percentages:
        financial_info['percentages'] = list(set(percentages))[:5]
    
    if DEBUG_MODE and financial_info:
        print(f"[DEBUG] Found financial terms: {financial_info}")
    return financial_info

def clean_llm_artifacts(text: str) -> str: 
    
    if "not found" in text.lower()[:50]:
        return "Not found"

    
    prompt_pollution_phrases = [
        r'Search the ENTIRE CONTRACT TEXT.*$',
        r'CRITICAL INSTRUCTIONS:.*$',
        r'RESPONSE \(copy the exact text.*\):.*$'
    ]
    for phrase in prompt_pollution_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE | re.DOTALL)

    
    if text.lower().startswith(('here is', 'here\'s', 'based on', 'according to', 'i found', 'i have extracted', 'i located')):
        if ':' in text[:100]:
            text = text.split(':', 1)[1].strip()
        elif '\n' in text[:100]:
            lines = text.split('\n')
            text = '\n'.join(lines[1:]).strip()
    
    
    text = re.sub(r'^Article\s+\d+\.?\d*:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^Section\s+\d+\.?\d*:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    
    if 'Note:' in text or 'note:' in text:
        text = re.split(r'Note:', text, flags=re.IGNORECASE)[0].strip()
    
    
    end_chatter = [
        r'Since there is no.*$',
        r'There is no explicit.*$',
        r'The contract does not.*$',
        r'However, I did find.*$' 
    ]
    for chatter in end_chatter:
         text = re.sub(chatter, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    
    text = re.sub(r'```.*?\n', '', text, flags=re.DOTALL)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    
    return text.strip()


def validate_extraction(text: str, clause_type: str, primary_keywords: List[str], all_keywords: List[str]) -> bool:
       
    if len(text) < 25:
        if DEBUG_MODE:
            print(f"[DEBUG] Validation failed for {clause_type}: Too short ({len(text)} chars)")
        return False
    
    text_lower = text.lower().strip()
    if text_lower in ['not found', 'no clause found', 'clause does not exist', 'not applicable', 'n/a']:
        if DEBUG_MODE:
            print(f"[DEBUG] Validation failed for {clause_type}: Explicit 'Not found'")
        return False
    
    
    if re.search(r'\[(?:PARTY|AMOUNT|DATE|NAME|COMPANY)\]', text, re.IGNORECASE):
        if DEBUG_MODE:
            print(f"[DEBUG] Validation failed for {clause_type}: Template placeholders")
        return False
    
    
    word_count = len(re.findall(r'[a-zA-Z]{3,}', text))
    if word_count < 10:
        if DEBUG_MODE:
            print(f"[DEBUG] Validation failed for {clause_type}: Not enough words ({word_count})")
        return False

    if not any(kw.lower() in text_lower for kw in all_keywords):
        if DEBUG_MODE:
            print(f"[DEBUG] Validation failed for {clause_type}: Missing *any* relevant keywords (e.g., {all_keywords[0]})")
        return False

    if not any(kw.lower() in text_lower for kw in primary_keywords):
        if DEBUG_MODE:
            print(f"[DEBUG] Validation failed for {clause_type}: Missing PRIMARY keywords (e.g., {primary_keywords[0]})")
        return False

    
    if DEBUG_MODE:
        print(f"[DEBUG] Validation PASSED for {clause_type}! ({len(text)} chars, {word_count} words)")
    return True

def extract_single_clause(llm, vector_store, full_text: str, contract_id: str, 
                          clause_type_name: str, 
                          primary_keywords: List[str], 
                          all_keywords: List[str], 
                          section_headers: List[str]) -> str:
   
    if DEBUG_MODE:
        print(f"\n[DEBUG] --- Extracting: {clause_type_name} ---")

    context = get_enriched_context(vector_store, full_text, contract_id, all_keywords, MAX_CONTEXT_CHARS)
    
    if DEBUG_MODE:
        print(f"[DEBUG] Sending {len(context)} chars to LLM for {clause_type_name}")

    prompt = f"""You are a legal document extraction system. Find and copy the EXACT clause text from the contract.

CONTRACT TEXT:
{context}

CRITICAL INSTRUCTIONS:
1. Search the ENTIRE CONTRACT TEXT thoroughly for the clause below.
2. Copy ONLY text that appears VERBATIM - word for word.
3. Do NOT paraphrase, summarize, or rewrite.
4. Do NOT add explanations or commentary.
5. Stop extracting text when the clause is complete, before the next section begins (e.g., 'ARTICLE 8' or '9. WARRANTIES').
6. Only write "Not found" if absolutely NO relevant text exists.

EXTRACT THIS ONE CLAUSE:

# --- CORRECT ---
{clause_type_name.upper()}
Search for: {', '.join(all_keywords)}
Look in sections: {', '.join(section_headers)}
Extract EVERYTHING about this topic, including all sub-sections.

RESPONSE (copy the exact text or "Not found"):"""
    
    try:
        result = llm.invoke(prompt).strip()
        
        text = clean_llm_artifacts(result)
        
        if validate_extraction(text, clause_type_name, primary_keywords, all_keywords):
            return text
        else:
            return "Not found"
            
    except Exception as e:
        print(f"Error during {clause_type_name} extraction: {e}")
        return "Error"


def generate_summary(llm, vector_store, full_text: str, contract_id: str, party_names: List[str], financial_info: Dict) -> str:
    
    if DEBUG_MODE:
        print(f"\n[DEBUG] --- Generating: Summary ---")
        
    summary_keywords = ['agreement', 'purpose', 'services', 'obligations', 'payment', 'fees', 'compensation', 'term', 'duration']
    context = get_enriched_context(vector_store, full_text, contract_id, summary_keywords, MAX_CONTEXT_CHARS)
    
    financial_str = ""
    if 'amounts' in financial_info and financial_info['amounts']:
        financial_str += f"Dollar amounts: {', '.join(financial_info['amounts'])}\n"
    if 'percentages' in financial_info and financial_info['percentages']:
        financial_str += f"Percentages: {', '.join(financial_info['percentages'])}\n"
    if not financial_str:
        financial_str = "No specific amounts identified"
    
    party_str = ', '.join(party_names) if party_names else 'Parties not clearly identified'
    
    prompt = f"""Generate a precise, factual 100-150 word contract summary.

CONTRACT TEXT:
{context}

PARTIES: {party_str}
FINANCIAL TERMS: {financial_str}

RULES:
1. Use ONLY information in CONTRACT TEXT above
2. Do NOT invent names, amounts, dates, obligations
3. Use exact party names and dollar amounts from FINANCIAL TERMS
4. Do NOT use placeholders like [PARTY] or [AMOUNT]

INCLUDE:
- Agreement PURPOSE
- PARTIES involved (exact names)
- KEY OBLIGATIONS of each party
- FINANCIAL TERMS (exact amounts found, e.g., $480,000)
- IMPORTANT CONDITIONS (term, termination)

REQUIREMENTS:
- ONE flowing paragraph (no bullet points)
- Professional tone, present tense
- 100-150 words

SUMMARY (start immediately):"""
    
    try:
        summary = llm.invoke(prompt).strip()
        summary = clean_llm_artifacts(summary)
        
        if len(summary) < 80: # Summary is too short to be useful
             if DEBUG_MODE:
                print(f"[DEBUG] Summary failed validation: Too short ({len(summary)} chars)")
             return "Error: Summary too short"
        
        # Word count enforcement
        words = summary.split()
        if len(words) > 165:
            summary = ' '.join(words[:150])
            if '.' in summary:
                last_period = summary.rfind('.')
                if last_period > len(summary) * 0.75:
                    summary = summary[:last_period + 1]
        
        if DEBUG_MODE:
            print(f"[DEBUG] Summary generation PASSED.")
        return summary
    except Exception as e:
        return f"Error: {str(e)}"


CLAUSE_DEFINITIONS = {
    'termination_clause': {
        'name': 'Termination Clause',
        'primary_keywords': ['termination', 'terminate', 'breach', 'default', 'expire', 'cancel', 'dissolution'],
        'all_keywords': ['termination', 'terminate', 'breach', 'default', 'cure', 'expire', 'end', 'cancel', 'dissolution'],
        'sections': ['Termination', 'Term', 'Duration', 'Default', 'Breach']
    },
    'confidentiality_clause': {
        'name': 'Confidentiality Clause',
        'primary_keywords': ['confidential', 'proprietary', 'non-disclosure', 'secrecy'],
        'all_keywords': ['confidential', 'proprietary', 'disclosure', 'secret', 'non-disclosure', 'nda'],
        'sections': ['Confidentiality', 'Non-Disclosure', 'Proprietary Information']
    },
    'liability_clause': {
        'name': 'Liability / Indemnification Clause',
        'primary_keywords': ['liability', 'indemnification', 'indemnify', 'hold harmless'], # Stricter!
        'all_keywords': ['liability', 'indemnification', 'indemnify', 'damages', 'hold harmless', 'limitation of liability', 'warranty', 'disclaimer'],
        'sections': ['Indemnification', 'Liability', 'Limitation of Liability', 'Warranty', 'Disclaimer']
    }
}

def main():
    print("\n" + "=" * 80)
    print("LEGAL CONTRACT ANALYSIS - v3 FINAL PIPELINE")
    print(f"Processing up to {MAX_CONTRACTS} contracts (Full Re-run)")
    print("=" * 80)
    
    start_time = datetime.now()
    
    processed_ids, results = (set(), [])
    if RESUME_FROM_CHECKPOINT:
        loaded = load_checkpoint()
        if loaded[0]:
            processed_ids, results = loaded
    
    all_documents, full_texts = load_contracts(PDF_DIR, limit=MAX_CONTRACTS)
    
    if not all_documents:
        print("ERROR: No documents loaded.")
        return

    documents_to_process = [doc for doc in all_documents if doc.metadata['source'] not in processed_ids] if processed_ids else all_documents
    
    if not documents_to_process:
        print("\n All contracts already processed!")
        return results
    
    if processed_ids:
        print(f"\n Skipping {len(processed_ids)} contracts")
        print(f"   Remaining: {len(documents_to_process)}")
    
    print(f"\n  Estimated time: {len(documents_to_process) * 90 / 60:.1f} minutes")
    
    vector_store = create_vector_store(all_documents)
    
    print("\n" + "=" * 80)
    print("INITIALIZING LLM")
    print("=" * 80)
    llm = OllamaLLM(model="llama3.1:8b", temperature=0.1, num_predict=3072, top_k=20, top_p=0.9)
    print("✓ LLM initialized")
    
    print("\n" + "=" * 80)
    print("EXTRACTING CLAUSES & SUMMARIES (v3 High-Accuracy Mode)")
    print("=" * 80)
    
    for idx, doc in enumerate(tqdm(documents_to_process, desc="Processing"), 1):
        contract_id = doc.metadata['source']
        full_text = full_texts[contract_id]
        
        if not DEBUG_MODE:
             print(f"\nProcessing: {contract_id}") 
        
        party_names = extract_party_names(full_text)
        financial_info = extract_financial_terms(full_text)
        
        row = {'contract_id': contract_id}
        
       
        row['summary'] = generate_summary(llm, vector_store, full_text, contract_id, party_names, financial_info)
        
       
        for clause_key, definition in CLAUSE_DEFINITIONS.items():
            row[clause_key] = extract_single_clause(
                llm=llm,
                vector_store=vector_store,
                full_text=full_text,
                contract_id=contract_id,
                clause_type_name=definition['name'],
                primary_keywords=definition['primary_keywords'], # Pass primary
                all_keywords=definition['all_keywords'],         # Pass all
                section_headers=definition['sections']
            )
        
        results.append(row)
        
     
        if (len(results)) % 10 == 0:
            save_checkpoint(results, len(results))
    
  
    if results:
        save_checkpoint(results, len(results))

    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = Path(OUTPUT_DIR) / f"contract_analysis_{timestamp}.csv"
    fieldnames = ['contract_id', 'summary', 'termination_clause', 'confidentiality_clause', 'liability_clause']
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f" CSV: {csv_file}")
    
    json_file = Path(OUTPUT_DIR) / f"contract_analysis_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f" JSON: {json_file}")
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)
    
    total_processed = len(results)
    avg_time = (duration / len(documents_to_process)) if documents_to_process else 0 

    print(f"Total Contracts: {total_processed}")
    print(f"Contracts Processed this run: {len(documents_to_process)}")
    print(f"Duration: {duration/60:.1f} min")
    print(f"Avg: {avg_time:.1f}s per contract (this run)")
    
    
    term_success = sum(1 for r in results if r['termination_clause'] not in ['Not found', 'Error'])
    conf_success = sum(1 for r in results if r['confidentiality_clause'] not in ['Not found', 'Error'])
    liab_success = sum(1 for r in results if r['liability_clause'] not in ['Not found', 'Error'])
    
    print(f"\nSuccess Rates (Total):")
    if total_processed > 0:
        print(f"  Termination: {term_success}/{total_processed} ({term_success/total_processed*100:.1f}%)")
        print(f"  Confidentiality: {conf_success}/{total_processed} ({conf_success/total_processed*100:.1f}%)")
        print(f"  Liability: {liab_success}/{total_processed} ({liab_success/total_processed*100:.1f}%)")
    else:
        print("  No results to calculate rates.")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = main()