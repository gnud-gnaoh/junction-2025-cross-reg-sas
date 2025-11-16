# Regulatory Overlap & Contradiction Detection Pipeline

This project builds an end-to-end AI pipeline to identify **overlaps**, **contradictions**, and **missing links** across large collections of regulatory texts. It combines preprocessing, rule-based classification, clustering, SAS-based data engineering, and legal-domain language models to streamline regulatory analysis at scale.

The **results** for our project can be found in the csv files:
- contradictions_Risk_DepositRunOff.csv
- contradictions_Risk_MaturityMismatch.csv
- contradictions_Risk_ReputationImpact.csv
- entailments_Risk_DepositRunOff.csv
- entailments_Risk_MaturityMismatch.csv
- entailments_Risk_ReputationImpact.csv

---

## 1. Preprocessing

### Input
- Hundreds of thousands of regulatory paragraphs stored in individual JSON files.

### Process
1. Read JSON files.  
2. Extract paragraphs.  
3. Use **SAS** with custom Python scripts to build the data pipeline that:
   - Ingests and validates extracted paragraphs  
   - Performs initial formatting and dataset structuring  
   - Outputs a consolidated CSV with:
     - `Paragraph_ID`
     - `Filename`
     - `Paragraph`

### Outcome
A unified, SAS-engineered dataset ready for classification.

---

## 2. TF-IDF + K-Means Classification Pipeline

### Step A — Cleaning & Filtering
- SAS pipeline outputs are further processed to:
  - Split paragraphs into classes based on keyword presence  
  - Remove unclassified or empty paragraphs  
  - Drop clusters dominated by irrelevant topics (e.g., *fishing*, *agriculture*, etc.)  
- This eliminates ~70% of noise and non-financial content.

### Step B — Risk Classification
- Use **25 predefined financial risk categories**, each with associated keywords.  
- Assign paragraphs to one or more risks based on keyword frequency.  
- (Optional) Re-cluster unclassified paragraphs to detect missing or weak keywords.

### Outcome
A cleaned dataset of paragraphs mapped to financial risk categories.

---

## 3. Similarity, Overlap, and Contradiction Detection

### Step A — BERT Similarity Analysis
- Use **BERT fine-tuned on legal text** to detect semantically similar paragraphs.  
- Identify pairs or groups of paragraphs that discuss the same theme.

### Step B — Deep Analysis (DeepSeek v3)
For each similar pair:
- Detect **overlaps** (duplicated regulatory requirements)  
- Identify **contradictions** (conflicting mandates)  
- Filter out unrelated matches

### Step C — Resolution Proposals
- Use GenAI to propose **harmonised solutions** for contradictions.

---

## 4. Output

### Deliverables
- Full report containing:
  - All detected overlaps  
  - All detected contradictions  
  - Clusters of closely related regulatory text  
- Suggested harmonisation or resolution for each contradiction.

### Benefits
- Automates manual analysis that normally takes months.  
- Highlights hidden inconsistencies and duplicated compliance burdens.  
- Scales across thousands of pages from multiple regulatory authorities.

---

## 5. Tools & Models

- **SAS** for data ingestion, transformation, and pipeline automation  
- **TF-IDF** for feature extraction  
- **K-Means** for clustering and noise removal  
- **Keyword-based risk classification**  
- **BERT (legal fine-tuned)** for semantic similarity detection  
- **DeepSeek v3** for overlap/contradiction reasoning  
- **GenAI** for proposed resolutions  

---
