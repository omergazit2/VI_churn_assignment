# VI_churn_assignment

## Project Goal

This project develops a churn prediction model for WellCo to identify members at high risk of leaving. The model produces a ranked list of members who would benefit most from targeted outreach interventions.

I was unable to complete this assignment because it requires understanding causal inference, which is not part of my current knowledge base. Given the limited time available, I could not adequately learn this topic to the level needed for success.




## Running Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebooks in order:
   - [notebooks/EDA.ipynb](notebooks/EDA.ipynb) - Exploratory data analysis
   - [notebooks/model.ipynb](notebooks/model.ipynb) - Model training and evaluation

## Data Explanation

The dataset contains member churn information across multiple data sources:

### 1. Churn Labels ([data/schema_churn_labels.md](data/schema_churn_labels.md))
- **member_id** (int): Unique member identifier
- **signup_date** (date): Member signup date (YYYY-MM-DD)
- **churn** (int, {0,1}): Target label indicating near-term churn after the observation window
- **outreach** (int, {0,1}): Binary flag indicating whether the member received outreach (treatment)

### 2. App Usage ([data/schema_app_usage.md](data/schema_app_usage.md))
- **member_id** (int): Unique member identifier
- **event_type** (string): Application event type (e.g., "session")
- **timestamp** (datetime, UTC): Event timestamp within the fixed observation window

### 3. Web Visits ([data/schema_web_visits.md](data/schema_web_visits.md))
- **member_id** (int): Unique member identifier
- **url** (string): Page URL visited
- **title** (string): Page title
- **description** (string): Short description of the page content
- **timestamp** (datetime, UTC): Event timestamp within the fixed observation window

### 4. Claims ([data/schema_claims.md](data/schema_claims.md))
- **member_id** (int): Unique member identifier
- **icd_code** (string): ICD-10 diagnosis code (e.g., E11.9, I10, Z71.3)
- **diagnosis_date** (date): Diagnosis date (YYYY-MM-DD), within the observation window


## Feature Extraction

I explored two feature extraction approaches:

1. **Data aggregation over the entire time window** - Aggregating user behavior and characteristics across the full observation period
2. **Temporal feature extraction** - Extracting time-sensitive features to capture recent behavioral patterns

Due to time constraints, the initial temporal feature extraction approach did not yield improvements in model performance. Further refinement of the temporal features would have required significant additional time, so I proceeded with the data aggregation method, which proved more effective for this iteration.


## Modeling Strategy

I implemented two approaches to address the churn prediction problem:

### 1. Simple Churn Prediction
A single model trained on all clients to predict the probability of churn. This baseline approach identifies clients at high risk of leaving, regardless of intervention strategy.

### 2. Two-Model Uplift Approach
This approach aims to identify clients who would benefit most from outreach interventions:

- **Model T (Treatment)**: Trained on clients who received outreach
- **Model C (Control)**: Trained on clients who did not receive outreach

**Uplift Calculation:**
For each client, the uplift is calculated as the difference in predicted churn probability between the two models:
```
Uplift = P(churn | no outreach) - P(churn | outreach)
```

Clients are then prioritized for outreach based on their uplift scores. Higher uplift indicates clients who are most likely to benefit from intervention - they have high churn risk without outreach but lower risk with outreach.

This approach targets resources toward clients where outreach makes the greatest difference, rather than simply targeting all high-risk clients.
