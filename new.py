"""
Advanced life insurance persistency analysis
- Left truncation (entry times)
- Exact time (days) preserved
- Time-varying covariates support (CoxTimeVaryingFitter)
- Frailty/cluster adjustments and stratified Cox
- Cox PH with diagnostics and AFT fallback
- Automatic PDF report generation with KM plots, group KM, residuals, Cox tables

How to use:
1. Update DATA_PATH to point at your CSV (or upload file into Kaggle/Colab and set that path).
2. If your dataset contains multiple rows per policy (time-varying records), the script will
   try to detect that and run a CoxTimeVarying model. Otherwise it runs standard Cox PH.
3. If you want me to execute the script for you, upload the CSV here or paste the Kaggle path.

Note: This script uses lifelines and pandas. Run in Kaggle/Colab.
"""

# -------------------------
# Install required libs
# -------------------------
# In Kaggle/Colab uncomment the pip install line if lifelines is missing
# !pip install --quiet lifelines==0.27.4

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter, CoxTimeVaryingFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Configuration (edit)
# -------------------------
DATA_PATH = '/kaggle/input/life-assurance-dataset.csv'  # <-- change this to your CSV path
OUTPUT_DIR = './persistency_outputs'
TIME_UNIT = 'days'  # use exact time available (days)
ADVANCED_HANDLING = True  # left-truncation, time-varying, frailty/cluster

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Column name candidates
# -------------------------
column_candidates = {
    'policy_id': ['policy_id','policy_no','policyno','id','policyid'],
    'policyholder_id': ['holder_id','policyholder_id','customer_id','client_id','cust_id'],
    'issue_date': ['issue_date','inception_date','start_date','effective_date'],
    'entry_date': ['entry_date','observed_start','start_observed','first_observed_date'],
    'lapse_date': ['lapse_date','surrender_date','end_date','termination_date','cancellation_date'],
    'last_observed_date': ['last_observed_date','last_date','cutoff_date','data_cutoff','observed_date'],
    'status': ['status','policy_status','current_status'],
    'premium': ['premium','premium_amount','annual_premium','paid_premium'],
    'payment_frequency': ['payment_frequency','pay_mode','payment_mode','mode'],
    'policy_type': ['policy_type','plan_type','product','policy_plan'],
    'age': ['age','age_at_issue','age_at_inception'],
    'gender': ['gender','sex']
}

# -------------------------
# Utility: detect columns
# -------------------------

def detect_columns(df, candidates):
    found = {}
    cols = set(df.columns.str.lower())
    for key, opts in candidates.items():
        found[key] = None
        for o in opts:
            if o.lower() in cols:
                # find exact casing
                for col in df.columns:
                    if col.lower() == o.lower():
                        found[key] = col
                        break
                if found[key]:
                    break
    return found

# -------------------------
# Helper: try to coerce dates
# -------------------------

def parse_dates_safe(df, col):
    if col is None:
        return pd.Series([pd.NaT]*len(df))
    return pd.to_datetime(df[col], errors='coerce')

# -------------------------
# Load data
# -------------------------
print('Loading:', DATA_PATH)
try:
    raw = pd.read_csv(DATA_PATH, low_memory=False)
except Exception as e:
    raise RuntimeError(f"Could not read DATA_PATH={DATA_PATH}: {e}")

print('Rows:', raw.shape[0], 'Cols:', raw.shape[1])

# -------------------------
# Detect columns and harmonize
# -------------------------
found = detect_columns(raw, column_candidates)
print('Detected columns:')
for k,v in found.items():
    print(f'  {k}: {v}')

if not found['issue_date']:
    raise ValueError('Issue date column is required (no candidate detected).')

# Make a working copy
df = raw.copy()
# parse dates
df['issue_date'] = parse_dates_safe(df, found['issue_date'])
df['lapse_date'] = parse_dates_safe(df, found['lapse_date'])
df['last_observed_date'] = parse_dates_safe(df, found['last_observed_date'])
df['entry_date'] = parse_dates_safe(df, found['entry_date'])

# If last_observed_date missing, create cutoff from max(lapse_date) or today
if df['last_observed_date'].isna().all():
    fallback = df['lapse_date'].max()
    if pd.isna(fallback):
        fallback = pd.to_datetime('today')
    df['last_observed_date'] = fallback

# If entry_date missing, we'll default to issue_date later (no left-truncation)

# -------------------------
# Build time variables: exact days
# -------------------------
# end_date: lapse_date if present else last_observed_date
df['end_date'] = df['lapse_date'].fillna(df['last_observed_date'])

# event indicator: 1 if lapse_date present OR status indicates lapsed
df['event'] = 0
mask_lapse_date = df['lapse_date'].notna()
df.loc[mask_lapse_date, 'event'] = 1
if found['status'] and found['status'] in df.columns:
    low_status = df[found['status']].astype(str).str.lower()
    lapsed_vals = ['lapsed','surrendered','terminated','cancelled','canceled']
    df.loc[low_status.isin(lapsed_vals), 'event'] = 1

# handle entry for left truncation: prefer explicit entry_date, else fall back to issue_date
if df['entry_date'].notna().any():
    df['entry'] = df['entry_date']
else:
    df['entry'] = df['issue_date']

# compute durations as integer days (exact)
# duration = (end_date - entry).days  (for left truncation we measure from entry to end)
# Also compute duration_since_issue for descriptive reasons

df['duration_days'] = (df['end_date'] - df['entry']).dt.days
# duration_since_issue (end - issue) for some analyses
df['dur_since_issue_days'] = (df['end_date'] - df['issue_date']).dt.days

# drop rows with invalid / negative durations
valid = df['duration_days'].notna() & (df['duration_days'] >= 0)
df = df.loc[valid].copy()
print('After filtering invalid durations, rows =', len(df))

# convert premium, age
if found['premium'] and found['premium'] in df.columns:
    df['premium'] = pd.to_numeric(df[found['premium']], errors='coerce')
else:
    df['premium'] = np.nan

if found['age'] and found['age'] in df.columns:
    df['age'] = pd.to_numeric(df[found['age']], errors='coerce')
else:
    df['age'] = np.nan

# payment frequency and policy_type and gender
for c in ['payment_frequency','policy_type','gender','policyholder_id','policy_id']:
    col = found.get(c)
    if col and col in df.columns:
        df[c] = df[col]

# normalize payment frequency
if 'payment_frequency' in df.columns:
    df['payment_frequency'] = df['payment_frequency'].astype(str).str.title()

# create age groups
bins = [0,25,35,45,55,150]
labels = ['<25','25-34','35-44','45-54','55+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Save cleaned snapshot
df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_snapshot.csv'), index=False)

# -------------------------
# Decide whether dataset is time-varying (multiple rows per policy)
# Condition: if policy_id exists and some policy_id appears more than once -> time-varying
# -------------------------
use_time_varying = False
if found['policy_id'] and found['policy_id'] in df.columns:
    counts = df[found['policy_id']].value_counts()
    if counts.max() > 1:
        use_time_varying = True

print('Use time-varying format:', use_time_varying)

# -------------------------
# OBJECTIVE 1: Kaplan-Meier with left truncation support
# -------------------------
kmf = KaplanMeierFitter()
# lifelines accepts `entry` param for left-truncation
entry_col = 'entry' if 'entry' in df.columns else None
if entry_col:
    # durations are in days; pass duration as integer days
    T = df['duration_days']
else:
    T = df['duration_days']
E = df['event']

# Fit KM using days as timeline; but we'll plot x-axis in days and also convert to years where useful
kmf.fit(T, event_observed=E, entry=df['entry'] if entry_col else None, label='All policies')

# Survival estimates at selected times (1,3,5 years) in days
def to_days(years):
    return int(round(years * 365.25))

s_times = {1: to_days(1), 3: to_days(3), 5: to_days(5)}
 surv_probs = {}
for yrs, td in s_times.items():
    try:
        p = float(kmf.survival_function_at_times(td))
    except Exception:
        p = float(kmf.predict(td))
    surv_probs[yrs] = p

print('\nSurvival probabilities (exact days):')
for yrs, p in surv_probs.items():
    print(f' - at {yrs} year(s): survival = {p:.4f}')

# Plot KM (x axis in years for readability)
plt.figure(figsize=(8,5))
ax = kmf.plot(ci_show=True)
ax.set_xlabel('Days since entry (x1000)')
ax.set_ylabel('Survival probability')
ax.set_title('Kaplan-Meier: Policy persistency (days, left-truncation handled)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'km_all_policies_days.png'), dpi=200)
plt.close()

# -------------------------
# OBJECTIVE 2: Stratified KM & Log-rank tests
# -------------------------
group_cols = ['payment_frequency','policy_type','age_group','gender']
group_km_results = {}
for g in group_cols:
    if g not in df.columns:
        continue
    if df[g].nunique(dropna=True) < 2:
        continue
    # dropna for group
    sub = df.dropna(subset=[g, 'duration_days', 'event', 'entry']) if entry_col else df.dropna(subset=[g, 'duration_days', 'event'])
    # Plot by group
    plt.figure(figsize=(9,6))
    for grp in sorted(sub[g].dropna().unique()):
        mask = sub[g] == grp
        if mask.sum() < 5:
            continue
        km = KaplanMeierFitter()
        if entry_col:
            km.fit(sub.loc[mask,'duration_days'], event_observed=sub.loc[mask,'event'], entry=sub.loc[mask,'entry'], label=str(grp))
        else:
            km.fit(sub.loc[mask,'duration_days'], event_observed=sub.loc[mask,'event'], label=str(grp))
        km.plot(ci_show=False)
    plt.title(f'KM by {g} (days)')
    plt.xlabel('Days since entry')
    plt.ylabel('Survival probability')
    plt.legend(title=g)
    plt.grid(True)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'km_by_{g}_days.png')
    plt.savefig(fname, dpi=200)
    plt.close()

    # global log-rank test (if enough groups)
    try:
        lr = multivariate_logrank_test(sub['duration_days'], sub[g], sub['event'])
        group_km_results[g] = {'p_value': lr.p_value}
        print(f'Log-rank {g} p-value: {lr.p_value:.4g}')
    except Exception as e:
        print('Log-rank error for', g, e)

# -------------------------
# OBJECTIVE 3: Cox PH, PH checks, frailty/cluster, stratified Cox, AFT fallback
# -------------------------
# Prepare covariates for the baseline Cox (time-invariant)
covars = []
if 'age' in df.columns and df['age'].notna().sum() > 0:
    covars.append('age')
if 'gender' in df.columns and df['gender'].notna().sum() > 0:
    covars.append('gender')
if 'payment_frequency' in df.columns and df['payment_frequency'].notna().sum() > 0:
    covars.append('payment_frequency')
if 'policy_type' in df.columns and df['policy_type'].notna().sum() > 0:
    covars.append('policy_type')
if 'premium' in df.columns and df['premium'].notna().sum() > 0:
    covars.append('premium')

print('Candidate covariates for Cox:', covars)

# If time-varying use CoxTimeVaryingFitter
if use_time_varying:
    print('Preparing time-varying dataset (CoxTimeVaryingFitter)...')
    # Expected data layout: multiple rows per policy with start_date (or issue_date) and end_date
    # If the dataset already has start/end columns, use them; otherwise try to infer
    # We'll construct a tidy dataframe with columns: id, start, stop, event, covariates...

    id_col = found['policy_id'] if found['policy_id'] in df.columns else 'policy_id'

    # Heuristic: if dataset has columns named 'start_date'/'stop_date' or 'change_date' use them
    start_candidates = ['start_date','period_start','segment_start','active_from']
    stop_candidates = ['stop_date','period_end','segment_end','active_to']
    start_col = None
    stop_col = None
    for c in start_candidates:
        if c in df.columns:
            start_col = c
            break
    for c in stop_candidates:
        if c in df.columns:
            stop_col = c
            break

    # If start/stop not available, we will use the entry and end_date and assume each row is a time window
    if start_col is None:
        df['_tv_start'] = df['entry']
    else:
        df['_tv_start'] = pd.to_datetime(df[start_col], errors='coerce')
    if stop_col is None:
        df['_tv_stop'] = df['end_date']
    else:
        df['_tv_stop'] = pd.to_datetime(df[stop_col], errors='coerce')

    # Convert to numeric durations (days from a reference) for CoxTimeVaryingFitter (it accepts datetimes too)
    tv_df = df[[found['policy_id']] + ['_tv_start','_tv_stop','event'] + [c for c in covars if c in df.columns]].copy()
    tv_df = tv_df.rename(columns={found['policy_id']:'id'})

    # lifelines expects columns 'id','start','stop', and 'event' where event=1 only on the final interval.
    # Ensure event is set only on rows where stop == end_date
    tv_df['start'] = (tv_df['_tv_start'] - pd.Timestamp('1970-01-01')) / np.timedelta64(1, 'D')
    tv_df['stop'] = (tv_df['_tv_stop'] - pd.Timestamp('1970-01-01')) / np.timedelta64(1, 'D')

    # Make event only on rows where stop equals policy end_date
    # We'll map end_date per id
    end_map = df.groupby('id')['end_date'].max()
    tv_df = tv_df.merge(end_map.rename('policy_end'), left_on='id', right_index=True)
    tv_df['policy_end_d'] = (tv_df['policy_end'] - pd.Timestamp('1970-01-01')) / np.timedelta64(1, 'D')
    tv_df['event_interval'] = ((tv_df['stop'] == tv_df['policy_end_d']) & (df['event'] == 1)).astype(int).values

    # Drop rows where start>=stop
    tv_df = tv_df.loc[tv_df['stop'] > tv_df['start']].copy()

    # One-hot encode categorical covariates in tv_df
    tv_covs = [c for c in covars if c in tv_df.columns]
    tv_df = pd.get_dummies(tv_df, columns=[c for c in tv_covs if tv_df[c].dtype=='object'], drop_first=True)

    # Keep only required cols for CoxTimeVaryingFitter
    tv_cols = ['id','start','stop','event_interval'] + [c for c in tv_df.columns if c not in ['id','_tv_start','_tv_stop','start','stop','event','policy_end','policy_end_d','event_interval']]
    tv_final = tv_df[['id','start','stop','event_interval'] + tv_cols[4:]]
    tv_final.rename(columns={'event_interval':'event'}, inplace=True)

    print('TV rows:', tv_final.shape)
    # Fit CoxTimeVaryingFitter
    ctv = CoxTimeVaryingFitter()
    try:
        ctv.fit(tv_final, id_col='id', start_col='start', stop_col='stop', event_col='event', show_progress=True)
        print('\nCoxTimeVarying summary:')
        print(ctv.summary)
        ctv.summary.to_csv(os.path.join(OUTPUT_DIR, 'ctv_summary.csv'))
    except Exception as e:
        print('CoxTimeVarying fitting failed:', e)

    # For baseline display also fit a standard Cox on the snapshot of last covariates
    # Snapshot defined as last row per id
    snapshot = df.sort_values(['policy_id','_tv_stop']).groupby('policy_id').last().reset_index()
    baseline_df = snapshot[['duration_days','event'] + [c for c in covars if c in snapshot.columns]].dropna()
    baseline_df = pd.get_dummies(baseline_df, columns=[c for c in covars if baseline_df[c].dtype=='object'], drop_first=True)
    cph = CoxPHFitter()
    try:
        cph.fit(baseline_df, duration_col='duration_days', event_col='event', show_progress=False)
        cph.summary.to_csv(os.path.join(OUTPUT_DIR,'cox_snapshot_summary.csv'))
        print('\nCox (snapshot) fitted. Summary saved.')
    except Exception as e:
        print('Snapshot Cox failed:', e)

else:
    # Time-invariant Cox PH
    print('Preparing time-invariant Cox PH...')
    model_df = df[['duration_days','event'] + [c for c in covars if c in df.columns]].dropna().copy()
    model_df = pd.get_dummies(model_df, columns=[c for c in covars if model_df[c].dtype=='object'], drop_first=True)
    print('Rows for Cox:', model_df.shape[0])

    cph = CoxPHFitter()
    # Use entry_col for left-truncation if present
    try:
        if 'entry' in df.columns:
            # convert entry to numeric days (relative to epoch) for CoxPH
            model_df['entry_days'] = (df.loc[model_df.index,'entry'] - pd.Timestamp('1970-01-01')) / np.timedelta64(1,'D')
            # lifelines CoxPHFitter.fit accepts entry_col argument
            cph.fit(model_df, duration_col='duration_days', event_col='event', entry_col='entry_days', show_progress=False)
        else:
            cph.fit(model_df, duration_col='duration_days', event_col='event', show_progress=False)
        print('\nCox PH summary:')
        print(cph.summary)
        cph.summary.to_csv(os.path.join(OUTPUT_DIR,'cox_summary.csv'))
    except Exception as e:
        print('Cox PH fitting failed:', e)

    # PH assumption checks
    try:
        cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        print('PH assumptions check produced warnings/errors:', e)

    # If PH violated, consider stratified Cox or AFT
    # We'll automatically try Weibull AFT as fallback
    try:
        aft = WeibullAFTFitter()
        aft.fit(model_df, duration_col='duration_days', event_col='event')
        print('\nWeibull AFT summary:')
        print(aft.summary)
        aft.summary.to_csv(os.path.join(OUTPUT_DIR,'weibull_aft_summary.csv'))
    except Exception as e:
        print('Weibull AFT fitting failed:', e)

# -------------------------
# Frailty / Clustering: if policyholder_id exists we can use cluster_col to get robust SE
# -------------------------
if ADVANCED_HANDLING and found.get('policyholder_id') and found['policyholder_id'] in df.columns and 'cph' in globals():
    try:
        model_df2 = df[['duration_days','event','policyholder_id'] + [c for c in covars if c in df.columns]].dropna()
        model_df2 = pd.get_dummies(model_df2, columns=[c for c in covars if model_df2[c].dtype=='object'], drop_first=True)
        cph_cluster = CoxPHFitter()
        cph_cluster.fit(model_df2, duration_col='duration_days', event_col='event', cluster_col='policyholder_id')
        cph_cluster.summary.to_csv(os.path.join(OUTPUT_DIR,'cox_cluster_summary.csv'))
        print('\nCox PH with cluster robust SE fitted. Summary saved.')
    except Exception as e:
        print('Clustered Cox failed:', e)

# -------------------------
# Save key outputs and create PDF report
# -------------------------
pdf_path = os.path.join(OUTPUT_DIR, 'persistency_advanced_report.pdf')
with PdfPages(pdf_path) as pdf:
    # Title page
    fig, ax = plt.subplots(figsize=(11,8.5))
    ax.axis('off')
    ax.text(0.5,0.7,'Policy Persistency Analysis (Advanced)', ha='center', fontsize=20, weight='bold')
    ax.text(0.5,0.6,f'Dataset: {os.path.basename(DATA_PATH)}', ha='center', fontsize=10)
    ax.text(0.5,0.55,f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}', ha='center', fontsize=9)
    pdf.savefig(fig); plt.close(fig)

    # KM plot
    try:
        img = plt.imread(os.path.join(OUTPUT_DIR,'km_all_policies_days.png'))
        fig, ax = plt.subplots(figsize=(11,8.5))
        ax.imshow(img)
        ax.axis('off')
        pdf.savefig(fig); plt.close(fig)
    except Exception:
        pass

    # group KM images
    for g in group_cols:
        path = os.path.join(OUTPUT_DIR, f'km_by_{g}_days.png')
        if os.path.exists(path):
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11,8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig); plt.close(fig)

    # Cox summary table(s)
    if 'cph' in globals():
        try:
 """
Advanced life insurance persistency analysis
- Left truncation (entry times)
- Exact time (days) preserved
- Time-varying covariates support (CoxTimeVaryingFitter)
- Frailty/cluster adjustments and stratified Cox
- Cox PH with diagnostics and AFT fallback
- Automatic PDF report generation with KM plots, group KM, residuals, Cox tables

How to use:
1. Update DATA_PATH to point at your CSV (or upload file into Kaggle/Colab and set that path).
2. If your dataset contains multiple rows per policy (time-varying records), the script will
   try to detect that and run a CoxTimeVarying model. Otherwise it runs standard Cox PH.
3. If you want me to execute the script for you, upload the CSV here or paste the Kaggle path.

Note: This script uses lifelines and pandas. Run in Kaggle/Colab.
"""

# -------------------------
# Install required libs
# -------------------------
# In Kaggle/Colab uncomment the pip install line if lifelines is missing
# !pip install --quiet lifelines==0.27.4

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter, CoxTimeVaryingFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Configuration (edit)
# -------------------------
DATA_PATH = '/kaggle/input/life-assurance-dataset.csv'  # <-- change this to your CSV path
OUTPUT_DIR = './persistency_outputs'
TIME_UNIT = 'days'  # use exact time available (days)
ADVANCED_HANDLING = True  # left-truncation, time-varying, frailty/cluster

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Column name candidates
# -------------------------
column_candidates = {
    'policy_id': ['policy_id','policy_no','policyno','id','policyid'],
    'policyholder_id': ['holder_id','policyholder_id','customer_id','client_id','cust_id'],
    'issue_date': ['issue_date','inception_date','start_date','effective_date'],
    'entry_date': ['entry_date','observed_start','start_observed','first_observed_date'],
    'lapse_date': ['lapse_date','surrender_date','end_date','termination_date','cancellation_date'],
    'last_observed_date': ['last_observed_date','last_date','cutoff_date','data_cutoff','observed_date'],
    'status': ['status','policy_status','current_status'],
    'premium': ['premium','premium_amount','annual_premium','paid_premium'],
    'payment_frequency': ['payment_frequency','pay_mode','payment_mode','mode'],
    'policy_type': ['policy_type','plan_type','product','policy_plan'],
    'age': ['age','age_at_issue','age_at_inception'],
    'gender': ['gender','sex']
}

# -------------------------
# Utility: detect columns
# -------------------------

def detect_columns(df, candidates):
    found = {}
    cols = set(df.columns.str.lower())
    for key, opts in candidates.items():
        found[key] = None
        for o in opts:
            if o.lower() in cols:
                # find exact casing
                for col in df.columns:
                    if col.lower() == o.lower():
                        found[key] = col
                        break
                if found[key]:
                    break
    return found

# -------------------------
# Helper: try to coerce dates
# -------------------------

def parse_dates_safe(df, col):
    if col is None:
        return pd.Series([pd.NaT]*len(df))
    return pd.to_datetime(df[col], errors='coerce')

# -------------------------
# Load data
# -------------------------
print('Loading:', DATA_PATH)
try:
    raw = pd.read_csv(DATA_PATH, low_memory=False)
except Exception as e:
    raise RuntimeError(f"Could not read DATA_PATH={DATA_PATH}: {e}")

print('Rows:', raw.shape[0], 'Cols:', raw.shape[1])

# -------------------------
# Detect columns and harmonize
# -------------------------
found = detect_columns(raw, column_candidates)
print('Detected columns:')
for k,v in found.items():
    print(f'  {k}: {v}')

if not found['issue_date']:
    raise ValueError('Issue date column is required (no candidate detected).')

# Make a working copy
df = raw.copy()
# parse dates
df['issue_date'] = parse_dates_safe(df, found['issue_date'])
df['lapse_date'] = parse_dates_safe(df, found['lapse_date'])
df['last_observed_date'] = parse_dates_safe(df, found['last_observed_date'])
df['entry_date'] = parse_dates_safe(df, found['entry_date'])

# If last_observed_date missing, create cutoff from max(lapse_date) or today
if df['last_observed_date'].isna().all():
    fallback = df['lapse_date'].max()
    if pd.isna(fallback):
        fallback = pd.to_datetime('today')
    df['last_observed_date'] = fallback

# If entry_date missing, we'll default to issue_date later (no left-truncation)

# -------------------------
# Build time variables: exact days
# -------------------------
# end_date: lapse_date if present else last_observed_date
df['end_date'] = df['lapse_date'].fillna(df['last_observed_date'])

# event indicator: 1 if lapse_date present OR status indicates lapsed
df['event'] = 0
mask_lapse_date = df['lapse_date'].notna()
df.loc[mask_lapse_date, 'event'] = 1
if found['status'] and found['status'] in df.columns:
    low_status = df[found['status']].astype(str).str.lower()
    lapsed_vals = ['lapsed','surrendered','terminated','cancelled','canceled']
    df.loc[low_status.isin(lapsed_vals), 'event'] = 1

# handle entry for left truncation: prefer explicit entry_date, else fall back to issue_date
if df['entry_date'].notna().any():
    df['entry'] = df['entry_date']
else:
    df['entry'] = df['issue_date']

# compute durations as integer days (exact)
# duration = (end_date - entry).days  (for left truncation we measure from entry to end)
# Also compute duration_since_issue for descriptive reasons

df['duration_days'] = (df['end_date'] - df['entry']).dt.days
# duration_since_issue (end - issue) for some analyses
df['dur_since_issue_days'] = (df['end_date'] - df['issue_date']).dt.days

# drop rows with invalid / negative durations
valid = df['duration_days'].notna() & (df['duration_days'] >= 0)
df = df.loc[valid].copy()
print('After filtering invalid durations, rows =', len(df))

# convert premium, age
if found['premium'] and found['premium'] in df.columns:
    df['premium'] = pd.to_numeric(df[found['premium']], errors='coerce')
else:
    df['premium'] = np.nan

if found['age'] and found['age'] in df.columns:
    df['age'] = pd.to_numeric(df[found['age']], errors='coerce')
else:
    df['age'] = np.nan

# payment frequency and policy_type and gender
for c in ['payment_frequency','policy_type','gender','policyholder_id','policy_id']:
    col = found.get(c)
    if col and col in df.columns:
        df[c] = df[col]

# normalize payment frequency
if 'payment_frequency' in df.columns:
    df['payment_frequency'] = df['payment_frequency'].astype(str).str.title()

# create age groups
bins = [0,25,35,45,55,150]
labels = ['<25','25-34','35-44','45-54','55+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Save cleaned snapshot
df.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_snapshot.csv'), index=False)

# -------------------------
# Decide whether dataset is time-varying (multiple rows per policy)
# Condition: if policy_id exists and some policy_id appears more than once -> time-varying
# -------------------------
use_time_varying = False
if found['policy_id'] and found['policy_id'] in df.columns:
    counts = df[found['policy_id']].value_counts()
    if counts.max() > 1:
        use_time_varying = True

print('Use time-varying format:', use_time_varying)

# -------------------------
# OBJECTIVE 1: Kaplan-Meier with left truncation support
# -------------------------
kmf = KaplanMeierFitter()
# lifelines accepts `entry` param for left-truncation
entry_col = 'entry' if 'entry' in df.columns else None
if entry_col:
    # durations are in days; pass duration as integer days
    T = df['duration_days']
else:
    T = df['duration_days']
E = df['event']

# Fit KM using days as timeline; but we'll plot x-axis in days and also convert to years where useful
kmf.fit(T, event_observed=E, entry=df['entry'] if entry_col else None, label='All policies')

# Survival estimates at selected times (1,3,5 years) in days
def to_days(years):
    return int(round(years * 365.25))

s_times = {1: to_days(1), 3: to_days(3), 5: to_days(5)}
 surv_probs = {}
for yrs, td in s_times.items():
    try:
        p = float(kmf.survival_function_at_times(td))
    except Exception:
        p = float(kmf.predict(td))
    surv_probs[yrs] = p

print('\nSurvival probabilities (exact days):')
for yrs, p in surv_probs.items():
    print(f' - at {yrs} year(s): survival = {p:.4f}')

# Plot KM (x axis in years for readability)
plt.figure(figsize=(8,5))
ax = kmf.plot(ci_show=True)
ax.set_xlabel('Days since entry (x1000)')
ax.set_ylabel('Survival probability')
ax.set_title('Kaplan-Meier: Policy persistency (days, left-truncation handled)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'km_all_policies_days.png'), dpi=200)
plt.close()

# -------------------------
# OBJECTIVE 2: Stratified KM & Log-rank tests
# -------------------------
group_cols = ['payment_frequency','policy_type','age_group','gender']
group_km_results = {}
for g in group_cols:
    if g not in df.columns:
        continue
    if df[g].nunique(dropna=True) < 2:
        continue
    # dropna for group
    sub = df.dropna(subset=[g, 'duration_days', 'event', 'entry']) if entry_col else df.dropna(subset=[g, 'duration_days', 'event'])
    # Plot by group
    plt.figure(figsize=(9,6))
    for grp in sorted(sub[g].dropna().unique()):
        mask = sub[g] == grp
        if mask.sum() < 5:
            continue
        km = KaplanMeierFitter()
        if entry_col:
            km.fit(sub.loc[mask,'duration_days'], event_observed=sub.loc[mask,'event'], entry=sub.loc[mask,'entry'], label=str(grp))
        else:
            km.fit(sub.loc[mask,'duration_days'], event_observed=sub.loc[mask,'event'], label=str(grp))
        km.plot(ci_show=False)
    plt.title(f'KM by {g} (days)')
    plt.xlabel('Days since entry')
    plt.ylabel('Survival probability')
    plt.legend(title=g)
    plt.grid(True)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'km_by_{g}_days.png')
    plt.savefig(fname, dpi=200)
    plt.close()

    # global log-rank test (if enough groups)
    try:
        lr = multivariate_logrank_test(sub['duration_days'], sub[g], sub['event'])
        group_km_results[g] = {'p_value': lr.p_value}
        print(f'Log-rank {g} p-value: {lr.p_value:.4g}')
    except Exception as e:
        print('Log-rank error for', g, e)

# -------------------------
# OBJECTIVE 3: Cox PH, PH checks, frailty/cluster, stratified Cox, AFT fallback
# -------------------------
# Prepare covariates for the baseline Cox (time-invariant)
covars = []
if 'age' in df.columns and df['age'].notna().sum() > 0:
    covars.append('age')
if 'gender' in df.columns and df['gender'].notna().sum() > 0:
    covars.append('gender')
if 'payment_frequency' in df.columns and df['payment_frequency'].notna().sum() > 0:
    covars.append('payment_frequency')
if 'policy_type' in df.columns and df['policy_type'].notna().sum() > 0:
    covars.append('policy_type')
if 'premium' in df.columns and df['premium'].notna().sum() > 0:
    covars.append('premium')

print('Candidate covariates for Cox:', covars)

# If time-varying use CoxTimeVaryingFitter
if use_time_varying:
    print('Preparing time-varying dataset (CoxTimeVaryingFitter)...')
    # Expected data layout: multiple rows per policy with start_date (or issue_date) and end_date
    # If the dataset already has start/end columns, use them; otherwise try to infer
    # We'll construct a tidy dataframe with columns: id, start, stop, event, covariates...

    id_col = found['policy_id'] if found['policy_id'] in df.columns else 'policy_id'

    # Heuristic: if dataset has columns named 'start_date'/'stop_date' or 'change_date' use them
    start_candidates = ['start_date','period_start','segment_start','active_from']
    stop_candidates = ['stop_date','period_end','segment_end','active_to']
    start_col = None
    stop_col = None
    for c in start_candidates:
        if c in df.columns:
            start_col = c
            break
    for c in stop_candidates:
        if c in df.columns:
            stop_col = c
            break

    # If start/stop not available, we will use the entry and end_date and assume each row is a time window
    if start_col is None:
        df['_tv_start'] = df['entry']
    else:
        df['_tv_start'] = pd.to_datetime(df[start_col], errors='coerce')
    if stop_col is None:
        df['_tv_stop'] = df['end_date']
    else:
        df['_tv_stop'] = pd.to_datetime(df[stop_col], errors='coerce')

    # Convert to numeric durations (days from a reference) for CoxTimeVaryingFitter (it accepts datetimes too)
    tv_df = df[[found['policy_id']] + ['_tv_start','_tv_stop','event'] + [c for c in covars if c in df.columns]].copy()
    tv_df = tv_df.rename(columns={found['policy_id']:'id'})

    # lifelines expects columns 'id','start','stop', and 'event' where event=1 only on the final interval.
    # Ensure event is set only on rows where stop == end_date
    tv_df['start'] = (tv_df['_tv_start'] - pd.Timestamp('1970-01-01')) / np.timedelta64(1, 'D')
    tv_df['stop'] = (tv_df['_tv_stop'] - pd.Timestamp('1970-01-01')) / np.timedelta64(1, 'D')

    # Make event only on rows where stop equals policy end_date
    # We'll map end_date per id
    end_map = df.groupby('id')['end_date'].max()
    tv_df = tv_df.merge(end_map.rename('policy_end'), left_on='id', right_index=True)
    tv_df['policy_end_d'] = (tv_df['policy_end'] - pd.Timestamp('1970-01-01')) / np.timedelta64(1, 'D')
    tv_df['event_interval'] = ((tv_df['stop'] == tv_df['policy_end_d']) & (df['event'] == 1)).astype(int).values

    # Drop rows where start>=stop
    tv_df = tv_df.loc[tv_df['stop'] > tv_df['start']].copy()

    # One-hot encode categorical covariates in tv_df
    tv_covs = [c for c in covars if c in tv_df.columns]
    tv_df = pd.get_dummies(tv_df, columns=[c for c in tv_covs if tv_df[c].dtype=='object'], drop_first=True)

    # Keep only required cols for CoxTimeVaryingFitter
    tv_cols = ['id','start','stop','event_interval'] + [c for c in tv_df.columns if c not in ['id','_tv_start','_tv_stop','start','stop','event','policy_end','policy_end_d','event_interval']]
    tv_final = tv_df[['id','start','stop','event_interval'] + tv_cols[4:]]
    tv_final.rename(columns={'event_interval':'event'}, inplace=True)

    print('TV rows:', tv_final.shape)
    # Fit CoxTimeVaryingFitter
    ctv = CoxTimeVaryingFitter()
    try:
        ctv.fit(tv_final, id_col='id', start_col='start', stop_col='stop', event_col='event', show_progress=True)
        print('\nCoxTimeVarying summary:')
        print(ctv.summary)
        ctv.summary.to_csv(os.path.join(OUTPUT_DIR, 'ctv_summary.csv'))
    except Exception as e:
        print('CoxTimeVarying fitting failed:', e)

    # For baseline display also fit a standard Cox on the snapshot of last covariates
    # Snapshot defined as last row per id
    snapshot = df.sort_values(['policy_id','_tv_stop']).groupby('policy_id').last().reset_index()
    baseline_df = snapshot[['duration_days','event'] + [c for c in covars if c in snapshot.columns]].dropna()
    baseline_df = pd.get_dummies(baseline_df, columns=[c for c in covars if baseline_df[c].dtype=='object'], drop_first=True)
    cph = CoxPHFitter()
    try:
        cph.fit(baseline_df, duration_col='duration_days', event_col='event', show_progress=False)
        cph.summary.to_csv(os.path.join(OUTPUT_DIR,'cox_snapshot_summary.csv'))
        print('\nCox (snapshot) fitted. Summary saved.')
    except Exception as e:
        print('Snapshot Cox failed:', e)

else:
    # Time-invariant Cox PH
    print('Preparing time-invariant Cox PH...')
    model_df = df[['duration_days','event'] + [c for c in covars if c in df.columns]].dropna().copy()
    model_df = pd.get_dummies(model_df, columns=[c for c in covars if model_df[c].dtype=='object'], drop_first=True)
    print('Rows for Cox:', model_df.shape[0])

    cph = CoxPHFitter()
    # Use entry_col for left-truncation if present
    try:
        if 'entry' in df.columns:
            # convert entry to numeric days (relative to epoch) for CoxPH
            model_df['entry_days'] = (df.loc[model_df.index,'entry'] - pd.Timestamp('1970-01-01')) / np.timedelta64(1,'D')
            # lifelines CoxPHFitter.fit accepts entry_col argument
            cph.fit(model_df, duration_col='duration_days', event_col='event', entry_col='entry_days', show_progress=False)
        else:
            cph.fit(model_df, duration_col='duration_days', event_col='event', show_progress=False)
        print('\nCox PH summary:')
        print(cph.summary)
        cph.summary.to_csv(os.path.join(OUTPUT_DIR,'cox_summary.csv'))
    except Exception as e:
        print('Cox PH fitting failed:', e)

    # PH assumption checks
    try:
        cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        print('PH assumptions check produced warnings/errors:', e)

    # If PH violated, consider stratified Cox or AFT
    # We'll automatically try Weibull AFT as fallback
    try:
        aft = WeibullAFTFitter()
        aft.fit(model_df, duration_col='duration_days', event_col='event')
        print('\nWeibull AFT summary:')
        print(aft.summary)
        aft.summary.to_csv(os.path.join(OUTPUT_DIR,'weibull_aft_summary.csv'))
    except Exception as e:
        print('Weibull AFT fitting failed:', e)

# -------------------------
# Frailty / Clustering: if policyholder_id exists we can use cluster_col to get robust SE
# -------------------------
if ADVANCED_HANDLING and found.get('policyholder_id') and found['policyholder_id'] in df.columns and 'cph' in globals():
    try:
        model_df2 = df[['duration_days','event','policyholder_id'] + [c for c in covars if c in df.columns]].dropna()
        model_df2 = pd.get_dummies(model_df2, columns=[c for c in covars if model_df2[c].dtype=='object'], drop_first=True)
        cph_cluster = CoxPHFitter()
        cph_cluster.fit(model_df2, duration_col='duration_days', event_col='event', cluster_col='policyholder_id')
        cph_cluster.summary.to_csv(os.path.join(OUTPUT_DIR,'cox_cluster_summary.csv'))
        print('\nCox PH with cluster robust SE fitted. Summary saved.')
    except Exception as e:
        print('Clustered Cox failed:', e)

# -------------------------
# Save key outputs and create PDF report
# -------------------------
pdf_path = os.path.join(OUTPUT_DIR, 'persistency_advanced_report.pdf')
with PdfPages(pdf_path) as pdf:
    # Title page
    fig, ax = plt.subplots(figsize=(11,8.5))
    ax.axis('off')
    ax.text(0.5,0.7,'Policy Persistency Analysis (Advanced)', ha='center', fontsize=20, weight='bold')
    ax.text(0.5,0.6,f'Dataset: {os.path.basename(DATA_PATH)}', ha='center', fontsize=10)
    ax.text(0.5,0.55,f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}', ha='center', fontsize=9)
    pdf.savefig(fig); plt.close(fig)

    # KM plot
    try:
        img = plt.imread(os.path.join(OUTPUT_DIR,'km_all_policies_days.png'))
        fig, ax = plt.subplots(figsize=(11,8.5))
        ax.imshow(img)
        ax.axis('off')
        pdf.savefig(fig); plt.close(fig)
    except Exception:
        pass

    # group KM images
    for g in group_cols:
        path = os.path.join(OUTPUT_DIR, f'km_by_{g}_days.png')
        if os.path.exists(path):
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11,8.5))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig); plt.close(fig)

    # Cox summary table(s)
    if 'cph' in globals():
        try:
 
