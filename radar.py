"""Purchase-time return prediction with a purged chronological evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score)

NUMERIC = ['price', 'discount', 'quantity', 'customer_tenure_days']
CATEGORICAL = ['category', 'channel']
FEATURES = NUMERIC + CATEGORICAL
WINDOW_DAYS = 30


def generate(n=12000, seed=42):
    if n < 500: raise ValueError('Generate at least 500 orders')
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp('2024-01-01') + pd.to_timedelta(np.linspace(0, 540, n), unit='D')
    frame = pd.DataFrame({
        'ordered_at': dates,
        'price': np.round(rng.lognormal(3.8, .6, n), 2),
        'discount': rng.uniform(0, .5, n),
        'quantity': rng.integers(1, 5, n),
        'customer_tenure_days': rng.integers(0, 1500, n),
        'category': rng.choice(['apparel', 'electronics', 'home'], n),
        'channel': rng.choice(['web', 'mobile'], n)})
    logit = (-2.5 + .005*frame.price + 2*frame.discount + .22*(frame.quantity-1)
             - .0005*frame.customer_tenure_days + 1.1*(frame.category=='apparel')
             + .2*(frame.channel=='mobile'))
    frame['returned'] = rng.binomial(1, 1/(1+np.exp(-logit)))
    # Every order's label is regarded as complete only after its return window closes.
    frame['label_available_at'] = frame.ordered_at + pd.Timedelta(days=WINDOW_DAYS)
    return frame


def validate(frame):
    required = FEATURES + ['ordered_at', 'label_available_at', 'returned']
    if set(frame.columns) != set(required):
        raise ValueError('Expected exactly the documented schema; reject unknown or post-purchase columns')
    data = frame.copy()
    for col in ['ordered_at', 'label_available_at']:
        data[col] = pd.to_datetime(data[col], errors='raise', utc=True)
        if data[col].isna().any(): raise ValueError('Missing dates')
    if data.empty or data.returned.isna().any() or not data.returned.isin([0,1]).all():
        raise ValueError('Binary nonmissing returned labels required')
    for name in NUMERIC:
        data[name] = pd.to_numeric(data[name], errors='raise')
        if not np.isfinite(data[name].dropna()).all(): raise ValueError('Nonfinite numeric feature')
    if (data.price < 0).any() or not data.discount.dropna().between(0,1).all():
        raise ValueError('Invalid price or discount')
    if (data.quantity.dropna() < 1).any() or (data.customer_tenure_days.dropna() < 0).any():
        raise ValueError('Invalid quantity or tenure')
    for col in CATEGORICAL:
        if not data[col].dropna().map(lambda value: isinstance(value,str)).all():
            raise ValueError('Categorical values must be strings')
    if (data.label_available_at < data.ordered_at + pd.Timedelta(days=WINDOW_DAYS)).any():
        raise ValueError('Labels must mature after the 30-day return window')
    return data.sort_values('ordered_at', kind='stable').reset_index(drop=True)


def split(frame):
    data = validate(frame)
    n = len(data)
    if n < 100: raise ValueError('At least 100 orders required')
    validation_start = data.ordered_at.iloc[int(n*.6)]
    test_start = data.ordered_at.iloc[int(n*.8)]
    train = data[(data.ordered_at < validation_start) & (data.label_available_at < validation_start)]
    validation = data[(data.ordered_at >= validation_start) & (data.ordered_at < test_start)
                      & (data.label_available_at < test_start)]
    test = data[data.ordered_at >= test_start]
    for part in [train, validation, test]:
        if len(part) < 20 or part.returned.nunique() != 2:
            raise ValueError('Each chronological split needs at least 20 rows and both label classes')
    return train, validation, test


def pipeline():
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    category = Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                         ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return Pipeline([('features', ColumnTransformer([('numeric', numeric, NUMERIC),
                                                    ('category', category, CATEGORICAL)])),
                     ('model', LogisticRegression(max_iter=1000, random_state=42))])


def cost(y, probabilities, threshold, missed=8., false_alarm=2.):
    predicted = np.asarray(probabilities) >= threshold
    y = np.asarray(y)
    return float((missed*np.sum((y==1)&~predicted) + false_alarm*np.sum((y==0)&predicted))/len(y))


def choose_threshold(y, probabilities):
    # Tune on validation labels only, including flag-everything and flag-nothing options.
    thresholds = np.r_[np.linspace(0,1,201), 1.01]
    return float(min(thresholds, key=lambda t: cost(y, probabilities, t)))


def metrics(y, probabilities, threshold):
    prediction = np.asarray(probabilities) >= threshold
    return {'roc_auc': float(roc_auc_score(y, probabilities)),
            'average_precision': float(average_precision_score(y, probabilities)),
            'brier_score': float(brier_score_loss(y, probabilities)),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction, zero_division=0)),
            'confusion_matrix_tn_fp_fn_tp': confusion_matrix(y, prediction, labels=[0,1]).ravel().tolist(),
            'assumed_cost_per_order': cost(y, probabilities, threshold)}


def explain(model, row):
    transformed = model.named_steps['features'].transform(row[FEATURES])
    classifier = model.named_steps['model']
    contributions = transformed[0] * classifier.coef_[0]
    names = model.named_steps['features'].get_feature_names_out()
    intercept = float(classifier.intercept_[0])
    return {'intercept_log_odds': intercept,
            'contributions_log_odds': dict(zip(names, map(float, contributions))),
            'sum_log_odds': float(intercept+contributions.sum()),
            'probability': float(model.predict_proba(row[FEATURES])[0,1]),
            'meaning': 'Exact additive model contributions in log-odds, not causal effects.'}


def run(frame):
    train, validation, test = split(frame)
    model = pipeline().fit(train[FEATURES], train.returned)
    val_probs = model.predict_proba(validation[FEATURES])[:,1]
    threshold = choose_threshold(validation.returned, val_probs)
    test_probs = model.predict_proba(test[FEATURES])[:,1]
    prior = float(train.returned.mean())
    baseline_threshold = choose_threshold(validation.returned, np.full(len(validation), prior))
    report = {'data_origin': 'Caller-supplied dataset; CLI demo uses synthetic data',
              'rows': {'train':len(train),'validation':len(validation),'test':len(test),
                       'purged': len(frame)-len(train)-len(validation)-len(test)},
              'periods': {name:{'first_order':str(part.ordered_at.min()),'last_order':str(part.ordered_at.max()),
                                'latest_label':str(part.label_available_at.max())}
                          for name,part in [('train',train),('validation',validation),('test',test)]},
              'features': FEATURES, 'threshold_from_validation': threshold,
              'test_prevalence':float(test.returned.mean()),
              'model':metrics(test.returned,test_probs,threshold),
              'training_prior_baseline':metrics(test.returned,np.full(len(test),prior),baseline_threshold),
              'baseline_threshold_from_validation':baseline_threshold,
              'cost_assumptions':{'missed_return':8,'false_alarm':2,'units':'illustrative cost units, not measured savings'},
              'example_explanation':explain(model,test.iloc[[0]]),
              'limitations':['Synthetic data does not establish real-world performance.',
                             'Costs are assumptions; no intervention benefit has been measured.',
                             'Strict column allowlist does not prove source features were collected at purchase time.',
                             'Customers may recur across time splits; this evaluates future orders, not unseen customers.',
                             'All test outcomes must have fully matured before offline evaluation.']}
    predictions=test[['ordered_at','returned']].copy()
    predictions['return_probability']=test_probs
    predictions['flag_for_review']=test_probs>=threshold
    return model, report, predictions


def save_chart(predictions, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import PrecisionRecallDisplay
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    PrecisionRecallDisplay.from_predictions(predictions.returned,predictions.return_probability,ax=axes[0])
    axes[0].axhline(predictions.returned.mean(),ls='--',color='gray',label='Prevalence')
    axes[0].legend()
    observed, predicted = calibration_curve(predictions.returned,predictions.return_probability,n_bins=8,strategy='quantile')
    axes[1].plot(predicted,observed,'o-',label='Model')
    axes[1].plot([0,1],[0,1],'--',color='gray')
    axes[1].set(xlabel='Mean predicted probability',ylabel='Observed return rate',title='Test calibration')
    fig.suptitle('Return Radar · held-out evaluation')
    fig.tight_layout()
    fig.savefig(out/'evaluation.png',dpi=160)
    fig.savefig(out/'evaluation.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,help='Use a CSV matching the documented schema')
    parser.add_argument('--out',type=Path,default=Path('work/demo'))
    args=parser.parse_args()
    frame=pd.read_csv(args.csv) if args.csv else generate()
    _, report, predictions=run(frame)
    report['data_origin']='User-supplied CSV; provenance not verified' if args.csv else 'Synthetic seeded demo (seed 42)'
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    predictions.to_csv(args.out/'test_predictions.csv',index=False)
    save_chart(predictions,args.out)
    print(json.dumps({'rows':report['rows'],'model':report['model'],'baseline':report['training_prior_baseline'],
                      'output':str(args.out)},indent=2))

if __name__=='__main__': main()
