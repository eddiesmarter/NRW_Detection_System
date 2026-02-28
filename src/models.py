from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def get_stacked_ensemble():
    """
    Combines base learners into a stacked model to leverage complementary strengths.
    Ref: [cite: 358, 561]
    """
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, kernel='rbf')),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ]
    
    # Meta-learner: Stacked Logistic Regression [cite: 356, 358]
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(),
        cv=5
    )
    return stack
