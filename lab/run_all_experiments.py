import sys
sys.path.insert(0, '.')
import csv
from pathlib import Path
from data_loader import load_data
from models.factory import create_model

results_file = Path('results/baseline_results.csv')
results_file.parent.mkdir(parents=True, exist_ok=True)

experiments = [
    ('lr', 'german_credit'),
    ('rf', 'german_credit'),
    ('xgboost', 'german_credit'),
    ('deepfm', 'german_credit'),
    ('tabnet', 'german_credit'),
    ('llm', 'german_credit', True),
    ('lr', 'credit_card'),
    ('rf', 'credit_card'),
    ('xgboost', 'credit_card'),
    ('deepfm', 'credit_card'),
    ('tabnet', 'credit_card'),
    ('llm', 'credit_card', True),
]

for exp in experiments:
    model_type = exp[0]
    dataset = exp[1]
    mock_mode = exp[2] if len(exp) > 2 else False

    print(f'Running {model_type} on {dataset} (mock={mock_mode})...')

    try:
        if model_type == 'llm':
            model = create_model(model_type, mock_mode=mock_mode, random_state=42)
        elif model_type == 'xgboost':
            model = create_model(model_type, random_state=42)
        else:
            model = create_model(model_type, random_state=42)

        X_train, X_test, y_train, y_test = load_data(dataset)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        result = {
            'model_type': model_type,
            'dataset': dataset,
            'accuracy': f"{metrics['accuracy']:.4f}",
            'precision': f"{metrics['precision']:.4f}",
            'recall': f"{metrics['recall']:.4f}",
            'f1': f"{metrics['f1']:.4f}",
            'auc_roc': f"{metrics['auc_roc']:.4f}",
            'auc_pr': f"{metrics['auc_pr']:.4f}",
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

        file_exists = results_file.exists()
        with open(results_file, 'a', newline='') as f:
            fieldnames = ['model_type', 'dataset', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'train_samples', 'test_samples']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

        print(f"  Saved: {result}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\nAll results saved to {results_file}")
