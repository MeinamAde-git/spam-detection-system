import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os


def plot_label_distribution(df, save_path="outputs/01_label_distribution.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    counts = df['label'].value_counts()
    labels = ['Ham (Legitimate)', 'Spam']
    colors = ['#2ed573', '#ff4757']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(counts, labels=labels, colors=colors,
           autopct='%1.1f%%', explode=(0, 0.05),
           startangle=140, textprops={'fontsize': 12})
    ax.set_title('Email Label Distribution', fontsize=15, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {save_path}")


def plot_training_loss(model, save_path="outputs/02_training_loss.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(model.loss_curve_) + 1)
    ax.plot(epochs, model.loss_curve_, color='orange',
            linewidth=2.5, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {save_path}")


def plot_confusion_matrix(cm, save_path="outputs/03_confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                linewidths=2, annot_kws={'size': 18, 'weight': 'bold'},
                ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=15, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {save_path}")


def plot_top_features(vectorizer, model, save_path="outputs/04_top_features.png", top_n=15):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    feature_names = vectorizer.get_feature_names_out()
    weights = np.abs(model.coefs_[0]).sum(axis=1)
    top_idx = np.argsort(weights)[-top_n:]
    top_words = [feature_names[i] for i in top_idx]
    top_weights = weights[top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_words, top_weights, color='steelblue', height=0.6)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {save_path}")


def plot_metrics_bar(metrics, save_path="outputs/05_metrics_summary.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    names  = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        metrics['accuracy']  * 100,
        metrics['precision'] * 100,
        metrics['recall']    * 100,
        metrics['f1']        * 100,
    ]
    colors = ['#1e90ff', '#ffa502', '#2ed573', '#ff4757']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}%',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=15)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: {save_path}")


def generate_all_plots(df, model, vectorizer, metrics):
    print("\n📈 Generating visualizations...")
    plot_label_distribution(df)
    plot_training_loss(model)
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_top_features(vectorizer, model)
    plot_metrics_bar(metrics)
    print("✅ All plots saved to outputs/ folder!")


if __name__ == "__main__":
    from dataset import create_dataset
    from preprocessing import preprocess_dataframe, extract_features, split_data
    from model import build_model, train_model, evaluate_model

    df = create_dataset()
    df = preprocess_dataframe(df)
    X, y, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model()
    model = train_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    generate_all_plots(df, model, vectorizer, metrics)