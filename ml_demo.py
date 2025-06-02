import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load the extracted features and prepare for machine learning"""
    print("Loading extracted features...")
    df = pd.read_csv('d:/HAND_GESTURE/Data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Classes: {sorted(df['label'].unique())}")
    print(f"Class distribution:")
    print(df['label'].value_counts())
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder

def train_and_evaluate_models(X, y, label_encoder):
    """Train and evaluate different ML models"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        if name == 'Random Forest':
            # Random Forest doesn't require scaling
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            # SVM and KNN benefit from scaling
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=label_encoder.classes_))
    
    return results, models, scaler, (X_test, y_test, X_test_scaled)

def analyze_feature_importance(rf_model, feature_names):
    """Analyze feature importance from Random Forest"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance
    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features for Hand Gesture Recognition')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('d:/HAND_GESTURE/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance

def create_confusion_matrix(y_test, y_pred, label_encoder, model_name):
    """Create and display confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'd:/HAND_GESTURE/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_prediction(models, scaler, X_test, y_test, X_test_scaled, label_encoder):
    """Demonstrate prediction on a few test samples"""
    print("\n" + "="*60)
    print("PREDICTION DEMONSTRATION")
    print("="*60)
      # Select a few random samples
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\nSample {i+1}:")
        print(f"True label: {label_encoder.classes_[y_test[idx]]}")
        
        sample_rf = X_test.iloc[idx:idx+1]
        sample_scaled = X_test_scaled[idx:idx+1]
        
        print("Predictions:")
        # Random Forest prediction
        rf_pred = models['Random Forest'].predict(sample_rf)[0]
        print(f"  Random Forest: {label_encoder.classes_[rf_pred]}")
        
        # SVM prediction
        svm_pred = models['SVM'].predict(sample_scaled)[0]
        print(f"  SVM: {label_encoder.classes_[svm_pred]}")
        
        # KNN prediction
        knn_pred = models['KNN'].predict(sample_scaled)[0]
        print(f"  KNN: {label_encoder.classes_[knn_pred]}")

def main():
    print("Hand Gesture Recognition - Machine Learning Demo")
    print("="*60)
    
    # Load and prepare data
    X, y, label_encoder = load_and_prepare_data()
    
    # Train and evaluate models
    results, models, scaler, test_data = train_and_evaluate_models(X, y, label_encoder)
    X_test, y_test, X_test_scaled = test_data
    
    # Compare model performance
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    best_model = max(results, key=results.get)
    print(f"\nBest performing model: {best_model} ({results[best_model]:.4f})")
    
    # Feature importance analysis (Random Forest)
    feature_importance = analyze_feature_importance(models['Random Forest'], X.columns)
    
    # Create confusion matrix for the best model
    if best_model == 'Random Forest':
        y_pred_best = models[best_model].predict(X_test)
    else:
        y_pred_best = models[best_model].predict(X_test_scaled)
    
    create_confusion_matrix(y_test, y_pred_best, label_encoder, best_model)
    
    # Demonstrate predictions
    demonstrate_prediction(models, scaler, X_test, y_test, X_test_scaled, label_encoder)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Successfully extracted {X.shape[1]} meaningful features from hand landmarks")
    print(f"✅ Trained and evaluated 3 different ML models")
    print(f"✅ Best model: {best_model} with {results[best_model]:.2%} accuracy")
    print(f"✅ Feature importance analysis completed")
    print(f"✅ Visualization saved to d:/HAND_GESTURE/")
    
    # Save the trained models
    import pickle
    
    model_data = {
        'models': models,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': list(X.columns),
        'results': results
    }
    
    with open('d:/HAND_GESTURE/trained_models.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Trained models saved to d:/HAND_GESTURE/trained_models.pkl")

if __name__ == "__main__":
    main()
