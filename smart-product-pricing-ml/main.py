import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def enhanced_extract_features(content):
    """Enhanced feature extraction from catalog_content"""
    if pd.isna(content):
        content = ""
    
    features = {}
    
    # Extract brand (more robust pattern)
    brand_match = re.search(r'Item Name:\s*([^,\n]+)', content)
    features['brand'] = brand_match.group(1).strip() if brand_match else "unknown"
    
    # Extract numeric value and unit
    value_match = re.search(r'Value:\s*([\d.]+)', content)
    unit_match = re.search(r'Unit:\s*([^\n,\\.]+)', content)
    
    features['value'] = float(value_match.group(1)) if value_match else 1.0
    features['unit'] = unit_match.group(1).strip().lower() if unit_match else "unknown"
    
    # Extract pack quantity with multiple patterns
    pack_patterns = [
        r'Pack of\s*(\d+)',
        r'(\d+)\s*pack',
        r'(\d+)\s*ct',
        r'(\d+)\s*count'
    ]
    
    features['pack_size'] = 1
    for pattern in pack_patterns:
        pack_match = re.search(pattern, content.lower())
        if pack_match:
            features['pack_size'] = int(pack_match.group(1))
            break
    
    # Enhanced text cleaning
    clean_text = content.lower()
    # Remove metadata markers
    clean_text = re.sub(r'item name:|bullet point \d+:|product description:|value:.*|unit:.*', '', clean_text)
    # Remove special characters but keep words
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    features['clean_text'] = clean_text
    features['text_length'] = len(clean_text)
    features['word_count'] = len(clean_text.split())
    
    # Enhanced keyword features
    keywords = {
        'organic': ['organic', 'bio', 'biological'],
        'gluten_free': ['gluten-free', 'gluten free', 'no gluten'],
        'premium': ['premium', 'gourmet', 'artisan', 'craft'],
        'natural': ['natural', 'all natural', '100% natural'],
        'healthy': ['healthy', 'health', 'nutritious', 'wholesome'],
        'spicy': ['spicy', 'hot', 'chili', 'pepper'],
        'sweet': ['sweet', 'sugar', 'honey', 'caramel'],
        'sour': ['sour', 'tangy', 'tart', 'citrus'],
        'crunchy': ['crunchy', 'crispy', 'crisp'],
        'soft': ['soft', 'tender', 'smooth']
    }
    
    for feature, word_list in keywords.items():
        features[f'has_{feature}'] = int(any(word in clean_text for word in word_list))
    
    # Extract potential weight/volume from text
    weight_patterns = [
        r'(\d+)\s*oz',
        r'(\d+)\s*ounce',
        r'(\d+)\s*lb',
        r'(\d+)\s*pound',
        r'(\d+)\s*g',
        r'(\d+)\s*gram',
        r'(\d+)\s*kg',
        r'(\d+)\s*kilo'
    ]
    
    features['detected_weight'] = 1.0
    for pattern in weight_patterns:
        weight_match = re.search(pattern, clean_text)
        if weight_match:
            features['detected_weight'] = float(weight_match.group(1))
            break
    
    return features

def robust_preprocess_data(df, is_train=True, le_brand=None, le_unit=None):
    """Robust preprocessing that handles test set unknowns"""
    print("Extracting enhanced features from catalog content...")
    
    # Use parallel processing for feature extraction
    extracted_features = df['catalog_content'].apply(enhanced_extract_features)
    feature_df = pd.DataFrame(extracted_features.tolist())
    
    # Combine with original data
    df_processed = df.copy()
    for col in feature_df.columns:
        df_processed[col] = feature_df[col]
    
    # Handle categorical encoding robustly
    if is_train or le_brand is None:
        le_brand = LabelEncoder()
        le_unit = LabelEncoder()
        
        # Fit on training data
        df_processed['brand_encoded'] = le_brand.fit_transform(df_processed['brand'].fillna('unknown'))
        df_processed['unit_encoded'] = le_unit.fit_transform(df_processed['unit'].fillna('unknown'))
    else:
        # Transform test data, handling unknown categories
        df_processed['brand_encoded'] = df_processed['brand'].apply(
            lambda x: le_brand.transform([x])[0] if x in le_brand.classes_ else len(le_brand.classes_)
        )
        df_processed['unit_encoded'] = df_processed['unit'].apply(
            lambda x: le_unit.transform([x])[0] if x in le_unit.classes_ else len(le_unit.classes_)
        )
    
    # Calculate price per unit for training data
    if is_train and 'price' in df_processed.columns:
        df_processed['price_per_unit'] = df_processed['price'] / (df_processed['value'] * df_processed['detected_weight'])
    
    return df_processed, le_brand, le_unit

def create_tfidf_features(df, tfidf_vectorizer=None, fit_vectorizer=True):
    """Create TF-IDF features"""
    print("Creating TF-IDF features...")
    
    if fit_vectorizer or tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        tfidf_features = tfidf_vectorizer.fit_transform(df['clean_text'].fillna(''))
    else:
        tfidf_features = tfidf_vectorizer.transform(df['clean_text'].fillna(''))
    
    return tfidf_features, tfidf_vectorizer

def create_lightgbm_model():
    """Create LightGBM model with optimal parameters"""
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=128,
        max_depth=10,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
        subsample=0.8,
        colsample_bytree=0.8
    )
    return model

def calculate_comprehensive_errors(y_true, y_pred, dataset_name=""):
    """Calculate comprehensive error metrics"""
    print(f"\n📊 COMPREHENSIVE ERROR METRICS - {dataset_name}")
    print("=" * 50)
    
    # Basic regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"R² Score: {r2:.4f}")
    
    # Additional metrics
    mse = mean_squared_error(y_true, y_pred)
    mean_actual = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    std_actual = np.std(y_true)
    std_pred = np.std(y_pred)
    
    print(f"Mean Squared Error (MSE): ${mse:.4f}")
    print(f"Mean Actual Price: ${mean_actual:.4f}")
    print(f"Mean Predicted Price: ${mean_pred:.4f}")
    print(f"Std Actual Price: ${std_actual:.4f}")
    print(f"Std Predicted Price: ${std_pred:.4f}")
    
    # Error distribution
    errors = y_pred - y_true
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(np.abs(errors))
    
    print(f"Mean Error: ${mean_error:.4f}")
    print(f"Std Error: ${std_error:.4f}")
    print(f"Max Absolute Error: ${max_error:.4f}")
    
    # Percentage within thresholds
    within_10_percent = np.mean(np.abs(errors) <= 0.1 * y_true) * 100
    within_20_percent = np.mean(np.abs(errors) <= 0.2 * y_true) * 100
    within_50_percent = np.mean(np.abs(errors) <= 0.5 * y_true) * 100
    
    print(f"Within 10% of actual: {within_10_percent:.2f}%")
    print(f"Within 20% of actual: {within_20_percent:.2f}%")
    print(f"Within 50% of actual: {within_50_percent:.2f}%")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'mse': mse,
        'within_10_percent': within_10_percent,
        'within_20_percent': within_20_percent,
        'within_50_percent': within_50_percent
    }

def main():
    """Main execution function with comprehensive error analysis"""
    try:
        # Load data
        print("Loading data...")
        train_df = pd.read_csv("/kaggle/input/hackathon-ds/student_resource/dataset/train.csv")
        test_df = pd.read_csv("/kaggle/input/hackathon-ds/student_resource/dataset/test.csv")
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Phase 1: Preprocess data
        print("\n" + "="*50)
        print("PHASE 1: DATA PREPROCESSING")
        print("="*50)
        
        train_processed, le_brand, le_unit = robust_preprocess_data(train_df, is_train=True)
        test_processed, _, _ = robust_preprocess_data(test_df, is_train=False, le_brand=le_brand, le_unit=le_unit)
        
        # Phase 2: Feature Engineering
        print("\n" + "="*50)
        print("PHASE 2: FEATURE ENGINEERING")
        print("="*50)
        
        # Create TF-IDF features
        tfidf_features_train, tfidf_vectorizer = create_tfidf_features(train_processed)
        tfidf_features_test, _ = create_tfidf_features(test_processed, tfidf_vectorizer, fit_vectorizer=False)
        
        # Select numerical features
        numerical_features = [
            'value', 'pack_size', 'brand_encoded', 'unit_encoded', 
            'text_length', 'word_count', 'detected_weight',
            'has_organic', 'has_gluten_free', 'has_premium', 
            'has_natural', 'has_healthy', 'has_spicy', 'has_sweet',
            'has_sour', 'has_crunchy', 'has_soft'
        ]
        
        X_numerical_train = train_processed[numerical_features].values.astype(np.float32)
        X_numerical_test = test_processed[numerical_features].values.astype(np.float32)
        
        # Combine features
        print("Combining features...")
        X_train = np.hstack([X_numerical_train, tfidf_features_train[:, :1500].toarray()])
        X_test = np.hstack([X_numerical_test, tfidf_features_test[:, :1500].toarray()])
        
        # Target variable (log transform for better performance)
        y_train = train_processed['price'].values
        y_train_log = np.log1p(y_train)  # log(1 + price)
        
        print(f"Final training features shape: {X_train.shape}")
        print(f"Final test features shape: {X_test.shape}")
        
        # Phase 3: Model Training with Validation
        print("\n" + "="*50)
        print("PHASE 3: MODEL TRAINING & VALIDATION")
        print("="*50)
        
        # Train-validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_log, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Convert back to original scale for validation
        y_val_original = np.expm1(y_val)
        
        # Train LightGBM model with correct early stopping syntax
        print("Training LightGBM model...")
        model = create_lightgbm_model()
        
        # Use callbacks for early stopping (correct syntax)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        
        # Phase 4: Comprehensive Error Analysis
        print("\n" + "="*50)
        print("PHASE 4: COMPREHENSIVE ERROR ANALYSIS")
        print("="*50)
        
        # Predict on validation set
        val_pred_log = model.predict(X_val)
        val_pred = np.expm1(val_pred_log)
        
        # Calculate validation errors
        val_metrics = calculate_comprehensive_errors(y_val_original, val_pred, "VALIDATION SET")
        
        # Predict on training set (for model assessment)
        train_pred_log = model.predict(X_tr)
        train_pred_original = np.expm1(train_pred_log)
        y_tr_original = np.expm1(y_tr)
        
        # Calculate training errors
        train_metrics = calculate_comprehensive_errors(y_tr_original, train_pred_original, "TRAINING SET")
        
        # Phase 5: Final Model Training and Prediction
        print("\n" + "="*50)
        print("PHASE 5: FINAL PREDICTIONS")
        print("="*50)
        
        # Train final model on all training data
        print("Training final model on all training data...")
        final_model = create_lightgbm_model()
        final_model.fit(X_train, y_train_log)
        
        # Make predictions on test set
        print("Making predictions on test set...")
        test_pred_log = final_model.predict(X_test)
        test_pred = np.expm1(test_pred_log)
        
        # Make predictions on training set (for adding to train dataset)
        print("Making predictions on training set...")
        train_full_pred_log = final_model.predict(X_train)
        train_full_pred = np.expm1(train_full_pred_log)
        
        # Calculate final training errors
        final_train_metrics = calculate_comprehensive_errors(y_train, train_full_pred, "FULL TRAINING SET")
        
        # Phase 6: Save Results
        print("\n" + "="*50)
        print("PHASE 6: SAVING RESULTS")
        print("="*50)
        
        # Create enhanced test file with predictions
        test_with_predictions = test_df.copy()
        test_with_predictions['predicted_price'] = test_pred
        
        # Create enhanced train file with predictions
        train_with_predictions = train_df.copy()
        train_with_predictions['predicted'] = train_full_pred
        
        # Save files
        test_output_file = '/kaggle/working/test_with_predictions.csv'
        train_output_file = '/kaggle/working/train_with_predictions.csv'
        submission_file = '/kaggle/working/submission_lightgbm.csv'
        
        test_with_predictions.to_csv(test_output_file, index=False)
        train_with_predictions.to_csv(train_output_file, index=False)
        
        # Create submission file (only sample_id and price)
        submission = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': test_pred
        })
        submission.to_csv(submission_file, index=False)
        
        print(f"✅ Test file with predictions saved to: {test_output_file}")
        print(f"✅ Train file with predictions saved to: {train_output_file}")
        print(f"✅ Submission file saved to: {submission_file}")
        
        # Print summary statistics
        print(f"\n📈 PREDICTION SUMMARY")
        print("=" * 30)
        print(f"Training set predictions:")
        print(f"  - Mean actual price: ${y_train.mean():.2f}")
        print(f"  - Mean predicted price: ${train_full_pred.mean():.2f}")
        print(f"  - Final training RMSE: ${final_train_metrics['rmse']:.4f}")
        print(f"  - Final training R²: {final_train_metrics['r2']:.4f}")
        
        print(f"\nTest set predictions:")
        print(f"  - Mean predicted price: ${test_pred.mean():.2f}")
        print(f"  - Min predicted price: ${test_pred.min():.2f}")
        print(f"  - Max predicted price: ${test_pred.max():.2f}")
        
        # Show some examples from test set
        print(f"\n🔍 SAMPLE TEST PREDICTIONS")
        print("=" * 30)
        sample_indices = np.random.choice(len(test_pred), min(5, len(test_pred)), replace=False)
        for idx in sample_indices:
            sample_id = test_df.iloc[idx]['sample_id']
            catalog_preview = test_df.iloc[idx]['catalog_content'][:100] + "..."
            print(f"   Sample {sample_id}: ${test_pred[idx]:.2f}")
            print(f"     Preview: {catalog_preview}")
            print()
            
        # Feature importance (if available)
        try:
            feature_names = numerical_features + [f'tfidf_{i}' for i in range(1500)]
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(model.feature_importances_)],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES")
            print("=" * 40)
            print(importance_df.head(10).to_string(index=False))
            
        except Exception as e:
            print(f"Note: Could not display feature importance: {e}")
        
        # Compare training vs validation performance
        print(f"\n📊 MODEL PERFORMANCE COMPARISON")
        print("=" * 40)
        print(f"Metric           | Training  | Validation")
        print(f"-----------------|-----------|-----------")
        print(f"RMSE             | ${train_metrics['rmse']:.4f} | ${val_metrics['rmse']:.4f}")
        print(f"MAE              | ${train_metrics['mae']:.4f} | ${val_metrics['mae']:.4f}")
        print(f"R² Score         | {train_metrics['r2']:.4f}   | {val_metrics['r2']:.4f}")
        print(f"Within 10%       | {train_metrics['within_10_percent']:.2f}%   | {val_metrics['within_10_percent']:.2f}%"
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
