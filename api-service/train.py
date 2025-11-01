import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import joblib
import os
import argparse
from datetime import datetime
from utils.cleaner import clean_text_for_model
from pymongo import MongoClient

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed. Install with: pip install mlflow")

class ExpenseModelTrainer:

    def __init__(self,data_source='csv',csv_path='data/sample_data.csv',mongo_uri=None,db_name='expense_db',collection_name='expenses',use_mlflow=False,experiment_name='expense_categorization'):
        # mongodb needs to be checked

        self.data_source = data_source
        self.csv_path = csv_path
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.results_log = []
        
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)

    def load_data_from_csv(self):
        print(f"Loading data from CSV: {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        return df    

    def load_data_from_mongodb(self):
        print(f"Loading data from MongoDB: {self.db_name}.{self.collection_name}...")
        
        if not self.mongo_uri:
            raise ValueError("MongoDB URI not provided")
        
        client = MongoClient(self.mongo_uri)
        db= client[self.db_name]
        collection = db[self.collection_name]

        #fetch all documents
        cursor= collection.find({})
        df= pd.DataFrame(list(cursor))

        #drop the _id
        if '_id' in df.columns:
            df=df.drop('_id',axis=1)

        client.close()
        print(f"Loaded {len(df)} records from MongoDB")
        return df
    
    def load_and_prepare_data(self):
        
        if self.data_source == 'csv':
            df= self.load_data_from_csv()
        elif self.data_source == 'mongodb':
            df= self.load_data_from_mongodb()
        else:
              raise ValueError(f"Invalid data source: {self.data_source}")
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        #check for the req columns
        required_columns =['vendor','description','amount','category']
        missing_cols= [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        #remove rows with missing category
        initial_size = len(df)
        df= df.dropna(subset=['category'])
        print(f"Removed {initial_size - len(df)} rows with missing categories")
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Categories: {df['category'].unique()}")
        print(f"Category distribution:\n{df['category'].value_counts()}")
        
        return df
    
    def create_combined_features(self,df):

        #clean individual fields
        df['vendor_clean'] = df['vendor'].apply(clean_text_for_model)
        df['description_clean'] = df['description'].apply(clean_text_for_model)
        df['amount_str'] = df['amount'].apply(lambda x: f"amount_{str(int(float(x))) if not pd.isna(x) else '0'}")

         # Combine all features
        df['combined_text'] = (
            df['vendor_clean'] + ' ' + 
            df['description_clean'] + ' ' + 
            df['amount_str']
        )

        return df
    
    def get_model_configs(self):

        configs = {
            'Logistic Regression': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', LogisticRegression(random_state=42))
                ]),
                'param_grid': {
                    'tfidf__max_features': [3000, 5000],
                    'tfidf__ngram_range': [(1, 1), (1, 2)],
                    'clf__C': [0.1, 1.0, 10.0],
                    'clf__max_iter': [1000]
                }
            },
            'Naive Bayes': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', MultinomialNB())
                ]),
                'param_grid': {
                    'tfidf__max_features': [3000, 5000],
                    'tfidf__ngram_range': [(1, 1), (1, 2)],
                    'clf__alpha': [0.1, 0.5, 1.0]
                }
            },
            'Linear SVM': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', LinearSVC(random_state=42))
                ]),
                'param_grid': {
                    'tfidf__max_features': [3000, 5000],
                    'tfidf__ngram_range': [(1, 1), (1, 2)],
                    'clf__C': [0.1, 1.0, 10.0],
                    'clf__max_iter': [2000]
                }
            },
            'Random Forest': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', RandomForestClassifier(random_state=42))
                ]),
                'param_grid': {
                    'tfidf__max_features': [2000, 3000],
                    'tfidf__ngram_range': [(1, 1), (1, 2)],
                    'clf__n_estimators': [50, 100],
                    'clf__max_depth': [None, 20]
                }
            }
        }
        
        return configs
    
    def calculate_metrics(self, y_true, y_pred, average='weighted'):
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0)
        }
        return metrics
    
    def train_with_grid_search(self,x_train,x_test,y_train,y_test,use_grid_search=True):

        configs=self.get_model_configs()
        results={}
        best_model = None
        best_accuracy = 0
        best_name = ""
        best_params = None

        print("TRAINING AND EVALUATING MODELS")

        for name,config in configs.items():
            print (f"training :{name}")

            if self.use_mlflow:
                mlflow.start_run(run_name=name)

            try:
                if use_grid_search:
                    print("performing gridsearch for hyperparameter tuning")
                    grid_search = GridSearchCV(
                        config['pipeline'],
                        config['param_grid'],
                        cv=3,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=1
                    )
                    grid_search.fit(x_train,y_train)
                    model=grid_search.best_estimator_
                    best_grid_params= grid_search.best_params_
                    print(f"Best parameters: {best_grid_params}")
                else:
                    model = config['pipeline']
                    model.fit(x_train, y_train)
                    best_grid_params = None 

                y_pred = model.predict(x_test)

                metrics = self.calculate_metrics(y_test, y_pred)
                
                print(f"\nTest Metrics:")
                print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
                print(f"  F1-Score:  {metrics['f1_score']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")

                 
                # Cross-validation
                cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
                print(f"\nCross-validation scores: {cv_scores}")
                print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Classification report
                print(f"\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                # Store results
                result_entry = {
                    'model_name': name,
                    'timestamp': datetime.now().isoformat(),
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'best_params': str(best_grid_params) if best_grid_params else 'default',
                    'data_source': self.data_source
                }
                
                results[name] = result_entry
                self.results_log.append(result_entry)
                
                # Log to MLflow
                if self.use_mlflow:
                    mlflow.log_params(best_grid_params if best_grid_params else {})
                    mlflow.log_metrics(metrics)
                    mlflow.log_metric('cv_mean', cv_scores.mean())
                    mlflow.log_metric('cv_std', cv_scores.std())
                    mlflow.sklearn.log_model(model, "model")
                
                
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_model = model
                    best_name = name
                    best_params = best_grid_params
            
            finally:
                if self.use_mlflow:
                    mlflow.end_run()


        print("RESULTS SUMMARY")
        
        print(f"{'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}")
       
        
        for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{name:<25} {res['accuracy']:.4f}     {res['f1_score']:.4f}     "
                  f"{res['precision']:.4f}     {res['recall']:.4f}")
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {best_name}")
        print(f"BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        if best_params:
            print(f"BEST PARAMS: {best_params}")
        print(f"{'='*70}")
        
        return best_model, best_name, best_accuracy, results  
    
    def save_results_to_csv(self,filepath='models/training_results.csv'):

        os.makedirs('models',exist_ok=True)

        df_results= pd.DataFrame(self.results_log)

        #append to existing file if it exists
        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            df_results = pd.concat([df_existing, df_results], ignore_index=True)

            df_results.to_csv(filepath,index=False)
            print(f"\n training results saved")

    def save_model(self,model,model_name,accuracy,categories,train_size,test_size):

        os.makedirs('models',exist_ok=True)

        model_path = 'models/classifier.joblib'
        joblib.dump(model,model_path)
        print(f"\n model saved to :{model_path}")

        #save metadata
        metadata = {
            'model_name': model_name,
            'accuracy': accuracy,
            'categories': categories,
            'training_samples': train_size,
            'test_samples': test_size,
            'trained_at': datetime.now().isoformat(),
            'data_source': self.data_source
        }

        metadata_path = 'models/metadata.joblib'
        joblib.dump(metadata, metadata_path)
        print(f"   Metadata saved to: {metadata_path}")

    def load_existing_model(self):
       
        model_path = 'models/classifier.joblib'
        metadata_path = 'models/metadata.joblib'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        
        model = joblib.load(model_path)
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
        
        print(f"\n  Loaded existing model from: {model_path}")
        if metadata:
            print(f"   Model type: {metadata.get('model_name', 'Unknown')}")
            print(f"   Previous accuracy: {metadata.get('accuracy', 0)*100:.2f}%")
            print(f"   Trained at: {metadata.get('trained_at', 'Unknown')}")
        
        return model, metadata  

    def train(self,use_grid_search=True):

        df=self.load_and_prepare_data()
        df=self.create_combined_features(df)

        x= df['combined_text']
        y= df['category']

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

        print(f"\nTraining set size: {len(x_train)}")
        print(f"Test set size: {len(x_test)}")

        best_model,best_name,best_accuracy,results=self.train_with_grid_search(x_train,x_test,y_train,y_test,use_grid_search)

        if best_accuracy >= 0.80:
            print(f"\n SUCCESS! Best model accuracy ({best_accuracy*100:.2f}%) meets the ≥80% threshold")
        else:
            print(f"\n  WARNING: Best model accuracy ({best_accuracy*100:.2f}%) is below 80% threshold")

        
        self.save_model(best_model, best_name, best_accuracy,
                       y.unique().tolist(), len(x_train), len(x_test))

        self.save_results_to_csv()
        
        return best_model, best_accuracy, results
    

    
def main():
    """Main function with CLI arguments."""
    parser = argparse.ArgumentParser(description='Train expense categorization model')
    parser.add_argument('--source', choices=['csv', 'mongodb'], default='csv',
                       help='Data source: csv or mongodb')
    parser.add_argument('--csv-path', default='data/sample_data.csv',
                       help='Path to CSV file')
    parser.add_argument('--mongo-uri', default=None,
                       help='MongoDB connection URI')
    parser.add_argument('--db-name', default='expense_db',
                       help='MongoDB database name')
    parser.add_argument('--collection', default='expenses',
                       help='MongoDB collection name')
    parser.add_argument('--use-mlflow', action='store_true',
                       help='Enable MLflow tracking')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='Disable GridSearchCV (faster but less optimal)')
    parser.add_argument('--reload-model', action='store_true',
                       help='Load existing model before training')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ExpenseModelTrainer(
        data_source=args.source,
        csv_path=args.csv_path,
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        collection_name=args.collection,
        use_mlflow=args.use_mlflow
    )
    
    # Optionally load existing model
    if args.reload_model:
        try:
            old_model, old_metadata = trainer.load_existing_model()
            print("Previous model loaded. Will compare with new training results.\n")
        except FileNotFoundError as e:
            print(f"No existing model found: {e}\n")
    
    # Train model
    
    print("STARTING TRAINING PIPELINE")
    
    
    model, accuracy, results = trainer.train(use_grid_search=not args.no_grid_search)
    
    
    print("TRAINING COMPLETE!")
    
    
    return model, accuracy


if __name__ == "__main__":
    model, accuracy = main()

































