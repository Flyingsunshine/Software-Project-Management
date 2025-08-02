import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib
import os

class SprintBacklogAnalyzer:
    def __init__(self, data_path=None, df=None):
        """Initialize the analyzer with either a path to CSV or a dataframe"""
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.models = {}
        self.preprocessors = {}
        self.feature_importances = {}
        self.results = {}
        
    def preprocess_data(self):
        """Clean and preprocess the data"""
        df = self.df.copy()
        
        # Handle missing values
        df['DEADLINE (MINUTES)'] = df['DEADLINE (MINUTES)'].fillna(df['DURATION (MINUTES)'].median())
        
        # Feature extraction from ticket titles
        df['TASK_TYPE'] = df['TICKET TITLE'].apply(lambda x: x.split(' - ')[0] if ' - ' in x else x.split(' ')[0])
        
        # Create additional features
        df['DEADLINE_MET'] = df['DURATION (MINUTES)'] <= df['DEADLINE (MINUTES)']
        df['EFFICIENCY_RATIO'] = df['STORY POINT'] / df['DURATION (MINUTES)']
        
        # Categorical variables mapping for better interpretability
        priority_map = {'Highest': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        df['PRIORITY_NUM'] = df['PRIORITY'].map(priority_map)
        
        # Extract sprint number
        df['SPRINT_NUM'] = df['SPRINT NAME'].str.extract(r'(\d+)').astype(int)
        
        self.processed_df = df
        return df
    
    def extract_text_features(self, train_data, test_data=None):
        """Extract features from ticket titles using TF-IDF"""
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
        if test_data is not None:
            # For training and testing separately
            train_text_features = tfidf.fit_transform(train_data['TICKET TITLE'])
            test_text_features = tfidf.transform(test_data['TICKET TITLE'])
            self.tfidf_vectorizer = tfidf
            return train_text_features, test_text_features
        else:
            # For single transformation
            text_features = tfidf.fit_transform(train_data['TICKET TITLE'])
            self.tfidf_vectorizer = tfidf
            return text_features
    
    def build_duration_prediction_model(self):
        """Build a model to predict task duration"""
        df = self.processed_df.copy()
        
        # Define features and target
        X = df[['PROJECT NAME (ANONYMISED)', 'SPRINT_NUM', 'TASK_TYPE', 'STORY POINT', 
                'PRIORITY', 'DEADLINE (MINUTES)']]
        y = df['DURATION (MINUTES)']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get text features
        train_text, test_text = self.extract_text_features(df.loc[X_train.index], df.loc[X_test.index])
        
        # Define preprocessing for numerical features
        numerical_features = ['STORY POINT', 'DEADLINE (MINUTES)', 'SPRINT_NUM']
        numerical_transformer = StandardScaler()
        
        # Define preprocessing for categorical features
        categorical_features = ['PROJECT NAME (ANONYMISED)', 'TASK_TYPE', 'PRIORITY']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Prepare train and test features
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Add text features
        if isinstance(X_train_processed, np.ndarray):
            X_train_processed = np.hstack((X_train_processed, train_text.toarray()))
            X_test_processed = np.hstack((X_test_processed, test_text.toarray()))
        else:
            # If sparse matrix
            from scipy.sparse import hstack
            X_train_processed = hstack([X_train_processed, train_text])
            X_test_processed = hstack([X_test_processed, test_text])
        
        # Train models
        models = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            # Train the model
            model.fit(X_train_processed, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"Model: {name}")
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R²: {r2:.2f}")
            print("-------------------")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['r2'])
        print(f"Best model: {best_model_name} with R² of {results[best_model_name]['r2']:.2f}")
        
        # Save best model and preprocessor
        self.models['duration'] = results[best_model_name]['model']
        self.preprocessors['duration'] = preprocessor
        self.results['duration'] = results
        
        # Feature importance for random forest
        if 'rf' in results:
            # Get feature names
            feature_names = []
            # Get numerical feature names
            for name in numerical_features:
                feature_names.append(name)
            
            # Get one-hot encoded feature names
            for i, name in enumerate(categorical_features):
                categories = preprocessor.transformers_[1][1].categories_[i]
                for category in categories:
                    feature_names.append(f"{name}_{category}")
            
            # Add text feature names (simplified)
            for i in range(train_text.shape[1]):
                feature_names.append(f"text_feature_{i}")
            
            # Get feature importances
            importances = results['rf']['model'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Limit to top 20 features
            top_n = min(20, len(feature_names))
            
            self.feature_importances['duration'] = {
                'names': [feature_names[i] for i in indices[:top_n]],
                'scores': [importances[i] for i in indices[:top_n]]
            }
        
        return results
    
    def build_deadline_classification_model(self):
        """Build a model to predict if a task will meet its deadline"""
        df = self.processed_df.copy()
        
        # Define features and target
        X = df[['PROJECT NAME (ANONYMISED)', 'SPRINT_NUM', 'TASK_TYPE', 
                'STORY POINT', 'PRIORITY', 'DEADLINE (MINUTES)']]
        y = df['DEADLINE_MET']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get text features
        train_text, test_text = self.extract_text_features(df.loc[X_train.index], df.loc[X_test.index])
        
        # Define preprocessing for numerical features
        numerical_features = ['STORY POINT', 'DEADLINE (MINUTES)', 'SPRINT_NUM']
        numerical_transformer = StandardScaler()
        
        # Define preprocessing for categorical features
        categorical_features = ['PROJECT NAME (ANONYMISED)', 'TASK_TYPE', 'PRIORITY']
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Prepare train and test features
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Add text features
        if isinstance(X_train_processed, np.ndarray):
            X_train_processed = np.hstack((X_train_processed, train_text.toarray()))
            X_test_processed = np.hstack((X_test_processed, test_text.toarray()))
        else:
            # If sparse matrix
            from scipy.sparse import hstack
            X_train_processed = hstack([X_train_processed, train_text])
            X_test_processed = hstack([X_test_processed, test_text])
        
        # Train logistic regression model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'report': report
        }
        
        print(f"Deadline Classification Model")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)
        
        # Save model and preprocessor
        self.models['deadline'] = model
        self.preprocessors['deadline'] = preprocessor
        self.results['deadline'] = results
        
        return results
    
    def predict_task_duration(self, project, sprint, title, story_point, priority, deadline):
        """Predict the duration of a new task"""
        if 'duration' not in self.models:
            raise ValueError("Duration prediction model not trained yet. Call build_duration_prediction_model() first.")
        
        # Create a dataframe with the new task
        task = pd.DataFrame({
            'PROJECT NAME (ANONYMISED)': [project],
            'SPRINT_NUM': [int(sprint.replace('Sprint ', ''))],
            'TICKET TITLE': [title],
            'TASK_TYPE': [title.split(' - ')[0] if ' - ' in title else title.split(' ')[0]],
            'STORY POINT': [story_point],
            'PRIORITY': [priority],
            'DEADLINE (MINUTES)': [deadline]
        })
        
        # Preprocess the features
        preprocessor = self.preprocessors['duration']
        X = task[['PROJECT NAME (ANONYMISED)', 'SPRINT_NUM', 'TASK_TYPE', 'STORY POINT', 
                 'PRIORITY', 'DEADLINE (MINUTES)']]
        
        # Transform numerical and categorical features
        X_processed = preprocessor.transform(X)
        
        # Transform text features
        text_processed = self.tfidf_vectorizer.transform(task['TICKET TITLE'])
        
        # Combine features
        if isinstance(X_processed, np.ndarray):
            X_final = np.hstack((X_processed, text_processed.toarray()))
        else:
            # If sparse matrix
            from scipy.sparse import hstack
            X_final = hstack([X_processed, text_processed])
        
        # Make prediction
        predicted_duration = self.models['duration'].predict(X_final)[0]
        
        return predicted_duration
    
    def predict_deadline_met(self, project, sprint, title, story_point, priority, deadline):
        """Predict if a task will meet its deadline"""
        if 'deadline' not in self.models:
            raise ValueError("Deadline classification model not trained yet. Call build_deadline_classification_model() first.")
        
        # Create a dataframe with the new task
        task = pd.DataFrame({
            'PROJECT NAME (ANONYMISED)': [project],
            'SPRINT_NUM': [int(sprint.replace('Sprint ', ''))],
            'TICKET TITLE': [title],
            'TASK_TYPE': [title.split(' - ')[0] if ' - ' in title else title.split(' ')[0]],
            'STORY POINT': [story_point],
            'PRIORITY': [priority],
            'DEADLINE (MINUTES)': [deadline]
        })
        
        # Preprocess the features
        preprocessor = self.preprocessors['deadline']
        X = task[['PROJECT NAME (ANONYMISED)', 'SPRINT_NUM', 'TASK_TYPE', 'STORY POINT', 
                 'PRIORITY', 'DEADLINE (MINUTES)']]
        
        # Transform numerical and categorical features
        X_processed = preprocessor.transform(X)
        
        # Transform text features
        text_processed = self.tfidf_vectorizer.transform(task['TICKET TITLE'])
        
        # Combine features
        if isinstance(X_processed, np.ndarray):
            X_final = np.hstack((X_processed, text_processed.toarray()))
        else:
            # If sparse matrix
            from scipy.sparse import hstack
            X_final = hstack([X_processed, text_processed])
        
        # Make prediction
        deadline_met = self.models['deadline'].predict(X_final)[0]
        deadline_prob = self.models['deadline'].predict_proba(X_final)[0][1]  # Probability of meeting deadline
        
        return deadline_met, deadline_prob
    
    def visualize_duration_distribution(self):
        """Visualize the distribution of task durations"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.processed_df['DURATION (MINUTES)'], bins=30, kde=True, ax=ax)
        ax.set_title('Distribution of Task Durations')
        ax.set_xlabel('Duration (minutes)')
        ax.set_ylabel('Count')
        
        return fig
    
    def visualize_story_points_vs_duration(self):
        """Visualize the relationship between story points and duration"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='STORY POINT', y='DURATION (MINUTES)', data=self.processed_df, ax=ax)
        ax.set_title('Task Duration by Story Points')
        ax.set_xlabel('Story Points')
        ax.set_ylabel('Duration (minutes)')
        
        return fig
    
    def visualize_priority_vs_deadline_met(self):
        """Visualize the relationship between priority and meeting deadlines"""
        fig, ax = plt.subplots(figsize=(10, 6))
        priority_deadline = self.processed_df.groupby('PRIORITY')['DEADLINE_MET'].mean().reset_index()
        priority_deadline = priority_deadline.sort_values(by='PRIORITY_NUM', key=lambda x: x.map({
            'Highest': 4, 'High': 3, 'Medium': 2, 'Low': 1
        }))
        
        sns.barplot(x='PRIORITY', y='DEADLINE_MET', data=priority_deadline, ax=ax)
        ax.set_title('Proportion of Tasks Meeting Deadline by Priority')
        ax.set_xlabel('Priority')
        ax.set_ylabel('Proportion Meeting Deadline')
        
        return fig
    
    def visualize_feature_importance(self):
        """Visualize feature importances for the duration prediction model"""
        if 'duration' not in self.feature_importances:
            raise ValueError("Feature importances not available. Train Random Forest model first.")
        
        importances = self.feature_importances['duration']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(importances['names']))
        ax.barh(y_pos, importances['scores'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importances['names'])
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for Duration Prediction')
        
        return fig
    
    def save_models(self, directory='models'):
        """Save trained models and preprocessors to disk"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(directory, f"{model_name}_model.pkl"))
        
        for preprocessor_name, preprocessor in self.preprocessors.items():
            joblib.dump(preprocessor, os.path.join(directory, f"{preprocessor_name}_preprocessor.pkl"))
        
        # Save TF-IDF vectorizer
        if hasattr(self, 'tfidf_vectorizer'):
            joblib.dump(self.tfidf_vectorizer, os.path.join(directory, "tfidf_vectorizer.pkl"))
    
    def load_models(self, directory='models'):
        """Load trained models and preprocessors from disk"""
        for model_file in os.listdir(directory):
            if model_file.endswith('_model.pkl'):
                model_name = model_file.split('_')[0]
                self.models[model_name] = joblib.load(os.path.join(directory, model_file))
            
            if model_file.endswith('_preprocessor.pkl'):
                preprocessor_name = model_file.split('_')[0]
                self.preprocessors[preprocessor_name] = joblib.load(os.path.join(directory, model_file))
            
            if model_file == "tfidf_vectorizer.pkl":
                self.tfidf_vectorizer = joblib.load(os.path.join(directory, model_file))

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with data
    analyzer = SprintBacklogAnalyzer(data_path="Sample Agile Data for KIIT - v2.csv")
    
    # Preprocess data
    analyzer.preprocess_data()
    
    # Build and train models
    analyzer.build_duration_prediction_model()
    analyzer.build_deadline_classification_model()
    
    # Save models for later use
    analyzer.save_models()
    
    # Make some predictions
    predicted_duration = analyzer.predict_task_duration(
        project="ACME",
        sprint="Sprint 4",
        title="Frontend - Implement new login form",
        story_point=2,
        priority="High",
        deadline=120
    )
    
    deadline_met, deadline_prob = analyzer.predict_deadline_met(
        project="ACME",
        sprint="Sprint 4",
        title="Frontend - Implement new login form",
        story_point=2,
        priority="High",
        deadline=120
    )
    
    print(f"Predicted duration: {predicted_duration:.2f} minutes")
    print(f"Will meet deadline: {deadline_met} (Probability: {deadline_prob:.2f})")
    
    # Create visualizations
    analyzer.visualize_duration_distribution()
    analyzer.visualize_story_points_vs_duration()
    analyzer.visualize_priority_vs_deadline_met()
    
    try:
        analyzer.visualize_feature_importance()
    except ValueError as e:
        print(e)
    
    plt.show()