from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from ML_minor import SprintBacklogAnalyzer

app = Flask(__name__, static_folder='static')


# Initialize analyzer
analyzer = None

@app.route('/upload_backlog', methods=['POST'])
def upload_backlog():
    global analyzer
    
    if analyzer is None:
        return jsonify({'error': 'Please upload and process historical data first'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save file with its original name
        file_path = os.path.join('data', 'backlog_' + file.filename)
        file.save(file_path)
        
        try:
            # Process the backlog
            backlog_df = analyzer.process_pending_backlog(file_path)
            
            return jsonify({
                'success': True,
                'message': 'Backlog file uploaded successfully',
                'ticket_count': len(backlog_df)
            })
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/prioritize_backlog', methods=['POST'])
def prioritize_backlog():
    global analyzer
    
    if analyzer is None:
        return jsonify({'error': 'No data uploaded yet'})
    
    if not hasattr(analyzer, 'pending_backlog'):
        return jsonify({'error': 'No backlog uploaded yet'})
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Predict durations for backlog items
        analyzer.predict_backlog_durations()
        
        # Prioritize the backlog
        prioritized = analyzer.prioritize_backlog()
        
        # Generate CSV - use absolute path
        output_path = os.path.abspath(os.path.join('data', 'prioritized_backlog.csv'))
        analyzer.generate_prioritized_csv(output_path)
        
        # Verify file was created
        file_exists = os.path.exists(output_path)
        
        # Get summary statistics - convert NumPy types to Python native types
        summary = {
            'total_tickets': int(len(prioritized)),
            'predicted_to_meet_deadline': int(prioritized['deadline_met'].sum()),
            'avg_duration': float(prioritized['predicted_duration'].mean()),
            'avg_deadline_probability': float(prioritized['deadline_probability'].mean()),
            'file_created': file_exists  # For debugging
        }
        
        return jsonify({
            'success': True,
            'message': 'Backlog prioritized successfully',
            'summary': summary,
            'download_path': output_path
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_prioritized')
def download_prioritized():
    file_path = os.path.abspath(os.path.join('data', 'prioritized_backlog.csv'))
    if os.path.exists(file_path):
        return send_file(file_path, 
                         mimetype='text/csv',
                         download_name='prioritized_backlog.csv',
                         as_attachment=True)
    else:
        return jsonify({'error': 'Prioritized backlog file not found'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global analyzer
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save file with its original name
        file_path = os.path.join('data', file.filename)
        file.save(file_path)
        
        # Initialize analyzer with data
        analyzer = SprintBacklogAnalyzer(data_path=file_path)
        
        # Preprocess data
        analyzer.preprocess_data()
        
        # Get basic statistics
        stats = {
            'total_tickets': len(analyzer.processed_df),
            'projects': analyzer.processed_df['PROJECT NAME (ANONYMISED)'].nunique(),
            'sprints': analyzer.processed_df['SPRINT NAME'].nunique(),
            'avg_duration': analyzer.processed_df['DURATION (MINUTES)'].mean(),
            'avg_story_points': analyzer.processed_df['STORY POINT'].mean(),
            'deadline_met_rate': analyzer.processed_df['DEADLINE_MET'].mean() if 'DEADLINE_MET' in analyzer.processed_df.columns else None
        }
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'stats': stats
        })
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/train', methods=['POST'])
def train_models():
    global analyzer
    
    if analyzer is None:
        return jsonify({'error': 'No data uploaded yet'})
    
    try:
        # Train models
        duration_results = analyzer.build_duration_prediction_model()
        deadline_results = analyzer.build_deadline_classification_model()
        
        # Save models
        if not os.path.exists('models'):
            os.makedirs('models')
        analyzer.save_models()
        
        # Prepare results summary
        duration_metrics = {
            'best_model': max(duration_results, key=lambda x: duration_results[x]['r2']),
            'r2': duration_results[max(duration_results, key=lambda x: duration_results[x]['r2'])]['r2'],
            'rmse': duration_results[max(duration_results, key=lambda x: duration_results[x]['r2'])]['rmse']
        }
        
        deadline_metrics = {
            'accuracy': deadline_results['accuracy']
        }
        
        return jsonify({
            'success': True,
            'duration_metrics': duration_metrics,
            'deadline_metrics': deadline_metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    global analyzer
    
    if analyzer is None:
        return jsonify({'error': 'No models trained yet'})
    
    try:
        data = request.json
        
        # Extract input parameters
        project = data.get('project')
        sprint = data.get('sprint')
        title = data.get('title')
        story_point = float(data.get('story_point'))
        priority = data.get('priority')
        deadline = float(data.get('deadline'))
        
        # Make predictions
        predicted_duration = analyzer.predict_task_duration(
            project=project,
            sprint=sprint,
            title=title,
            story_point=story_point,
            priority=priority,
            deadline=deadline
        )
        
        deadline_met, deadline_prob = analyzer.predict_deadline_met(
            project=project,
            sprint=sprint,
            title=title,
            story_point=story_point,
            priority=priority,
            deadline=deadline
        )
        
        return jsonify({
            'success': True,
            'predicted_duration': float(predicted_duration),
            'deadline_met': bool(deadline_met),
            'deadline_probability': float(deadline_prob)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/visualize/<viz_type>')
def visualize(viz_type):
    global analyzer
    
    if analyzer is None:
        return jsonify({'error': 'No data uploaded yet'})
    
    try:
        # Create figure based on visualization type
        if viz_type == 'duration_distribution':
            fig = analyzer.visualize_duration_distribution()
        elif viz_type == 'story_points_vs_duration':
            fig = analyzer.visualize_story_points_vs_duration()
        elif viz_type == 'priority_vs_deadline':
            fig = analyzer.visualize_priority_vs_deadline_met()
        elif viz_type == 'feature_importance' and hasattr(analyzer, 'feature_importances') and 'duration' in analyzer.feature_importances:
            fig = analyzer.visualize_feature_importance()
        else:
            return jsonify({'error': 'Invalid visualization type or data not available'})
        
        # Convert plot to base64 encoded image
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': plot_url
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)