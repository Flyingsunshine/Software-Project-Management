$(document).ready(function() {
    // File upload form submission
    $('#uploadForm').submit(function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                if (response.success) {
                    $('#uploadResult').html(`
                        <div class="alert alert-success">
                            ${response.message}<br>
                            Total tickets: ${response.stats.total_tickets}<br>
                            Projects: ${response.stats.projects}<br>
                            Sprints: ${response.stats.sprints}<br>
                            Avg Duration: ${response.stats.avg_duration.toFixed(2)} minutes<br>
                            Avg Story Points: ${response.stats.avg_story_points.toFixed(2)}<br>
                            Deadline Met Rate: ${(response.stats.deadline_met_rate * 100).toFixed(2)}%
                        </div>
                    `);
                } else {
                    $('#uploadResult').html(`<div class="alert alert-danger">${response.error}</div>`);
                }
            },
            error: function() {
                $('#uploadResult').html('<div class="alert alert-danger">Server error occurred</div>');
            }
        });
    });

    // Train models button click
    $('#trainButton').click(function() {
        $.ajax({
            url: '/train',
            type: 'POST',
            success: function(response) {
                if (response.success) {
                    $('#trainingResult').html(`
                        <div class="alert alert-success">
                            Models trained successfully<br>
                            Duration Model (${response.duration_metrics.best_model}):<br>
                            R² Score: ${response.duration_metrics.r2.toFixed(4)}<br>
                            RMSE: ${response.duration_metrics.rmse.toFixed(2)} minutes<br>
                            Deadline Classification Model:<br>
                            Accuracy: ${(response.deadline_metrics.accuracy * 100).toFixed(2)}%
                        </div>
                    `);
                } else {
                    $('#trainingResult').html(`<div class="alert alert-danger">${response.error}</div>`);
                }
            },
            error: function() {
                $('#trainingResult').html('<div class="alert alert-danger">Server error occurred</div>');
            }
        });
    });

    // Prediction form submission
    $('#predictionForm').submit(function(e) {
        e.preventDefault();
        var formData = {
            project: $('#project').val(),
            sprint: $('#sprint').val(),
            title: $('#title').val(),
            story_point: $('#storyPoint').val(),
            priority: $('#priority').val(),
            deadline: $('#deadline').val()
        };
        
        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(formData),
            success: function(response) {
                if (response.success) {
                    $('#predictionResult').html(`
                        <div class="alert alert-info">
                            Predicted Duration: ${response.predicted_duration.toFixed(2)} minutes<br>
                            Deadline Met: ${response.deadline_met ? 'Yes' : 'No'}<br>
                            Deadline Probability: ${(response.deadline_probability * 100).toFixed(2)}%
                        </div>
                    `);
                } else {
                    $('#predictionResult').html(`<div class="alert alert-danger">${response.error}</div>`);
                }
            },
            error: function() {
                $('#predictionResult').html('<div class="alert alert-danger">Server error occurred</div>');
            }
        });
    });

    // Visualization buttons
    $('.btn-secondary').click(function() {
        var vizType = $(this).attr('id').replace('Button', '');
        $.ajax({
            url: '/visualize/' + vizType,
            type: 'GET',
            success: function(response) {
                if (response.success) {
                    $('#visualizationResult').html(`<img src="data:image/png;base64,${response.image}" class="img-fluid">`);
                } else {
                    $('#visualizationResult').html(`<div class="alert alert-danger">${response.error}</div>`);
                }
            },
            error: function() {
                $('#visualizationResult').html('<div class="alert alert-danger">Server error occurred</div>');
            }
        });
    });

    // Backlog upload form submission
    $('#backlogUploadForm').submit(function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        
        $.ajax({
            url: '/upload_backlog',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                if (response.success) {
                    $('#backlogUploadResult').html(`
                        <div class="alert alert-success">
                            ${response.message}<br>
                            Number of tickets: ${response.ticket_count}
                        </div>
                    `);
                    $('#prioritizationSection').show();
                } else {
                    $('#backlogUploadResult').html(`<div class="alert alert-danger">${response.error}</div>`);
                }
            },
            error: function() {
                $('#backlogUploadResult').html('<div class="alert alert-danger">Server error occurred</div>');
            }
        });
    });

    // Prioritize backlog button
    $('#prioritizeButton').click(function() {
        $.ajax({
            url: '/prioritize_backlog',
            type: 'POST',
            success: function(response) {
                if (response.success) {
                    var summary = response.summary;
                    $('#prioritizationResult').html(`
                        <div class="alert alert-success">
                            ${response.message}<br>
                            Total tickets: ${summary.total_tickets}<br>
                            Predicted to meet deadline: ${summary.predicted_to_meet_deadline} 
                            (${(summary.predicted_to_meet_deadline/summary.total_tickets*100).toFixed(1)}%)<br>
                            Average predicted duration: ${summary.avg_duration.toFixed(2)} minutes<br>
                            Average deadline probability: ${(summary.avg_deadline_probability*100).toFixed(1)}%
                        </div>
                    `);
                    $('#downloadLink').show();
                } else {
                    $('#prioritizationResult').html(`<div class="alert alert-danger">${response.error}</div>`);
                }
            },
            error: function() {
                $('#prioritizationResult').html('<div class="alert alert-danger">Server error occurred</div>');
            }
        });
    });
});
