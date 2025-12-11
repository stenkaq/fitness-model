from flask import render_template, request, redirect, url_for, session, flash
from backend.utils.helpers import login_required

def register_user_routes(app, db, recommender):
    @app.route('/dashboard')
    @login_required
    def dashboard():
        user_id = session['user_id']
        
        liked_indices = db.get_user_liked_programs(user_id)
        prefs = db.get_user_preferences(user_id)
        
        if prefs and (prefs.get('level') or prefs.get('equipment')):
            recommendations = recommender.filter_by_preferences(
                level=prefs.get('level'),
                equipment=prefs.get('equipment'),
                goal=prefs.get('goal'),
                limit=10
            )
        elif liked_indices:
            recommendations = recommender.recommend_personalized(liked_indices, top_n=10)
        else:
            recommendations = recommender.get_all_programs(limit=10)
        
        interactions = db.get_user_interactions(user_id, limit=10)
        
        return render_template('dashboard.html', 
                             recommendations=recommendations,
                             interactions=interactions,
                             liked_count=len(liked_indices),
                             logged_in=True)
    
    @app.route('/preferences', methods=['GET', 'POST'])
    @login_required
    def preferences():
        user_id = session['user_id']
        
        if request.method == 'POST':
            goals = request.form.getlist('goal')
            goal = str(goals) if goals else None
            level = request.form.get('level')
            equipment = request.form.get('equipment')
            duration = request.form.get('preferred_duration')
            
            db.update_user_preferences(user_id, goal, level, equipment, duration)
            flash('Preferences updated successfully!', 'success')
            return redirect(url_for('dashboard'))
        
        prefs = db.get_user_preferences(user_id)
        unique_values = recommender.get_unique_values()
        
        return render_template('preferences.html', 
                             preferences=prefs,
                             unique_values=unique_values,
                             logged_in=True)
