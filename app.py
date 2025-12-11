from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from database import FitnessDatabase
from recommender import FitnessRecommender
from functools import wraps

app = Flask(__name__)
app.secret_key = 'very-secret-key'

db = FitnessDatabase('fitness_app.db')
recommender = FitnessRecommender(
    exercises_path='data/PROGRAM_EXERCISES.csv',
    summary_path='data/PROGRAM_SUMMARY.csv'
)

@app.template_filter('format_list')
def format_list(value):
    import ast
    if isinstance(value, str):
        if value.startswith('[') and value.endswith(']'):
            try:
                value = ast.literal_eval(value)
            except:
                pass
    if isinstance(value, list):
        return ', '.join([str(item).capitalize() for item in value])
    return str(value).capitalize()

@app.template_filter('parse_list')
def parse_list(value):
    import ast
    if isinstance(value, str):
        if value.startswith('[') and value.endswith(']'):
            try:
                return ast.literal_eval(value)
            except:
                pass
    if isinstance(value, list):
        return value
    return []

@app.template_filter('clean_number')
def clean_number(value):
    """Remove .0 from numbers that are whole numbers"""
    try:
        num = float(value)
        if num == int(num):
            return int(num)
        return num
    except (ValueError, TypeError):
        return value

@app.template_filter('format_description')
def format_description_filter(text):
    """Format program descriptions with proper HTML structure"""
    from backend.utils.helpers import format_description
    return format_description(text)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Home page"""
    programs = recommender.get_all_programs(limit=12)
    return render_template('index.html', programs=programs, logged_in='user_id' in session)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User signup"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_id = db.create_user(username, email, password)
        
        if user_id:
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username or email already exists', 'error')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = db.verify_user(username, password)
        
        if user:
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with personalized recommendations"""
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
    """User preferences page"""
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

@app.route('/programs')
def programs():
    """Browse all programs"""
    goal = request.args.get('goal', 'all')
    level = request.args.get('level', 'all')
    equipment = request.args.get('equipment', 'all')
    search = request.args.get('search', '')
    
    if search:
        programs_list = recommender.search_programs(search, top_n=50)
    else:
        programs_list = recommender.filter_programs(goal, level, equipment, limit=50)
    
    unique_values = recommender.get_unique_values()
    
    return render_template('programs.html',
                         programs=programs_list,
                         unique_values=unique_values,
                         current_filters={'goal': goal, 'level': level, 'equipment': equipment},
                         logged_in='user_id' in session)

@app.route('/program/<int:program_id>')
def program_detail(program_id):
    """Program detail page"""
    program = recommender.get_program_by_index(program_id)
    
    if not program:
        flash('Program not found', 'error')
        return redirect(url_for('programs'))

    exercises_data = recommender.get_program_exercises(program['title'])
    
    similar = recommender.recommend_similar(program_id, top_n=5)
    
    liked = False
    completed = False
    if 'user_id' in session:
        user_id = session['user_id']
        liked = db.check_user_interaction(user_id, program_id, 'like')
        completed = db.check_user_interaction(user_id, program_id, 'complete')
    
    return render_template('program_detail.html',
                         program=program,
                         exercises_data=exercises_data,
                         similar=similar,
                         liked=liked,
                         completed=completed,
                         logged_in='user_id' in session)

@app.route('/api/like/<int:program_id>', methods=['POST'])
@login_required
def like_program(program_id):
    """Toggle like/unlike a program"""
    user_id = session['user_id']
    program = recommender.get_program_by_index(program_id)
    
    if program:
        is_liked = db.check_user_interaction(user_id, program_id, 'like')
        
        if is_liked:
            db.remove_interaction(user_id, program_id, 'like')
            return jsonify({'success': True, 'message': 'Program unliked!', 'liked': False})
        else:
            db.add_interaction(user_id, program_id, program['title'], 'like')
            return jsonify({'success': True, 'message': 'Program liked!', 'liked': True})
    
    return jsonify({'success': False, 'message': 'Program not found'}), 404

@app.route('/api/complete/<int:program_id>', methods=['POST'])
@login_required
def complete_program(program_id):
    user_id = session['user_id']
    program = recommender.get_program_by_index(program_id)
    
    if program:
        is_completed = db.check_user_interaction(user_id, program_id, 'complete')
        
        if is_completed:
            db.remove_interaction(user_id, program_id, 'complete')
            return jsonify({'success': True, 'message': 'Program unmarked!', 'completed': False})
        else:
            db.add_interaction(user_id, program_id, program['title'], 'complete')
            return jsonify({'success': True, 'message': 'Program completed!', 'completed': True})
    
    return jsonify({'success': False, 'message': 'Program not found'}), 404

@app.route('/api/rate/<int:program_id>', methods=['POST'])
@login_required
def rate_program(program_id):
    """Rate a program"""
    user_id = session['user_id']
    rating = request.json.get('rating')
    program = recommender.get_program_by_index(program_id)
    
    if program and rating and 1 <= rating <= 5:
        db.add_interaction(user_id, program_id, program['title'], 'rate', rating)
        return jsonify({'success': True, 'message': f'Rated {rating} stars!'})
    
    return jsonify({'success': False, 'message': 'Invalid rating'}), 400

@app.route('/api/recommendations/<int:program_id>')
def get_recommendations(program_id):
    """Get recommendations for a program (API endpoint)"""
    recommendations = recommender.recommend_similar(program_id, top_n=5)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
