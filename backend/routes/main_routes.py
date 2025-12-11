from flask import render_template, request, redirect, url_for, session, flash
from backend.utils.helpers import login_required

def register_main_routes(app, recommender, db):
    @app.route('/')
    def index():
        programs = recommender.get_all_programs(limit=12)
        return render_template('index.html', programs=programs, logged_in='user_id' in session)
    
    @app.route('/programs')
    def programs():
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
