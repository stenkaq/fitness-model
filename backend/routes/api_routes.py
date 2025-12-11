from flask import session, jsonify, request
from backend.utils.helpers import login_required

def register_api_routes(app, db, recommender):
    @app.route('/api/like/<int:program_id>', methods=['POST'])
    @login_required
    def like_program(program_id):
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
        user_id = session['user_id']
        rating = request.json.get('rating')
        program = recommender.get_program_by_index(program_id)
        
        if program and rating and 1 <= rating <= 5:
            db.add_interaction(user_id, program_id, program['title'], 'rate', rating)
            return jsonify({'success': True, 'message': f'Rated {rating} stars!'})
        
        return jsonify({'success': False, 'message': 'Invalid rating'}), 400
    
    @app.route('/api/recommendations/<int:program_id>')
    def get_recommendations(program_id):
        recommendations = recommender.recommend_similar(program_id, top_n=5)
        return jsonify(recommendations)
