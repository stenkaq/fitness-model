from flask import Flask
from backend.models.database import FitnessDatabase
from backend.utils.recommender import FitnessRecommender
from backend.routes.auth_routes import register_auth_routes
from backend.routes.user_routes import register_user_routes
from backend.routes.main_routes import register_main_routes
from backend.routes.api_routes import register_api_routes
from backend.utils.helpers import format_list, parse_list, clean_number, format_description

app = Flask(__name__)
app.secret_key = 'very-secret-key'

db = FitnessDatabase('data/fitness_app.db')
recommender = FitnessRecommender(
    exercises_path='data/PROGRAM_EXERCISES.csv',
    summary_path='data/PROGRAM_SUMMARY.csv',
    model_cache='data/model_cache.pkl'
)

app.jinja_env.filters['format_list'] = format_list
app.jinja_env.filters['parse_list'] = parse_list
app.jinja_env.filters['clean_number'] = clean_number
app.jinja_env.filters['format_description'] = format_description

register_auth_routes(app, db)
register_user_routes(app, db, recommender)
register_main_routes(app, recommender, db)
register_api_routes(app, db, recommender)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
