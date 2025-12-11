import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

class FitnessRecommender:
    def __init__(self, exercises_path, summary_path, model_cache='model_cache.pkl'):
        self.exercises_path = exercises_path
        self.summary_path = summary_path
        self.model_cache = model_cache
        
        self.prog_df = None
        self.exercises_df = None
        self.features_clean = None
        self.similarity_matrix = None
        self.kmeans = None
        self.text_model = None
        self.label_encoders = {}
        self.scaler = None
        
        if os.path.exists(model_cache):
            self.load_model()
        else:
            self.train_model()
            self.save_model()
    
    def clean_text(self, text):
        if pd.isna(text): 
            return ''
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.strip()
    
    def is_level_appropriate(self, program_level, user_level):
        level_hierarchy = {
            'beginner': 0,
            'intermediate': 1,
            'advanced': 2,
            'expert': 3
        }
        
        user_level_norm = user_level.lower().strip() if user_level else 'beginner'
        program_level_norm = str(program_level).lower().strip() if pd.notna(program_level) else 'beginner'
        
        user_rank = level_hierarchy.get(user_level_norm, 0)
        program_rank = level_hierarchy.get(program_level_norm, 0)
        
        return program_rank <= user_rank
    
    def train_model(self):
        print("Training fitness recommendation model...")
        
        ex_df = pd.read_csv(self.exercises_path)
        self.exercises_df = ex_df
        prog_df = pd.read_csv(self.summary_path)
        
        prog_df['goal'] = prog_df['goal'].fillna('unknown')
        prog_df['level'] = prog_df['level'].fillna('unknown')
        prog_df['equipment'] = prog_df['equipment'].fillna('unknown').str.strip()
        
        prog_df['program_length_original'] = prog_df['program_length']
        prog_df['time_per_workout_original'] = prog_df['time_per_workout']
        
        self.scaler = MinMaxScaler()
        prog_df[['program_length', 'time_per_workout']] = self.scaler.fit_transform(
            prog_df[['program_length', 'time_per_workout']]
        )
        
        prog_df['clean_title'] = prog_df['title'].apply(self.clean_text)
        prog_df['clean_desc'] = prog_df['description'].apply(self.clean_text)
        
        top_exercises = (ex_df.groupby('title')['exercise_name']
                        .apply(lambda x: ' '.join(x.dropna().astype(str).str.lower()
                                                 .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                                                 .unique()[:5])))
        
        prog_df = prog_df.merge(top_exercises.rename('common_exercises'), on='title', how='left')
        prog_df['common_exercises'] = prog_df['common_exercises'].fillna('')
        
        prog_df['text_input'] = (
            prog_df['clean_title'] + ' ' +
            prog_df['clean_desc'] + ' ' +
            prog_df['common_exercises']
        )
        
        for col in ['goal', 'level', 'equipment']:
            le = LabelEncoder()
            prog_df[col + '_enc'] = le.fit_transform(prog_df[col])
            self.label_encoders[col] = le
        
        print("Generating text embeddings...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', 
                                             device='cuda' if torch.cuda.is_available() else 'cpu')
        
        batch_texts = prog_df['text_input'].tolist()
        text_embeddings = self.text_model.encode(batch_texts, show_progress_bar=True, batch_size=128)
        
        embedding_df = pd.DataFrame(
            text_embeddings,
            columns=[f'emb_{i}' for i in range(text_embeddings.shape[1])],
            index=prog_df.index
        )
        prog_df = pd.concat([prog_df, embedding_df], axis=1)
        
        metadata_cols = ['goal_enc', 'level_enc', 'equipment_enc', 'program_length', 'time_per_workout']
        embedding_cols = [f'emb_{i}' for i in range(text_embeddings.shape[1])]
        all_features = metadata_cols + embedding_cols
        
        imputer = SimpleImputer(strategy='mean')
        self.features_clean = imputer.fit_transform(prog_df[all_features])
        
        print("Clustering programs...")
        NUM_CLUSTERS = 25
        self.kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
        prog_df['cluster_id'] = self.kmeans.fit_predict(self.features_clean)
        
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.features_clean)
        
        self.prog_df = prog_df
        print("Model training complete!")
    
    def save_model(self):
        cache_data = {
            'prog_df': self.prog_df,
            'exercises_df': self.exercises_df,
            'features_clean': self.features_clean,
            'similarity_matrix': self.similarity_matrix,
            'kmeans': self.kmeans,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }
        with open(self.model_cache, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Model saved to {self.model_cache}")
    
    def load_model(self):
        print(f"Loading model from {self.model_cache}...")
        with open(self.model_cache, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.prog_df = cache_data['prog_df']
        self.exercises_df = cache_data.get('exercises_df', None)
        self.features_clean = cache_data['features_clean']
        self.similarity_matrix = cache_data['similarity_matrix']
        self.kmeans = cache_data['kmeans']
        self.label_encoders = cache_data['label_encoders']
        self.scaler = cache_data['scaler']
        
        if self.exercises_df is None:
            self.exercises_df = pd.read_csv(self.exercises_path)
        
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2',
                                             device='cuda' if torch.cuda.is_available() else 'cpu')
        print("Model loaded successfully!")
    
    def get_all_programs(self, limit=None):
        cols = ['title', 'goal', 'level', 'equipment', 'program_length_original', 'time_per_workout_original', 'cluster_id']
        if 'total_exercises' in self.prog_df.columns:
            cols.append('total_exercises')
        
        df = self.prog_df[cols].copy()

        df['program_length'] = df['program_length_original']
        df['time_per_workout'] = df['time_per_workout_original']
        df = df.drop(['program_length_original', 'time_per_workout_original'], axis=1)
        df['program_index'] = df.index
        
        if limit:
            return df.head(limit).to_dict('records')
        return df.to_dict('records')
    
    def get_program_by_index(self, index):
        if index < 0 or index >= len(self.prog_df):
            return None
        
        program = self.prog_df.iloc[index]
        return {
            'program_index': index,
            'title': program['title'],
            'description': program.get('description', ''),
            'goal': program['goal'],
            'level': program['level'],
            'equipment': program['equipment'],
            'program_length': program.get('program_length_original', program['program_length']),
            'time_per_workout': program.get('time_per_workout_original', program['time_per_workout']),
            'total_exercises': program.get('total_exercises', 0),
            'created': program.get('created', ''),
            'last_edit': program.get('last_edit', ''),
            'cluster_id': program['cluster_id']
        }
    
    def search_programs(self, query, top_n=10):
        query_clean = self.clean_text(query)
        
        matches = self.prog_df[
            self.prog_df['text_input'].str.contains(query_clean, case=False, na=False)
        ]
        
        if len(matches) == 0:
            query_embedding = self.text_model.encode([query_clean])
            
            embedding_cols = [f'emb_{i}' for i in range(384)]
            program_embeddings = self.prog_df[embedding_cols].values
            
            similarities = cosine_similarity(query_embedding, program_embeddings).flatten()
            top_indices = similarities.argsort()[-top_n:][::-1]
            matches = self.prog_df.iloc[top_indices]
        
        results = []
        for idx, row in matches.head(top_n).iterrows():
            result = {
                'program_index': idx,
                'title': row['title'],
                'goal': row['goal'],
                'level': row['level'],
                'equipment': row['equipment'],
                'program_length': row.get('program_length_original', row['program_length']),
                'time_per_workout': row.get('time_per_workout_original', row['time_per_workout'])
            }
            if 'total_exercises' in row:
                result['total_exercises'] = row['total_exercises']
            results.append(result)
        
        return results
    
    def filter_programs(self, goal=None, level=None, equipment=None, limit=50):
        import ast
        filtered = self.prog_df.copy()
        
        if goal and goal != 'all':
            def check_goal(val):
                if pd.isna(val):
                    return False
                if isinstance(val, str) and val.startswith('['):
                    try:
                        goal_list = ast.literal_eval(val)
                        if isinstance(goal_list, list):
                            return any(goal.lower() in str(g).lower() for g in goal_list)
                    except:
                        pass
                return goal.lower() in str(val).lower()
            
            filtered = filtered[filtered['goal'].apply(check_goal)]
        
        if level and level != 'all':
            def check_level(val):
                if pd.isna(val):
                    return False
                if isinstance(val, str) and val.startswith('['):
                    try:
                        level_list = ast.literal_eval(val)
                        if isinstance(level_list, list):
                            return any(level.lower() in str(l).lower() for l in level_list)
                    except:
                        pass
                return level.lower() in str(val).lower()
            
            filtered = filtered[filtered['level'].apply(check_level)]
        
        if equipment and equipment != 'all':
            filtered = filtered[filtered['equipment'].str.lower() == equipment.lower()]
        
        results = []
        for idx, row in filtered.head(limit).iterrows():
            result = {
                'program_index': idx,
                'title': row['title'],
                'goal': row['goal'],
                'level': row['level'],
                'equipment': row['equipment'],
                'program_length': row.get('program_length_original', row['program_length']),
                'time_per_workout': row.get('time_per_workout_original', row['time_per_workout'])
            }
            if 'total_exercises' in row:
                result['total_exercises'] = row['total_exercises']
            results.append(result)
        
        return results
    
    def filter_by_preferences(self, level=None, equipment=None, goal=None, limit=10):
        import ast
        filtered = self.prog_df.copy()
        
        if level:
            def check_level(val):
                if pd.isna(val):
                    return False
                if isinstance(val, str) and val.startswith('['):
                    try:
                        level_list = ast.literal_eval(val)
                        if isinstance(level_list, list):
                            return any(self.is_level_appropriate(str(l), level) for l in level_list)
                    except:
                        pass
                return self.is_level_appropriate(str(val), level)
            
            filtered = filtered[filtered['level'].apply(check_level)]
        
        if equipment:
            filtered = filtered[filtered['equipment'].str.lower() == equipment.lower()]
        
        if goal:
            def check_goal(val):
                if pd.isna(val):
                    return False
                if isinstance(val, str) and val.startswith('['):
                    try:
                        goal_list = ast.literal_eval(val)
                        if isinstance(goal_list, list):
                            if isinstance(goal, str) and goal.startswith('['):
                                try:
                                    pref_goals = ast.literal_eval(goal)
                                    return any(pg.lower() in str(g).lower() for pg in pref_goals for g in goal_list)
                                except:
                                    pass
                            return any(goal.lower() in str(g).lower() for g in goal_list)
                    except:
                        pass
                return goal.lower() in str(val).lower()
            
            filtered = filtered[filtered['goal'].apply(check_goal)]
        
        results = []
        for idx, row in filtered.head(limit).iterrows():
            result = {
                'program_index': idx,
                'title': row['title'],
                'goal': row['goal'],
                'level': row['level'],
                'equipment': row['equipment'],
                'program_length': row.get('program_length_original', row['program_length']),
                'time_per_workout': row.get('time_per_workout_original', row['time_per_workout'])
            }
            if 'total_exercises' in row:
                result['total_exercises'] = row['total_exercises']
            results.append(result)
        
        return results
    
    def recommend_similar(self, program_index, top_n=5):
        if program_index < 0 or program_index >= len(self.prog_df):
            return []
        
        sim_scores = self.similarity_matrix[program_index]
        indices = np.argsort(sim_scores)[::-1]
        indices = indices[indices != program_index]
        top_indices = indices[:top_n]
        
        recommendations = []
        for idx in top_indices:
            program = self.prog_df.iloc[idx]
            recommendations.append({
                'program_index': int(idx),
                'title': program['title'],
                'goal': program['goal'],
                'level': program['level'],
                'equipment': program['equipment'],
                'similarity_score': float(sim_scores[idx])
            })
        
        return recommendations
    
    def recommend_personalized(self, liked_program_indices, top_n=5):
        if not liked_program_indices:
            return []
        
        liked_indices = [i for i in liked_program_indices if 0 <= i < len(self.prog_df)]
        if not liked_indices:
            return []
        
        user_vector = self.features_clean[liked_indices].mean(axis=0).reshape(1, -1)
        
        user_sim = cosine_similarity(user_vector, self.features_clean).flatten()
        
        excluded = set(liked_indices)
        top_indices = [i for i in user_sim.argsort()[::-1] if i not in excluded][:top_n]
        
        recommendations = []
        for idx in top_indices:
            program = self.prog_df.iloc[idx]
            recommendations.append({
                'program_index': int(idx),
                'title': program['title'],
                'goal': program['goal'],
                'level': program['level'],
                'equipment': program['equipment'],
                'match_score': float(user_sim[idx])
            })
        
        return recommendations
    
    def recommend_within_cluster(self, program_index, top_n=5):
        if program_index < 0 or program_index >= len(self.prog_df):
            return []
        
        user_cluster = self.prog_df.iloc[program_index]['cluster_id']
        cluster_mask = (self.prog_df['cluster_id'] == user_cluster).values
        
        sim_scores = self.similarity_matrix[program_index] * cluster_mask
        indices = np.argsort(sim_scores)[::-1]
        indices = indices[indices != program_index]
        top_indices = indices[:top_n]
        
        recommendations = []
        for idx in top_indices:
            program = self.prog_df.iloc[idx]
            recommendations.append({
                'program_index': int(idx),
                'title': program['title'],
                'goal': program['goal'],
                'level': program['level'],
                'equipment': program['equipment'],
                'cluster_id': int(user_cluster)
            })
        
        return recommendations
    
    def get_program_exercises(self, program_title):
        if self.exercises_df is None:
            return None
        
        program_exercises = self.exercises_df[self.exercises_df['title'] == program_title].copy()
        
        if len(program_exercises) == 0:
            return None
        
        exercises_by_week = {}
        
        for _, row in program_exercises.iterrows():
            week = row.get('week', 1)
            day = row.get('day', 1)
            
            try:
                week = int(float(week)) if pd.notna(week) else 1
                day = int(float(day)) if pd.notna(day) else 1
            except:
                week = 1
                day = 1
            
            if week not in exercises_by_week:
                exercises_by_week[week] = {}
            
            if day not in exercises_by_week[week]:
                exercises_by_week[week][day] = []
            
            exercise_info = {
                'exercise_name': row.get('exercise_name', 'Unknown'),
                'sets': row.get('sets', ''),
                'reps': row.get('reps', ''),
                'intensity': row.get('intensity', ''),
                'number_of_exercises': row.get('number_of_exercises', '')
            }
            
            exercises_by_week[week][day].append(exercise_info)
        
        return {
            'exercises_by_week': exercises_by_week,
            'total_weeks': max(exercises_by_week.keys()) if exercises_by_week else 0,
            'program_info': {
                'title': program_exercises.iloc[0].get('title', ''),
                'description': program_exercises.iloc[0].get('description', ''),
                'level': program_exercises.iloc[0].get('level', ''),
                'goal': program_exercises.iloc[0].get('goal', ''),
                'equipment': program_exercises.iloc[0].get('equipment', ''),
                'program_length': program_exercises.iloc[0].get('program_length', ''),
                'time_per_workout': program_exercises.iloc[0].get('time_per_workout', ''),
                'created': program_exercises.iloc[0].get('created', ''),
                'last_edit': program_exercises.iloc[0].get('last_edit', '')
            }
        }
    
    def get_unique_values(self):
        import ast
        
        all_goals = set()
        all_levels = set()
        all_equipment = set()
        
        for goal in self.prog_df['goal'].dropna():
            if isinstance(goal, str) and goal.startswith('['):
                try:
                    goals_list = ast.literal_eval(goal)
                    if isinstance(goals_list, list):
                        all_goals.update(goals_list)
                    else:
                        all_goals.add(str(goal))
                except:
                    all_goals.add(str(goal))
            else:
                all_goals.add(str(goal))
        
        for level in self.prog_df['level'].dropna():
            if isinstance(level, str) and level.startswith('['):
                try:
                    levels_list = ast.literal_eval(level)
                    if isinstance(levels_list, list):
                        all_levels.update(levels_list)
                    else:
                        all_levels.add(str(level))
                except:
                    all_levels.add(str(level))
            else:
                all_levels.add(str(level))
        
        for equip in self.prog_df['equipment'].dropna():
            all_equipment.add(str(equip).strip())
        
        all_goals.discard('unknown')
        all_levels.discard('unknown')
        all_equipment.discard('unknown')
        
        return {
            'goals': sorted(list(all_goals)),
            'levels': sorted(list(all_levels)),
            'equipment': sorted(list(all_equipment))
        }
