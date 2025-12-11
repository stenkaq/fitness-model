import sqlite3
import hashlib
from datetime import datetime

class FitnessDatabase:
    def __init__(self, db_path='fitness_app.db'):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                pref_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                goal TEXT,
                level TEXT,
                equipment TEXT,
                preferred_duration INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                program_index INTEGER NOT NULL,
                program_title TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                rating INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            password_hash = self.hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
    def verify_user(self, username, password):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute('''
            SELECT user_id, username, email FROM users
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {'user_id': user[0], 'username': user[1], 'email': user[2]}
        return None
    
    def update_user_preferences(self, user_id, goal=None, level=None, equipment=None, preferred_duration=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT pref_id FROM user_preferences WHERE user_id = ?', (user_id,))
        exists = cursor.fetchone()
        
        if exists:
            cursor.execute('''
                UPDATE user_preferences
                SET goal = COALESCE(?, goal),
                    level = COALESCE(?, level),
                    equipment = COALESCE(?, equipment),
                    preferred_duration = COALESCE(?, preferred_duration),
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (goal, level, equipment, preferred_duration, user_id))
        else:
            cursor.execute('''
                INSERT INTO user_preferences (user_id, goal, level, equipment, preferred_duration)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, goal, level, equipment, preferred_duration))
        
        conn.commit()
        conn.close()
    
    def get_user_preferences(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT goal, level, equipment, preferred_duration
            FROM user_preferences WHERE user_id = ?
        ''', (user_id,))
        
        prefs = cursor.fetchone()
        conn.close()
        
        if prefs:
            return {
                'goal': prefs[0],
                'level': prefs[1],
                'equipment': prefs[2],
                'preferred_duration': prefs[3]
            }
        return None
    
    def add_interaction(self, user_id, program_index, program_title, interaction_type, rating=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_interactions (user_id, program_index, program_title, interaction_type, rating)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, program_index, program_title, interaction_type, rating))
        
        conn.commit()
        conn.close()
    
    def get_user_liked_programs(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT program_index FROM user_interactions
            WHERE user_id = ? AND interaction_type IN ('like', 'complete')
            ORDER BY created_at DESC
        ''', (user_id,))
        
        programs = [row[0] for row in cursor.fetchall()]
        conn.close()
        return programs
    
    def get_user_interactions(self, user_id, limit=20):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT program_index, program_title, interaction_type, rating, created_at
            FROM user_interactions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        interactions = cursor.fetchall()
        conn.close()
        
        return [{
            'program_index': row[0],
            'program_title': row[1],
            'interaction_type': row[2],
            'rating': row[3],
            'created_at': row[4]
        } for row in interactions]
    
    def check_user_interaction(self, user_id, program_index, interaction_type):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT interaction_id FROM user_interactions
            WHERE user_id = ? AND program_index = ? AND interaction_type = ?
        ''', (user_id, program_index, interaction_type))
        
        exists = cursor.fetchone()
        conn.close()
        return exists is not None
    
    def remove_interaction(self, user_id, program_index, interaction_type):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM user_interactions
            WHERE user_id = ? AND program_index = ? AND interaction_type = ?
        ''', (user_id, program_index, interaction_type))
        
        conn.commit()
        deleted_count = cursor.rowcount
        conn.close()
        return deleted_count > 0
