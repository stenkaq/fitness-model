from flask import render_template, request, redirect, url_for, session, flash
from backend.utils.helpers import login_required

def register_auth_routes(app, db):
    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
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
        session.clear()
        flash('You have been logged out', 'info')
        return redirect(url_for('index'))
