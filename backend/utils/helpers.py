from flask import session, redirect, url_for, flash
from functools import wraps
import ast

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def format_list(value):
    if isinstance(value, str):
        if value.startswith('[') and value.endswith(']'):
            try:
                value = ast.literal_eval(value)
            except:
                pass
    if isinstance(value, list):
        return ', '.join([str(item).capitalize() for item in value])
    return str(value).capitalize()

def parse_list(value):
    if isinstance(value, str):
        if value.startswith('[') and value.endswith(']'):
            try:
                return ast.literal_eval(value)
            except:
                pass
    if isinstance(value, list):
        return value
    return []

def clean_number(value):
    try:
        num = float(value)
        if num == int(num):
            return int(num)
        return num
    except (ValueError, TypeError):
        return value

def format_description(text):
    if not text:
        return ""
    
    lines = text.split('\n')
    formatted_html = []
    current_list = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                formatted_html.append('<ul class="formatted-list">')
                formatted_html.extend(current_list)
                formatted_html.append('</ul>')
                current_list = []
                in_list = False
            continue
        
        if line.endswith(':') and len(line) < 80:
            if in_list:
                formatted_html.append('<ul class="formatted-list">')
                formatted_html.extend(current_list)
                formatted_html.append('</ul>')
                current_list = []
                in_list = False
            
            formatted_html.append(f'<h5 class="section-header">{line}</h5>')
        
        elif line[0].isdigit() and len(line) > 2 and line[1] in ['.', ')']:
            if not in_list:
                in_list = True
            current_list.append(f'<li>{line[2:].strip()}</li>')
        
        elif line.startswith(('-', '*', '•', '[')):
            if not in_list:
                in_list = True
            content = line[1:].strip()
            if line.startswith('['):
                content = line
            current_list.append(f'<li>{content}</li>')
        
        else:
            if in_list:
                formatted_html.append('<ul class="formatted-list">')
                formatted_html.extend(current_list)
                formatted_html.append('</ul>')
                current_list = []
                in_list = False
            
            formatted_html.append(f'<p class="formatted-paragraph">{line}</p>')
    
    if in_list:
        formatted_html.append('<ul class="formatted-list">')
        formatted_html.extend(current_list)
        formatted_html.append('</ul>')
    
    return '\n'.join(formatted_html)
