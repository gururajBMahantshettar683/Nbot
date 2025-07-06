from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_bcrypt import Bcrypt
from flask_session import Session
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timedelta, date
import os
import pandas as pd
import re
from dotenv import load_dotenv
from difflib import get_close_matches
from rag_pipeline import answer_query
import logging
from werkzeug.exceptions import HTTPException
from bson.errors import InvalidId

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
bcrypt = Bcrypt(app)

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_DB", "mongodb://localhost:27017"), serverSelectionTimeoutMS=5000)
db = client['nutrition_app']
users_collection = db['users']
intake_logs_collection = db['intake_logs']
symptom_logs_collection = db['symptom_logs']
foods_collection = db['foods']
chats_collection = db['chats']
goals_collection = db['goals']

# Load and normalize dataset
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anuvaad_INDB_2024.11_converted.csv')
COMMON_FOODS = []
df = pd.DataFrame()
try:
    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset file '{CSV_PATH}' not found in {os.getcwd()}")
    else:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
        if 'food_name' not in df.columns:
            print("Error: 'food_name' column not found in dataset")
        else:
            df['food_name'] = df['food_name'].astype(str).str.strip().str.lower()
            df = df[df['food_name'].notna() & (df['food_name'] != '')]
            COMMON_FOODS = sorted(df['food_name'].unique().tolist())
            print(f"Loaded {len(COMMON_FOODS)} normalized food names")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Cache dataset
app.config['FOOD_DF'] = df.copy()

NUTRIENT_COLUMNS = [
    'energy_kcal', 'carb_g', 'protein_g', 'fat_g', 'calcium_mg', 'iron_mg', 'vitc_mg'
]

# Set up logging
logging.basicConfig(level=logging.INFO)

# Global error handler for JSON errors
def make_json_error(message, code=500):
    response = jsonify({'error': message, 'code': code})
    response.status_code = code
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, HTTPException):
        return make_json_error(e.description, e.code)
    # Log and return JSON for all other errors
    logging.exception("Unhandled Exception: %s", e)
    return make_json_error(str(e), 500)

@app.route('/')
def home():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    try:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        if user is None:
            session.pop('user_id', None)
            session.pop('username', None)
            flash('User not found. Please log in again.', 'danger')
            return redirect(url_for('login'))
        today = datetime.utcnow().strftime('%Y-%m-%d')
        return render_template('home.html', 
                             email=user['email'], 
                             username=user.get('username', user['email']), 
                             dob=user.get('dob'), 
                             gender=user.get('gender'), 
                             common_foods=COMMON_FOODS,
                             foods_available=len(COMMON_FOODS) > 0, 
                             today=today)
    except Exception as e:
        session.pop('user_id', None)
        session.pop('username', None)
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    try:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        if user is None:
            session.pop('user_id', None)
            session.pop('username', None)
            flash('User not found. Please log in again.', 'danger')
            return redirect(url_for('login'))
        
        if request.method == 'POST':
            dob = request.form.get('dob')
            gender = request.form.get('gender')
            username = request.form.get('username', '').strip()
            
            # Validate inputs
            errors = []
            if not username:
                errors.append('Username is required.')
            elif not re.match(r'^[a-zA-Z0-9]{3,20}$', username):
                errors.append('Username must be 3-20 alphanumeric characters.')
            elif username != user['username'] and users_collection.find_one({'username': username}):
                errors.append('Username already taken.')
            if dob:
                try:
                    dob_date = datetime.strptime(dob, '%Y-%m-%d')
                    today = datetime.utcnow()
                    if dob_date > today:
                        errors.append('Date of birth cannot be in the future.')
                except ValueError:
                    errors.append('Invalid date format.')
            if gender not in ['male', 'female', 'other', '']:
                errors.append('Invalid gender selection.')
            
            if errors:
                for error in errors:
                    flash(error, 'danger')
            else:
                update_data = {'username': username}
                if dob:
                    update_data['dob'] = dob
                if gender:
                    update_data['gender'] = gender
                if update_data:
                    users_collection.update_one(
                        {'_id': ObjectId(session['user_id'])},
                        {'$set': update_data}
                    )
                    session['username'] = username  # Update session
                    flash('Profile updated successfully!', 'success')
                return redirect(url_for('profile'))
        
        today = datetime.utcnow().strftime('%Y-%m-%d')
        return render_template('profile.html', 
                             email=user['email'], 
                             username=user.get('username', user['email']), 
                             dob=user.get('dob', ''),
                             gender=user.get('gender', ''),
                             today=today)
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(url_for('login'))

@app.route('/dashboard_data', methods=['GET'])
def dashboard_data():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    time_frame = request.args.get('time_frame', 'daily')
    if time_frame not in ['daily', 'weekly', 'monthly']:
        return jsonify({'error': 'Invalid time frame'}), 400

    user_id = ObjectId(session['user_id'])
    now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    date_param = request.args.get('date')
    days_param = request.args.get('days', '7')

    try:
        days = int(days_param)
        if days not in [7, 14, 30]:
            days = 7
    except ValueError:
        days = 7

    if time_frame == 'daily':
        if date_param:
            try:
                selected_date = datetime.strptime(date_param, '%Y-%m-%d')
                start_date = selected_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date + timedelta(days=1)
                labels = [start_date.strftime('%Y-%m-%d')]
            except ValueError:
                return jsonify({'error': 'Invalid date format'}), 400
        else:
            end_date = now
            start_date = end_date - timedelta(days=days)
            labels = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        group_by = {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date'}}
    elif time_frame == 'weekly':
        start_date = now - timedelta(days=now.weekday() + 28)
        end_date = now + timedelta(days=1)
        group_by = {'$week': '$date'}
        labels = [f"Week {i+1}" for i in range(4)]
    else:  # monthly
        start_date = (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=365)).replace(day=1)
        end_date = now + timedelta(days=1)
        group_by = {'$dateToString': {'format': '%Y-%m', 'date': '$date'}}
        labels = [(now - timedelta(days=30*i)).strftime('%Y-%m') for i in range(12, -1, -1)]

    pipeline = [
        {'$match': {
            'user_id': user_id,
            'date': {'$gte': start_date, '$lt': end_date}
        }},
        {'$group': {
            '_id': group_by,
            **{nutrient: {'$sum': f'$nutrients.{nutrient}'} for nutrient in NUTRIENT_COLUMNS}
        }},
        {'$sort': {'_id': 1}}
    ]

    try:
        data = list(intake_logs_collection.aggregate(pipeline))
        result = {nutrient: [0] * len(labels) for nutrient in NUTRIENT_COLUMNS}
        for entry in data:
            if time_frame == 'weekly':
                label = f"Week {int(entry['_id']) % 4 + 1}"
            else:
                label = entry['_id']
            if label in labels:
                idx = labels.index(label)
                for nutrient in NUTRIENT_COLUMNS:
                    result[nutrient][idx] = entry.get(nutrient, 0)
        return jsonify({'labels': labels, 'data': result})
    except Exception as e:
        print(f"Error fetching dashboard data: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        data = request.get_json()
        query = data.get('message', '')
        user_id = session.get('user_id')

        if not query or not user_id:
            return jsonify({'error': 'Invalid request'}), 400

        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Conversation history management
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        # Append current user message
        session['conversation_history'].append({'role': 'user', 'content': query})
        # Keep only the last 10 turns (user+bot)
        session['conversation_history'] = session['conversation_history'][-10:]

        # Get recent food logs
        recent_food_logs = intake_logs_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('date', -1).limit(5)

        user['recent_foods'] = [
            str(log.get('food_name', '')).strip().lower()
            for log in recent_food_logs
            if 'food_name' in log and isinstance(log['food_name'], str)
        ]

        # Get recent symptom logs
        recent_symptoms = symptom_logs_collection.find(
            {'user_id': ObjectId(user_id)}
        ).sort('date', -1).limit(5)

        user['recent_symptoms'] = [
            {
                'symptom': s.get('symptom'),
                'severity': s.get('severity'),
                'description': s.get('description'),
                'date': s.get('date'),
            }
            for s in recent_symptoms
        ]

        # Pass to RAG pipeline with conversation context
        conversation_context = session['conversation_history']
        answer, docs = answer_query(query, user=user, conversation_context=conversation_context)
        response_text = getattr(answer, "content", str(answer))

        # Append bot response to conversation history
        session['conversation_history'].append({'role': 'assistant', 'content': response_text})
        session['conversation_history'] = session['conversation_history'][-10:]

        return jsonify({'response': response_text}), 200
    else:
        return render_template('chat.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password'].strip()
        dob = request.form['dob'].strip()
        gender = request.form['gender'].strip()
        username = request.form['username'].strip()

        if not all([email, password, dob, gender, username]):
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))

        if '@' not in email or '.' not in email:
            flash('Invalid email format.', 'danger')
            return redirect(url_for('register'))

        if not re.match(r'^[a-zA-Z0-9]{3,20}$', username):
            flash('Username must be 3-20 alphanumeric characters.', 'danger')
            return redirect(url_for('register'))

        try:
            dob_date = datetime.strptime(dob, '%Y-%m-%d')
            today = datetime.utcnow()
            if dob_date > today:
                flash('Date of birth cannot be in the future.', 'danger')
                return redirect(url_for('register'))
        except ValueError:
            flash('Invalid date of birth format (use YYYY-MM-DD).', 'danger')
            return redirect(url_for('register'))

        if gender not in ['male', 'female', 'other']:
            flash('Invalid gender selection.', 'danger')
            return redirect(url_for('register'))

        if users_collection.find_one({'email': email}):
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))

        if users_collection.find_one({'username': username}):
            flash('Username already taken.', 'danger')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        result = users_collection.insert_one({
            'email': email,
            'password': hashed_password,
            'dob': dob,
            'gender': gender,
            'username': username
        })
        session['user_id'] = str(result.inserted_id)
        session['username'] = username
        # Create unique index for username
        try:
            users_collection.create_index('username', unique=True)
        except Exception as e:
            print(f"Warning: Could not create unique index for username: {e}")
        flash('Registration successful! Welcome!', 'success')
        return redirect(url_for('home'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password'].strip()

        user = users_collection.find_one({'email': email})
        if user and bcrypt.check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user.get('username', email)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/intake', methods=['POST'])
def log_intake():
    if 'user_id' not in session:
        flash('Please log in to log intake.', 'warning')
        return redirect(url_for('login'))

    food_name = request.form['food_name'].strip().lower()
    quantity = request.form['quantity'].strip()
    unit = request.form['unit'].strip()
    meal_type = request.form['meal_type'].strip()

    if not all([food_name, quantity, unit, meal_type]):
        flash('All fields are required.', 'danger')
        return redirect(url_for('home'))

    try:
        quantity = float(quantity)
        if quantity <= 0:
            raise ValueError
    except ValueError:
        flash('Quantity must be a positive number.', 'danger')
        return redirect(url_for('home'))

    if unit not in ['grams', 'servings']:
        flash('Invalid unit.', 'danger')
        return redirect(url_for('home'))

    if meal_type not in ['breakfast', 'lunch', 'dinner', 'snack']:
        flash('Invalid meal type.', 'danger')
        return redirect(url_for('home'))

    if food_name not in COMMON_FOODS:
        suggestions = get_close_matches(food_name, COMMON_FOODS, n=3)
        if suggestions:
            flash(f"Food not found. Did you mean: {', '.join(suggestions)}?", 'info')
        else:
            flash('Food not found in dataset. Please select a valid food.', 'danger')
        return redirect(url_for('home'))

    df = app.config['FOOD_DF']
    food_data = df[df['food_name'] == food_name]
    if food_data.empty:
        flash('Food not found in dataset. Please select a valid food.', 'danger')
        return redirect(url_for('home'))
    food_data = food_data.iloc[0]

    if unit == 'grams':
        nutrients = {col: float(food_data[col]) * (quantity / 100)
                     for col in NUTRIENT_COLUMNS if pd.notna(food_data[col])}
    else:
        nutrients = {col: float(food_data[f'unit_serving_{col}']) * quantity
                     for col in NUTRIENT_COLUMNS
                     if f'unit_serving_{col}' in food_data and pd.notna(food_data[f'unit_serving_{col}'])}

    intake_logs_collection.insert_one({
        'user_id': ObjectId(session['user_id']),
        'date': datetime.utcnow(),
        'meal_type': meal_type,
        'food_name': food_name,
        'quantity': quantity,
        'unit': unit,
        'serving_unit': food_data.get('servings_unit', 'unit'),
        'nutrients': nutrients
    })

    flash('Intake logged successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/symptoms', methods=['POST'])
def log_symptoms():
    if 'user_id' not in session:
        flash('Please log in to log symptoms.', 'warning')
        return redirect(url_for('login'))

    symptom = request.form['symptom'].strip().lower()
    severity = request.form['severity'].strip()
    description = request.form['description'].strip()

    if not all([symptom, severity]):
        flash('Symptom and severity are required.', 'danger')
        return redirect(url_for('home'))

    try:
        severity = int(severity)
        if severity < 1 or severity > 5:
            raise ValueError
    except ValueError:
        flash('Severity must be between 1 and 5.', 'danger')
        return redirect(url_for('home'))

    if symptom not in ['fatigue', 'headache', 'nausea', 'dizziness', 'custom']:
        flash('Invalid symptom.', 'danger')
        return redirect(url_for('home'))

    if symptom == 'custom' and not description:
        flash('Description required for custom symptom.', 'danger')
        return redirect(url_for('home'))

    symptom_logs_collection.insert_one({
        'user_id': ObjectId(session['user_id']),
        'date': datetime.utcnow(),
        'symptom': symptom,
        'severity': severity,
        'description': description
    })

    flash('Symptom logged successfully!', 'success')
    return redirect(url_for('home'))

# --- Goals/Tasks Section ---

@app.route('/goals', methods=['GET', 'POST'])
def goals():
    if 'user_id' not in session:
        flash('Please log in to access your goals.', 'warning')
        return redirect(url_for('login'))
    user_id = session['user_id']
    if request.method == 'POST':
        goal_text = request.form.get('goal', '').strip()
        if not goal_text:
            flash('Goal cannot be empty.', 'danger')
        else:
            goals_collection.insert_one({
                'user_id': user_id,
                'goal': goal_text,
                'created_at': datetime.utcnow(),
                'completed': False
            })
            flash('Goal added!', 'success')
        return redirect(url_for('goals'))
    # GET: show goals
    user_goals = list(goals_collection.find({'user_id': user_id}))
    return render_template('goals.html', goals=user_goals)

@app.route('/api/goals', methods=['GET', 'POST'])
def api_goals():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user_id = session['user_id']
    if request.method == 'POST':
        data = request.get_json(force=True)
        goal_text = data.get('goal', '').strip()
        if not goal_text:
            return jsonify({'error': 'Goal cannot be empty.'}), 400
        goal = {
            'user_id': user_id,
            'goal': goal_text,
            'created_at': datetime.utcnow(),
            'completed': False
        }
        goals_collection.insert_one(goal)
        return jsonify({'message': 'Goal added!'}), 201
    # GET: return all goals for user
    user_goals = list(goals_collection.find({'user_id': user_id}))
    for g in user_goals:
        g['_id'] = str(g['_id'])
    return jsonify(user_goals)

@app.route('/api/goals/<goal_id>', methods=['POST'])
def update_goal(goal_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.get_json(force=True)
        completed = data.get('completed', None)
        if completed is None:
            return jsonify({'error': 'Missing completed status'}), 400
        result = goals_collection.update_one(
            {'_id': ObjectId(goal_id), 'user_id': session['user_id']},
            {'$set': {'completed': bool(completed)}}
        )
        if result.matched_count:
            return jsonify({'message': 'Goal updated.'})
        else:
            return jsonify({'error': 'Goal not found.'}), 404
    except (InvalidId, Exception):
        return jsonify({'error': 'Invalid goal ID.'}), 400

# Helper: Create a new chat session
def create_chat_session(user_id=None, title=None, first_message=None):
    # Use first_message as title if no title provided
    chat_title = title or (first_message[:40] + ("..." if len(first_message) > 40 else "")) if first_message else "New Chat"
    chat = {
        "user_id": user_id,
        "title": chat_title,
        "created_at": datetime.utcnow(),
        "messages": []
    }
    result = chats_collection.insert_one(chat)
    return str(result.inserted_id)

# Helper: Add a message to a chat session
def add_message_to_chat(chat_id, role, content):
    chats_collection.update_one(
        {"_id": ObjectId(chat_id)},
        {"$push": {"messages": {"role": role, "content": content, "timestamp": datetime.utcnow()}}}
    )

# Helper: Get chat history
def get_chat_history(chat_id):
    chat = chats_collection.find_one({"_id": ObjectId(chat_id)})
    if chat:
        return chat["messages"]
    return []

# Helper: List all chats for a user
def list_chats(user_id=None):
    query = {"user_id": user_id} if user_id else {}
    chats = chats_collection.find(query).sort("created_at", -1)
    return [{"_id": str(chat["_id"]), "title": chat.get("title", "Chat"), "created_at": chat["created_at"]} for chat in chats]

# Helper: Delete a chat session
def delete_chat_session(chat_id, user_id=None):
    query = {"_id": ObjectId(chat_id)}
    if user_id:
        query["user_id"] = user_id
    result = chats_collection.delete_one(query)
    return result.deleted_count > 0

# Flask routes for chat management
@app.route("/api/chats", methods=["GET"])
def api_list_chats():
    try:
        user_id = session.get("user_id")
        chats = list_chats(user_id)
        return jsonify(chats)
    except Exception as e:
        logging.exception("Error in api_list_chats")
        return make_json_error(str(e))

@app.route("/api/chats", methods=["POST"])
def api_create_chat():
    try:
        user_id = session.get("user_id")
        data = request.json or {}
        title = data.get("title")
        first_message = data.get("first_message")
        chat_id = create_chat_session(user_id, title, first_message)
        return jsonify({"chat_id": chat_id})
    except Exception as e:
        logging.exception("Error in api_create_chat")
        return make_json_error(str(e))

@app.route("/api/chats/<chat_id>", methods=["GET"])
def api_get_chat(chat_id):
    try:
        messages = get_chat_history(chat_id)
        return jsonify(messages)
    except Exception as e:
        logging.exception("Error in api_get_chat")
        return make_json_error(str(e))

@app.route("/api/chats/<chat_id>/message", methods=["POST"])
def api_add_message(chat_id):
    try:
        data = request.json
        role = data.get("role")
        content = data.get("content")
        add_message_to_chat(chat_id, role, content)
        return jsonify({"status": "ok"})
    except Exception as e:
        logging.exception("Error in api_add_message")
        return make_json_error(str(e))

@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def api_delete_chat(chat_id):
    try:
        user_id = session.get("user_id")
        success = delete_chat_session(chat_id, user_id)
        if success:
            return jsonify({"status": "deleted"})
        else:
            return make_json_error("Chat not found or not authorized", 404)
    except Exception as e:
        logging.exception("Error in api_delete_chat")
        return make_json_error(str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
