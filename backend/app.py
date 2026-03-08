import os
import io
import json
import zipfile
import pandas as pd
import numpy as np
import pickle
import requests
import base64
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify, redirect
from flask_cors import CORS

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

# Load .env from frontend folder (where GitHub credentials are stored)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'frontend', '.env'))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# GitHub OAuth credentials
GITHUB_CLIENT_ID = os.getenv('GITHUB_CLIENT_ID')
GITHUB_CLIENT_SECRET = os.getenv('GITHUB_CLIENT_SECRET')

# Gemini API credentials
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# GLOBAL MODEL STORE
store = {
    "model": None,
    "image_extractor": MobileNetV2(weights='imagenet', include_top=False, pooling='avg'),
    "feature_names": [],
    "train_cols": [],
    "classes": [],
    "data_type": None,
    "mode": None,
    "metrics": {},
    "feature_importance": {}
}


@app.route('/')
def index():
    return render_template('index.html')


# ============================================
# TRAIN CSV MODEL
# ============================================

@app.route('/train_csv', methods=['POST'])
def train_csv():

    mode = request.form.get('mode')
    target = request.form.get('target')
    selected_features = request.form.get('features', "").split(',')

    file = request.files['file']

    df = pd.read_csv(file).dropna()

    X = df[selected_features]
    y = df[target]

    # One-hot encoding
    X_encoded = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42
    )

    # Extract model parameters from request with defaults
    model_params = {
        'n_estimators': int(request.form.get('n_estimators', 300)),
        'max_depth': int(request.form.get('max_depth', 20)) if request.form.get('max_depth') else None,
        'min_samples_split': int(request.form.get('min_samples_split', 2)),
        'min_samples_leaf': int(request.form.get('min_samples_leaf', 1)),
        'random_state': 42
    }

    if mode == "Classification":

        model = RandomForestClassifier(**model_params)

    else:

        model = RandomForestRegressor(**model_params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # METRICS
    if mode == "Classification":

        acc = accuracy_score(y_test, y_pred)

        store["metrics"] = {
            "accuracy": float(acc),
            "samples": len(df)
        }

    else:

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        store["metrics"] = {
            "r2_score": float(r2),
            "mse": float(mse)
        }

    # FEATURE IMPORTANCE
    importances = model.feature_importances_

    store.update({
        "mode": mode,
        "data_type": "csv",
        "train_cols": X_encoded.columns.tolist(),
        "feature_names": selected_features,
        "model": model,
        "feature_importance": dict(zip(X_encoded.columns, importances))
    })

    return jsonify({
        "status": f"CSV {mode} Model Trained",
        "metrics": store["metrics"],
        "features": selected_features
    })


# ============================================
# TRAIN IMAGE MODEL
# ============================================

@app.route('/train_images', methods=['POST'])
def train_images():

    zip_files = request.files.getlist('zip_files')
    
    # Get custom class names from form data, or fall back to filenames
    custom_class_names = request.form.get('class_names', '').split(',')
    custom_class_names = [name.strip() for name in custom_class_names if name.strip()]

    X_features = []
    y_labels = []

    class_names = []

    for i, zf in enumerate(zip_files):

        # Use custom class name if provided, otherwise use filename
        if custom_class_names and i < len(custom_class_names):
            class_name = custom_class_names[i]
        else:
            class_name = zf.filename.replace('.zip', '')
        class_names.append(class_name)

        with zipfile.ZipFile(zf, 'r') as z:

            for file_name in z.namelist():

                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):

                    with z.open(file_name) as f:

                        img_data = f.read()

                        img = image.load_img(
                            io.BytesIO(img_data),
                            target_size=(224, 224)
                        )

                        x = image.img_to_array(img)

                        x = np.expand_dims(x, axis=0)

                        x = preprocess_input(x)

                        feat = store["image_extractor"].predict(x, verbose=0)

                        X_features.append(feat.flatten())

                        y_labels.append(class_name)

    X_features = np.array(X_features)

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y_labels,
        test_size=0.2,
        random_state=42,
        stratify=y_labels if len(set(y_labels)) > 1 else None
    )

    # Extract model parameters from request with defaults
    model_params = {
        'n_estimators': int(request.form.get('n_estimators', 300)),
        'max_depth': int(request.form.get('max_depth', 20)) if request.form.get('max_depth') else None,
        'min_samples_split': int(request.form.get('min_samples_split', 2)),
        'min_samples_leaf': int(request.form.get('min_samples_leaf', 1)),
        'random_state': 42
    }

    model = RandomForestClassifier(**model_params)

    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    store["metrics"] = {
        "accuracy": float(acc),
        "samples": len(y_labels)
    }

    store.update({
        "mode": "Classification",
        "data_type": "image",
        "classes": class_names,
        "model": model,
        "feature_importance": {}  # No feature importance for image models
    })

    return jsonify({
        "status": "Image Classifier Trained",
        "classes": class_names
    })


# ============================================
# PREDICTION
# ============================================

@app.route('/predict', methods=['POST'])
def predict():

    if not store["model"]:
        return jsonify({"error": "No model trained"}), 400

    # -------------------
    # CSV Prediction
    # -------------------

    if store["data_type"] == 'csv':

        df_input = pd.DataFrame([request.json['inputs']])

        # convert numeric strings
        df_input = df_input.apply(pd.to_numeric, errors="coerce")

        df_input = pd.get_dummies(df_input).reindex(
            columns=store["train_cols"],
            fill_value=0
        )

        pred = store["model"].predict(df_input)[0]

        response = {
            "prediction": str(pred)
        }

        if store["mode"] == "Classification":

            proba = store["model"].predict_proba(df_input)[0].tolist()

            response["probabilities"] = proba

        return jsonify(response)

    # -------------------
    # IMAGE Prediction
    # -------------------

    elif store["data_type"] == 'image':

        file = request.files['image']

        img = image.load_img(
            io.BytesIO(file.read()),
            target_size=(224, 224)
        )

        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        feat = store["image_extractor"].predict(x, verbose=0)

        pred = store["model"].predict(feat.reshape(1, -1))[0]

        return jsonify({"prediction": str(pred)})


# ============================================
# MODEL METRICS API
# ============================================

@app.route('/metrics')
def metrics():

    if not store["metrics"]:
        return jsonify({"error": "Model not trained"}), 400

    return jsonify(store["metrics"])


# ============================================
# FEATURE IMPORTANCE API
# ============================================

@app.route('/feature_importance')
def feature_importance():

    if not store["model"]:
        return jsonify({"error": "Model not trained"}), 400

    # Return empty list for image models (no meaningful feature importance)
    if not store["feature_importance"]:
        return jsonify([])

    sorted_feats = sorted(
        store["feature_importance"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    return jsonify(sorted_feats)


# ============================================
# MODEL INFO
# ============================================

@app.route('/model_info')
def model_info():

    if not store["model"]:
        return jsonify({"error": "No model trained"}), 400

    # CSV models
    if store["data_type"] == "csv":
        return jsonify({
            "data_type": "csv",
            "mode": store["mode"],
            "features": store["feature_names"],
            "n_features_after_encoding": len(store["train_cols"])
        })

    # Image models
    elif store["data_type"] == "image":
        return jsonify({
            "data_type": "image",
            "mode": "Classification",
            "classes": store["classes"],
            "feature_extractor": "MobileNetV2"
        })


# ============================================
# EXPORT MODEL
# ============================================

@app.route('/export_model')
def export_model():

    if not store["model"]:
        return jsonify({"error": "No model trained"}), 400

    # Prepare model data for export
    model_data = {
        "model": store["model"],
        "data_type": store["data_type"],
        "mode": store["mode"],
        "classes": store["classes"] if store["data_type"] == "image" else [],
        "feature_names": store["feature_names"] if store["data_type"] == "csv" else [],
        "train_cols": store["train_cols"] if store["data_type"] == "csv" else []
    }

    # Include image extractor for image models
    if store["data_type"] == "image":
        model_data["image_extractor"] = store["image_extractor"]

    # Serialize the model data using pickle
    model_bytes = pickle.dumps(model_data)

    # Set filename based on data type
    filename = f"trained_model_{store['data_type']}.pkl"

    return model_bytes, 200, {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': f'attachment; filename={filename}'
    }


# ============================================
# EXPLAIN MODEL (Gemini)
# ============================================

@app.route('/explain_model')
def explain_model():
    """Use Gemini to explain the trained model"""
    if not store["model"]:
        return jsonify({"error": "No model trained"}), 400
    
    # Build model info in the same format as export_model
    model_info = {
        "data_type": store["data_type"],
        "mode": store["mode"],
        "classes": store["classes"] if store["data_type"] == "image" else [],
        "feature_names": store["feature_names"] if store["data_type"] == "csv" else [],
        "train_cols": store["train_cols"] if store["data_type"] == "csv" else [],
        "metrics": store["metrics"],
        "model_type": "RandomForestClassifier" if store["mode"] == "Classification" else "RandomForestRegressor",
        "feature_extractor": "MobileNetV2" if store["data_type"] == "image" else None
    }
    
    prompt = f"""Explain how this model works in under 3 sentences:

Model Information (same format as exported model file):
{json.dumps(model_info, indent=2)}"""
    
    try:
        explanation = call_gemini(prompt)
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# GEMINI API HELPER
# ============================================

def call_gemini(prompt, system_instruction=None):
    """Call Gemini API with a prompt"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    else:
        raise Exception(f"Gemini API error: {response.text}")


# ============================================
# GITHUB OAUTH
# ============================================

@app.route('/auth/github')
def github_auth():
    """Redirect user to GitHub for OAuth login with prompt in state"""
    user_prompt = request.args.get('prompt', '')
    # Encode prompt in state parameter (base64 for URL safety)
    state = base64.b64encode(user_prompt.encode()).decode()
    url = f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&scope=repo&state={state}"
    return redirect(url)


@app.route('/auth/github/callback')
def github_callback():
    """Handle GitHub OAuth callback, create repo and show loading page"""
    code = request.args.get('code')
    state = request.args.get('state', '')
    
    # Decode user prompt from state
    try:
        user_prompt = base64.b64decode(state).decode() if state else ''
    except:
        user_prompt = ''
    
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    try:
        # Exchange code for access token
        token_response = requests.post(
            'https://github.com/login/oauth/access_token',
            headers={'Accept': 'application/json'},
            json={
                'client_id': GITHUB_CLIENT_ID,
                'client_secret': GITHUB_CLIENT_SECRET,
                'code': code
            }
        )
        token_data = token_response.json()
        access_token = token_data.get('access_token')
        
        if not access_token:
            return jsonify({"error": "Failed to get access token"}), 400
        
        # Create repository first
        result = create_and_push_repo(access_token)
        repo_name = result['name']
        repo_url = result['html_url']
        
        # Return loading page that will call /generate_codebase via JavaScript
        return f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    color: white;
                    margin: 0;
                }}
                .container {{
                    text-align: center;
                    max-width: 500px;
                    padding: 40px;
                    background: rgba(255,255,255,0.05);
                    border-radius: 16px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                }}
                h1 {{ color: #00F0FF; margin-bottom: 20px; }}
                p {{ color: #ccc; margin-bottom: 30px; }}
                .loader {{
                    width: 60px;
                    height: 60px;
                    border: 4px solid rgba(255,255,255,0.1);
                    border-top: 4px solid #39FF14;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin: 30px auto;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
                .status {{
                    font-size: 14px;
                    color: #888;
                    margin-top: 20px;
                }}
                .status.error {{
                    color: #ff4444;
                }}
                .dots {{
                    display: inline-block;
                }}
                .dots::after {{
                    content: '';
                    animation: dots 1.5s steps(4, end) infinite;
                }}
                @keyframes dots {{
                    0%, 20% {{ content: ''; }}
                    40% {{ content: '.'; }}
                    60% {{ content: '..'; }}
                    80%, 100% {{ content: '...'; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Generating Your App</h1>
                <div class="loader" id="loader"></div>
                <p id="message">AI is writing your codebase<span class="dots"></span></p>
                <p class="status" id="status">This may take a minute</p>
            </div>
            
            <script>
                const accessToken = "{access_token}";
                const repoName = "{repo_name}";
                const repoUrl = "{repo_url}";
                const userPrompt = decodeURIComponent("{base64.b64encode(user_prompt.encode()).decode()}");
                
                async function generateCode() {{
                    try {{
                        const response = await fetch('/generate_codebase', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json'
                            }},
                            body: JSON.stringify({{
                                access_token: accessToken,
                                repo_name: repoName,
                                prompt: atob(userPrompt)
                            }})
                        }});
                        
                        const data = await response.json();
                        
                        if (response.ok && data.success) {{
                            document.getElementById('message').innerHTML = 'Success! Redirecting to your repo...';
                            document.getElementById('status').textContent = '';
                            document.getElementById('loader').style.display = 'none';
                            setTimeout(() => {{
                                window.location.href = repoUrl;
                            }}, 1500);
                        }} else {{
                            throw new Error(data.error || 'Generation failed');
                        }}
                    }} catch (error) {{
                        document.getElementById('message').textContent = 'Error: ' + error.message;
                        document.getElementById('status').className = 'status error';
                        document.getElementById('status').innerHTML = '<a href="' + repoUrl + '" style="color: #00F0FF;">Go to repository anyway</a>';
                        document.getElementById('loader').style.display = 'none';
                    }}
                }}
                
                // Start generation immediately
                generateCode();
            </script>
        </body>
        </html>
        """
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_codebase', methods=['POST'])
def generate_codebase():
    """Generate codebase with Gemini and push to GitHub"""
    data = request.json
    access_token = data.get('access_token')
    repo_name = data.get('repo_name')
    user_prompt = data.get('prompt')
    
    if not all([access_token, repo_name, user_prompt]):
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        headers = {
            'Authorization': f'token {access_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Get the repo owner
        user_response = requests.get('https://api.github.com/user', headers=headers)
        owner = user_response.json()['login']
        
        # Build context about the trained model
        model_context = f"""
        Trained Model Information:
        - Data Type: {store.get('data_type', 'N/A')}
        - Mode: {store.get('mode', 'N/A')}
        - Features: {', '.join(store.get('feature_names', [])) or 'N/A'}
        - Classes: {', '.join(store.get('classes', [])) or 'N/A'}
        - Training Columns: {store.get('train_cols', [])}
        - Metrics: {store.get('metrics', {})}
        """
        
        # Build the exact model file structure for Gemini
        model_file_structure = f"""
        THE TRAINED MODEL FILE (trained_model.pkl) STRUCTURE:
        The pickle file contains a dictionary with these exact keys:
        {{
            "model": <sklearn RandomForest{'Classifier' if store.get('mode') == 'Classification' else 'Regressor'} object>,
            "data_type": "{store.get('data_type')}",
            "mode": "{store.get('mode')}",
            "classes": {store.get('classes', []) if store.get('data_type') == 'image' else []},
            "feature_names": {store.get('feature_names', []) if store.get('data_type') == 'csv' else []},
            "train_cols": {store.get('train_cols', []) if store.get('data_type') == 'csv' else []}
            {', "image_extractor": <MobileNetV2 Keras model>' if store.get('data_type') == 'image' else ''}
        }}
        """
        
        # Build exact usage code based on model type
        if store.get('data_type') == 'csv':
            usage_code = f"""
        EXACT CODE TO LOAD AND USE THIS MODEL:
        ```python
        import pickle
        import pandas as pd
        
        # Load the model
        with open('trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        train_cols = model_data['train_cols']  # {store.get('train_cols', [])}
        mode = model_data['mode']  # "{store.get('mode')}"
        
        # To make a prediction, create input with these exact feature names: {store.get('feature_names', [])}
        # Example:
        input_data = {{
            {', '.join([f'"{f}": <value>' for f in store.get('feature_names', [])])}
        }}
        
        # Prepare input (must match training columns after one-hot encoding)
        df_input = pd.DataFrame([input_data])
        df_input = pd.get_dummies(df_input).reindex(columns=train_cols, fill_value=0)
        
        # Predict
        prediction = model.predict(df_input)[0]
        
        # For classification, get probabilities:
        if mode == "Classification":
            probabilities = model.predict_proba(df_input)[0]
        ```
        """
        else:  # image model
            usage_code = f"""
        EXACT CODE TO LOAD AND USE THIS MODEL:
        ```python
        import pickle
        import numpy as np
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        import io
        
        # Load the model
        with open('trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        image_extractor = model_data['image_extractor']  # MobileNetV2 feature extractor
        classes = model_data['classes']  # {store.get('classes', [])}
        
        # To classify an image:
        def predict_image(img_file):
            # Load and preprocess image (MUST be 224x224)
            img = image.load_img(io.BytesIO(img_file.read()), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Extract features using MobileNetV2
            features = image_extractor.predict(x, verbose=0)
            
            # Predict class
            prediction = model.predict(features.reshape(1, -1))[0]
            return prediction  # Returns one of: {store.get('classes', [])}
        ```
        """
        
        # Generate codebase with Gemini
        generation_prompt = f"""
        {user_prompt}
        
        Model Context:
        {model_context}
        
        {model_file_structure}
        
        {usage_code}
        
        IMPORTANT REQUIREMENTS:
        1. Generate a complete, working codebase STRUCTURED FOR VERCEL DEPLOYMENT
        2. The model MUST be loaded from 'trained_model.pkl' using the EXACT code pattern shown above
        3. VERCEL STRUCTURE: The main Flask app MUST be in 'api/index.py' with the app instance named 'app'
        4. Include all necessary files (Python files in api/, HTML templates in api/templates/, requirements.txt in root)
        5. Make sure the code handles the specific data type ({store.get('data_type')}) and mode ({store.get('mode')})
        6. USE THE EXACT LOADING AND PREDICTION CODE PROVIDED ABOVE - do not deviate from it
        7. The requirements.txt must include: flask, pandas, numpy, scikit-learn{', tensorflow-cpu>=2.15 (NOT tensorflow, use tensorflow-cpu). DO NOT use strict version pins like ==, use >= instead' if store.get('data_type') == 'image' else ''}
        8. The trained_model.pkl file will be in the ROOT directory, so load it with: open('trained_model.pkl', 'rb')
        9. For templates, use: app = Flask(__name__, template_folder='templates') and put HTML files in api/templates/
        
        VERCEL FILE STRUCTURE:
        - api/index.py (main Flask app with 'app' variable)
        - api/templates/*.html (HTML templates)
        - requirements.txt (in root)
        - trained_model.pkl (in root - already provided)
        
        OUTPUT FORMAT:
        Return the files in this exact format, with each file separated by "---FILE---":
        
        ---FILE---
        FILENAME: api/index.py
        CONTENT:
        <Flask app code here>
        ---FILE---
        FILENAME: api/templates/index.html
        CONTENT:
        <HTML content here>
        ---FILE---
        FILENAME: requirements.txt
        CONTENT:
        <dependencies here>
        
        Start with api/index.py and include ALL necessary files.
        """
        
        system_instruction = """You are an expert Python developer specializing in Vercel deployments. Generate clean, production-ready code.
        Always include proper error handling, comments, and a requirements.txt file.
        For web apps, create attractive, modern UIs. Follow the exact output format specified.
        CRITICAL: Structure the app for Vercel serverless deployment with Flask in api/index.py.
        CRITICAL: In requirements.txt, NEVER use strict version pins (==). Use >= for minimum versions only.
        CRITICAL: For image/tensorflow projects, use 'tensorflow-cpu>=2.15' NOT 'tensorflow'."""
        
        generated_code = call_gemini(generation_prompt, system_instruction)
        
        # Parse the generated files
        files = parse_generated_files(generated_code)
        
        if not files:
            raise Exception("Failed to parse generated files")
        
        # Push each file to GitHub
        for file_path, content in files.items():
            push_file_to_github(headers, owner, repo_name, file_path, content)
        
        # Push vercel.json for Vercel deployment
        vercel_config = '''{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}'''
        push_file_to_github(headers, owner, repo_name, "vercel.json", vercel_config)
        
        # Push the trained model file
        model_data = {
            "model": store["model"],
            "data_type": store["data_type"],
            "mode": store["mode"],
            "classes": store["classes"] if store["data_type"] == "image" else [],
            "feature_names": store["feature_names"] if store["data_type"] == "csv" else [],
            "train_cols": store["train_cols"] if store["data_type"] == "csv" else []
        }
        
        if store["data_type"] == "image":
            model_data["image_extractor"] = store["image_extractor"]
        
        model_bytes = pickle.dumps(model_data)
        push_binary_file_to_github(headers, owner, repo_name, "trained_model.pkl", model_bytes)
        
        # Update README
        readme_content = f"""# {repo_name}

Created with **Universal Teachable Machine** + **Gemini AI**

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/{owner}/{repo_name})

## About This Project
{user_prompt}

## Model Info
- Data Type: {store.get('data_type', 'N/A')}
- Mode: {store.get('mode', 'N/A')}
- Features: {', '.join(store.get('feature_names', [])) or 'N/A'}
- Classes: {', '.join(store.get('classes', [])) or 'N/A'}

## Deployment
Connect this repository to [Vercel](https://vercel.com) for automatic deployment.

## Files Generated
{chr(10).join([f'- `{f}`' for f in files.keys()])}
- `trained_model.pkl`
- `vercel.json`

---
*Generated by Universal Teachable Machine with Gemini AI*
"""
        
        readme_response = requests.get(
            f"https://api.github.com/repos/{owner}/{repo_name}/contents/README.md",
            headers=headers
        )
        sha = readme_response.json().get('sha') if readme_response.status_code == 200 else None
        
        update_data = {
            'message': 'Update README with project details',
            'content': base64.b64encode(readme_content.encode()).decode()
        }
        if sha:
            update_data['sha'] = sha
        
        requests.put(
            f"https://api.github.com/repos/{owner}/{repo_name}/contents/README.md",
            headers=headers,
            json=update_data
        )
        
        # Return success JSON response
        return jsonify({
            "success": True,
            "files": list(files.keys()) + ['trained_model.pkl', 'vercel.json']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def create_and_push_repo(token):
    """Create a new GitHub repo and push initial files"""
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Create repository
    import time
    repo_name = f"teachable-machine-{int(time.time())}"
    
    create_response = requests.post(
        'https://api.github.com/user/repos',
        headers=headers,
        json={
            'name': repo_name,
            'auto_init': True,
            'description': 'Created with Universal Teachable Machine'
        }
    )
    
    if create_response.status_code != 201:
        raise Exception(f"Failed to create repo: {create_response.json()}")
    
    repo_data = create_response.json()
    owner = repo_data['owner']['login']
    
    # Get SHA of existing README (created by auto_init)
    readme_response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/contents/README.md",
        headers=headers
    )
    
    sha = readme_response.json().get('sha') if readme_response.status_code == 200 else None
    
    # Update README with custom content
    readme_content = f"""# {repo_name}

Created with **Universal Teachable Machine**

## Model Info
- Data Type: {store.get('data_type', 'N/A')}
- Mode: {store.get('mode', 'N/A')}
- Features: {', '.join(store.get('feature_names', [])) or 'N/A'}

## Usage
Load the exported model with pickle:
```python
import pickle
with open('trained_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
```
"""
    
    update_data = {
        'message': 'Initial commit from Universal Teachable Machine',
        'content': base64.b64encode(readme_content.encode()).decode()
    }
    if sha:
        update_data['sha'] = sha
    
    requests.put(
        f"https://api.github.com/repos/{owner}/{repo_name}/contents/README.md",
        headers=headers,
        json=update_data
    )
    
    return {
        'html_url': repo_data['html_url'],
        'name': repo_name
    }


def parse_generated_files(generated_code):
    """Parse the generated code into individual files"""
    files = {}
    
    # Split by the file separator
    parts = generated_code.split('---FILE---')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Look for FILENAME: and CONTENT: markers
        if 'FILENAME:' in part and 'CONTENT:' in part:
            try:
                # Extract filename
                filename_start = part.index('FILENAME:') + len('FILENAME:')
                filename_end = part.index('CONTENT:')
                filename = part[filename_start:filename_end].strip()
                
                # Extract content
                content_start = part.index('CONTENT:') + len('CONTENT:')
                content = part[content_start:].strip()
                
                # Clean up the filename (remove any leading/trailing quotes or spaces)
                filename = filename.strip('\'"')
                
                if filename and content:
                    files[filename] = content
            except ValueError:
                continue
    
    return files


def push_file_to_github(headers, owner, repo_name, file_path, content):
    """Push a single file to GitHub repository"""
    # Check if file exists
    check_response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}",
        headers=headers
    )
    
    sha = check_response.json().get('sha') if check_response.status_code == 200 else None
    
    data = {
        'message': f'Add {file_path}',
        'content': base64.b64encode(content.encode()).decode()
    }
    
    if sha:
        data['sha'] = sha
    
    response = requests.put(
        f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}",
        headers=headers,
        json=data
    )
    
    if response.status_code not in [200, 201]:
        print(f"Warning: Failed to push {file_path}: {response.text}")


def push_binary_file_to_github(headers, owner, repo_name, file_path, content_bytes):
    """Push a binary file to GitHub repository"""
    # Check if file exists
    check_response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}",
        headers=headers
    )
    
    sha = check_response.json().get('sha') if check_response.status_code == 200 else None
    
    data = {
        'message': f'Add {file_path}',
        'content': base64.b64encode(content_bytes).decode()
    }
    
    if sha:
        data['sha'] = sha
    
    response = requests.put(
        f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}",
        headers=headers,
        json=data
    )
    
    if response.status_code not in [200, 201]:
        print(f"Warning: Failed to push {file_path}: {response.text}")


# ============================================

if __name__ == '__main__':
    app.run(host='localhost', debug=True)