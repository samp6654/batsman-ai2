from flask import (
    Flask, request, jsonify, send_from_directory,
    redirect, url_for, session
)
from flask_login import (
    LoginManager, UserMixin,
    login_user, logout_user, login_required, current_user
)
from authlib.integrations.flask_client import OAuth
from flask_cors import CORS
import os
from model import analyze_video
from database import (
    init_db, create_user, find_user_by_email,
    find_user_by_google_id, verify_password,
    get_user_by_id, update_user_google
)

print("Flask starting...")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "batsman-ai-secret-2024-xK9pQ")

# ── CORS: allow the Render frontend to talk to this backend ──────────────────
ALLOWED_ORIGINS = [
    "https://batsman-ai2-1.onrender.com",
    "https://bat-3.onrender.com",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]
CORS(app, supports_credentials=True, origins=ALLOWED_ORIGINS)

# ── Session cookie settings for HTTPS (Render) ───────────────────────────────
is_production = os.environ.get("RENDER", False)  # Render sets RENDER=true
app.config["SESSION_COOKIE_SECURE"]   = bool(is_production)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "None" if is_production else "Lax"

# ── Flask-Login ──────────────────────────────────────────────────────────────
login_manager = LoginManager(app)


class User(UserMixin):
    def __init__(self, data):
        self.id     = data["id"]
        self.name   = data["name"]
        self.email  = data["email"]
        self.avatar = data.get("avatar", "")

    def get_id(self):
        return str(self.id)


@login_manager.user_loader
def load_user(user_id):
    data = get_user_by_id(int(user_id))
    return User(data) if data else None


@login_manager.unauthorized_handler
def unauthorized():
    # API requests get JSON 401; page requests get a redirect
    if request.path.startswith("/api/") or request.method == "POST":
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    return redirect("/")


# ── Google OAuth ─────────────────────────────────────────────────────────────
oauth = OAuth(app)
google_oauth = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# ── Database init ─────────────────────────────────────────────────────────────
init_db()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Static pages ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory("static", "login.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route("/signup", methods=["POST"])
def signup():
    data     = request.json or {}
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"status": "error", "message": "All fields are required."})
    if len(password) < 6:
        return jsonify({"status": "error", "message": "Password must be at least 6 characters."})
    if find_user_by_email(email):
        return jsonify({"status": "error", "message": "That email is already registered. Please log in."})

    user_data = create_user(name=name, email=email, password=password)
    login_user(User(user_data))
    return jsonify({
        "status": "success",
        "user": {"name": user_data["name"], "email": user_data["email"], "avatar": user_data["avatar"]}
    })


@app.route("/login", methods=["POST"])
def login():
    data     = request.json or {}
    email    = data.get("email", "").strip()
    password = data.get("password", "")

    user_data = find_user_by_email(email)
    if not user_data or not verify_password(user_data, password):
        return jsonify({"status": "error", "message": "Invalid email or password."})

    login_user(User(user_data))
    return jsonify({
        "status": "success",
        "user": {"name": user_data["name"], "email": user_data["email"], "avatar": user_data["avatar"]}
    })


@app.route("/logout", methods=["POST"])
def logout():
    logout_user()
    return jsonify({"status": "success"})


# ── Google OAuth flow ─────────────────────────────────────────────────────────
@app.route("/auth/google")
def google_login():
    if not os.environ.get("GOOGLE_CLIENT_ID"):
        # Redirect back with a friendly error flag
        return redirect("/?google_error=not_configured")
    redirect_uri = url_for("google_callback", _external=True)
    return google_oauth.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    try:
        token     = google_oauth.authorize_access_token()
        user_info = token.get("userinfo")
    except Exception:
        return redirect("/?google_error=auth_failed")

    if not user_info:
        return redirect("/?google_error=auth_failed")

    google_id = user_info["sub"]
    email     = user_info.get("email", "")
    name      = user_info.get("name", email.split("@")[0])
    avatar    = user_info.get("picture", "")

    # Find existing Google-linked account
    user_data = find_user_by_google_id(google_id)

    if not user_data:
        # Try to link to an existing email account
        user_data = find_user_by_email(email)
        if user_data:
            update_user_google(user_data["id"], google_id, avatar)
            user_data = get_user_by_id(user_data["id"])
        else:
            # Brand-new user via Google
            user_data = create_user(name=name, email=email, google_id=google_id, avatar=avatar)

    login_user(User(user_data))
    return redirect("/video-upload.html")


# ── Current user API ──────────────────────────────────────────────────────────
@app.route("/api/me")
@login_required
def me():
    return jsonify({
        "id":     current_user.id,
        "name":   current_user.name,
        "email":  current_user.email,
        "avatar": current_user.avatar,
    })


# ── Analysis routes ───────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    file = request.files["video"]
    shot = request.form.get("shot", "Unknown")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = analyze_video(filepath, shot=shot)
    result["shot"]     = shot
    result["filename"] = file.filename
    return jsonify(result)


@app.route("/history")
@login_required
def history():
    return jsonify([])


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)