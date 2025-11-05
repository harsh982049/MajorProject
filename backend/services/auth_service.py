# services/auth_service.py

from flask import jsonify
from sqlalchemy import or_
from models import User
from extensions import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token
import datetime

def register_user(data):
    """
    Minimal backend logic for user registration.
    Assumes the frontend has validated all fields
    (non-empty, matching passwords, etc.).
    """
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    
    # Check if a user with the same username or email already exists
    if User.query.filter(or_(User.username == username, User.email == email)).first():
        return jsonify({"error": "User already exists"}), 400

    # Hash the password before storing
    hashed_password = generate_password_hash(password)

    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "message": "User registered successfully",
        "user": new_user.to_dict()
    }), 201


def login_user(data):
    """
    Minimal backend logic for user login.
    Assumes the frontend has validated that
    username/password are non-empty, etc.
    """
    username = data.get("username")
    password = data.get("password")

    # Find user by username
    user = User.query.filter_by(username=username).first()

    # Verify hashed password
    if user and check_password_hash(user.password, password):
        # Generate a JWT token; token expires in 1 hour
        access_token = create_access_token(
            identity=str(user.id),
            expires_delta=datetime.timedelta(days=30)
        )
        return jsonify({
            "message": "Login successful",
            "token": access_token,
            "user": user.to_dict()
        }), 200

    return jsonify({"error": "Invalid username or password"}), 401
