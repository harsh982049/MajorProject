# config.py

class Config:
    # Replace username, password, host, and databasename with your MySQL credentials
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:9ox(CK^C<W@localhost/majorproject"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "your-secret-key"
    JWT_SECRET_KEY = "your-jwt-secret-key"
