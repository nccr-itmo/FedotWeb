import os

import requests
from dotenv import load_dotenv
from werkzeug.contrib.fixers import ProxyFix

from app import create_app, db, socketio

if __name__ == "__main__":
    load_dotenv("local.env")
    load_dotenv("oauth.env")
    load_dotenv("mongo_conn_string.env")

    env = os.getenv('MONGO_CONN_STRING')
    print(env)

    app = create_app(os.getenv("FLASK_ENV") or "dev")

    app.wsgi_app = ProxyFix(app.wsgi_app)

    db.create_all(app=app)
    host = os.getenv("FLASK_HOST")
    port = os.getenv("FLASK_PORT")
    if not host:
        socketio.run(app, use_reloader=False)
    else:
        socketio.run(app, use_reloader=False, host=host, port=port)
