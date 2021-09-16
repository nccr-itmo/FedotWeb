import os

import requests
from flask import Flask, jsonify, Blueprint
from flask_cors import CORS, cross_origin
from flask_login import LoginManager
from flask_pymongo import PyMongo
from flask_restx import Api
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

storage = PyMongo()

socketio = SocketIO()

login_manager = LoginManager()


def create_app(env=None):
    print('Create app')
    from app.config import config_by_name

    template_dir = os.path.abspath('frontend/build')
    static_dir = os.path.abspath('frontend/build/static')

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.config['CORS_HEADERS'] = 'Content-Type'

    app.config.from_object(config_by_name[env or "dev"])

    api_blueprint = Blueprint("api", __name__, url_prefix="/api")
    api = Api(api_blueprint, title="Fedot Web API", version="0.1.0")

    socketio.init_app(app)

    db.init_app(app)

    storage.init_app(app)

    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @app.route("/health")
    @cross_origin()
    def health():
        return jsonify("healthy")

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    # @app.before_first_request
    # def create_default_user():
    #     host = os.getenv("FLASK_HOST")
    #     port = os.getenv("FLASK_PORT")
    #     if not host or not port:
    #         host = 'http://127.0.0.1'
    #         port = 5000
    #     r = requests.post(f'{host}:{port}/api/token/signup', data={'email': 'guest', 'password': 'guest'})
    #     print(r.status_code, r.reason)

    from app.web.auth.controller import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    from app.web.mod_base.controller import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from app.api.routes import register_routes
    register_routes(api, app)
    app.register_blueprint(api_blueprint)

    return app
