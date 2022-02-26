from flask_cors import CORS
from flask import Flask, jsonify, g
from config import Config

from vis import vis
from api import api

import argparse

def create_app(config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)

    @app.route('/config')
    def config():
        return jsonify([str(k) for k in list(app.config.items())])

    @app.route('/hello')
    def hello():
        return 'hello world'

    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(vis, url_prefix='/')

    return app


parser = argparse.ArgumentParser()
parser.add_argument('--host', default='0.0.0.0',
                    help='Port in which to run the API')
parser.add_argument('--port', default=8001,
                    help='Port in which to run the API')
parser.add_argument('--debug', action='store_true',
                    help='If true, run Flask in debug mode')

_args, unknown = parser.parse_known_args()

application = create_app(vars(_args))

if __name__ == '__main__':
    application.run(
        debug=_args.debug,
        host=_args.host,
        port=int(_args.port)
    )
