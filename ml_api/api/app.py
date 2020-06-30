from flask import Flask
from api.controller import prediction_app


def create_app(*, config_object) -> Flask:
	flask_app = Flask('ml_api')
	flask_app.config.from_object(config_object)

	# register blueprint
	flask_app.register_blueprint(prediction_app)

	return flask_app
