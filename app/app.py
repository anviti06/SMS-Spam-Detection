
from flask import (Flask, render_template, request,Blueprint,session , redirect , url_for)
#from flask_sqlalchemy import SQLAlchemy
from datetime import timedelta
import datetime

def create_app():
    
    app = Flask(__name__, instance_relative_config=False,template_folder='../templates', static_folder='../static')
    #app.config.from_object('config.Config')
   
    # Initialize Plugins
    #db.init_app(app)
    
    
    #Importing Files
    from .train import trainmodel
    
    
    with app.app_context():
    
        #Register Blueprint
        from .home import home_bp as home_blueprint
        app.register_blueprint(home_blueprint)
        
        print(trainmodel())  

        return app
        
