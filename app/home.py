from flask import Blueprint, render_template 
from .train import spam_filter 
home_bp = Blueprint('home_bp', __name__)

@home_bp.route('/')
def main(methods = ['POST']):
	return "Welcome to SMS Spam Detection. Kindly Enter Your message on top"


@home_bp.route('/<string:s>')
def home(s , methods = ['POST' , 'GET']):
    return spam_filter(s);
        
    
