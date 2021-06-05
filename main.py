
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


sid = SentimentIntensityAnalyzer()

app = Flask(__name__)
cors = CORS(app)
api = Api(app)


class Messages(Resource):
    def get(self):
        return "Get Request not supported"

    def post(self):
        message = request.json['message']
        return(sid.polarity_scores(message))


api.add_resource(Messages, '/predict')
