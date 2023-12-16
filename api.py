from flask import Flask, request
from flask_restx import Api, Resource, fields, Namespace
import pickle
import numpy as np

app = Flask(__name__)
api = Api(app, title='Attack Prediction API', version='1.0')
my_namespace = Namespace('My Custom Namespace', description='API')

# Load the model and label encoder
with open("C:\\Users\\ASUS\\Desktop\\BigData\\Final Project\\CODE\\models\\random_forest_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("C:\\Users\\ASUS\\Desktop\\BigData\\Final Project\\CODE\\models\\label_encoder.pkl", 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define the API model for Swagger documentation
input_model = api.model('InputData', {
    'flag_S0': fields.Float(required=True),
    'flag_SF': fields.Float(required=True),
    'src_bytes': fields.Integer(required=True),
    'dst_bytes': fields.Integer(required=True),
    'logged_in': fields.Integer(required=True),
    'count': fields.Integer(required=True),
    'serror_rate': fields.Float(required=True),
    'srv_serror_rate': fields.Float(required=True),
    'dst_host_srv_count': fields.Integer(required=True),
    'dst_host_same_srv_rate': fields.Float(required=True),
    'dst_host_diff_srv_rate': fields.Float(required=True),
    'dst_host_same_src_port_rate': fields.Float(required=True),
    'dst_host_serror_rate': fields.Float(required=True),
    'dst_host_srv_serror_rate': fields.Float(required=True)
})
api.add_namespace(my_namespace)
@api.route('/predict')
class Predict(Resource):
    @api.expect(input_model)
    def post(self):
        data = request.json

        input_data = np.array([[
            data['flag_S0'],
            data['flag_SF'],
            data['src_bytes'],
            data['dst_bytes'],
            data['logged_in'],
            data['count'],
            data['serror_rate'],
            data['srv_serror_rate'],
            data['dst_host_srv_count'],
            data['dst_host_same_srv_rate'],
            data['dst_host_diff_srv_rate'],
            data['dst_host_same_src_port_rate'],
            data['dst_host_serror_rate'],
            data['dst_host_srv_serror_rate']
        ]])

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        attack_label = label_encoder.inverse_transform(prediction)

        response = {
            'probability_of_attack': prediction_proba.max(),
            'type_of_attack': attack_label[0]
        }

        return response

if __name__ == '__main__':
    app.run(debug=True, port=9000)
