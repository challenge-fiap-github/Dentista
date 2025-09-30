from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)
CORS(app)

# Carregar dados
csv_file = 'odonto_vision_data.csv'  
data = pd.read_csv(csv_file)

# Preparação dos dados
X = data.drop('fraude', axis=1)
y = data['fraude']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Salvar modelo treinado
joblib.dump(model, 'modelo_fraude.pkl')

@app.route('/prever', methods=['POST'])
def prever_fraude():
    """Recebe dados JSON e retorna a previsão de fraude"""
    try:
        dados = request.json
        df_input = pd.DataFrame([dados])  
        
        # Carregar modelo treinado
        modelo = joblib.load('modelo_fraude.pkl')
        
        # Fazer previsão
        previsao = modelo.predict(df_input)[0]
        
        return jsonify({"fraude_detectada": bool(previsao), "mensagem": "Fraude detectada!" if previsao else "Transação segura."})
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
