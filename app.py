# app.py
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

# Initialiser l'application Dash
app = Dash(__name__)


# Mise en page de l'application
app.layout = html.Div(children=[
    html.H1(children='Hello Dash !!!!!'),

])

# Lancer l'application
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)