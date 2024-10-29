from dash import Dash, dcc, html
import pandas as pd
import plotly.express as px

# Charger le dataset
url = "spotify_songs.csv"
spotify_songs = pd.read_csv(url)

# Créer une figure
fig = px.histogram(
    spotify_songs,
    x="track_popularity",
    title="Distribution de la popularité des chansons",
    color="playlist_genre",
    barmode="overlay",
)

# Créer l'application Dash
app = Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": "#f8f9fa", "padding": "20px"},
    children=[
        html.H1(
            children="Analyse des Chansons Spotify",
            style={"textAlign": "center", "color": "#343a40"},
        ),
        html.Div(
            children="""Visualisez la popularité des chansons sur Spotify à travers différents genres.""",
            style={
                "textAlign": "center",
                "color": "#6c757d",
                "fontSize": "18px",
                "margin-bottom": "30px",
            },
        ),
        dcc.Graph(id="popularity-histogram", figure=fig),
        html.Footer(
            style={"textAlign": "center", "margin-top": "50px", "color": "#6c757d"},
            children="2024 Analyse de données avec Dash",
        ),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
