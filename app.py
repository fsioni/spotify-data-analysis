import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

# ====== FONCTION D'ANALYSE AVANCÉE ======
def prepare_advanced_analysis(df):
    """
    Prépare les analyses avancées du dataset Spotify.
    Retourne un dictionnaire contenant les différentes analyses.
    """
    # Extraire l'année de la date de sortie en gérant les différents formats
    def extract_year(date_str):
        try:
            if len(str(date_str)) == 4:  # Format "YYYY"
                return int(date_str)
            else:  # Format "YYYY-MM-DD" ou autre
                return pd.to_datetime(date_str).year
        except:
            return None
    
    df['release_year'] = df['track_album_release_date'].apply(extract_year)
    
    # Caractéristiques audio principales
    audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    # 1. Analyse des tendances temporelles
    yearly_trends = df.groupby('release_year')[audio_features].mean()
    
    # 2. Analyse des "signatures sonores" par genre
    genre_signatures = df.groupby('playlist_genre')[audio_features].agg(['mean', 'std'])
    
    # 3. Détection des morceaux atypiques
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(df[audio_features])
    
    # Calculer les scores Z pour chaque caractéristique
    z_scores = np.abs(stats.zscore(features_normalized))
    outliers = (z_scores > 3).any(axis=1)
    df['is_outlier'] = outliers
    
    # 4. Clustering pour trouver des "types" de chansons
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(features_normalized)
    
    # Caractériser chaque cluster
    cluster_profiles = pd.DataFrame()
    for i in range(n_clusters):
        cluster_data = df[df['cluster'] == i]
        profile = {
            'size': len(cluster_data),
            'main_genre': cluster_data['playlist_genre'].mode().iloc[0],
            'avg_popularity': cluster_data['track_popularity'].mean(),
            'top_artist': cluster_data['track_artist'].mode().iloc[0]
        }
        for feature in audio_features:
            profile[f'avg_{feature}'] = cluster_data[feature].mean()
        
        cluster_profiles[f'Cluster_{i}'] = pd.Series(profile)
    
    # 5. Score de complexité musicale
    df['complexity_score'] = (
        df['instrumentalness'] * 0.3 +
        df['speechiness'] * 0.2 +
        df['tempo'].rank(pct=True) * 0.2 +
        df['loudness'].rank(pct=True) * 0.15 +
        df['energy'] * 0.15
    )
    
    # 6. Analyse des corrélations par genre
    correlation_by_genre = {}
    for genre in df['playlist_genre'].unique():
        genre_data = df[df['playlist_genre'] == genre][audio_features]
        correlation_by_genre[genre] = genre_data.corr()
    
    # 7. PCA pour visualisation
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_normalized)
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]
    
    # 8. Facteur de danse composite
    df['dance_factor'] = (
        df['danceability'] * 0.4 +
        df['energy'] * 0.3 +
        df['tempo'].rank(pct=True) * 0.2 +
        df['valence'] * 0.1
    )
    
    return {
        'yearly_trends': yearly_trends,
        'genre_signatures': genre_signatures,
        'cluster_profiles': cluster_profiles,
        'correlation_by_genre': correlation_by_genre,
        'df_enriched': df
    }

# ====== CONFIGURATION DES STYLES ======
COLORS = {
    'background': '#1DB954',  # Vert Spotify
    'text': '#191414',        # Noir Spotify
    'accent': '#1ED760',      # Vert clair Spotify
    'secondary': '#535353',   # Gris Spotify
    'white': '#FFFFFF'
}

# ====== CHARGEMENT ET PRÉPARATION DES DONNÉES ======
spotify_songs = pd.read_csv("spotify_songs.csv")
analysis_results = prepare_advanced_analysis(spotify_songs)
df_enriched = analysis_results['df_enriched']
yearly_trends = analysis_results['yearly_trends']
cluster_profiles = analysis_results['cluster_profiles']

# ====== CRÉATION DE L'APPLICATION DASH ======
app = Dash(__name__)

app.layout = html.Div(
    style={
        'backgroundColor': COLORS['white'],
        'minHeight': '100vh',
        'fontFamily': 'Helvetica, Arial, sans-serif'
    },
    children=[
        # En-tête
        html.Div(
            style={
                'backgroundColor': COLORS['background'],
                'padding': '20px',
                'color': COLORS['white'],
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            },
            children=[
                html.H1(
                    "Analyse Avancée des Tendances Musicales Spotify",
                    style={'textAlign': 'center', 'marginBottom': '10px'}
                ),
                html.P(
                    "Explorez les tendances et patterns cachés dans les données Spotify",
                    style={'textAlign': 'center', 'fontSize': '18px'}
                )
            ]
        ),

        # Corps principal avec les contrôles
        html.Div(
            style={'padding': '20px'},
            children=[
                # Contrôles
                html.Div([
                    html.Div([
                        html.Label("Type d'analyse:"),
                        dcc.Dropdown(
                            id='analysis-type',
                            options=[
                                {'label': 'Carte des genres (PCA)', 'value': 'pca'},
                                {'label': 'Évolution temporelle', 'value': 'temporal'},
                                {'label': 'Analyse des clusters', 'value': 'clusters'},
                                {'label': 'Complexité musicale', 'value': 'complexity'}
                            ],
                            value='pca'
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("Filtrer par genre:"),
                        dcc.Dropdown(
                            id='genre-filter',
                            options=[{'label': genre, 'value': genre} 
                                    for genre in df_enriched['playlist_genre'].unique()],
                            multi=True
                        )
                    ], style={'width': '30%', 'display': 'inline-block'})
                ], style={'marginBottom': '30px'}),

                # Graphiques
                html.Div([
                    # Graphique principal
                    html.Div(
                        style={
                            'backgroundColor': '#ffffff',
                            'padding': '20px',
                            'borderRadius': '10px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                            'marginBottom': '20px'
                        },
                        children=[dcc.Graph(id='main-graph')]
                    ),

                    # Statistiques des clusters
                    html.Div(
                        style={
                            'backgroundColor': '#ffffff',
                            'padding': '20px',
                            'borderRadius': '10px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                            'marginBottom': '20px'
                        },
                        children=[
                            html.H3("Statistiques des clusters"),
                            dcc.Graph(id='cluster-stats')
                        ]
                    ),

                    # Distribution de la complexité
                    html.Div(
                        style={
                            'backgroundColor': '#ffffff',
                            'padding': '20px',
                            'borderRadius': '10px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        },
                        children=[
                            html.H3("Distribution de la complexité musicale"),
                            dcc.Graph(id='complexity-dist')
                        ]
                    )
                ])
            ]
        ),

        # Pied de page
        html.Footer(
            style={
                'textAlign': 'center',
                'padding': '20px',
                'backgroundColor': COLORS['secondary'],
                'color': COLORS['white'],
                'marginTop': '30px'
            },
            children=["© 2024 Analyse de données Spotify avec Dash"]
        )
    ]
)

# ====== CALLBACKS ======
@app.callback(
    [Output('main-graph', 'figure'),
     Output('cluster-stats', 'figure'),
     Output('complexity-dist', 'figure')],
    [Input('analysis-type', 'value'),
     Input('genre-filter', 'value')]
)
def update_graphs(analysis_type, selected_genres):
    """Mise à jour des graphiques en fonction des sélections de l'utilisateur"""
    
    # Filtrage des données
    if selected_genres:
        df_filtered = df_enriched[df_enriched['playlist_genre'].isin(selected_genres)]
    else:
        df_filtered = df_enriched

    # Création du graphique principal
    if analysis_type == 'pca':
        main_fig = px.scatter(
            df_filtered,
            x='pca_1',
            y='pca_2',
            color='playlist_genre',
            hover_data=['track_name', 'track_artist', 'dance_factor'],
            title="Carte des genres musicaux (PCA)",
            template="plotly_white"
        )
        main_fig.update_traces(marker=dict(size=8))

    elif analysis_type == 'temporal':
        temporal_data = df_filtered.groupby('release_year').agg({
            'danceability': 'mean',
            'energy': 'mean',
            'valence': 'mean',
            'complexity_score': 'mean'
        }).reset_index()
        
        main_fig = px.line(
            temporal_data,
            x='release_year',
            y=['danceability', 'energy', 'valence', 'complexity_score'],
            title="Évolution des caractéristiques musicales",
            template="plotly_white"
        )

    elif analysis_type == 'clusters':
        main_fig = px.scatter(
            df_filtered,
            x='energy',
            y='danceability',
            color='cluster',
            size='track_popularity',
            hover_data=['track_name', 'track_artist'],
            title="Analyse des clusters",
            template="plotly_white"
        )

    else:  # complexity
        main_fig = px.histogram(
            df_filtered,
            x='complexity_score',
            color='playlist_genre',
            title="Distribution de la complexité musicale",
            template="plotly_white",
            marginal="box"
        )

    # Statistiques des clusters
    cluster_stats = go.Figure(data=[
        go.Bar(
            name='Popularité moyenne',
            x=[f'Cluster {i}' for i in range(5)],
            y=df_filtered.groupby('cluster')['track_popularity'].mean()
        ),
        go.Bar(
            name='Complexité moyenne',
            x=[f'Cluster {i}' for i in range(5)],
            y=df_filtered.groupby('cluster')['complexity_score'].mean()
        )
    ])
    cluster_stats.update_layout(
        title="Statistiques par cluster",
        barmode='group',
        template="plotly_white"
    )

    # Distribution de la complexité
    complexity_fig = px.violin(
        df_filtered,
        x='playlist_genre',
        y='complexity_score',
        box=True,
        points='all',
        title="Distribution de la complexité musicale par genre",
        template="plotly_white"
    )
    
    return main_fig, cluster_stats, complexity_fig

# ====== LANCEMENT DE L'APPLICATION ======
if __name__ == '__main__':
    app.run_server(debug=True)