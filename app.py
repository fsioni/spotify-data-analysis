import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

# ====== CONFIGURATION DES STYLES ======
COLORS = {
    'background': '#1DB954',  # Vert Spotify
    'text': '#191414',        # Noir Spotify
    'accent': '#1ED760',      # Vert clair Spotify
    'secondary': '#535353',   # Gris Spotify
    'white': '#FFFFFF'
}

# ====== LISTE DES CARACTÉRISTIQUES AUDIO ======
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# ====== PARAMÈTRES DE CLUSTERING ======
n_clusters = 5  # Nombre de clusters défini globalement

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

    # 1. Analyse des tendances temporelles
    yearly_trends = df.groupby('release_year')[audio_features].mean()

    # 2. Analyse des "signatures sonores" par genre
    genre_signatures = df.groupby('playlist_genre')[audio_features].agg(['mean', 'std'])

    # 3. Analyse des corrélations par genre
    correlation_by_genre = {}
    for genre in df['playlist_genre'].unique():
        genre_data = df[df['playlist_genre'] == genre][audio_features]
        correlation = genre_data.corr()
        correlation_by_genre[genre] = correlation

    # 4. Clustering pour trouver des "types" de chansons
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(df[audio_features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(features_normalized)
    df['cluster'] = df['cluster'].astype(str)  # Convertir en string pour catégories

    # 5. Score de complexité musicale
    df['complexity_score'] = (
        df['instrumentalness'] * 0.3 +
        df['speechiness'] * 0.2 +
        df['tempo'].rank(pct=True) * 0.2 +
        df['loudness'].rank(pct=True) * 0.15 +
        df['energy'] * 0.15
    )

    # 6. Normalisation de la Popularité
    df['popularity_scaled'] = df['track_popularity'] / 100  # Popularité entre 0 et 1

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

    # 9. Caractérisation des clusters
    cluster_profiles_list = []
    for i in range(n_clusters):
        cluster_id = str(i)
        cluster_data = df[df['cluster'] == cluster_id]
        if not cluster_data.empty:
            profile = {
                'size': len(cluster_data),
                'main_genre': cluster_data['playlist_genre'].mode().iloc[0] if not cluster_data['playlist_genre'].mode().empty else 'N/A',
                'avg_popularity': cluster_data['popularity_scaled'].mean(),
                'top_artist': cluster_data['track_artist'].mode().iloc[0] if not cluster_data['track_artist'].mode().empty else 'N/A'
            }
            for feature in audio_features:
                profile[f'avg_{feature}'] = cluster_data[feature].mean()
        else:
            # Gérer le cas où un cluster pourrait être vide
            profile = {
                'size': 0,
                'main_genre': 'N/A',
                'avg_popularity': 0,
                'top_artist': 'N/A'
            }
            for feature in audio_features:
                profile[f'avg_{feature}'] = 0
        cluster_profiles_list.append(pd.Series(profile, name=f'Cluster_{i}'))

    # Concaténer toutes les séries en un DataFrame
    cluster_profiles = pd.concat(cluster_profiles_list, axis=1).transpose()

    return {
        'yearly_trends': yearly_trends,
        'genre_signatures': genre_signatures,
        'cluster_profiles': cluster_profiles,
        'correlation_by_genre': correlation_by_genre,
        'df_enriched': df
    }

# ====== CHARGEMENT ET PRÉPARATION DES DONNÉES ======
spotify_songs = pd.read_csv("spotify_songs.csv")
analysis_results = prepare_advanced_analysis(spotify_songs)
df_enriched = analysis_results['df_enriched']
yearly_trends = analysis_results['yearly_trends']
cluster_profiles = analysis_results['cluster_profiles']
correlation_by_genre = analysis_results['correlation_by_genre']

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

                    # Graphique de corrélation
                    html.Div(
                        style={
                            'backgroundColor': '#ffffff',
                            'padding': '20px',
                            'borderRadius': '10px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        },
                        children=[
                            html.H3("Graphique de corrélation par genre"),
                            dcc.Graph(id='correlation-graph')
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
     Output('correlation-graph', 'figure')],
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
        # Définir une palette de couleurs distinctes
        palette = pc.qualitative.Plotly  # Vous pouvez choisir d'autres palettes comme 'Set1', 'Dark2', etc.

        # Ajuster la palette si nécessaire
        if len(palette) < n_clusters:
            palette = palette * (n_clusters // len(palette) + 1)

        cluster_colors = {str(i): palette[i] for i in range(n_clusters)}

        main_fig = px.scatter(
            df_filtered,
            x='energy',
            y='danceability',
            color='cluster',  # Assurez-vous que 'cluster' est catégorique
            color_discrete_map=cluster_colors,  # Appliquer la palette de couleurs
            size='popularity_scaled',
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
        # Barre pour la popularité moyenne
        go.Bar(
            name='Popularité moyenne',
            x=[f'Cluster {i}' for i in range(n_clusters)],
            y=df_filtered.groupby('cluster')['popularity_scaled'].mean(),
            marker_color=COLORS['accent']
        ),
        # Barre pour la complexité moyenne
        go.Bar(
            name='Complexité moyenne',
            x=[f'Cluster {i}' for i in range(n_clusters)],
            y=df_filtered.groupby('cluster')['complexity_score'].mean(),
            marker_color=COLORS['secondary']
        )
    ])
    cluster_stats.update_layout(
        title="Statistiques par cluster",
        barmode='group',
        xaxis_title="Clusters",
        yaxis_title="Valeur Moyenne",
        template="plotly_white",
        legend_title="Métriques",
        yaxis=dict(range=[0, 1])  # Fixer l'échelle entre 0 et 1
    )

    # Graphique de corrélation
    if selected_genres and len(selected_genres) == 1:
        # Si un seul genre est sélectionné, afficher sa corrélation
        genre = selected_genres[0]
        corr_matrix = correlation_by_genre.get(genre)
        if corr_matrix is not None:
            correlation_fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title=f"Corrélation des caractéristiques audio pour le genre: {genre}",
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
    else:
        # Si plusieurs genres ou aucun sont sélectionnés, afficher une corrélation moyenne
        if selected_genres:
            corr_matrices = [correlation_by_genre[genre] for genre in selected_genres if genre in correlation_by_genre]
            if corr_matrices:
                # Calculer la moyenne des matrices de corrélation
                avg_corr = sum(corr_matrices) / len(corr_matrices)
            else:
                avg_corr = pd.DataFrame()
        else:
            # Utiliser toutes les données si aucun genre n'est sélectionné
            avg_corr = df_filtered[audio_features].corr()
        
        if not avg_corr.empty:
            correlation_fig = px.imshow(
                avg_corr,
                text_auto=True,
                aspect="auto",
                title="Corrélation des caractéristiques audio (Moyenne des genres sélectionnés)",
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
        else:
            correlation_fig = go.Figure()
            correlation_fig.update_layout(
                title="Aucune donnée disponible pour les genres sélectionnés.",
                template="plotly_white"
            )

    return main_fig, cluster_stats, correlation_fig

# ====== LANCEMENT DE L'APPLICATION ======
if __name__ == '__main__':
    app.run_server(debug=True)
