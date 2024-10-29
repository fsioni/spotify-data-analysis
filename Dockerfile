# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier seulement requirements.txt pour installer les dépendances en cache
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers
COPY . .

# Exposer le port pour Dash
EXPOSE 8050

# Démarrer l'application
CMD ["python", "app.py"]