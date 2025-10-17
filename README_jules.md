# Jules AI - Assistant IA Développeur

![Version](https://img.shields.io/badge/version-3.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)

Un assistant IA puissant pour générer des projets de développement complets et professionnels.
Ce projet a ete inspirer d'un projet existant concu par **@Reacknadam** , dans ce fork nous avons fait les meilleurs ameliorations possibles.

## 🚀 Fonctionnalités

- 🤖 **Génération de code IA** avec Google Gemini
- 🏗️ **Création de projets complets** (sites web, APIs, applications)
- 📁 **Structure automatique** dans le dossier `GenAI/`
- 🎯 **Interface Rich** avec progression visuelle
- 🔧 **Gestion des dépendances** automatique

--- 

## 📦 Installation

Cette etape est enormement important , car sans cela l'agent risquerai de ne pas fonctionner .

```bash
# Cloner le repository
https://github.com/Onestepcom00/agent_ai
cd jules_ai
```

Vous allez devoir par la suite installer les dependances .
**A noter** : meme si vous oubliez de le faire lorsque vous allez lancer l'outil , il va automatiquement l'installer .

```python
# Installer les dépendances
pip install -r requirements.txt
```

```python
# Lancer Jules AI
python jules_fork.py
```

## 🎯 Utilisation

bash
### Lancer avec le dossier par défaut

```python
python jules_fork.py
```

### Spécifier un dossier personnalisé

```python
python jules_fork.py --project /chemin/vers/mon/dossier
```

**Commandes disponibles** :
- crée moi un site e-commerce - Génère un site e-commerce complet
- crée une API REST avec FastAPI - Crée une API REST professionnelle

- **aide** - Affiche l'aide complète
- **exit** - Quitte l'application

## 📁 Structure

```text
GenAI/
├── projet-1/
│   ├── src/
│   ├── assets/
│   └── README.md
├── projet-2/
│   ├── frontend/
│   ├── backend/
│   └── package.json
└── .jules_memory.json
```

## 🔧 Dépendances
- Rich : Interface console avancée
- Google GenAI : Accès à Gemini AI
- Pathlib : Gestion des chemins (inclus avec Python)

---
## 🤝 Contribution
Les contributions sont les bienvenues ! N'hésitez pas à :

### Fork le projet

- Créer une branche feature (`git checkout -b feature/AmazingFeature`)
- Commit vos changements (`git commit -m 'Add AmazingFeature'`)
- Push sur la branche (`git push origin feature/AmazingFeature`)
- Ouvrir une Pull Request

---
## 👥 Auteurs
Votre Nom - Développement initial - VotreGitHub

---
## 🙏 Remerciements
Reacknadam Pour le code de base 
Google pour l'API Gemini
Will McGugan pour la librairie Rich
La communauté Python