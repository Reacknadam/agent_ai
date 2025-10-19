# 🧠 Jules v2.9 Enhanced  
### L’agent IA qui voit, cherche, code, apprend… et **t’épate**.

---

<p align="center">
  <img src="docs/ia.png" width="90%" alt="Bannière Jules">
</p>

---

## 🌟 30 secondes pour être convaincu

| ✨ Fonctionnalité | 🔍 Ce que ça fait | 🎯 Pourquoi tu vas l’adorer |
|---|---|---|
| **Vision** 📸 | Glisse une **capture d’écran** → Jules génère le **HTML/CSS** correspondant en 5 s. | Plus besoin de coder une maquette à la main. |
| **Recherche sémantique** 🔎 | « *Où est-ce que je gère les tokens JWT ?* » → Jules **comprend la question** et te sort les bons fichiers. | Fini `grep -r` qui te retourne 300 faux positifs. |
| **Apprentissage perso** 🧠 | Tu lui montres **une fois** une compétence → il **s’en souvient** pour toutes les prochaines sessions. | Il devient **vraiment ton assistant**, pas un générique. |
| **Patch intelligent** ✏️ | Il **modifie** le code sans tout ré-écrire (ajoute un log, wrap try/except, etc.). | Historique Git propre + review facile. |
| **Auto-commit & push** 🚀 | Chaque série d’actions → **commit + push** avec un message clair. | Zero oubli, zero « WIP fix lol ». |
| **Interface web** 🎨 | Un **chat Gradio** avec aperçu image, arborescence, logs en temps réel. | Tu peux tout faire **sans toucher le terminal**. |

---

## 📸

| Capture → Code en 1 clic | Recherche sémantique | Patch « try/except » auto |
|---|---|---|
| ![vision-demo](docs/ia.PNG) | 

> Les GIFs sont stockés dans `docs/` (légers, < 2 Mo chacun).  
> Clique dessus pour voir la **vitesse d'exécution réelle**.

---

## 🛠️ Installation ultra-rapide

```bash
# 1. Clone
git clone https://github.com/Reacknadam/agent-ia.git 

# 2. Environnement (optionnel mais recommandé)
python -m venv venv && source venv/bin/activate  # Windows : venv\Scripts\activate

# 3. Installe tout (inclut vision + sémantique)
pip install -r requirements.txt
# requirements.txt contient :
# google-genai, gradio, rich, sentence-transformers, pillow, pyyaml, vulture, ruff

# 4. Clé API (obtenue ici : https://aistudio.google.com/app/apikey)
export GEMINI_API_KEY="ta_clé"
# ou directement dans le code : GEMINI_API_KEY_HARDCODED = "ta_clé"

# 5. Go !
python main.py --web  # Interface web
# ou
python main.py        # CLI stylée
```

---

## 🎯 5 exemples ultra-courts

| Tu dis… | Jules fait… |
|---|---|
| « *Ajoute un log DEBUG après le calcul de total dans maths.py* » | Trouve la ligne, insère, vérifie, commit. |
| « *Maquette cette capture* » (tu glisses l’image) | Génère `generated_ui.html` + `tailwind.css`. |
| « *Où est la validation des tokens ?* » | Retourne fichiers + lignes exactes. |
| « *Apprends que je veux toujours un decorator @timer sur mes fonctions lentes* » | Mémorise → l’appliquera automatiquement plus tard. |
| « *Corrige les imports manquants et applique ruff* » | Fix + lint + commit. |

---

## 📁 Arborescence générée

```
jules-v2/
├── main.py                 # ⬅ tout le code (monofichier = facile à lire)
├── requirements.txt
├── docs/                   # images & gifs
│   ├── jules-banner.png
│   ├── vision-demo.gif
│   ├── semantic-demo.gif
│   └── patch-demo.gif
├── README.md               # ⬅ vous êtes ici
└── .gitignore
```

---

## 🧪 Tests & Qualité

```bash
# Lint + auto-fix
ruff check --fix main.py

# Sécurité
bandit -r main.py

# Benchmarks (intégrés)
python main.py --benchmark  # lance la suite de projets tests
```

---

## 📈 Roadmap ouverte

- [ ] Support **Vue / React** pour le vision-to-code  
- [ ] **Voice input** (Whisper local)  
- [ ] **Plugins pip** extérieurs (`jules-plugin-django`, `jules-plugin-fastapi`)  
- [ ] **Mode équipe** : partage des compétences via serveur central  

---

## 🙋‍♂️ Besoin d’aide ?

- 📬 **Discussions** GitHub → question / idée  
- 🐛 **Issues** → bug / amélioration  
- 📖 **Wiki** → tutoriels pas-à-pas (bientôt)

---

## 🎉 Licence

MIT → fais-en **ce que tu veux**, même dans un projet commercial.

---

**⭐ Star le repo** si Jules t’a fait gagner 5 minutes (ou plus).  
**🔔 Watch** pour être notifié des nouvelles features improbables.