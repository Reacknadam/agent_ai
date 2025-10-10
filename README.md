

# Jules v2++ (agent_ai)

**Un agent IA "codeur" single-file (Python)** — interface CLI (+ option Web via Gradio), intégration Google Gemini (`google-genai`) si disponible, mémoire locale, historique d'actions, commandes avancées (analyse, audit, refactor, génération de tests, docs...), et simulation/sandbox d'exécution shell.  
Basé uniquement sur le fichier `main.py`.  [oai_citation:1‡GitHub](https://github.com/Reacknadam/agent_ai/raw/7ce76e9fd232673ad4bfed2ec5c3246f13a0e577/main.py)

---

## Fonctionnalitées clés

- Agent interactif CLI capable de traiter des **ACTION:** structurées (ex: `ACTION:read|path|200`, `ACTION:shell|ls -la`, `ACTION:analyze|full`).  
- Intégration optionnelle **Google Gemini** via le package `google-genai` (si installé et clé fournie).  [oai_citation:2‡GitHub](https://github.com/Reacknadam/agent_ai/raw/7ce76e9fd232673ad4bfed2ec5c3246f13a0e577/main.py)  
- UI Web optionnelle via **Gradio** (`--web`) pour interface graphique légère.  [oai_citation:3‡GitHub](https://github.com/Reacknadam/agent_ai/raw/7ce76e9fd232673ad4bfed2ec5c3246f13a0e577/main.py)  
- Mémoire locale des conversations et **historique d'actions** (JSON). Fichier mémoire par défaut : `.jules_memory.json`.  [oai_citation:4‡GitHub](https://github.com/Reacknadam/agent_ai/raw/7ce76e9fd232673ad4bfed2ec5c3246f13a0e577/main.py)  
- Ensemble de commandes « plug-in » implémentées : `progress`, `suggest`, `analyze:full`, `audit:security`, `analyze:static`, `refactor:auto`, `testgen:multi`, `doc:full`, `suggest:update`, `project:status`, `replay`, `shell:sandbox`, lecture/écriture de fichiers, etc. (voir `main.py` pour la liste complète et les comportements).  [oai_citation:5‡GitHub](https://github.com/Reacknadam/agent_ai/raw/7ce76e9fd232673ad4bfed2ec5c3246f13a0e577/main.py)

---

## Prérequis

- Python 3.8+ (ou 3.9+)  
- Recommandé d'utiliser un environnement virtuel (venv/virtualenv)

Dépendances (optionnelles / selon usages) — installer celles dont tu as besoin :

```bash
pip install google-genai gradio rich

Si google-genai ou gradio ne sont pas installés, le script se lance mais certaines fonctions (Gemini ou UI web) seront désactivées ou simulées. Le fichier main.py affiche un avertissement si un package optionnel manque.  ￼

⸻

Installation rapide

git clone https://github.com/Reacknadam/agent_ai.git
cd agent_ai
# (optionnel) créer un virtualenv
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sur Windows

# installer uniquement ce dont tu as besoin, par exemple:
pip install google-genai gradio rich


⸻

Lancer l’agent

Exemple minimal (mode CLI) :

python main.py --project ./mon_projet

Exemple avec UI Web (Gradio) :

python main.py --project ./mon_projet --web

Le header du fichier indique l’usage suivant : python main.py --project ./mon_proj --web. Adapte le chemin --project à ton code/rep cible.  ￼

⸻

Configuration de l’accès Gemini (Google GenAI)

Le fichier contient une constante API_KEY (par défaut définie dans main.py). Ne laisse pas de clés en clair dans le dépôt public.
Options sûres recommandées :
	•	Remplacer la valeur API_KEY dans main.py par ta clé (temporaire) ou mieux : modifier le script pour lire la clé depuis une variable d’environnement (GEMINI_API_KEY) et ne pas committer la clé.
	•	Si tu utilises une clé publique ou committée par erreur, révoque-la et crée-en une nouvelle.

Le script gère l’absence du package ou d’une clé en désactivant les fonctions Gemini et en simulant certaines réponses.  ￼

⸻

Format d’interaction / ACTIONs

Le système utilise une syntaxe ACTION: pour déclencher des commandes structurées depuis l’agent. Format général :

ACTION:<commande>|<arg1>|<arg2>|...

Exemples utiles (d’après main.py) :
	•	Lire un fichier (200 lignes) : ACTION:read|src/app.py|200
	•	Simuler/examiner une commande shell : ACTION:shell|ls -la
	•	Lancer une analyse complète : ACTION:analyze|full
	•	Demander des suggestions : ACTION:suggest|Optimiser le module d'authentification
	•	Générer des tests : ACTION:testgen|src/module.py
	•	Refactorer un fichier : ACTION:refactor|src/module.py|performance
	•	Proposer mise à jour des dépendances : ACTION:suggest|update|requirements.txt
	•	Rejouer une action : ACTION:replay|<log_id>

Beaucoup de commandes vont retourner des blocs ACTION:write|<fichier>| pour proposer d’écrire un fichier ; l’écriture effective du disque doit être confirmée/acceptée selon l’implémentation. Vérifie la sortie avant d’accepter l’écriture.  ￼

⸻

Stockage / fichiers produits
	•	.jules_memory.json — mémoire des conversations / préférences (constante MEMORY_FILE).
	•	Un fichier d’historique d’actions (géré par la dataclass Memory) — contient les ActionLog (id, ts, cmd, args, durée, statut, résumé).
Ces fichiers sont créés localement dans le répertoire de travail du projet.  ￼

⸻

Sécurité & limites importantes
	•	Le script contient une whitelist (SAFE_SHELL_WHITELIST) pour limiter les commandes shell autorisées, mais le code le signale lui-même : la whitelist est une validation basique, pas une sécurité absolue. L’utilisateur reste responsable. Ne lance pas le script sur un système de production sans audit préalable.  ￼
	•	Le fichier contient par défaut une valeur pour API_KEY — ne la partage pas et remplace-la par une méthode sûre (variable d’environnement).  ￼
	•	Plusieurs analyses (audit, static, test coverage, synthèse) sont simulées ou placeholders : pour un audit réel, utilise des outils dédiés (bandit, mypy, flake8, pytest, coverage, etc.) et relie-les au script si tu veux un workflow complet.  ￼

⸻

Suggestions / amélioration rapide
	•	Extraire la clé API du fichier vers une variable d’environnement (GEMINI_API_KEY) ou un fichier .env non committé.
	•	Ajouter un requirements.txt minimal :

google-genai
gradio
rich


	•	Ajouter un mode --dry-run pour les actions dangereuses (écritures fichier / commandes shell).
	•	Documenter précisément le format ACTION: et fournir tests unitaires d’exemples (fixtures).

⸻

Contribution
	1.	Fork → branche → PR.
	2.	Respecte le style PEP8 et ajoute tests pour toute fonctionnalité nouvelle.
	3.	N’ajoute pas de clés API au dépôt.

⸻

Licence

Propose MIT si tu souhaites un repo permissif. (Ajoute LICENSE à la racine.)

⸻

Source

Ce README a été rédigé en s’appuyant uniquement sur l’analyse du fichier main.py présent dans ce dépôt (commit/URL fourni). Pour les détails d’implémentation, consulte directement main.py.  ￼

Si tu veux, je peux maintenant :
- Générer un `requirements.txt` suggéré et un `.gitignore` minimal.  
- Modifier le README pour ajouter des badges (PyPI, license
