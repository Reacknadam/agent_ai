#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jules AI - Agent IA Développeur avec Task Manager
Version: 3.0.0
Auteur: Exauce Stan Malka
GitHub: https://github.com/votre-repo/jules-ai
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

# Vérification et installation des dépendances
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich import box
except ImportError:
    print("📦 Installation des dépendances Rich...")
    os.system('pip install rich')
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich import box

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("📦 Installation de Google GenAI...")
    os.system('pip install google-genai')
    from google import genai
    from google.genai import types

# ----------------------------
# Configuration & Constants
# ----------------------------
APP_NAME = "Jules AI"
APP_VERSION = "3.0.0"
MEMORY_FILE = "data/.jules_memory.json"
MEMORY_MAX = 200 
API_KEY = "PLACER_VOTRE_API_KEY_ICI"
GENAI_DIR = "GenAI"  # Dossier dédié pour tous les projets générés

console = Console()

# ----------------------------
# Task Management System
# ----------------------------

class TaskType(Enum):
    CREATE_DIR = "create_dir"
    CREATE_FILE = "create_file"
    WRITE_CONTENT = "write_content"
    EXECUTE_SHELL = "execute_shell"
    GENERATE_CODE = "generate_code"

@dataclass
class Task:
    id: str
    type: TaskType
    description: str
    target_path: str
    content: Optional[str] = None
    command: Optional[str] = None
    dependencies: List[str] = None
    status: str = "pending"
    result: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class TaskPlan:
    project_name: str
    tasks: List[Task]
    created_at: str
    total_tasks: int = 0
    completed_tasks: int = 0
    
    def __post_init__(self):
        self.total_tasks = len(self.tasks)

class TaskManager:
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.current_plan: Optional[TaskPlan] = None
        self.verbose = True
        
    def log(self, message: str, message_type: str = "info"):
        """Journalisation verbose avec Rich"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if message_type == "success":
            console.print(f"[[bold green]{timestamp}[/]] ✅ {message}")
        elif message_type == "error":
            console.print(f"[[bold red]{timestamp}[/]] ❌ {message}")
        elif message_type == "warning":
            console.print(f"[[bold yellow]{timestamp}[/]] ⚠️  {message}")
        elif message_type == "info":
            console.print(f"[[bold blue]{timestamp}[/]] 🔄 {message}")
        elif message_type == "task":
            console.print(f"[[bold cyan]{timestamp}[/]] 🚀 {message}")
    
    def create_task_plan_from_prompt(self, user_prompt: str) -> TaskPlan:
        """Crée un plan de tâches détaillé avec Gemini"""
        with console.status("[bold green]🤖 Analyse de la demande et création du plan de tâches...") as status:
            time.sleep(1)
            
            prompt = f"""
            Tu es Jules, un architecte logiciel IA expert. Analyse la demande suivante et crée un plan COMPLET de développement.

            DEMANDE UTILISATEUR: {user_prompt}

            CRITÈRES IMPORTANTS:
            1. Structure le projet de manière PROFESSIONNELLE
            2. Pour CHAQUE fichier, fournis du CODE COMPLET et FONCTIONNEL
            3. Organise les tâches par ORDRE LOGIQUE avec dépendances
            4. Sois EXHAUSTIF - inclus tous les fichiers nécessaires
            5. Le code doit être PRÊT À L'EMPLOI et BIEN STRUCTURÉ

            FORMAT JSON OBLIGATOIRE:

            {{
                "project_name": "Nom précis et professionnel du projet",
                "tasks": [
                    {{
                        "id": "task_1",
                        "type": "create_dir|create_file|write_content|execute_shell|generate_code",
                        "description": "Description détaillée de la tâche",
                        "target_path": "chemin/relatif/depuis/GenAI",
                        "content": "CODE COMPLET ET FONCTIONNEL",
                        "command": "commande shell si nécessaire",
                        "dependencies": ["task_id_antérieure"]
                    }}
                ]
            }}

            EXEMPLES DE BONNES PRATIQUES:
            - Pour un site web: HTML/CSS/JS complets avec structure responsive
            - Pour une API: endpoints REST, modèles, middleware, tests
            - Pour une app: composants, routing, state management
            - Toujours inclure un README.md professionnel

            Réponds UNIQUEMENT avec le JSON, sans commentaires.
            """
            
            try:
                if genai is None:
                    raise Exception("Gemini non disponible")
                    
                client = genai.Client(api_key=API_KEY)
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                )
                
                json_str = response.text.strip()
                json_str = re.sub(r'```json\n?', '', json_str)
                json_str = re.sub(r'\n?```', '', json_str)
                
                plan_data = json.loads(json_str)
                
                tasks = []
                for task_data in plan_data["tasks"]:
                    task = Task(
                        id=task_data["id"],
                        type=TaskType(task_data["type"]),
                        description=task_data["description"],
                        target_path=task_data["target_path"],
                        content=task_data.get("content"),
                        command=task_data.get("command"),
                        dependencies=task_data.get("dependencies", [])
                    )
                    tasks.append(task)
                
                self.current_plan = TaskPlan(
                    project_name=plan_data["project_name"],
                    tasks=tasks,
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                self.log(f"Plan créé: {self.current_plan.project_name} avec {len(tasks)} tâches", "success")
                return self.current_plan
                
            except Exception as e:
                self.log(f"Erreur lors de la création du plan: {e}", "error")
                return self._create_fallback_plan(user_prompt)
    
    def _create_fallback_plan(self, description: str) -> TaskPlan:
        """Plan de secours avec structure basique"""
        self.log("Utilisation du plan de secours", "warning")
        
        tasks = [
            Task(id="task_1", type=TaskType.CREATE_DIR, description="Créer la racine du projet", target_path=".", dependencies=[]),
            Task(id="task_2", type=TaskType.CREATE_DIR, description="Créer le dossier source", target_path="src", dependencies=["task_1"]),
            Task(id="task_3", type=TaskType.CREATE_DIR, description="Créer le dossier des assets", target_path="assets", dependencies=["task_1"]),
            Task(id="task_4", type=TaskType.GENERATE_CODE, description="Créer le fichier principal", target_path="src/main.py", dependencies=["task_2"]),
            Task(id="task_5", type=TaskType.GENERATE_CODE, description="Créer le fichier de configuration", target_path="config.py", dependencies=["task_1"]),
            Task(id="task_6", type=TaskType.GENERATE_CODE, description="Créer le README du projet", target_path="README.md", dependencies=["task_1"]),
        ]
        
        self.current_plan = TaskPlan(
            project_name=f"Projet {description[:20]}...",
            tasks=tasks,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        return self.current_plan
    
    def display_plan(self) -> None:
        """Affiche le plan de manière visuelle avec Rich"""
        if not self.current_plan:
            console.print(Panel("❌ Aucun plan disponible", style="red"))
            return
        
        # Tableau des tâches
        table = Table(title=f"📋 Plan de Création: {self.current_plan.project_name}", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Type", style="magenta", width=12)
        table.add_column("Description", style="white", width=40)
        table.add_column("Fichier", style="green", width=25)
        table.add_column("Dépendances", style="yellow", width=15)
        
        for i, task in enumerate(self.current_plan.tasks, 1):
            deps = ", ".join(task.dependencies) if task.dependencies else "Aucune"
            table.add_row(
                str(i),
                task.type.value,
                task.description,
                task.target_path,
                deps
            )
        
        console.print(table)
        
        # Statistiques
        stats_table = Table(box=box.SIMPLE)
        stats_table.add_column("Statistique", style="bold")
        stats_table.add_column("Valeur", style="green")
        
        stats_table.add_row("Total des tâches", str(len(self.current_plan.tasks)))
        stats_table.add_row("Tâches de code", str(len([t for t in self.current_plan.tasks if t.type in [TaskType.WRITE_CONTENT, TaskType.GENERATE_CODE]])))
        stats_table.add_row("Dossiers à créer", str(len([t for t in self.current_plan.tasks if t.type == TaskType.CREATE_DIR])))
        
        console.print(Panel(stats_table, title="📊 Statistiques du Plan"))
    
    def execute_plan(self) -> Dict[str, Any]:
        """Exécute le plan de tâches avec progression visuelle"""
        if not self.current_plan or not self.current_plan.tasks:
            return {"success": False, "message": "Aucun plan de tâches disponible"}
        
        console.print(Panel("🚀 Démarrage de l'exécution du plan", style="bold green"))
        
        results = []
        successful_tasks = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Exécution du plan...", total=len(self.current_plan.tasks))
            
            for task in self.current_plan.tasks:
                # Vérifier les dépendances
                if not self._check_dependencies(task):
                    task.status = "failed"
                    task.result = "Dépendances non satisfaites"
                    results.append(asdict(task))
                    self.log(f"Tâche {task.id} bloquée: dépendances non satisfaites", "error")
                    progress.update(main_task, advance=1)
                    continue
                
                task.status = "running"
                
                # Task individuelle avec progression
                task_desc = f"{task.description[:50]}..." if len(task.description) > 50 else task.description
                task_progress = progress.add_task(f"🔧 {task_desc}", total=1)
                
                try:
                    result = self._execute_task(task)
                    task.status = "completed"
                    task.result = result
                    self.current_plan.completed_tasks += 1
                    successful_tasks += 1
                    
                    self.log(f"Terminé: {task.description}", "success")
                    progress.update(task_progress, completed=1)
                    
                except Exception as e:
                    task.status = "failed"
                    task.result = f"Erreur: {str(e)}"
                    self.log(f"Échec: {task.description} - {str(e)}", "error")
                    progress.update(task_progress, completed=1)
                
                results.append(asdict(task))
                progress.update(main_task, advance=1)
                progress.remove_task(task_progress)
                
                time.sleep(0.1)  # Petite pause pour la lisibilité
        
        success_rate = (successful_tasks / self.current_plan.total_tasks) * 100
        
        # Résumé final
        console.print(Panel(
            f"""🎉 **Exécution Terminée**

📊 **Statistiques:**
• Tâches totales: {self.current_plan.total_tasks}
• Tâches réussies: {successful_tasks}
• Taux de succès: {success_rate:.1f}%

📁 **Projet créé dans:** {self.project_path}""",
            style="green" if success_rate > 80 else "yellow" if success_rate > 50 else "red"
        ))
        
        return {
            "success": True,
            "plan": asdict(self.current_plan),
            "results": results,
            "success_rate": success_rate
        }
    
    def _check_dependencies(self, task: Task) -> bool:
        for dep_id in task.dependencies:
            dep_task = next((t for t in self.current_plan.tasks if t.id == dep_id), None)
            if not dep_task or dep_task.status != "completed":
                return False
        return True
    
    def _execute_task(self, task: Task) -> str:
        try:
            if task.type == TaskType.CREATE_DIR:
                return self._create_directory(task)
            elif task.type == TaskType.CREATE_FILE:
                return self._create_file(task)
            elif task.type == TaskType.WRITE_CONTENT:
                return self._write_content(task)
            elif task.type == TaskType.EXECUTE_SHELL:
                return self._execute_shell(task)
            elif task.type == TaskType.GENERATE_CODE:
                return self._generate_code(task)
            else:
                return f"Type de tâche non supporté: {task.type}"
        except Exception as e:
            raise Exception(f"Erreur d'exécution: {str(e)}")
    
    def _create_directory(self, task: Task) -> str:
        dir_path = self.project_path / task.target_path
        dir_path.mkdir(parents=True, exist_ok=True)
        return f"Dossier créé: {task.target_path}"
    
    def _create_file(self, task: Task) -> str:
        file_path = self.project_path / task.target_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        return f"Fichier créé: {task.target_path}"
    
    def _write_content(self, task: Task) -> str:
        file_path = self.project_path / task.target_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if task.content:
            file_path.write_text(task.content, encoding='utf-8')
            return f"Contenu écrit: {task.target_path} ({len(task.content)} caractères)"
        else:
            file_path.touch()
            return f"Fichier vide créé: {task.target_path}"
    
    def _execute_shell(self, task: Task) -> str:
        if not task.command:
            return "Aucune commande spécifiée"
        
        try:
            result = subprocess.run(
                task.command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.project_path,
                timeout=30
            )
            return f"Commande exécutée: {task.command} (code: {result.returncode})"
        except Exception as e:
            return f"Erreur commande: {str(e)}"
    
    def _generate_code(self, task: Task) -> str:
        try:
            if genai is None:
                return "Gemini non disponible"
            
            client = genai.Client(api_key=API_KEY)
            
            prompt = f"""
            Génère du code PROFESSIONNEL et COMPLET pour: {task.description}
            Fichier: {task.target_path}
            
            Le code doit être:
            - FONCTIONNEL et PRÊT À L'EMPLOI
            - BIEN STRUCTURÉ et COMMENTÉ
            - RESPECTER les bonnes pratiques
            - COMPLET avec toutes les fonctionnalités nécessaires
            
            Réponds UNIQUEMENT avec le code, sans explications.
            """
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            )
            
            generated_code = response.text.strip()
            
            file_path = self.project_path / task.target_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(generated_code, encoding='utf-8')
            
            return f"Code généré: {task.target_path} ({len(generated_code)} caractères)"
            
        except Exception as e:
            return f"Erreur génération: {str(e)}"

# ----------------------------
# Memory System
# ----------------------------

@dataclass
class Memory:
    filepath: Path
    
    def __post_init__(self):
        self.filepath = Path(self.filepath)
        if not self.filepath.exists():
            self._data = {"conversations": [], "prefs": {}}
            self._save()
        else:
            self._load()

    def _load(self):
        try:
            with self.filepath.open("r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception:
            self._data = {"conversations": [], "prefs": {}}

    def _save(self):
        try:
            with self.filepath.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"❌ Erreur sauvegarde mémoire: {e}", style="red")

    def push_conversation(self, user: str, assistant: str):
        self._data.setdefault("conversations", []).append({
            "ts": datetime.now().strftime("%Y%m%dT%H%M%SZ"),
            "user": user,
            "assistant": assistant
        })
        self._data["conversations"] = self._data["conversations"][-MEMORY_MAX:]
        self._save()

    def get_recent(self, n: int = 6) -> List[Dict[str, str]]:
        return self._data.get("conversations", [])[-n:]

# ----------------------------
# JulesAgent Principal
# ----------------------------

class JulesAgent:
    def __init__(self, project_path: str = "."):
        # Toujours utiliser le dossier GenAI
        base_path = Path(project_path)
        self.project_path = base_path / GENAI_DIR
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        self.memory = Memory(self.project_path / MEMORY_FILE)
        self.task_manager = TaskManager(self.project_path)
        self.gemini = self._init_gemini()
        self.waiting_confirmation = False
        
        self._display_welcome()
    
    def _display_welcome(self):
        """Affiche l'écran d'accueil avec Rich"""
        welcome_text = f"""
# 🚀 {APP_NAME} v{APP_VERSION}

**Assistant IA Développeur Professionnel**

📁 **Dossier de travail:** `{self.project_path}`
🤖 **IA:** Google Gemini 2.0 Flash
🔧 **Fonctionnalités:** Génération de code, gestion de projets, architecture logicielle

## Commandes disponibles:
- `crée moi [projet]` - Génère un projet complet
- `aide` - Affiche l'aide
- `exit` - Quitte l'application

## Exemples:
- `crée moi un site e-commerce`
- `crée une API REST avec FastAPI`
- `développe une application React`
"""
        
        console.print(Panel(
            Markdown(welcome_text),
            style="bold blue",
            title="🎯 Bienvenue dans Jules AI",
            subtitle="Votre assistant développement IA"
        ))
    
    def _init_gemini(self):
        """Initialise le client Gemini"""
        if genai is None or not API_KEY:
            console.print("❌ Gemini non disponible - certaines fonctionnalités seront limitées", style="red")
            return None
        
        try:
            client = genai.Client(api_key=API_KEY)
            console.print("✅ Connexion à Gemini établie", style="green")
            return client
        except Exception as e:
            console.print(f"❌ Erreur connexion Gemini: {e}", style="red")
            return None
    
    def ask_gemini(self, user_input: str) -> str:
        """Pose une question à Gemini avec contexte"""
        if not self.gemini:
            return "🔧 Mode local. Pour la création de projets, utilisez: 'crée moi [description]'"
        
        recent = self.memory.get_recent(3)
        context = "\n".join([f"User: {r['user']}\nAssistant: {r['assistant']}" for r in recent])
        
        system_prompt = """Tu es Jules, un assistant IA développeur expert. 

Pour les demandes de CRÉATION (site web, application, API, script, etc.), réponds EXACTEMENT avec:
ACTION:CREATE|description_détaillée_du_projet

Pour les questions générales ou l'aide, réponds normalement.

Sois concis, professionnel et efficace."""
        
        full_prompt = f"{system_prompt}\n\nContexte récent:\n{context}\n\nUtilisateur: {user_input}"
        
        try:
            response = self.gemini.models.generate_content(
                model="gemini-2.0-flash",
                contents=[types.Content(role="user", parts=[types.Part(text=full_prompt)])],
            )
            return response.text
        except Exception as e:
            return f"❌ Erreur de communication avec Gemini: {str(e)}"
    
    def handle_create_project(self, description: str) -> str:
        """Gère la création d'un projet complet"""
        console.print(Panel(f"🎯 Analyse de la demande: {description}", style="bold cyan"))
        
        plan = self.task_manager.create_task_plan_from_prompt(description)
        
        if not plan.tasks:
            return "❌ Impossible de créer un plan pour ce projet."
        
        # Afficher le plan visuellement
        self.task_manager.display_plan()
        
        console.print(Panel(
            "🚀 **Voulez-vous exécuter ce plan?**\n\n"
            "Tapez 'oui' pour lancer la création\n"
            "Tapez 'non' pour annuler",
            style="yellow"
        ))
        
        self.waiting_confirmation = True
        return ""
    
    def execute_creation_plan(self) -> str:
        """Exécute le plan de création en cours"""
        if not self.task_manager.current_plan:
            return "❌ Aucun plan de création en attente."
        
        results = self.task_manager.execute_plan()
        
        # Afficher le résumé final
        success_rate = results["success_rate"]
        
        if success_rate > 80:
            style = "bold green"
            emoji = "🎉"
        elif success_rate > 50:
            style = "bold yellow" 
            emoji = "⚠️"
        else:
            style = "bold red"
            emoji = "❌"
        
        summary = f"""
{emoji} **Projet Créé: {self.task_manager.current_plan.project_name}**

📊 **Résultats:**
• Tâches totales: {self.task_manager.current_plan.total_tasks}
• Tâches réussies: {self.task_manager.current_plan.completed_tasks}
• Taux de succès: {success_rate:.1f}%

📁 **Emplacement:** `{self.project_path}`

💡 **Prochaines étapes:**
1. Naviguez vers le dossier du projet
2. Examinez les fichiers générés
3. Lancez votre projet!
"""
        
        console.print(Panel(summary, style=style))
        
        self.waiting_confirmation = False
        return ""
    
    def process_input(self, user_input: str) -> str:
        """Traite l'entrée utilisateur"""
        user_input = user_input.strip()
        
        if not user_input:
            return ""
            
        # Commandes spéciales
        if user_input.lower() == 'aide':
            return self._show_help()
        elif user_input.lower() == 'clear':
            console.clear()
            return ""
        
        # Gestion de la confirmation
        if self.waiting_confirmation:
            if user_input.lower() in ['oui', 'yes', 'y', 'o', 'ok']:
                return self.execute_creation_plan()
            elif user_input.lower() in ['non', 'no', 'n']:
                self.waiting_confirmation = False
                self.task_manager.current_plan = None
                return "❌ Création annulée."
        
        # Traitement normal
        gemini_response = self.ask_gemini(user_input)
        
        # Détection des demandes de création
        if "ACTION:CREATE" in gemini_response or any(word in user_input.lower() for word in ['crée', 'create', 'construis', 'build', 'génère', 'développe']):
            if "|" in gemini_response:
                description = gemini_response.split("|", 1)[1].strip()
            else:
                description = user_input
            
            return self.handle_create_project(description)
        else:
            # Affichage formaté de la réponse
            console.print(Panel(
                gemini_response,
                title="🤖 Réponse de Jules",
                style="blue"
            ))
            return gemini_response
    
    def _show_help(self) -> str:
        """Affiche l'aide"""
        help_text = """
## 📖 Aide de Jules AI

### Commandes principales:
- `crée moi [description]` - Crée un projet complet
- `aide` - Affiche cette aide
- `clear` - Nettoie l'écran
- `exit` - Quitte l'application

### Exemples de projets:
- `crée moi un site e-commerce avec React et Node.js`
- `crée une API REST avec authentification JWT`
- `développe un script Python pour analyser des données`
- `crée un portfolio personnel responsive`

### Fonctionnalités:
- 🏗️  Génération de code complet et fonctionnel
- 📁 Structure automatique des projets
- 🔧 Gestion des dépendances et configurations
- 🎯 Architecture professionnelle
- 📊 Suivi visuel de la progression

**Tous les projets sont créés dans le dossier `GenAI/`**
"""
        console.print(Panel(Markdown(help_text), style="green"))
        return ""

# ----------------------------
# Interface Console principale
# ----------------------------

def main_console():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} - Assistant IA Développeur")
    parser.add_argument("--project", "-p", default=".", help="Répertoire de base (GenAI sera créé dedans)")
    args = parser.parse_args()
    
    agent = JulesAgent(project_path=args.project)
    
    try:
        while True:
            try:
                user_input = console.input("\n[bold cyan]💬 Vous > [/]").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in {'exit', 'quit', 'q'}:
                    console.print(Panel("👋 À bientôt !", style="bold green"))
                    break
                
                # Traiter la demande
                response = agent.process_input(user_input)
                
                # Sauvegarder dans la mémoire (sauf pour les commandes spéciales)
                if user_input not in ['aide', 'clear'] and response:
                    agent.memory.push_conversation(user_input, response)
                
            except KeyboardInterrupt:
                console.print("\n\n🛑 Interruption - Au revoir!", style="yellow")
                break
            except Exception as e:
                console.print(f"\n💥 Erreur: {str(e)}", style="red")
                
    except Exception as e:
        console.print(f"\n💥 Erreur critique: {str(e)}", style="bold red")
        sys.exit(1)

if __name__ == "__main__":
    main_console()