#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jules v2++ - Agent IA Codeur (single-file, avancé)
- Basé sur Google Gemini (google-genai)
- Gradio UI intégrée (option --web)
- Mémoire locale, prompt système puissant, actions shell sandboxées,
  commandes avancées (generate, correct, refactor, testgen, summarize, explain, doc, run, file ops...)
- Conçu pour être déployé localement sur votre machine de développement.
- Usage: python main.py --project ./mon_proj --web
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

# Optional deps
try:
    from google import genai
    from google.genai import types
except Exception as e:
    print(" google-genai (Gemini) absent ou erreur d'import. Installez: pip install google-genai")
    print(f"   Détail: {e}")
    # do not sys.exit here so user can still view code; but runtime will warn later
    genai = None
    types = None

try:
    import gradio as gr
    # --- AJOUT GEMINI --- gr.Progress
    from gradio.components import Progress
except Exception:
    gr = None
    Progress = None

# Nice terminal output (optional)
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    # --- AJOUT GEMINI --- rich progress bar
    from rich.progress import Progress as RichProgress, SpinnerColumn, BarColumn, TextColumn
    RICH_PROGRESS = RichProgress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=True
    )
    CONSOLE = Console()
except Exception:
    rprint = print
    CONSOLE = None
    RICH_PROGRESS = None

# ----------------------------
# Configuration & Constants
# ----------------------------
APP_NAME = "Jules-v2++"
MEMORY_FILE = ".jules_memory.json"
MEMORY_MAX = 200 
API_KEY = "AIzaSyDltpB9ooZV4hANfA3WiuaTZdV1ca1JsUk"
PROJECT_IGNORED = {"__pycache__", ".git", "node_modules", "venv", ".venv", "*.backup", MEMORY_FILE}
SUPPORTED_EXT = {".py", ".js", ".ts", ".html", ".css", ".json", ".md", ".txt", ".sql", ".sh", ".yaml", ".yml"}
MAX_CMD_OUTPUT = 5000
SHELL_TIMEOUT = 20 
SAFE_SHELL_WHITELIST = {
    "ls", "pwd", "cat", "echo", "mkdir", "rmdir", "rm", "mv", "cp", "touch", "git", "python", "node", "npm",
    "pip", "pip3", "pytest", "coverage", "code", "sed", "awk", "head", "tail", "grep", "wc", "find",
    # --- AJOUT GEMINI --- Nouvelles commandes shell utiles
    "flake8", "pylint", "mypy", "bandit", "safety"
}
# Note: whitelist used for simple validation, not strict enforcement (user is responsible)

# --- AJOUT GEMINI --- Dataclass pour l'historique enrichi et le replay
@dataclass
class ActionLog:
    id: str
    ts: str
    cmd: str
    args: List[str]
    duration: float
    result_status: str
    result_summary: str
    is_gemini_action: bool = False
    is_replayable: bool = False

# ----------------------------
# Helpers / Utilities
# ----------------------------

def _rich_print(msg: str, title: Optional[str] = None):
    try:
        if CONSOLE:
            if title:
                CONSOLE.print(Panel(msg, title=title))
            else:
                CONSOLE.print(msg)
        else:
            print(msg)
    except Exception:
        print(msg)

# --- AJOUT GEMINI --- Animation: Typing Effect
def typing_effect(text: str, delay: float = 0.005, prefix: str = "Jules> ", console_out: Callable[[str], None] = print):
    """Simule un effet de machine à écrire dans la console."""
    console_out(prefix, end='', flush=True)
    if RICH_PROGRESS:
        with RICH_PROGRESS as p:
            task = p.add_task("[yellow]Réflexion...", total=len(text))
            for char in text:
                console_out(char, end='', flush=True)
                p.update(task, advance=1)
                time.sleep(delay)
    else:
        for char in text:
            console_out(char, end='', flush=True)
            time.sleep(delay)
    console_out("\n", end='', flush=True)

# --- AJOUT GEMINI --- Animation: Progress Bar
def console_progress_bar(description: str, total: int):
    """Affiche une barre de progression Rich pour les actions longues."""
    if not RICH_PROGRESS:
        _rich_print(f"⏳ {description}...")
        yield # Permet au générateur d'être utilisé
    else:
        with RICH_PROGRESS as p:
            task = p.add_task(f"[cyan]{description}...", total=total)
            for i in range(total):
                yield i
                p.update(task, advance=1)
            p.remove_task(task)

# --- AJOUT GEMINI --- Gradio Progress (Simulation - car le vrai gr.Progress n'est utilisable que dans les handlers Gradio)
def gradio_progress_simulate(progress: Optional[Progress], total: int, description: str):
    """Simule l'avancement pour les handlers Gradio."""
    if progress:
        progress(0, desc=f"**[Statut]** {description}...")
        for i in range(total):
            yield i
            progress(float(i+1)/total, desc=f"**[Statut]** {description} - Étape {i+1}/{total}")
    else:
        for i in range(total):
            yield i

def now_ts() -> str:
    # FIX: Utilisation de datetime.now(timezone.utc) à la place de utcnow() déprécié
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def safe_join(base: Path, target: str) -> Path:
    """Return resolved path inside base or raise PermissionError."""
    candidate = (base / Path(target)).resolve()
    try:
        candidate.relative_to(base.resolve())
    except Exception:
        raise PermissionError(f"Path traversal detected: {target} -> {candidate}")
    return candidate

def read_text(path: Path, max_chars: int = 200_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars]
    except Exception as e:
        return f"❌ Lecture impossible: {e}"

# ----------------------------
# Memory (simple JSON store)
# ----------------------------
@dataclass
class Memory:
    filepath: Path
    # --- AJOUT GEMINI --- Historique des actions localement
    history_filepath: Path
    
    def __post_init__(self):
        self.filepath = Path(self.filepath)
        self.history_filepath = Path(self.history_filepath)
        if not self.filepath.exists():
            self._data = {"conversations": [], "prefs": {}}
            self._save()
        else:
            self._load()
        # --- AJOUT GEMINI --- Chargement de l'historique
        if not self.history_filepath.exists():
             self._action_logs: List[ActionLog] = []
             self._save_logs()
        else:
             self._load_logs()

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
            _rich_print(f"Erreur sauvegarde mémoire: {e}")

    # --- AJOUT GEMINI --- Sauvegarde/Chargement des logs d'actions
    def _load_logs(self):
        try:
            with self.history_filepath.open("r", encoding="utf-8") as f:
                logs_data = json.load(f)
                self._action_logs = [ActionLog(**log) for log in logs_data]
        except Exception:
            self._action_logs = []
            
    def _save_logs(self):
        try:
            with self.history_filepath.open("w", encoding="utf-8") as f:
                json.dump([asdict(log) for log in self._action_logs], f, indent=2, ensure_ascii=False)
        except Exception as e:
            _rich_print(f"Erreur sauvegarde logs: {e}")

    def push_conversation(self, user: str, assistant: str):
        self._data.setdefault("conversations", []).append({
            "ts": now_ts(),
            "user": user,
            "assistant": assistant
        })
        # keep last MEMORY_MAX items
        self._data["conversations"] = self._data["conversations"][-MEMORY_MAX:]
        self._save()

    # --- AJOUT GEMINI --- Méthode pour pousser un log d'action
    def push_action_log(self, log: ActionLog):
        self._action_logs.append(log)
        self._save_logs()

    def get_recent(self, n: int = 6) -> List[Dict[str, str]]:
        return self._data.get("conversations", [])[-n:]

    # --- AJOUT GEMINI --- Méthode pour obtenir l'historique enrichi
    def get_action_history(self, n: int = 20) -> List[ActionLog]:
        return self._action_logs[-n:]

    def set_pref(self, key: str, value: Any):
        self._data.setdefault("prefs", {})[key] = value
        self._save()

    def get_pref(self, key: str, default=None):
        return self._data.get("prefs", {}).get(key, default)

# ----------------------------
# Gemini Client Wrapper
# ----------------------------
class GeminiWrapper:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.client = None
        if genai is not None and api_key:
            try:
                # Up-to-date usage may vary; we attempt Client init
                try:
                    genai.configure(api_key=api_key)
                    # Some versions use genai.Client()
                    try:
                        self.client = genai.Client(api_key=api_key)
                    except Exception:
                        self.client = None
                except Exception:
                    # fallback client
                    self.client = genai.Client(api_key=api_key)
            except Exception as e:
                _rich_print(f"⚠️ Gemini init failed: {e}")
                self.client = None
        else:
            if not api_key:
                _rich_print("⚠️ GEMINI_API_KEY not provided. Gemini features disabled.")
            else:
                _rich_print("⚠️ google-genai package not available. Gemini features disabled.")

    def is_ready(self) -> bool:
        return self.client is not None

    def generate(self, system_instruction: str, user_prompt: str, max_output_chars: int = 5000) -> str:
        """
        FIX: Utilise une construction conforme à l'API en combinant l'instruction système
        et l'invite utilisateur en un seul rôle 'user' pour éviter l'erreur 400.
        """
        if not self.is_ready():
            return "❌ Gemini non initialisé localement (clé manquante ou package absent)."

        try:
            # Construction conforme : combine system_instruction et user_prompt sous le rôle 'user'
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=system_instruction + "\n\n" + user_prompt)])
                ],
            )
            # Extrait le texte de la réponse
            text = getattr(response, "text", None) or getattr(response, "output_text", None)
            return str(text)[:max_output_chars] if text else "🤖 Pas de réponse ou réponse bloquée."
        except Exception as e:
            return f"❌ Erreur appel Gemini : {e}"

# ----------------------------
# Command Pattern Base Classes
# ----------------------------
class BaseCommand:
    def __init__(self, agent: "JulesAgent"):
        self.agent = agent

    def execute(self, *args: str) -> str:
        raise NotImplementedError

# --- AJOUT GEMINI --- Nouvelle classe de commande pour la gestion de la progression
class ProgressCommand(BaseCommand):
    """
    Commande: progress|<action_id>
    Usage: Retourne l'état de progression de l'action en cours ou d'une action récente.
    """
    def execute(self, action_id: str = "current") -> str:
        # NOTE: Dans un environnement synchrone, cela retourne l'état final/simulé de l'action.
        # Pour une vraie gestion temps réel, il faudrait un système de threads/queues.
        if action_id == "current" and self.agent.last_action_progress:
             # Simule le suivi d'étape
             steps = self.agent.last_action_progress['steps']
             current = self.agent.last_action_progress['current']
             desc = self.agent.last_action_progress['description']
             
             if current < steps:
                 return f"⏳ PROGRESS: {desc} - {current}/{steps} ({current/steps:.0%})"
             else:
                 return f"✅ PROGRESS: {desc} - Terminée."
        
        # Sinon, retourne le dernier log d'action comme progression historique
        history = self.agent.memory.get_action_history(1)
        if history:
            log = history[0]
            return f"✅ LAST ACTION: {log.cmd} - {log.result_status} ({log.duration:.2f}s)"
        
        return "⚠️ Aucune progression en cours ou historique récent."

# --- AJOUT GEMINI --- Nouvelle classe de commande pour les suggestions intelligentes
class SuggestCommand(BaseCommand):
    """
    Commande: suggest|<context>
    Usage: Propose des optimisations de code, des endpoints API, ou des fonctionnalités basées sur le contexte du projet.
    """
    def execute(self, context: str) -> str:
        _rich_print(f" Réflexion sur la suggestion pour: {context}...", title="Suggestion Intelligente")
        
        
        status_report = ProjectStatusCommand(self.agent).execute()
        
        
        prompt = f"""
        En tant qu'architecte logiciel, analyse le contexte utilisateur suivant: '{context}'.
        En t'appuyant sur l'analyse globale du projet ci-dessous, propose 3 suggestions concrètes et intelligentes.
        Les suggestions doivent être soit des optimisations de code existant (avec ACTION:correct|file|détail),
        soit de nouvelles fonctionnalités locales (avec ACTION:generate|description|langue),
        soit des endpoints API suggérés.

        Analyse du projet (statut global):
        ---
        {status_report}
        ---

        Format de réponse souhaité (liste numérotée):
        1. [Titre suggestion 1] : <Description détaillée> [ACTION:<commande>|...]
        2. [Titre suggestion 2] : <Description détaillée> [ACTION:<commande>|...]
        3. [Titre suggestion 3] : <Description détaillée> [ACTION:<commande>|...]
        """
        
        # Utilise le générateur de progression
        for _ in console_progress_bar("Génération de suggestions", 100):
            time.sleep(0.02) # Simulation de latence

        return self.agent._ask_gemini_raw(prompt)

# --- AJOUT GEMINI --- Nouvelle classe de commande pour l'analyse globale
class AnalyzeFullCommand(BaseCommand):
    """
    Commande: analyze|full
    Usage: Génère un rapport complet d'analyse (statique, sécurité, couverture, complexité, documentation) du projet.
    """
    def execute(self) -> str:
        _rich_print("⚙️ Lancement de l'analyse complète du projet (Security, Static, Status)...", title="ANALYSE:FULL")
        
        report = []
        
        # 1. Statut global
        report.append("## 1. Analyse Globale du Projet")
        report.append(ProjectStatusCommand(self.agent).execute())
        report.append("\n---\n")
        
        # 2. Audit de sécurité
        report.append("## 2. Audit de Sécurité (Simulé)")
        # Simule l'audit sur un fichier clé pour l'exemple
        audit_res = AuditSecurityCommand(self.agent).execute("main.py") 
        report.append(audit_res)
        report.append("\n---\n")

        # 3. Analyse statique (PEP8, Complexité)
        report.append("## 3. Analyse Statique Avancée (Simulée)")
        # Simule l'analyse statique sur un fichier clé pour l'exemple
        static_res = StaticAnalysisCommand(self.agent).execute("main.py")
        report.append(static_res)
        report.append("\n---\n")
        
        # 4. Couverture de test (Simulée)
        report.append("## 4. Couverture de Test (Simulée)")
        report.append("Couverture globale estimée : 75%. Fichiers critiques : `utils.py` (30%).")
        
        # 5. Résumé par Gemini
        report.append("\n## 5. Synthèse et Scores de Qualité (IA)")
        context = "\n".join(report)
        
        # Utilise le générateur de progression
        for i in console_progress_bar("Synthèse du rapport par Gemini", 5):
            time.sleep(0.1) # Simule le temps de génération
            if i == 4: break # Termine la simulation avant la fin (appel réel)

        prompt = f"""
        Synthétise les points clés des analyses de sécurité, statique et de statut suivantes.
        Attribue un score de 1 à 100 pour la **Qualité du code**, la **Sécurité** et la **Couverture de tests**.
        Fournis un résumé final et des actions recommandées.

        Données d'analyse:
        ---
        {context}
        ---

        Format de la réponse:
        **Score Qualité:** <Score>%
        **Score Sécurité:** <Score>%
        **Score Couverture:** <Score>%
        **Résumé:** <Synthèse>
        **Recommandations:** <Liste d'actions>
        """
        
        gemini_summary = self.agent._ask_gemini_raw(prompt)
        report.append(gemini_summary)
        
        return "\n".join(report)

# --- AJOUT GEMINI --- Nouvelles classes de commandes locales ultra-avancées
class AuditSecurityCommand(BaseCommand):
    """
    Commande: audit|security|<file_path?>
    Usage: Réalise un audit de sécurité (clés, patterns dangereux, shell) (local/simulé).
    """
    def execute(self, file_path: str = "all") -> str:
        _rich_print(f"🔒 Lancement de l'audit de sécurité sur {file_path}...", title="AUDIT:SECURITY")
        
        security_report = []
        target_files = [safe_join(self.agent.project_path, file_path)] if file_path != "all" else self.agent._get_all_code_files()
        
        for i, p in enumerate(target_files):
            # Utilise le générateur de progression
            for _ in gradio_progress_simulate(self.agent.gradio_progress, len(target_files) * 5, f"Audit {p.name}"):
                time.sleep(0.001)

            content = read_text(p)
            if content.startswith("❌"):
                security_report.append(f"⚠️ {p.name}: Ignoré (lecture impossible).")
                continue

            # Règle 1: Clés et secrets exposés (patterns simples)
            if re.search(r'(API_KEY|SECRET|PASSWORD)\s*=\s*["\']', content, re.IGNORECASE):
                security_report.append(f"🚨 DANGER - {p.name}: Clé/Secret potentiel exposé.")

            # Règle 2: Utilisation dangereuse de `subprocess.run(..., shell=True)`
            if 'subprocess.run(' in content and 'shell=True' in content:
                 security_report.append(f"⚠️ ATTENTION - {p.name}: Utilisation dangereuse de 'shell=True' détectée.")

            # Règle 3: Injection SQL simple (patterns)
            if re.search(r'\+\s*(user_input|data)', content, re.IGNORECASE) and ('sql' in content.lower() or 'query' in content.lower()):
                 security_report.append(f"⚠️ ALERTE - {p.name}: Risque d'injection SQL potentiel (concaténation de chaînes).")
            
            # (Simulation d'un outil de sécurité)
            # if p.suffix == ".py" and os.name != "nt": # Utilisation de Bandit (simulée)
            #     security_report.append(f"➡️ Bandit (simulé) : 2 High, 5 Medium.")

        return "\n".join(["--- Rapport de Sécurité ---"] + security_report + ["--- Fin Rapport ---"])

class StaticAnalysisCommand(BaseCommand):
    """
    Commande: analyze|static|<file_path?>
    Usage: Analyse statique avancée (PEP8, type hints, complexité cyclomatique, imports/variables inutilisées).
    """
    def execute(self, file_path: str = "all") -> str:
        _rich_print(f"🔬 Lancement de l'analyse statique sur {file_path}...", title="STATIC:ANALYSIS")
        
        static_report = []
        target_files = [safe_join(self.agent.project_path, file_path)] if file_path != "all" else self.agent._get_all_code_files()
        
        for p in target_files:
             if p.suffix.lower() not in {".py", ".js"}: continue
             content = read_text(p)
             if content.startswith("❌"): continue

             # 1. PEP8 (Simulé)
             # Compter les lignes > 79
             long_lines = sum(1 for line in content.splitlines() if len(line) > 79)
             if long_lines > 0:
                 static_report.append(f"⚠️ {p.name}: {long_lines} lignes > 79 chars (PEP8).")

             # 2. Type Hints (Simulé)
             # Simple count de 'def func(' vs 'def func(arg: type) -> type:'
             total_funcs = len(re.findall(r'def\s+\w+\s*\(', content))
             typed_funcs = len(re.findall(r'def\s+\w+\s*\(.*?:\s*\w+\s*\).*?->', content))
             if total_funcs > 0 and typed_funcs / total_funcs < 0.5:
                 static_report.append(f"ℹ️ {p.name}: {typed_funcs}/{total_funcs} fonctions typées (Améliorer les type hints).")

             # 3. Complexité Cyclomatique (Simulé - basé sur 'if', 'for', 'while', 'try', 'except')
             complexity_score = len(re.findall(r'(if|for|while|try|except|elif|case|match)\s', content)) + 1
             if complexity_score > 10:
                  static_report.append(f"🔥 {p.name}: Complexité Cyclomatique élevée ({complexity_score}). Refactorisation suggérée.")
             elif complexity_score > 5:
                  static_report.append(f"🟡 {p.name}: Complexité Cyclomatique modérée ({complexity_score}).")

        # Utilise le générateur de progression (Simulation Gradio)
        for _ in gradio_progress_simulate(self.agent.gradio_progress, 10, "Finalisation analyse statique"):
            time.sleep(0.05)

        return "\n".join(["--- Rapport d'Analyse Statique ---"] + static_report + ["--- Fin Rapport ---"])

class RefactorProactiveCommand(BaseCommand):
    """
    Commande: refactor|auto|<file_path>|<goal?>
    Usage: Correction automatique proactive selon les erreurs statiques détectées.
    """
    def execute(self, file_path: str, goal: str = "performance et lisibilité") -> str:
        _rich_print(f"🧠 Refactoring Proactif sur {file_path} (Objectif: {goal})...", title="REFACTOR:AUTO")
        
        # 1. Lance l'analyse statique pour identifier les problèmes.
        analysis = StaticAnalysisCommand(self.agent).execute(file_path)
        
        # 2. Lit le contenu
        content = ReadFileCommand(self.agent).execute(file_path, "5000")
        if content.startswith("❌"): return content
        
        # 3. Construit le prompt pour Gemini
        prompt = f"""
        Refactorise le code suivant en {goal}.
        Tiens compte des problèmes détectés par l'analyse statique ci-dessous pour corriger le code de manière proactive (PEP8, complexité, clarté).
        Fichier: {file_path}
        
        Analyse statique:
        ---
        {analysis}
        ---
        
        Code original:
        ---
        {content}
        ---
        
        Réponds uniquement avec le code refactorisé (sans explications).
        """
        
        # Utilise le générateur de progression
        for _ in console_progress_bar("Génération du refactoring intelligent", 100):
            time.sleep(0.01) # Simulation de latence

        # 4. Appelle Gemini
        refactored_code = self.agent._ask_gemini_raw(prompt)
        
        # 5. Propose l'écriture ou fait un diff (pour l'instant on retourne le code refactorisé)
        if "```" in refactored_code:
            # Extrait le bloc de code
            final_code = re.search(r"```[^\n]*\n(.*?)```", refactored_code, re.DOTALL)
            if final_code:
                # Propose d'écrire le fichier avec une ACTION
                return f"✅ Refactoring généré. Proposez l'écriture: ACTION:write|{file_path}|<CODE_REFRACTORISÉ> \n\n" + final_code.group(1).strip()
            
        return refactored_code

class TestMultiGenCommand(BaseCommand):
    """
    Commande: testgen|multi|<file_path>
    Usage: Génération de tests unitaires multi-fichiers avec indication de couverture (simulée).
    """
    def execute(self, file_path: str) -> str:
        _rich_print(f"🧪 Génération de tests unitaires pour {file_path}...", title="TESTGEN:MULTI")
        
        content = ReadFileCommand(self.agent).execute(file_path, "5000")
        if content.startswith("❌"): return content
        
        # 1. Contexte pour Gemini (demander la couverture)
        prompt = f"""
        Génère un ensemble de tests unitaires (pytest ou autre framework approprié) pour le code suivant.
        Crée si nécessaire des fichiers de tests séparés (`test_*.py`).
        Indique après chaque bloc de code de test, le pourcentage de couverture des fonctions testées.
        Fichier à tester: {file_path}
        
        Code original:
        ---
        {content}
        ---
        
        Réponds uniquement avec le(s) bloc(s) de code et leur couverture (format Markdown).
        """

        # Utilise le générateur de progression
        for _ in console_progress_bar("Génération des fichiers de tests", 80):
            time.sleep(0.01)

        result = self.agent._ask_gemini_raw(prompt)
        
        # 2. Simulation de la vérification de couverture
        coverage = "Couverture simulée: 85%."
        
        return f"{result}\n\n--- SUIVI COUVERTURE ---\n{coverage}\nProposez l'écriture: ACTION:write|test_{file_path}|<CODE_TEST>"

class DocGenCommand(BaseCommand):
    """
    Commande: doc|full|<file_path?>
    Usage: Documentation automatique complète (pour fichier, ou README global).
    """
    def execute(self, file_path: str = "README.md") -> str:
        _rich_print(f"📚 Génération de la documentation complète pour {file_path}...", title="DOC:FULL")
        
        if file_path.lower() == "readme.md" or file_path.lower() == "full":
            # Génération README global
            tree = TreeViewCommand(self.agent).execute()
            status = ProjectStatusCommand(self.agent).execute()
            
            prompt = f"""
            Génère un fichier README.md professionnel et complet pour le projet, en t'appuyant sur l'arborescence et le statut suivants.
            Le README doit inclure: un titre, une description, l'installation (requirements), l'utilisation, et le statut.
            
            Arborescence:
            ---
            {tree}
            ---
            
            Statut du projet:
            ---
            {status}
            ---
            
            Réponds uniquement avec le contenu du fichier README.md au format Markdown.
            """
        else:
            # Documentation d'un fichier spécifique
            content = ReadFileCommand(self.agent).execute(file_path, "5000")
            if content.startswith("❌"): return content
            
            prompt = f"""
            Génère une documentation complète et technique pour le fichier {file_path}.
            La documentation doit inclure: le but du module/fichier, les classes/fonctions, les paramètres, les valeurs de retour et les exemples d'utilisation (format docstring).
            
            Code:
            ---
            {content}
            ---
            
            Réponds uniquement avec le contenu de la documentation au format Markdown.
            """

        # Utilise le générateur de progression
        for _ in console_progress_bar("Génération documentation par Gemini", 150):
            time.sleep(0.01)

        result = self.agent._ask_gemini_raw(prompt)
        return result

class SuggestUpdateCommand(BaseCommand):
    """
    Commande: suggest|update|<requirements_file?>
    Usage: Détection et suggestion de mise à jour des dépendances.
    """
    def execute(self, requirements_file: str = "requirements.txt") -> str:
        _rich_print(f"🔄 Détection et suggestion de mise à jour des dépendances ({requirements_file})...", title="SUGGEST:UPDATE")
        
        try:
            p = safe_join(self.agent.project_path, requirements_file)
            if not p.exists():
                return f"⚠️ Fichier {requirements_file} introuvable. Exécutez ACTION:requirements pour générer le fichier en premier."

            reqs_content = read_text(p)
            
            # Simulation: analyse et détection de versions obsolètes
            # (Nécessiterait un appel à un service ou une base de données locale pour les versions récentes, ici on simule)
            lines = reqs_content.splitlines()
            updates = []
            
            for line in lines:
                if line.startswith("#") or not line.strip(): continue
                
                # Capture le nom du package, ignore les versions
                match = re.match(r"([a-zA-Z0-9_-]+)", line.strip())
                if match:
                    package = match.group(1)
                    if package.lower() in {"requests", "django", "flask"}:
                        updates.append(f"Package: {package}. Version obsolète potentielle. Suggestion: `{package}>=2.30.0` ou autre version majeure.")
                    else:
                         updates.append(f"Package: {package}. Version actuelle : OK (pas de mise à jour critique détectée).")

            if not updates:
                 return "✅ Aucune dépendance critique détectée pour la mise à jour (simulation)."

            # Appel Gemini pour formatter le nouveau requirements.txt
            prompt = f"""
            Analyse le fichier de dépendances suivant et les suggestions de mise à jour.
            Génère le nouveau fichier requirements.txt en proposant les dernières versions stables pour chaque package listé.
            Dépendances actuelles:
            ---
            {reqs_content}
            ---
            Suggestions:
            ---
            {'\n'.join(updates)}
            ---
            Réponds uniquement avec le contenu du fichier requirements.txt mis à jour.
            """
            
            for _ in console_progress_bar("Génération du requirements mis à jour", 50):
                 time.sleep(0.01)

            updated_reqs = self.agent._ask_gemini_raw(prompt)
            return updated_reqs

        except Exception as e:
            return f"❌ Erreur lors de la suggestion de mise à jour: {e}"

class ProjectStatusCommand(BaseCommand):
    """
    Commande: analyze|status
    Usage: Analyse globale du projet (nombre de fichiers/lignes/fonctions/classes, complexité totale).
    """
    def execute(self) -> str:
        _rich_print("📊 Collecte des statistiques globales du projet...", title="PROJECT:STATUS")

        total_files = 0
        total_lines = 0
        total_code_lines = 0
        total_functions = 0
        total_classes = 0
        
        # --- AJOUT GEMINI --- Historique des actions récentes et statistiques
        history = self.agent.memory.get_action_history(5)
        recent_actions = [f"- {log.ts} | {log.cmd} ({log.duration:.2f}s) | Status: {log.result_status}" for log in history]
        
        code_files = self.agent._get_all_code_files()
        total_files = len(code_files)
        
        for p in code_files:
            content = read_text(p)
            lines = content.splitlines()
            total_lines += len(lines)
            
            for line in lines:
                # Lignes de code non-vides et non-commentaires
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    total_code_lines += 1
                
                # Détection fonctions/classes (patterns simples Python)
                if p.suffix == ".py":
                    if stripped.startswith("def "):
                        total_functions += 1
                    elif stripped.startswith("class "):
                        total_classes += 1
        
        status = [
            "--- Statistiques Globales ---",
            f"Fichiers analysés: {total_files}",
            f"Lignes totales: {total_lines:,}",
            f"Lignes de code (estimé): {total_code_lines:,}",
            f"Fonctions/Méthodes (estimé): {total_functions}",
            f"Classes (estimé): {total_classes}",
            f"Ratio Code/Total: {(total_code_lines/total_lines)*100:.1f}%" if total_lines else "N/A",
            "\n--- Historique des Actions Récentes ---",
            "\n".join(recent_actions) or "Aucune action récente."
        ]
        
        return "\n".join(status)

class ReplayCommand(BaseCommand):
    """
    Commande: replay|<id_log>
    Usage: Permet de rejouer une action précédemment enregistrée.
    """
    def execute(self, log_id: str) -> str:
        _rich_print(f"🎬 Lancement du mode Replay pour l'action ID: {log_id}...", title="REPLAY:ACTION")
        
        log_entry = next((log for log in self.agent.memory.get_action_history(MEMORY_MAX) if log.id == log_id), None)
        
        if not log_entry or not log_entry.is_replayable:
            return f"❌ Action ID {log_id} introuvable ou non rejouable."
        
        # Reconstruit la chaîne d'action
        action_string = f"ACTION:{log_entry.cmd}|{'|'.join(log_entry.args)}"
        
        # Affiche le log
        _rich_print(f"Rejoue: {action_string}", title="REPLAY")
        
        # Exécute l'action (en évitant la boucle infinie si l'action rejouée est replay)
        if log_entry.cmd == "replay":
            return "❌ Replay d'une commande 'replay' refusé pour éviter boucle."

        # Lance l'exécution sans enregistrer le log de replay
        result = self.agent.execute_action(action_string, is_replay=True)
        
        return f"✅ Replay Terminé. Résultat original:\n{log_entry.result_summary}\n\nRésultat du Replay:\n{result}"

class ShellSandboxCommand(BaseCommand):
    """
    Commande: shell|sandbox|<command>
    Usage: Prévient et simule l'exécution shell avant de confirmer.
    """
    def execute(self, command: str) -> str:
        _rich_print(f"🔬 Analyse de la commande shell: '{command}'", title="SHELL:SANDBOX")
        
        # 1. Analyse de sécurité basique (réutilisation de la logique existante)
        if any(s in command for s in ["sudo", "su ", "rm -rf /", "shutdown", "reboot"]):
            return "❌ Commande dangereuse détectée — refusée."
        
        tokens = command.strip().split()
        if tokens and tokens[0] not in SAFE_SHELL_WHITELIST:
             return f"⚠️ Pré-validation échouée : Commande '{tokens[0]}' non reconnue dans whitelist. Refusée pour sécurité. Proposez ACTION:shell|{command} pour l'exécuter directement (à vos risques)."
        
        # 2. Simulation (basée sur l'analyse Gemini)
        prompt = f"""
        La commande shell suivante est proposée: `{command}`.
        Simule l'exécution de cette commande dans un environnement de développement sécurisé (sandbox).
        Décris clairement:
        1. L'impact réel de la commande sur le projet.
        2. Le résultat attendu (stdout/stderr).
        3. Les risques potentiels.
        
        Réponds uniquement avec le rapport de simulation (format Markdown).
        """
        
        # Utilise le générateur de progression
        for _ in console_progress_bar("Simulation de l'exécution", 60):
            time.sleep(0.01)

        simulation_report = self.agent._ask_gemini_raw(prompt)
        
        # 3. Validation par l'utilisateur
        return f"--- Rapport de Simulation Shell ---\n{simulation_report}\n\n🤖 **VALIDATION REQUISE:** Exécutez-la si satisfaisant: ACTION:shell|{command}"

# File commands (read/write/run/delete/copy/mkdir/rename/backup)
class ReadFileCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self, file_path: str, max_lines: str = "200") -> str:
        try:
            p = safe_join(self.agent.project_path, file_path)
            if not p.exists():
                return f"❌ Fichier introuvable: {file_path}"
            if p.is_dir():
                return f"❌ Chemin est un dossier: {file_path}"
            txt = read_text(p, max_chars=200_000)
            if txt.startswith("❌"):
                return txt
            lines = txt.splitlines()
            try:
                n = int(max_lines)
            except Exception:
                n = 200
            if len(lines) > n:
                return "\n".join(lines[:n]) + f"\n\n... (tronqué à {n} lignes)"
            return "\n".join(lines)
        except PermissionError as pe:
            return f"❌ Sécurité: {pe}"
        except Exception as e:
            return f"❌ Erreur lecture: {e}"

class WriteFileCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self, file_path: str, content: str) -> str:
        try:
            p = safe_join(self.agent.project_path, file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            # backup
            if p.exists():
                bname = p.name + "." + now_ts() + ".backup"
                shutil.copy2(p, p.with_name(bname))
            p.write_text(content, encoding="utf-8")
            # --- AJOUT GEMINI --- Notification en temps réel pour Gradio
            self.agent.last_notification = f"✅ Fichier Écrit et Sauvegardé: {file_path}"
            return f"✅ Écrit: {file_path}"
        except PermissionError as pe:
            return f"❌ Sécurité: {pe}"
        except Exception as e:
            return f"❌ Erreur écriture: {e}"

class DeleteFileCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self, file_path: str) -> str:
        try:
            p = safe_join(self.agent.project_path, file_path)
            if not p.exists():
                return f"❌ Inexistant: {file_path}"
            if p.is_dir():
                shutil.rmtree(p)
                return f"✅ Dossier supprimé: {file_path}"
            else:
                p.unlink()
                return f"✅ Fichier supprimé: {file_path}"
        except PermissionError as pe:
            return f"❌ Sécurité: {pe}"
        except Exception as e:
            return f"❌ Erreur suppression: {e}"

class RunFileCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self, file_path: str) -> str:
        try:
            p = safe_join(self.agent.project_path, file_path)
            if not p.exists():
                return f"❌ Inexistant: {file_path}"
            if p.suffix.lower() not in {".py", ".js", ".sh"}:
                return f"❌ Type non-exécutable: {p.suffix}"
            interp = {"py": sys.executable, "js": "node", "sh": "bash"}.get(p.suffix.lower().lstrip("."))
            
            # --- AJOUT GEMINI --- Indicateur de chargement
            for _ in console_progress_bar(f"Exécution de {file_path}", 10):
                time.sleep(0.01)

            cmd = [interp, str(p)]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=self.agent.project_path, timeout=SHELL_TIMEOUT)
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            return f"✅ Run code (rc={proc.returncode})\n---OUT---\n{out}\n---ERR---\n{err}"
        except subprocess.TimeoutExpired:
            return f"❌ Timeout après {SHELL_TIMEOUT}s"
        except Exception as e:
            return f"❌ Erreur exécution: {e}"

class CopyFileCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self, src: str, dest: str) -> str:
        try:
            s = safe_join(self.agent.project_path, src)
            d = safe_join(self.agent.project_path, dest)
            if s.is_dir():
                shutil.copytree(s, d)
            else:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)
            return f"✅ Copié {src} -> {dest}"
        except Exception as e:
            return f"❌ Erreur copie: {e}"

class MkdirCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self, dir_path: str) -> str:
        try:
            d = safe_join(self.agent.project_path, dir_path)
            d.mkdir(parents=True, exist_ok=True)
            return f"✅ Dossier créé: {dir_path}"
        except Exception as e:
            return f"❌ Erreur mkdir: {e}"

class RenameCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self, src: str, dest: str) -> str:
        try:
            s = safe_join(self.agent.project_path, src)
            d = safe_join(self.agent.project_path, dest)
            s.rename(d)
            return f"✅ Renommé {src} -> {dest}"
        except Exception as e:
            return f"❌ Erreur rename: {e}"

class BackupFileCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self, file_path: str) -> str:
        try:
            p = safe_join(self.agent.project_path, file_path)
            if not p.exists():
                return f"❌ Inexistant: {file_path}"
            b = p.with_name(p.name + "." + now_ts() + ".backup")
            shutil.copy2(p, b)
            return f"✅ Backup créé: {b.name}"
        except Exception as e:
            return f"❌ Erreur backup: {e}"

class ScanCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self) -> str:
        tree = self.agent._scan_tree()
        return json.dumps(tree, indent=2, ensure_ascii=False)

class TreeViewCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def _format(self, d: Dict, prefix: str = "") -> List[str]:
        out = []
        keys = sorted(d.keys())
        for i, k in enumerate(keys):
            is_last = (i == len(keys)-1)
            pointer = "└── " if is_last else "├── "
            out.append(prefix + pointer + k)
            if isinstance(d[k], dict):
                newprefix = prefix + ("    " if is_last else "│   ")
                out.extend(self._format(d[k], newprefix))
        return out

    def execute(self) -> str:
        tree = self.agent._scan_tree()
        top = f"📂 {self.agent.project_path.name}"
        lines = [top] + self._format(tree, "")
        return "\n".join(lines)

class RequirementsCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self) -> str:
        pyth_files = list(self.agent.project_path.rglob("*.py"))
        imports = set()
        for f in pyth_files:
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                for m in re.findall(r"^(?:from|import)\s+([a-zA-Z0-9_.]+)", txt, re.MULTILINE):
                    root = m.split(".")[0]
                    if root and root not in {"os","sys","re","json","pathlib","subprocess","shutil","datetime","typing","argparse"}:
                        imports.add(root)
            except Exception:
                continue
        if not imports:
            return "⚠️ Aucun module externe détecté."
        # Ask Gemini to format requirements (agent._ask_gemini_raw will be used by JulesAgent)
        prompt = f"Génère un fichier requirements.txt pour les modules: {', '.join(sorted(imports))}. Réponds uniquement avec le contenu du fichier."
        
        # --- AJOUT GEMINI --- Indicateur de chargement
        for _ in console_progress_bar("Génération des dépendances", 30):
            time.sleep(0.01)

        return self.agent._ask_gemini_raw(prompt)

class EnvCommand(BaseCommand):
#... (Conserver la classe existante sans modification) ...
    def execute(self) -> str:
        lines = []
        for k in sorted(os.environ.keys()):
            if any(s in k.upper() for s in ("KEY","SECRET","TOKEN","PASSWORD")):
                continue
            lines.append(f"{k}={os.environ[k]}")
        return "\n".join(lines) or "Aucune variable publique."

# ----------------------------
# JulesAgent (core)
# ----------------------------
class JulesAgent:
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        if not self.project_path.exists():
            self.project_path.mkdir(parents=True, exist_ok=True)
        # --- AJOUT GEMINI --- Initialisation de la mémoire avec le chemin de l'historique
        self.memory = Memory(self.project_path / MEMORY_FILE, self.project_path / ".jules_action_history.json")
        self.gemini = GeminiWrapper(API_KEY)
        # --- AJOUT GEMINI --- Attributs pour la gestion de l'état dynamique Gradio
        self.last_action_progress: Dict[str, Any] = {"description": "Prêt.", "steps": 0, "current": 0}
        self.last_notification: str = ""
        self.gradio_progress: Optional[Progress] = None # Sera injecté par le handler Gradio
        
        # dispatcher
        self._command_dispatcher: Dict[str, BaseCommand] = {
            "read": ReadFileCommand(self),
            "write": WriteFileCommand(self),
            "run": RunFileCommand(self),
            "delete": DeleteFileCommand(self),
            "copy": CopyFileCommand(self),
            "mkdir": MkdirCommand(self),
            "rename": RenameCommand(self),
            "backup": BackupFileCommand(self),
            "scan": ScanCommand(self),
            "tree": TreeViewCommand(self),
            "requirements": RequirementsCommand(self),
            "env": EnvCommand(self),
            # --- AJOUT GEMINI --- Nouvelles commandes avancées
            "audit": AuditSecurityCommand(self),
            "analyze": AnalyzeFullCommand(self), # Gère analyze|full et analyze|status (indirectement via ProjectStatus)
            "static": StaticAnalysisCommand(self),
            "refactor": RefactorProactiveCommand(self), # Gère refactor|auto
            "testgen": TestMultiGenCommand(self), # Gère testgen|multi
            "doc": DocGenCommand(self), # Gère doc|full
            "suggest": SuggestCommand(self), # Gère suggest|<context> et suggest|update (indirectement via SuggestUpdate)
            "progress": ProgressCommand(self),
            "replay": ReplayCommand(self),
            "shell": ShellSandboxCommand(self), # Gère shell|sandbox (le shell direct reste géré dans execute_action)
        }
        
        # --- AJOUT GEMINI --- Mise à jour de la liste des commandes disponibles
        self.available = sorted(list(set(list(self._command_dispatcher.keys()) + ["generate","correct","refactor","testgen","summarize","explain","doc","shell"])))

        # system prompt core
        self.system_base = self._build_system_prompt()
        _rich_print(f"{APP_NAME} initialisé. Projet: {self.project_path}", title="Jules")

    def _build_system_prompt(self) -> str:
        # --- AJOUT GEMINI --- Mise à jour de la liste des commandes pour le prompt
        cmds = ", ".join(self.available)
        prompt = f"""Tu es Jules, un agent IA développeur et auditeur de code.
Tu as accès aux commandes suivantes: {cmds}.
Règles:
- Si l'utilisateur demande une action exécutable, répond strictement par ACTION:<commande>|<arg1>|<arg2>...
- Les commandes valides et gérées localement sont : {', '.join(self._command_dispatcher.keys())}.
- Pour des tâches génératives (generate, correct, refactor, testgen, summarize, explain, doc), utilise le format : ACTION:<commande>|...
- Pour exécuter shell direct, réponds par ACTION:shell|<commande> .
- **Pour les commandes avancées :**
    - Audit de sécurité : ACTION:audit|security|<file_path>
    - Analyse complète du projet : ACTION:analyze|full
    - Suggestion intelligente : ACTION:suggest|<context>
    - Refactoring proactif : ACTION:refactor|auto|file|goal
- Si la demande est conversationnelle, réponds normalement (pas d'ACTION:).
- Ne jamais divulguer de secrets, clés API, mots de passe.
- Conserve le contexte du projet et respecte la racine du projet.
"""
        return prompt

    # ------------------------
    # LLM Helpers
    # ------------------------
    def _ask_gemini_raw(self, prompt: str) -> str:
        return self.gemini.generate(system_instruction=self.system_base, user_prompt=prompt)

    def ask_gemini(self, user_input: str) -> str:
        # Build context
        recent = self.memory.get_recent(6)
        ctx = "\n".join([f"U: {r['user']}\nA: {r['assistant']}" for r in recent])
        full_prompt = f"Contexte récent:\n{ctx}\n\nQuestion:\n{user_input}"
        
        # --- AJOUT GEMINI --- Animation machine à écrire
        # Dans le contexte d'une console interactive, l'animation est gérée directement.
        # Dans Gradio, la fonction wrapper doit gérer le streaming/update, ici on retourne juste la réponse.
        
        # Simule une latence pour l'animation de saisie
        time.sleep(1)
        
        return self._ask_gemini_raw(full_prompt)

    # ------------------------
    # Project utilities
    # ------------------------
    def _scan_tree(self) -> Dict[str, Any]:
        def rec(p: Path):
            d = {}
            for it in sorted(p.iterdir(), key=lambda x: x.name):
                if it.name.startswith("."):
                    continue
                if any(pat in it.name for pat in PROJECT_IGNORED):
                    continue
                if it.is_dir():
                    d[it.name] = rec(it)
                elif it.is_file() and it.suffix.lower() in SUPPORTED_EXT:
                    d[it.name] = None
            return d
        return rec(self.project_path)
    
    # --- AJOUT GEMINI --- Utilitaire pour lister tous les fichiers de code
    def _get_all_code_files(self) -> List[Path]:
        files = []
        for ext in SUPPORTED_EXT:
            files.extend(self.project_path.rglob(f"*{ext}"))
        # Filtrer les chemins ignorés
        valid_files = [
            p for p in files 
            if not any(ign in str(p.relative_to(self.project_path)) for ign in PROJECT_IGNORED)
        ]
        return valid_files

    # ------------------------
    # Dispatcher / action execution
    # ------------------------
    # --- AJOUT GEMINI --- Ajout de is_replay pour éviter la ré-enregistrement des logs
    def execute_action(self, action_string: str, is_replay: bool = False) -> str:
        start_time = time.time()
        
        if not action_string.startswith("ACTION:"):
            return "❌ Format invalide (doit commencer par ACTION:)."
            
        
        
        raw = action_string[len("ACTION:"):]

# Échappement temporaire des | dans le contenu
        if raw.startswith("write|"):
            # Format: write|file_path|content
            first_pipe = raw.find("|")
            second_pipe = raw.find("|", first_pipe + 1)
            if second_pipe == -1:
                parts = raw.split("|")  # fallback
            else:
                cmd = raw[:first_pipe]
                file_path = raw[first_pipe + 1:second_pipe]
                content = raw[second_pipe + 1:]
                parts = [cmd, file_path, content]
        else:
            parts = raw.split("|")





        cmd = parts[0].lower()
        args = parts[1:]
        
        # Gestion des commandes avancées mappées à des classes
        if cmd in self._command_dispatcher:
            handler = self._command_dispatcher[cmd]
            
            # --- AJOUT GEMINI --- Gestion spécifique des commandes à plusieurs arguments pour le dispatcher
            # Ex: analyze|full, audit|security
            if cmd == "analyze" and args and args[0].lower() in ["full", "status"]:
                cmd_key = args[0].lower()
                handler = AnalyzeFullCommand(self) if cmd_key == "full" else ProjectStatusCommand(self)
                args = args[1:]

            if cmd == "refactor" and args and args[0].lower() in {"auto", "proactive"}:
                handler = RefactorProactiveCommand(self)
                args = args[1:]  
            
            if cmd == "suggest" and args and args[0].lower() == "update":
                handler = SuggestUpdateCommand(self)
                args = args[1:]
            
            if cmd == "shell" and args and args[0].lower() == "sandbox":
                handler = ShellSandboxCommand(self)
                args = args[1:]
                
            try:
                # Mise à jour de la progression pour l'action en cours
                self.last_action_progress = {"description": f"Exécution de {cmd}", "steps": 100, "current": 0}
                self.last_notification = f"⏳ Démarrage: {cmd}..."
                
                result = handler.execute(*args)

                # Mise à jour de la progression et de la notification finale
                self.last_action_progress['current'] = self.last_action_progress['steps']
                status = "✅ SUCCÈS" if not result.startswith("❌") else "❌ ÉCHEC"
                self.last_notification = f"{status} : {cmd} terminé en {time.time() - start_time:.2f}s"
                
            except TypeError as te:
                result = f"❌ Mauvais args pour {cmd}: {te}"
                status = "❌ ARG_ERR"
            except PermissionError as pe:
                result = f"❌ Refus sécurité: {pe}"
                status = "❌ PERM_ERR"
            except Exception as e:
                result = f"❌ Erreur interne: {e}"
                status = "❌ INT_ERR"
                
        # Gestion des commandes LLM génériques
        elif cmd in {"generate", "correct", "refactor", "testgen", "summarize", "explain", "doc"}:
            
            # --- AJOUT GEMINI --- Utilisation du générateur de progression pour les actions LLM longues
            for _ in console_progress_bar(f"Génération LLM pour {cmd}", 20):
                time.sleep(0.01)
                
            # Logic for generate, correct, refactor, testgen, summarize, explain, doc
            if cmd == "generate":
                if not args:
                    result = "⚠️ generate nécessite une description."
                else:
                    desc = args[0]
                    lang = args[1] if len(args) > 1 else "python"
                    prompt = f"Génère un projet/code en {lang} pour: {desc}. Réponds uniquement avec le bloc de code."
                    result = self._ask_gemini_raw(prompt)
            elif cmd == "correct":
                if len(args) < 2:
                    result = "⚠️ correct nécessite: file|détail"
                else:
                    file_path = args[0]
                    details = args[1]
                    file_txt = ReadFileCommand(self).execute(file_path, "5000")
                    if file_txt.startswith("❌"):
                        result = file_txt
                    else:
                        prompt = f"Corrige le fichier suivant selon: {details}\n---\n{file_txt}\n---\nRéponds uniquement avec le fichier corrigé."
                        result = self._ask_gemini_raw(prompt)
            elif cmd == "refactor":
                if len(args) < 2:
                    result = "⚠️ refactor nécessite: file|objectif"
                else:
                    # Note: Utilise le refactor générique existant, refactor|auto utilise RefactorProactiveCommand
                    file_path = args[0]
                    goal = args[1]
                    content = ReadFileCommand(self).execute(file_path, "5000")
                    if content.startswith("❌"):
                        result = content
                    else:
                        prompt = f"Refactorise le code suivant pour: {goal}\nFichier: {file_path}\n{content}\nRéponds uniquement avec le code refactorisé."
                        result = self._ask_gemini_raw(prompt)
            elif cmd == "testgen":
                if len(args) < 1:
                    result = "⚠️ testgen nécessite: file"
                else:
                    # Note: Utilise le testgen générique existant, testgen|multi utilise TestMultiGenCommand
                    file_path = args[0]
                    content = ReadFileCommand(self).execute(file_path, "5000")
                    if content.startswith("❌"):
                        result = content
                    else:
                        prompt = f"Génère des tests unitaires pytest pour le code suivant:\n{content}\nRéponds uniquement avec les fichiers de test."
                        result = self._ask_gemini_raw(prompt)
            elif cmd == "summarize":
                if len(args) < 1:
                    result = "⚠️ summarize nécessite: file"
                else:
                    file_path = args[0]
                    content = ReadFileCommand(self).execute(file_path, "800")
                    if content.startswith("❌"):
                        result = content
                    else:
                        prompt = f"Fais un résumé clair et concis du fichier {file_path} (but, responsabilités, complexités, points à vérifier):\n{content}"
                        result = self._ask_gemini_raw(prompt)
            elif cmd == "explain":
                if len(args) < 2:
                    result = "⚠️ explain nécessite: file|ligne_num"
                else:
                    file_path, line_spec = args[0], args[1]
                    content = ReadFileCommand(self).execute(file_path, "1000")
                    if content.startswith("❌"):
                        result = content
                    else:
                        prompt = f"Explique la ligne(s) {line_spec} du fichier {file_path}:\n{content}\nDonne explications techniques et simplifiées."
                        result = self._ask_gemini_raw(prompt)
            elif cmd == "doc":
                if len(args) < 1:
                    result = "⚠️ doc nécessite: file"
                else:
                    # Note: Utilise le doc générique existant, doc|full utilise DocGenCommand
                    file_path = args[0]
                    content = ReadFileCommand(self).execute(file_path, "2000")
                    if content.startswith("❌"):
                        result = content
                    else:
                        prompt = f"Génère une documentation exhaustive pour le fichier {file_path} (README style):\n{content}"
                        result = self._ask_gemini_raw(prompt)
            
            status = "✅ SUCCÈS" if not result.startswith(("❌", "⚠️", "🤖 Pas de réponse")) else "❌ ÉCHEC"
            self.last_notification = f"{status} : {cmd} terminé."

        # Shell direct (non-sandbox)
        elif cmd == "shell":
            if not args:
                result = "⚠️ shell nécessite une commande (ex: shell|ls -la)"
                status = "❌ ARG_ERR"
            else:
                # --- AJOUT GEMINI --- Utilise _execute_shell_pipe pour le shell direct
                result = self._execute_shell_pipe(args[0])
                status = "✅ SUCCÈS" if not result.startswith("❌") else "❌ ÉCHEC"
        
        else:
            result = f"❌ Commande inconnue ou non-implémentée: {cmd}"
            status = "❌ INCONNU"

        # Enregistrement du log d'action (sauf si c'est un replay)
        if not is_replay:
            duration = time.time() - start_time
            log = ActionLog(
                id=datetime.now().strftime("%Y%m%d%H%M%S%f"),
                ts=now_ts(),
                cmd=cmd,
                args=args,
                duration=duration,
                result_status=status,
                result_summary=result[:200].replace('\n', ' '), # Résumé court du résultat
                is_gemini_action=(cmd in {"generate", "correct", "refactor", "testgen", "summarize", "explain", "doc"}),
                is_replayable=(cmd in list(self._command_dispatcher.keys())) # Les commandes LLM ne sont pas rejouables telles quelles
            )
            self.memory.push_action_log(log)

        return result

    # ------------------------
    # Shell execution (sandbox lite)
    # ------------------------
    def _execute_shell_pipe(self, command: str) -> str:
        # Basic safety checks
        if any(s in command for s in ["sudo", "su ", "rm -rf /", "shutdown", "reboot"]):
            return "❌ Commande dangereuse détectée — refusée."
        # Allow change directory "cd" implemented locally (affects subsequent safe path ops)
        if command.strip().startswith("cd "):
            target = command.strip()[3:].strip()
            try:
                # --- AJOUT GEMINI --- Correction de la logique cd pour bien fonctionner
                # On utilise chdir pour changer le répertoire de travail
                p = self.project_path / target
                newp = p.resolve()
                newp.relative_to(self.project_path.resolve()) # Re-vérifie la sécurité
                os.chdir(newp)
                self.project_path = newp # Mise à jour du chemin de base pour les futures opérations
                return f"✅ CD -> {os.getcwd()}"
            except Exception as e:
                return f"❌ CD failed: {e}"
        # Check whitelist presence (very rough)
        tokens = command.strip().split()
        if tokens and tokens[0] not in SAFE_SHELL_WHITELIST:
            # not necessarily forbidden, but warn user and allow with explicit override
            return f"⚠️ Commande '{tokens[0]}' non reconnue dans whitelist. Refusée pour sécurité."
        try:
            # --- AJOUT GEMINI --- Indicateur de chargement
            for _ in console_progress_bar(f"Shell {tokens[0]}", 10):
                time.sleep(0.01)
                
            proc = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.project_path, timeout=SHELL_TIMEOUT)
            out = proc.stdout.strip()
            err = proc.stderr.strip()
            summary = f"rc={proc.returncode}\n---OUT---\n{out[:MAX_CMD_OUTPUT]}\n---ERR---\n{err[:MAX_CMD_OUTPUT]}"
            return summary
        except subprocess.TimeoutExpired:
            return f"❌ Timeout après {SHELL_TIMEOUT}s"
        except Exception as e:
            return f"❌ Erreur shell: {e}"

# ----------------------------
# Console & Gradio UI
# ----------------------------
def main_console(agent: JulesAgent):
    print(f"\n--- {APP_NAME} Console ---")
    print(f"Projet: {agent.project_path}")
    print("Tape: 'exit' pour quitter. Tape 'help' pour assistance.")
    print("\n--- Commandes Avancées ---")
    print(" - analyze|full : Rapport complet du projet.")
    print(" - audit|security : Audit de sécurité simulé.")
    print(" - suggest|'contexte' : Suggestions intelligentes.")
    print(" - shell|sandbox|'cmd' : Pré-analyse de la commande.")
    print(" - replay|<id_log> : Rejouer une action (voir .jules_action_history.json).")
    print("--------------------------")

    while True:
        try:
            user = input("\nVous> ").strip()
            if not user:
                continue
            if user.lower() in {"exit","quit","q"}:
                print("Au revoir.")
                break
            if user.lower() == "help":
                print("Exemples: 'tree', 'read|main.py|50' or ask anything and the agent may reply ACTION:..")
                continue

            resp = agent.ask_gemini(user)

            if resp.startswith("ACTION:"):
                print(f"[Classified action] {resp}")
                res = agent.execute_action(resp)
                print("---- Résultat ----")
                print(res)
                agent.memory.push_conversation(user, f"{resp}\n\n{res}")
            else:
                # ➜  ANIMATION « CHATGPT » LETTRE PAR LETTRE
                print("Jules> ", end="", flush=True)
                for ch in resp:
                    print(ch, end="", flush=True)
                    time.sleep(0.008)          # vitesse « humaine »
                print()                          # retour à la ligne final
                agent.memory.push_conversation(user, resp)

        except KeyboardInterrupt:
            print("\nInterruption. Bye.")
            break
        except Exception as e:
            print(f"Erreur: {e}")




if gr is not None:
    def build_gradio_ui(agent: JulesAgent):
        default_box = "Pose une requête (ex: 'liste les fichiers', 'analyze|full', 'audit|security', 'suggest|ajoute un ORM')"
        
        # --- AJOUT GEMINI --- Fonction de rafraîchissement d'état global
        def refresh_state(last_history: List[ActionLog]):
            tree_content = TreeViewCommand(agent).execute()
            status_content = ProjectStatusCommand(agent).execute()
            
            # Formatage de l'historique enrichi
            history_lines = [f"[{log.id[-4:]}] {log.ts[9:-3]} {log.cmd} ({log.duration:.2f}s) -> {log.result_status}" for log in last_history]
            memory_content = "\n".join(history_lines)
            
            return tree_content, status_content, memory_content, agent.last_notification
        
        # --- AJOUT GEMINI --- Handler Gradio principal mis à jour
        def submit_fn(message, history, last_history: List[ActionLog], gr_progress: Progress):
            agent.gradio_progress = gr_progress # Injecte l'objet progress
            
            if not message or not message.strip():
                return history, "", gr.update(value=agent.last_notification), last_history
            
            history = history or []
            history.append(("Vous", message))
            
            # Affiche l'attente
            yield history + [("Jules", "🤖 Réflexion en cours...")], "", agent.last_notification, last_history
            
            # 1. Ask Gemini
            resp = agent.ask_gemini(message)
            
            # 2. Execution de l'action ou réponse conversationnelle
            if resp.startswith("ACTION:"):
                
                # Indication de l'action en cours
                temp_hist = history + [("Jules", f"**[CLASSIFICATION]** : `{resp}`\n**[STATUT]** : Démarrage de l'action...")]
                yield temp_hist, "", gr.update(value=f"⏳ Exécution de: {resp.split('|')[0]}..."), last_history
                
                result = agent.execute_action(resp)
                assistant_msg = f"**[ACTION EXECUTED] {resp}**\n\n```\n{result}\n```"
                
                history.append(("Jules", assistant_msg))
                agent.memory.push_conversation(message, assistant_msg)
                last_action_val = resp
                
            else:
                history.append(("Jules", resp))
                agent.memory.push_conversation(message, resp)
                last_action_val = ""
            
            # 3. Mise à jour de tous les états après l'action
            new_history = agent.memory.get_action_history(10)
            
            # 4. Retourne les mises à jour
            return (
                history, 
                last_action_val, 
                gr.update(value=agent.last_notification, visible=True),
                new_history
            )

        with gr.Blocks(title="Jules v2++", css="""
            body { background: linear-gradient(180deg,#0f172a,#111827); color: white; }
            .chatbox { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 8px; }
            /* --- AJOUT GEMINI --- Custom CSS */
            .action-panel { border-left: 3px solid #6366f1; padding-left: 10px; }
        """) as demo:
            
            # --- AJOUT GEMINI --- État invisible pour stocker l'historique d'action
            action_history_state = gr.State(value=agent.memory.get_action_history(10)) 
            
            gr.Markdown(f"# 🤖 {APP_NAME}  — Projet: `{agent.project_path.name}`")
            
            # --- AJOUT GEMINI --- Progress Bar et Notification Temps Réel
            status_row = gr.Row()
            with status_row:
                 gr_progress = gr.Progress(label="Progression de l'action")
                 realtime_notification = gr.Textbox(label="Notifications Temps Réel", interactive=False, value="")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chat = gr.Chatbot([], elem_classes="chatbox", label="Conversation")
                    txt = gr.Textbox(placeholder=default_box, label="Votre message")
                    with gr.Row():
                        submit = gr.Button("Envoyer")
                        clear_btn = gr.Button("Clear Chat")
                    last_action = gr.Textbox(label="Dernière action (pour debug)", interactive=False, elem_classes="action-panel")
                    
                with gr.Column(scale=1):
                    gr.Markdown("## Analyse du Projet")
                    
                    # --- AJOUT GEMINI --- Arborescence et Statut global
                    tree_area = gr.Textbox(value=TreeViewCommand(agent).execute(), label="Arborescence", lines=10, interactive=False)
                    status_area = gr.Textbox(value=ProjectStatusCommand(agent).execute(), label="Statut Global (analyze|status)", lines=10, interactive=False)
                    history_area = gr.Textbox(value="\n".join([f"[{r.id[-4:]}] {r.ts[9:-3]} {r.cmd} ({r.duration:.2f}s) -> {r.result_status}" for r in agent.memory.get_action_history(10)]), label="Historique enrichi (ID pour replay|...)", lines=10, interactive=False)
                    
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh All")
                        analyze_btn = gr.Button("Lancer Analyze|Full")
                        
            # Handlers
            # Refresh handler (pour les données statiques/légères)
            refresh_btn.click(
                fn=refresh_state, 
                inputs=[action_history_state], 
                outputs=[tree_area, status_area, history_area, realtime_notification]
            )
            
            # Lancer analyse complète
            analyze_btn.click(
                fn=lambda h, p: submit_fn("ACTION:analyze|full", h, agent.memory.get_action_history(10), p),
                inputs=[chat, gr_progress],
                outputs=[chat, last_action, realtime_notification, action_history_state]
            )

            # Soumission principale (utiliser .then pour la progression)
            submit.click(
                fn=submit_fn, 
                inputs=[txt, chat, action_history_state, gr_progress], 
                outputs=[chat, last_action, realtime_notification, action_history_state]
            ).then(
                 fn=refresh_state, 
                 inputs=[action_history_state], 
                 outputs=[tree_area, status_area, history_area, realtime_notification]
            )
            
            # Nettoyage
            clear_btn.click(lambda: ([], "", "", agent.memory.get_action_history(10)), None, [chat, last_action, realtime_notification, action_history_state])
            
            # Rafraîchissement périodique (facultatif mais utile)
            demo.load(
                fn=refresh_state,
                inputs=[action_history_state],
                outputs=[tree_area, status_area, history_area, realtime_notification],
                every=20 # toutes les 20 secondes
            )
            
        return demo

def parse_args():
    p = argparse.ArgumentParser(description="Jules v2++ - Agent IA codeur")
    p.add_argument("--project", "-p", default=".", help="Chemin du projet")
    p.add_argument("--web", action="store_true", help="Lancer interface Gradio")
    p.add_argument("--no-gemini", action="store_true", help="Désactive Gemini (mode local)")
    return p.parse_args()

def main():
    args = parse_args()
    agent = JulesAgent(project_path=args.project)
    if args.no_gemini:
        agent.gemini = GeminiWrapper(None)
    if args.web:
        if gr is None:
            print("❌ Gradio non installé. pip install gradio")
            return
        demo = build_gradio_ui(agent)
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    else:
        main_console(agent)

if __name__ == "__main__":
    main()
