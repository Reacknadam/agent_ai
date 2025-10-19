
# ==============================================================================
# Jules v2.9 Enhanced - main.py
# Agent IA Orchestrateur Central avec Vision, Recherche Sémantique & Kimi-like Robustness
# ==============================================================================
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import inspect  # ← ajouté pour introspection des signatures

# ==============================================================================
# 1. GESTION DES DÉPENDANCES ET INITIALISATION
# ==============================================================================
GEMINI_API_KEY_HARDCODED = "ALLEZ Y CHERCHER SUR GOOGLE AI "

# Dépendances optionnelles
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

try:
    import gradio as gr
    from rich.markdown import Markdown
    from rich.console import Console
    CLI_CONSOLE = Console()
except Exception:
    gr = None
    Markdown = None
    CLI_CONSOLE = None

# Pour la recherche sémantique
try:
    from sentence_transformers import SentenceTransformer, util  # pyright: ignore[reportMissingImports]
    SEMANTIC_SEARCH_AVAILABLE = True
except Exception:
    SEMANTIC_SEARCH_AVAILABLE = False

# ==============================================================================
# 2. STRUCTURES DE DONNÉES CLÉS
# ==============================================================================
@dataclass
class ActionTool:
    tool_name: str
    args: Dict[str, Any]
    reasoning: str = ""

@dataclass
class LogEntry:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    type: str = "USER_INPUT"
    content: str = ""
    success: Optional[bool] = None

# ==============================================================================
# 3. UTILITAIRE : AFFICHAGE LETTRE PAR LETTRE
# ==============================================================================
def stream_print(text: str, delay: float = 0.00008):
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    clean_text = re.sub(r'```.*?```', '[code]', clean_text, flags=re.DOTALL)
    for char in clean_text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# ==============================================================================
# 4. CONTEXTE ET SÉCURITÉ
# ==============================================================================
def get_gemini_key() -> Optional[str]:
    return GEMINI_API_KEY_HARDCODED if GEMINI_API_KEY_HARDCODED else None

def safe_join(base: Path, *paths: str) -> Optional[Path]:
    try:
        path = Path(base, *paths).resolve()
        if path.is_relative_to(base.resolve()):
            return path
    except Exception:
        pass
    return None

class Memory:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.action_log: List[LogEntry] = []
        self.learned_skills: Dict[str, str] = {}
        self._init_project_log()

    def _init_project_log(self):
        self.action_log.append(LogEntry(
            type="SYSTEM_INIT",
            content=f"Jules v2.9 initialisé. Projet: {self.project_path.resolve()}"
        ))

    def add_log(self, entry: LogEntry):
        self.action_log.append(LogEntry(
            timestamp=datetime.now(timezone.utc),
            type=entry.type,
            content=entry.content,
            success=entry.success
        ))

    def get_action_history(self, count: int = 5) -> str:
        return "\n".join([
            f"[{e.timestamp.strftime('%H:%M:%S')}] {e.type} ({'OK' if e.success else 'FAIL' if e.success is False else 'PENDING'}): {e.content[:150]}..."
            for e in self.action_log[-count:]
        ])

    def get_learned_skills_context(self) -> str:
        if not self.learned_skills:
            return "Aucune compétence spécifique apprise."
        skills_list = "\n".join([
            f"- **{name}** : {instruction}" 
            for name, instruction in self.learned_skills.items()
        ])
        return f"Compétences Apprises:\n{skills_list}"

    def get_project_tree(self) -> str:
        tree_lines = []
        max_depth = 3
        max_files = 100
        try:
            for root, dirs, files in os.walk(self.project_path):
                relative_path = Path(root).relative_to(self.project_path)
                level = len(relative_path.parts) if str(relative_path) != '.' else 0
                if level > max_depth or len(tree_lines) > max_files:
                    if len(tree_lines) <= max_files:
                        tree_lines.append(f"{' ' * 4 * max_depth}└── [Dossiers et fichiers supplémentaires omis...]")
                    break
                indent = ' ' * 4 * level
                if str(relative_path) != '.':
                    tree_lines.append(f"{indent}├── {relative_path.name}/")
                sub_indent = ' ' * 4 * (level + 1)
                for f in files:
                    tree_lines.append(f"{sub_indent}— {f}")
        except Exception:
            return "Erreur lors de la lecture de l'arborescence."
        if not tree_lines:
             return "Projet vide."
        return f"/{self.project_path.name}/\n" + "\n".join(tree_lines).lstrip()

# ==============================================================================
# 5. DÉFINITION DES OUTILS
# ==============================================================================
class Tool:
    name: str = "base_tool"
    description: str = "Description de l'outil de base."
    def __init__(self, memory: Memory):
        self.memory = memory
        self.project_path = memory.project_path
    def execute(self, **kwargs) -> Tuple[bool, str]:
        raise NotImplementedError

class ListFilesTool(Tool):
    name = "list_files"
    description = "Liste récursivement les fichiers et dossiers du projet."
    def execute(self, path: str = "") -> Tuple[bool, str]:
        target_path = self.project_path
        if path:
            target_path = safe_join(self.project_path, path)
            if not target_path:
                return False, f"Chemin invalide ou hors projet : {path}"
        try:
            tree_lines = []
            max_depth = 3
            max_lines = 100
            for root, dirs, files in os.walk(target_path):
                rel_root = Path(root).relative_to(target_path)
                level = len(rel_root.parts) if str(rel_root) != '.' else 0
                if level > max_depth or len(tree_lines) > max_lines:
                    tree_lines.append("    " * max_depth + "└── [...]")
                    break
                indent = '    ' * level
                if str(rel_root) != '.':
                    tree_lines.append(f"{indent}├── {rel_root.name}/")
                sub_indent = '    ' * (level + 1)
                for f in files:
                    tree_lines.append(f"{sub_indent}— {f}")
            return True, f"Arborescence sous '{path or '.'}':\n" + "\n".join(tree_lines)
        except Exception as e:
            return False, f"Erreur lors de la lecture de l'arborescence : {e}"

class ReadFileTool(Tool):
    name = "read_file"
    description = "Lit le contenu complet d'un fichier spécifié dans le projet."
    def execute(self, path: str) -> Tuple[bool, str]:
        file_path = safe_join(self.project_path, path)
        if not file_path:
            return False, f"Erreur de sécurité: Le chemin '{path}' est invalide ou en dehors du répertoire du projet."
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if len(content) > 15000:
                 content = content[:15000] + "\n[... CONTENU TRONQUÉ POUR LIMITER LE CONTEXTE IA ...]"
            return True, f"Contenu de '{path}' (taille {len(content)}):\n{content}"
        except FileNotFoundError:
            return False, f"Erreur: Fichier '{path}' non trouvé."
        except Exception as e:
            return False, f"Erreur lors de la lecture de '{path}': {e}"

class WriteFileTool(Tool):
    name = "write_file"
    description = "Écrit ou remplace le contenu d'un fichier."
    def execute(self, path: str, content: str) -> Tuple[bool, str]:
        file_path = safe_join(self.project_path, path)
        if not file_path:
            return False, f"Erreur de sécurité: Le chemin '{path}' est invalide ou en dehors du répertoire du projet."
        try:
            os.makedirs(file_path.parent, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"Fichier '{path}' écrit/mis à jour avec succès (taille: {len(content)})."
        except Exception as e:
            return False, f"Erreur lors de l'écriture de '{path}': {e}"

class RunShellTool(Tool):
    name = "run_shell"
    description = "Exécute une commande shell dans le répertoire du projet (sandbox)."
    ALLOWED_CMDS = ["ls", "git", "pip", "python", "npm", "yarn", "make", "docker", "mkdir", "touch"] 
    def execute(self, command: str) -> Tuple[bool, str]:
        if not any(command.strip().startswith(cmd) for cmd in self.ALLOWED_CMDS):
            return False, f"Commande non autorisée (Sandbox): '{command}'. Uniquement: {', '.join(self.ALLOWED_CMDS)}"
        try:
            result = subprocess.run(
                command,
                cwd=self.project_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=20
            )
            output_content = f"Code de sortie: {result.returncode}\n"
            output_content += f"Stdout:\n{result.stdout[-1000:]}\n"
            output_content += f"Stderr:\n{result.stderr[-1000:]}"
            return result.returncode == 0, output_content
        except subprocess.TimeoutExpired:
            return False, "Erreur: Timeout de la commande shell (20s)."
        except Exception as e:
            return False, f"Erreur d'exécution shell: {e}"

class LearnSkillTool(Tool):
    name = "learn_skill"
    description = "Outil de META-ACTION pour l'apprentissage. À utiliser si une tâche échoue par manque d'information."

    def execute(self, reason: str = "", instruction: str = "", **kwargs) -> Tuple[bool, str]:
        if not instruction:
            return False, "Erreur: L'instruction ne peut pas être vide pour apprendre une compétence."
        if not reason and 'skill_name' in kwargs:
            reason = kwargs.pop('skill_name')
        task_name = reason or "Nouvelle Compétence"
        self.memory.learned_skills[task_name] = instruction
        self.memory.add_log(LogEntry(
            type="SKILL_LEARNED",
            content=f"Nouvelle compétence '{task_name}' apprise: {instruction}",
            success=True
        ))
        return True, f"Compétence '{task_name}' enregistrée. L'agent attend la prochaine requête de l'utilisateur."

class SemanticCodeSearchTool(Tool):
    name = "semantic_code_search"
    description = "Recherche sémantique dans le code source du projet : pose une question en langage naturel et obtiens les fichiers pertinents."

    def __init__(self, memory: Memory):
        super().__init__(memory)
        self._model = None
        self._file_cache = None

    def _build_code_index(self):
        if self._file_cache is not None:
            return
        self._file_cache = []
        for root, _, files in os.walk(self.project_path):
            for f in files:
                if f.endswith(('.py', '.js', '.ts', '.html', '.css', '.json', '.md')):
                    fp = Path(root) / f
                    try:
                        with open(fp, 'r', encoding='utf-8', errors='ignore') as file:
                            content = file.read()
                        rel_path = fp.relative_to(self.project_path)
                        self._file_cache.append((str(rel_path), content))
                    except Exception:
                        continue
        if SEMANTIC_SEARCH_AVAILABLE and self._model is None:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')

    def execute(self, query: str) -> Tuple[bool, str]:
        self._build_code_index()
        if not self._file_cache:
            return False, "Aucun fichier de code trouvé dans le projet."

        if not SEMANTIC_SEARCH_AVAILABLE:
            cmd = ["rg", "-i", "--no-heading", "--line-number", query, str(self.project_path)]
            if not shutil.which("rg"):
                cmd = ["grep", "-r", "-n", "-i", query, str(self.project_path)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    return True, f"Résultats de recherche textuelle pour '{query}':\n{result.stdout[:2000]}"
                else:
                    return False, f"Aucun résultat trouvé pour '{query}' (recherche textuelle)."
            except Exception as e:
                return False, f"Échec de la recherche textuelle : {e}"

        try:
            query_emb = self._model.encode(query, convert_to_tensor=True)
            file_snippets = [f"{path}:\n{content[:500]}" for path, content in self._file_cache]
            file_embs = self._model.encode(file_snippets, convert_to_tensor=True)
            hits = util.semantic_search(query_emb, file_embs, top_k=5)
            results = []
            for hit in hits[0]:
                idx = hit['corpus_id']
                score = hit['score']
                if score > 0.3:
                    results.append(f"[Score: {score:.2f}] {file_snippets[idx][:200]}...")
            if results:
                return True, "Résultats sémantiques :\n" + "\n\n".join(results)
            else:
                return False, "Aucun résultat pertinent trouvé (recherche sémantique)."
        except Exception as e:
            return False, f"Erreur dans la recherche sémantique : {e}"

class VisionToCodeTool(Tool):
    name = "vision_to_code"
    description = "Génère du HTML/CSS à partir d'une capture d'écran fournie."

    def execute(self, image_path: str) -> Tuple[bool, str]:
        if not genai:
            return False, "Vision désactivée : Google GenAI non disponible."
        gemini_key = get_gemini_key()
        if not gemini_key:
            return False, "Clé API Gemini manquante pour la vision."
        image_path = safe_join(self.project_path, image_path)
        if not image_path or not image_path.exists():
            return False, f"Image introuvable : {image_path}"
        try:
            client = genai.Client(api_key=gemini_key)
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
            prompt = (
                "Analyse cette capture d'écran d'interface utilisateur. "
                "Génère un code HTML/CSS moderne, responsive et propre qui reproduit fidèlement ce design. "
                "Utilise Tailwind CSS si possible. Ne donne que le code, sans explication."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part(text=prompt),
                    types.Part(inline_data={"mime_type": "image/png", "data": img_data})
                ]
            )
            code = response.text.strip()
            output_path = "generated_ui.html"
            with open(self.project_path / output_path, "w", encoding="utf-8") as f:
                f.write(code)
            return True, f"✅ Code généré à partir de l'image et sauvegardé dans `{output_path}`.\n\n```html\n{code[:1000]}...\n```"
        except Exception as e:
            return False, f"❌ Échec de la génération vision → code : {e}"

# ==============================================================================
# 6. AGENT ORCHESTRATOR ET CORE
# ==============================================================================
class GeminiCore:
    def __init__(self, key: str, memory: Memory, tools: List[Tool]):
        self.client = genai.Client(api_key=key)
        self.model = "gemini-2.5-flash"
        self.memory = memory
        self.tools = {tool.name: tool for tool in tools}

    def _get_system_prompt(self, action_results: str = "") -> str:
        project_tree = self.memory.get_project_tree()
        action_log = self.memory.get_action_history(5)
        learned_skills = self.memory.get_learned_skills_context()

        tool_descriptions = "\n".join([
            f"  - **{t.name}** : {t.description}" +
            (f"  Args attendus : {list(inspect.signature(t.execute).parameters.keys())}" if t.name == "learn_skill" else "")
            for t in self.tools.values()
        ])

        return f"""
Tu es Jules v2.9, un **Agent IA Central** : ingénieur logiciel, expert en cybersécurité et orchestrateur de projet.
**MISSION PRINCIPALE (Décision Binaire) :**
1. **MODE: CHAT** : Question simple, conversation, résumé.
2. **MODE: PLAN** : Tâche nécessitant des outils (fichiers, shell, vision, recherche sémantique…).

## CONTEXTE
- **Chemin du Projet :** {self.memory.project_path.resolve()}
- **Arborescence (Aperçu) :**
{project_tree}
- **Dernières Actions :**
{action_log}
- **Compétences Apprises :**
{learned_skills}

## OUTILS DISPONIBLES (MODE PLAN UNIQUEMENT)
{tool_descriptions}
{action_results}

## FORMAT DE RÉPONSE OBLIGATOIRE
### CHAT
```json
{{"MODE": "CHAT", "RESPONSE_TEXT": "…"}}
```
### PLAN
```json
{{
    "MODE": "PLAN",
    "PLAN": [
        {{
            "step_id": 1,
            "reasoning": "…",
            "action": {{
                "tool_name": "nom_outil",
                "args": {{ "arg1": "valeur1" }}
            }}
        }}
    ]
}}
```
"""

    def process_request(self, user_prompt: str, action_results: str = "") -> Tuple[str, Optional[List[ActionTool]]]:
        system_prompt = self._get_system_prompt(action_results=action_results)
        prompt_content = f"OBJECTIF UTILISATEUR: {user_prompt}"
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt_content)])],
                config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.1)
            )
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            plan_data = json.loads(response_text)
            mode = plan_data.get('MODE', 'CHAT').upper()
            if mode == 'CHAT':
                return plan_data.get('RESPONSE_TEXT', "Réponse vide."), None
            elif mode == 'PLAN':
                actions = []
                for step in plan_data.get('PLAN', []):
                    action = step.get('action', {})
                    tool_name = action.get('tool_name')
                    if tool_name in self.tools:
                        actions.append(ActionTool(
                            tool_name=tool_name,
                            args=action.get('args', {}),
                            reasoning=step.get('reasoning', 'Raisonnement non fourni.')
                        ))
                    else:
                        self.memory.add_log(LogEntry(type="SYSTEM_PLANNING_ERROR", content=f"Outil invalide ignoré: {tool_name}", success=False))
                return "", actions
            else:
                return f"❌ Erreur: Mode invalide '{mode}'.", None
        except Exception as e:
            return f"❌ Erreur lors de la génération: {e}. Réponse brute:\n{response_text}", None

class JulesAgent:
    def __init__(self, project_path: str, no_gemini: bool = False):
        self.memory = Memory(project_path)
        self.tools: List[Tool] = [
            ListFilesTool(self.memory),
            ReadFileTool(self.memory),
            WriteFileTool(self.memory),
            RunShellTool(self.memory),
            LearnSkillTool(self.memory),
            SemanticCodeSearchTool(self.memory),
            VisionToCodeTool(self.memory),
        ]
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.no_gemini = no_gemini
        self.core: Optional[GeminiCore] = None
        if not no_gemini and genai:
            gemini_key = get_gemini_key()
            if gemini_key:
                self.core = GeminiCore(gemini_key, self.memory, self.tools)
            else:
                self.no_gemini = True
        if self.no_gemini:
            self.memory.add_log(LogEntry(type="SYSTEM_WARNING", content="Mode local/sans IA activé.", success=False))

    def process_query(self, query: str, progress_callback: Optional[Callable] = None) -> str:
        if not self.core:
            return "❌ Agent IA non opérationnel."

        self.memory.add_log(LogEntry(type="USER_INPUT", content=query))
        if progress_callback: progress_callback("🧠 Classification...", 5)

        chat_response, actions = self.core.process_request(query)
        if chat_response and not actions:
            self.memory.add_log(LogEntry(type="FINAL_RESPONSE", content=chat_response, success=True))
            if progress_callback: progress_callback("✅ Réponse instantanée", 100)
            return chat_response
        if not actions:
            return "⚠️ Plan vide."

        if progress_callback: progress_callback("🛠️ Exécution du plan...", 10)
        step_results_log = []
        max_retries = 2
        retry_count = 0
        current_actions = actions

        while retry_count <= max_retries:
            all_success = True
            for i, action in enumerate(current_actions):
                tool = self.tool_map.get(action.tool_name)
                if not tool:
                    result = f"❌ Outil '{action.tool_name}' introuvable."
                    self.memory.add_log(LogEntry(type="TOOL_RESULT", content=result, success=False))
                    step_results_log.append(result)
                    all_success = False
                    break

                # ← Garde-fou pour learn_skill
                if action.tool_name == "learn_skill":
                    action.args.setdefault("instruction", action.args.pop("skill_content", None))

                self.memory.add_log(LogEntry(type="TOOL_CALL", content=f"{action.tool_name} {action.args}"))
                success, result = tool.execute(**action.args)
                self.memory.add_log(LogEntry(type="TOOL_RESULT", content=result, success=success))
                step_results_log.append(f"{'✅' if success else '❌'} Étape {i+1} ({action.tool_name}): {result[:200]}...")

                if not success and action.tool_name != "list_files":
                    all_success = False
                    break
                if action.tool_name == "learn_skill" and success:
                    break

            if all_success:
                if len(current_actions) == 1:
                    first_action = current_actions[0]
                    if first_action.tool_name in ["list_files", "read_file"]:
                        last_log = self.memory.action_log[-1]
                        if last_log.type == "TOOL_RESULT" and last_log.success:
                            final_direct_output = last_log.content
                            self.memory.add_log(LogEntry(type="FINAL_RESPONSE", content=final_direct_output, success=True))
                            if progress_callback: progress_callback("✅ Résultat direct", 100)
                            return final_direct_output
                break

            retry_count += 1
            if retry_count > max_retries:
                break
            error_context = "\n".join(step_results_log[-3:])
            retry_prompt = (
                f"L'action suivante a échoué : '{query}'. "
                f"Derniers résultats :\n{error_context}\n"
                "Propose un **nouveau plan alternatif**."
            )
            if progress_callback:
                progress_callback(f"🔄 Tentative {retry_count}/2...", 50)
            _, new_actions = self.core.process_request(retry_prompt)
            if not new_actions:
                break
            current_actions = new_actions
            step_results_log.append(f"\n[🔄 Tentative {retry_count} : Nouveau plan]")

        final_context = "\n".join(step_results_log)
        if progress_callback: progress_callback("📝 Rapport final...", 95)
        final_prompt = f"Rédige une réponse professionnelle en français. Requête initiale: '{query}'. Résultats:\n{final_context}"
        final_response, _ = self.core.process_request(final_prompt)
        if final_response:
            self.memory.add_log(LogEntry(type="FINAL_RESPONSE", content=final_response, success=True))
            if progress_callback: progress_callback("✅ Tâche Complète", 100)
            return final_response
        else:
            fallback_msg = "✅ Tâche exécutée. Résultats bruts :\n" + final_context
            self.memory.add_log(LogEntry(type="FINAL_RESPONSE", content=fallback_msg, success=True))
            if progress_callback: progress_callback("✅ Tâche Complète (brut)", 100)
            return fallback_msg

# ==============================================================================
# 7. INTERFACE WEB (GRADIO) - avec support vision
# ==============================================================================
class WebUI:
    def __init__(self, agent: JulesAgent):
        self.agent = agent

    def _refresh_state(self):
        tree = self.agent.memory.get_project_tree()
        skills_status = f"Compétences Apprises: {len(self.agent.memory.learned_skills)}"
        history = "\n".join([
            f"[{e.timestamp.strftime('%H:%M:%S')}] {e.type} ({'OK' if e.success else 'FAIL' if e.success is False else '-'}):\n{e.content[:80].replace('\n', ' ')}..."
            for e in self.agent.memory.action_log[::-1][:20]
        ])
        return tree, skills_status, history

    def _process_query_web(self, chat_history, user_input, image_input):
        if not user_input.strip() and image_input is None:
            return chat_history, "", None, *self._refresh_state()
        gr_progress = gr.Progress()
        def web_progress_update(message, percentage):
            gr_progress(percentage / 100, desc=f"Jules : {message}")
        try:
            actual_query = user_input
            if image_input is not None:
                img_path = self.agent.memory.project_path / "uploaded_screenshot.png"
                from PIL import Image
                img = Image.fromarray(image_input)
                img.save(img_path)
                actual_query = f"Génère le code HTML/CSS à partir de cette capture d'écran : uploaded_screenshot.png"
            response = self.agent.process_query(actual_query, web_progress_update)
        except Exception as e:
            response = f"❌ Erreur : {e}"
        chat_history.append([user_input or "(image)", response])
        tree, skills_status, history = self._refresh_state()
        return chat_history, "", None, tree, skills_status, history

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Jules v2.9 Enhanced", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🤖 Jules v2.9 Enhanced - Chat, Code, Vision & Semantic Search")
            gr.Markdown(f"**Projet :** `{self.agent.memory.project_path.resolve()}`")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    chat = gr.Chatbot(label="Conversation", height=500, show_copy_button=True)
                    with gr.Row():
                        user_msg = gr.Textbox(label="Message", placeholder="Décris ce que tu veux faire…")
                        image_input = gr.Image(label="📸 Capture d’écran (optionnel)", type="numpy")
                    with gr.Row():
                        submit_btn = gr.Button("🚀 Exécuter", variant="primary")
                        clear_btn = gr.Button("🗑️ Effacer")
                with gr.Column(scale=1):
                    gr.Markdown("### Contexte")
                    tree_area = gr.Textbox(label="Arborescence", lines=10, interactive=False)
                    skills_status = gr.Textbox(label="Compétences", lines=1, interactive=False)
                    gr.Markdown("### Journal")
                    history_area = gr.Textbox(label="Actions récentes", lines=10, interactive=False)
                    initial_tree, initial_skills, initial_hist = self._refresh_state()
                    tree_area.value = initial_tree
                    skills_status.value = initial_skills
                    history_area.value = initial_hist

            outputs = [chat, user_msg, image_input, tree_area, skills_status, history_area]
            submit_btn.click(self._process_query_web, [chat, user_msg, image_input], outputs)
            user_msg.submit(self._process_query_web, [chat, user_msg, image_input], outputs)
            clear_btn.click(lambda: ([], "", None, *self._refresh_state()), None, outputs)
            demo.load(self._refresh_state, None, [tree_area, skills_status, history_area])
        return demo

# ==============================================================================
# 8. CLI
# ==============================================================================
def cli_interface(agent: JulesAgent):
    if CLI_CONSOLE is None:
        print("❌ 'rich' non installé.")
        sys.exit(1)
    CLI_CONSOLE.print(Markdown("# 🤖 Jules v2.9 Enhanced CLI"))
    CLI_CONSOLE.print(f"[bold blue]Projet:[/bold blue] [yellow]{agent.memory.project_path.resolve()}[/yellow]")
    CLI_CONSOLE.print("[dim]Commandes : 'exit', '/tree', ou demandez une tâche.[/dim]")
    if agent.no_gemini:
        CLI_CONSOLE.print("[bold red]⚠️ Mode sans IA[/bold red]")

    while True:
        try:
            user_input = input(f"\n[JULES] {agent.memory.project_path.name}$ ")
            if user_input.lower() in ["exit", "quit", "q"]:
                CLI_CONSOLE.print("[bold green]Au revoir ![/bold green]")
                break
            if user_input.strip() == "/tree":
                CLI_CONSOLE.print("[bold cyan]Arborescence :[/bold cyan]")
                CLI_CONSOLE.print(agent.memory.get_project_tree())
                continue
            if not user_input.strip():
                continue

            def cli_progress_update(message, percentage):
                CLI_CONSOLE.print(f"[dim]({percentage}%) {message}[/dim]")

            CLI_CONSOLE.print("[bold magenta]🧠 Traitement...[/bold magenta]")
            response = agent.process_query(user_input, cli_progress_update)
            CLI_CONSOLE.print("\n" + "="*50)
            CLI_CONSOLE.print("[bold magenta]Jules :[/bold magenta]")
            stream_print(response, delay=0.015)
            CLI_CONSOLE.print("="*50 + "\n")
        except KeyboardInterrupt:
            CLI_CONSOLE.print("\n[bold red]Interruption.[/bold red]")
            break
        except Exception as e:
            CLI_CONSOLE.print(f"[bold red]❌ Erreur CLI: {e}[/bold red]")

def parse_args():
    p = argparse.ArgumentParser(description="Jules v2.9 Enhanced")
    p.add_argument("--project", "-p", default=".", help="Chemin du projet")
    p.add_argument("--web", action="store_true", help="Lancer interface Web")
    p.add_argument("--no-gemini", action="store_true", help="Désactiver Gemini")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        if genai is None and not args.no_gemini:
            if CLI_CONSOLE: CLI_CONSOLE.print("[bold red]❌ google-genai non installé → mode sans IA.[/bold red]")
            args.no_gemini = True
        agent = JulesAgent(project_path=args.project, no_gemini=args.no_gemini)
    except Exception as e:
        print(f"❌ Erreur init: {e}")
        sys.exit(1)

    if args.web:
        if gr is None:
            print("❌ Gradio non installé. Exécutez : pip install gradio")
            return
        WebUI(agent).create_interface().launch(inbrowser=True)
    else:
        cli_interface(agent)

if __name__ == "__main__":
    main()