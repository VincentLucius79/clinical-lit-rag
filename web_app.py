"""
web_app.py

A minimal local web UI -- same retrieval + generation logic as query.py
(it imports get_answer() straight from there), with a browser front end
instead of a terminal. The only network call it makes is to your own
Ollama server.

Usage:
  python web_app.py
  Then open http://127.0.0.1:5050 in a browser.

By default it listens on 127.0.0.1 only, so nothing else on your network can
reach it. To reach it from another of your own devices, bind it to a private
interface you control (for example a Tailscale IP) rather than 0.0.0.0:

  CLINICAL_LIT_HOST=100.x.y.z python web_app.py
"""

import os

import chromadb
from flask import Flask, jsonify, render_template_string, request

from build_index import CHROMA_DIR, COLLECTION_NAME, OllamaEmbeddingFunction
from query import DEFAULT_MODEL, get_answer

HOST = os.environ.get("CLINICAL_LIT_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLINICAL_LIT_PORT", "5050"))

app = Flask(__name__)

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_embed_fn = OllamaEmbeddingFunction()
_collection = _client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embed_fn)

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical Literature RAG</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 1.3rem; }
  .note { font-size: 0.8rem; opacity: 0.7; margin-top: -8px; margin-bottom: 20px; }
  #chat { display: flex; flex-direction: column; gap: 16px; margin-bottom: 24px; }
  .msg { padding: 12px 16px; border-radius: 10px; line-height: 1.5; white-space: pre-wrap; }
  .question { background: rgba(37,99,235,0.12); align-self: flex-end; max-width: 80%; }
  .answer { background: rgba(107,114,128,0.12); }
  .sources { font-size: 0.8rem; opacity: 0.7; margin-top: 8px; }
  form { display: flex; gap: 8px; position: sticky; bottom: 16px; }
  input { flex: 1; padding: 10px 12px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.4); font-size: 1rem; }
  button { padding: 10px 18px; border-radius: 8px; border: none; background: #2563eb; color: white; font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  .loading { opacity: 0.6; font-style: italic; }
</style>
</head>
<body>
<h1>Clinical Literature RAG</h1>
<p class="note">Summarizes indexed research. Not medical advice.</p>
<div id="chat"></div>
<form id="form">
  <input id="question" placeholder="Ask about the indexed literature..." autocomplete="off" autofocus>
  <button id="send" type="submit">Ask</button>
</form>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('question');
const button = document.getElementById('send');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;

  const qDiv = document.createElement('div');
  qDiv.className = 'msg question';
  qDiv.textContent = q;
  chat.appendChild(qDiv);

  const aDiv = document.createElement('div');
  aDiv.className = 'msg answer loading';
  aDiv.textContent = 'Thinking...';
  chat.appendChild(aDiv);

  input.value = '';
  input.disabled = true;
  button.disabled = true;
  aDiv.scrollIntoView({behavior: 'smooth'});

  try {
    const resp = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await resp.json();
    aDiv.classList.remove('loading');
    if (data.error) {
      aDiv.textContent = data.error;
    } else {
      aDiv.textContent = data.answer;
      if (data.sources && data.sources.length) {
        const srcDiv = document.createElement('div');
        srcDiv.className = 'sources';
        srcDiv.textContent = 'Sources: ' + data.sources.map(s => `${s.title} (${s.date})`).join(' | ');
        aDiv.appendChild(srcDiv);
      }
    }
  } catch (err) {
    aDiv.classList.remove('loading');
    aDiv.textContent = 'Error: ' + err;
  } finally {
    input.disabled = false;
    button.disabled = false;
    input.focus();
    aDiv.scrollIntoView({behavior: 'smooth'});
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question."}), 400
    try:
        result = get_answer(_collection, DEFAULT_MODEL, question)
    except Exception as e:
        # Without this, an unhandled exception returns Flask's default HTML error
        # page, which breaks the front end's resp.json() with a confusing message.
        app.logger.exception("/api/ask failed")
        return jsonify({
            "error": f"Server error: {type(e).__name__}: {e}. "
                     f"Check the terminal running web_app.py for the full traceback."
        }), 500
    return jsonify(result)


if __name__ == "__main__":
    if HOST == "0.0.0.0":
        print("Warning: listening on every network interface. Prefer 127.0.0.1 or a private IP.")
    print(f"Web UI: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
