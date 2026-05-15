
"""
Adaptive Conversational History Tutor
Authors : Hugo Elhaik & Sami Rtel
Dataset Inspiration : QReCC — Anantha et al., NAACL 2021
"""

# ── Auto-launch Streamlit ───────────────────────────────────────────────────
import sys
import os
import subprocess

_is_streamlit = (
    "streamlit" in sys.modules
    or any("streamlit" in a for a in sys.argv)
)

if not _is_streamlit:

    current_file = globals().get("__file__")

    if current_file and os.path.isfile(current_file):

        print("🚀 Launching Streamlit...")

        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            current_file
        ])

    else:

        print("⚠️ Save the file first, then run again.")

    sys.exit(0)


# ── Imports ────────────────────────────────────────────────────────────────
import re
import math
import string
import streamlit as st

from collections import Counter


# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive History Tutor",
    page_icon="🏛️",
    layout="wide"
)


# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.banner {
    background: linear-gradient(135deg,#2c3e50,#4ca1af);
    border-radius:12px;
    padding:1.2rem 2rem;
    color:white;
    margin-bottom:1rem;
}
.banner h1 {
    margin:0;
    font-size:1.8rem;
}
.banner p {
    margin:.3rem 0 0;
    opacity:.85;
}
.badge {
    display:inline-block;
    background:rgba(255,255,255,.18);
    border-radius:20px;
    padding:2px 11px;
    font-size:.76rem;
    margin-right:5px;
    margin-top:.6rem;
    color:white;
}
.chip {
    display:inline-block;
    background:#e3f2fd;
    color:#1565c0;
    border-radius:10px;
    padding:1px 8px;
    font-size:.75rem;
    margin-right:4px;
}
.mode {
    border-radius:7px;
    padding:.35rem .9rem;
    font-weight:600;
    font-size:.82rem;
    display:inline-block;
    margin-bottom:.4rem;
}
.beginner{background:#e3f2fd;color:#1565c0}
.analogy {background:#fff9c4;color:#e65100}
.concise {background:#e8f5e9;color:#2e7d32}
.detailed{background:#f3e5f5;color:#6a1b9a}
.socratic{background:#fce4ec;color:#880e4f}
.conf {
    background:#fff3e0;
    border-left:4px solid #ff9800;
    border-radius:0 8px 8px 0;
    padding:.45rem .9rem;
    font-size:.84rem;
    color:#e65100;
    margin-bottom:.4rem;
}
.rewrite {
    background:#f0f4ff;
    border-left:3px solid #3f51b5;
    border-radius:0 6px 6px 0;
    padding:.35rem .8rem;
    font-size:.82rem;
    color:#283593;
    margin-bottom:.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── Banner ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>🏛️ Adaptive Conversational History Tutor</h1>
  <p>Context-aware educational conversational AI inspired by QReCC</p>

  <span class="badge">🧠 Conversational Memory</span>
  <span class="badge">✍️ Question Rewriting</span>
  <span class="badge">⚡ Pure Python</span>
  <span class="badge">🎓 Adaptive Teaching</span>
  <span class="badge">🔍 Retrieval</span>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ───────────────────────────────────────────────────────────────────────────
KB = [

    {
        "topic": "world war 1",
        "content": "World War I (1914–1918) began after the assassination of Archduke Franz Ferdinand. Major causes included militarism, alliances, imperialism, and nationalism."
    },

    {
        "topic": "world war 2",
        "content": "World War II (1939–1945) involved the Allies and Axis powers. Major causes included the Treaty of Versailles, expansionist policies of Nazi Germany, and economic instability."
    },

    {
        "topic": "french revolution",
        "content": "The French Revolution (1789–1799) overthrew the monarchy and promoted liberty, equality, and fraternity. It was driven by social inequality and financial crisis."
    },

    {
        "topic": "roman empire",
        "content": "The Roman Empire became one of the largest civilizations in history. It expanded through military conquest and influenced law, engineering, and governance."
    },

    {
        "topic": "napoleon",
        "content": "Napoleon Bonaparte was a French military leader who rose during the French Revolution and became Emperor of France. He led major European wars."
    },

    {
        "topic": "cold war",
        "content": "The Cold War was a geopolitical conflict between the United States and the Soviet Union after World War II. It involved ideological rivalry, nuclear arms races, and proxy wars."
    },

    {
        "topic": "renaissance",
        "content": "The Renaissance was a cultural movement beginning in Italy during the 14th century that revived interest in classical art, science, and humanism."
    },

    {
        "topic": "industrial revolution",
        "content": "The Industrial Revolution transformed economies through mechanization, factories, and technological innovation during the 18th and 19th centuries."
    },

    {
        "topic": "ancient egypt",
        "content": "Ancient Egypt was a civilization centered around the Nile River known for pyramids, pharaohs, and advances in writing and engineering."
    },

    {
        "topic": "alexander the great",
        "content": "Alexander the Great created one of the largest empires of the ancient world and spread Greek culture across Asia and Egypt."
    }
]


# ───────────────────────────────────────────────────────────────────────────
# TF-IDF RETRIEVAL
# ───────────────────────────────────────────────────────────────────────────
_STOP = {
    "the","a","an","is","are","was","were","be","and",
    "or","in","on","at","to","for","of","with","it",
    "its","i","you","we","do","not","this","that",
    "what","how","why","who","can","could","would"
}


def tokenize(text):

    return [
        t for t in re.sub(r"[^a-z\\s]", "", text.lower()).split()
        if t and t not in _STOP
    ]


@st.cache_resource(show_spinner="⚡ Building retrieval index...")
def build_index():

    docs = [k["content"] for k in KB]

    tokens = [tokenize(d) for d in docs]

    N = len(docs)

    df = Counter(
        t for ts in tokens for t in set(ts)
    )

    idf = {
        t: math.log((N+1)/(c+1))+1
        for t, c in df.items()
    }

    def vectorize(ts):

        tf = Counter(ts)

        n = max(len(ts), 1)

        return {
            t: (c/n) * idf.get(t, 1)
            for t, c in tf.items()
        }

    vectors = [vectorize(ts) for ts in tokens]

    return vectors, idf


vectors, idf = build_index()


def cosine(a, b):

    keys = set(a) & set(b)

    dot = sum(a[k] * b[k] for k in keys)

    ma = math.sqrt(sum(v*v for v in a.values()))

    mb = math.sqrt(sum(v*v for v in b.values()))

    return dot / (ma * mb) if ma and mb else 0.0


def retrieve(query, k=2):

    tokens = tokenize(query)

    n = max(len(tokens), 1)

    qv = {
        t: (Counter(tokens)[t]/n) * idf.get(t,1)
        for t in tokens
    }

    scored = sorted(
        enumerate(vectors),
        key=lambda x: cosine(qv, x[1]),
        reverse=True
    )

    return [KB[i] for i, _ in scored[:k]]


# ───────────────────────────────────────────────────────────────────────────
# TOPICS
# ───────────────────────────────────────────────────────────────────────────
TOPICS = [
    "world war 1",
    "world war 2",
    "french revolution",
    "roman empire",
    "napoleon",
    "cold war",
    "renaissance",
    "industrial revolution",
    "american revolution",
    "ancient egypt",
    "greek civilization",
    "ottoman empire",
    "hitler",
    "stalin",
    "julius caesar",
    "alexander the great",
    "middle ages",
    "black death",
    "colonialism",
    "imperialism"
]


# ───────────────────────────────────────────────────────────────────────────
# TOPIC EXTRACTION
# ───────────────────────────────────────────────────────────────────────────
def get_topic(history, question):

    q = question.lower()

    for topic in TOPICS:

        if topic in q:

            return topic

    for h in reversed(history[-4:]):

        for topic in TOPICS:

            if topic in h.lower():

                return topic

    return "history"


# ───────────────────────────────────────────────────────────────────────────
# QUESTION REWRITING
# ───────────────────────────────────────────────────────────────────────────
def rewrite(history, question):

    topic = get_topic(history, question)

    q = question.lower().strip().rstrip("?").strip()

    ellipsis_map = {

        "why":
        f"Why was {topic} important in history?",

        "how":
        f"How did {topic} influence history?",

        "what":
        f"What was {topic}?",

        "when":
        f"When did {topic} happen?",

        "who":
        f"Who was involved in {topic}?",

        "tell me more":
        f"Explain {topic} in more detail.",
    }

    if q in ellipsis_map:

        return ellipsis_map[q], "ellipsis_expansion", topic

    out = question

    changed = False

    for pattern, replacement in {

        r"\\bit\\b": topic,
        r"\\bthis\\b": topic,
        r"\\bthat\\b": topic,
        r"\\bthey\\b": f"people involved in {topic}",

    }.items():

        new_text, n = re.subn(
            pattern,
            replacement,
            out,
            flags=re.IGNORECASE
        )

        if n:

            out = new_text

            changed = True

    if changed:

        return out, "pronoun_resolution", topic

    return question, "passthrough", topic


# ───────────────────────────────────────────────────────────────────────────
# CONFUSION DETECTION
# ───────────────────────────────────────────────────────────────────────────
CONFUSION_PATTERNS = [
    r"don'?t understand",
    r"confused",
    r"not clear",
    r"explain again",
    r"simplify",
    r"what do you mean",
    r"still don'?t get",
    r"can you explain simply"
]


def detect_confusion(question):

    q = question.lower()

    for pattern in CONFUSION_PATTERNS:

        if re.search(pattern, q):

            return True

    return False


# ───────────────────────────────────────────────────────────────────────────
# TEACHING STYLES
# ───────────────────────────────────────────────────────────────────────────
STYLES = {

    "world war 1": {

        "beginner":
        "World War I started in 1914 after the assassination of Archduke Franz Ferdinand.",

        "analogy":
        "Imagine a small conflict triggering a chain reaction because countries were tightly connected through alliances.",

        "concise":
        "WWI: 1914–1918. Causes: alliances, militarism, nationalism, imperialism.",

        "detailed":
        "World War I involved major European powers divided into alliances. The assassination of Franz Ferdinand triggered existing tensions.",

        "socratic":
        "Why do you think alliances can turn a small conflict into a world war?"
    },

    "french revolution": {

        "beginner":
        "The French Revolution overthrew the monarchy and changed French society.",

        "analogy":
        "Imagine people rebelling against an unfair system where only the wealthy had power.",

        "concise":
        "French Revolution: 1789–1799. Goals: liberty, equality, fraternity.",

        "detailed":
        "The French Revolution was caused by inequality, financial crisis, and dissatisfaction with the monarchy.",

        "socratic":
        "What happens when ordinary citizens feel excluded from political power?"
    },

    "cold war": {

        "beginner":
        "The Cold War was a long rivalry between the USA and the Soviet Union after WWII.",

        "analogy":
        "It was like two rivals competing globally without directly fighting each other.",

        "concise":
        "Cold War: USA vs USSR, nuclear tension, proxy wars.",

        "detailed":
        "The Cold War involved ideological conflict between capitalism and communism, nuclear arms races, and geopolitical competition.",

        "socratic":
        "Why might countries avoid direct war if nuclear weapons exist?"
    }
}


# ───────────────────────────────────────────────────────────────────────────
# IMPORTANCE RESPONSES
# ───────────────────────────────────────────────────────────────────────────
IMPORTANCE = {

    "world war 1":
    "World War I changed global politics and contributed to the conditions leading to World War II.",

    "french revolution":
    "The French Revolution influenced democracy, human rights, and political revolutions worldwide.",

    "cold war":
    "The Cold War shaped modern geopolitics, nuclear policy, and international relations.",

    "roman empire":
    "The Roman Empire influenced modern law, governance, language, and engineering.",

    "renaissance":
    "The Renaissance transformed art, science, and intellectual thought in Europe."
}


# ───────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ───────────────────────────────────────────────────────────────────────────
def pipeline(history, question):

    rewritten, method, topic = rewrite(
        history,
        question
    )

    docs = retrieve(rewritten, k=2)

    confused = detect_confusion(question)

    q_low = question.lower()

    if confused:

        mode = "analogy"

    elif re.search(r"briefly|short|concise", q_low):

        mode = "concise"

    elif re.search(r"detail|in detail|thorough", q_low):

        mode = "detailed"

    elif re.search(r"guide me|socratic", q_low):

        mode = "socratic"

    else:

        mode = "beginner"

    if (
        re.search(r"important|why", q_low)
        and topic in IMPORTANCE
    ):

        response = IMPORTANCE[topic]

    elif topic in STYLES:

        response = STYLES[topic][mode]

    elif docs:

        response = docs[0]["content"]

    else:

        response = (
            "I could not find enough historical context. "
            "Please rephrase your question."
        )

    return {

        "rewritten": rewritten,

        "method": method,

        "topic": topic,

        "mode": mode,

        "response": response,

        "docs": docs,

        "confused": confused
    }


# ───────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ───────────────────────────────────────────────────────────────────────────
if "chat" not in st.session_state:

    st.session_state.chat = []

if "history" not in st.session_state:

    st.session_state.history = []


# ───────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.header("📘 Demo Examples")

    st.markdown("""
### Context Awareness
- What was the French Revolution?
- Why was it important?
- What happened after it?

### Confusion Detection
- Explain the Cold War.
- I still don't understand.

### Adaptive Teaching
- Explain World War 1 briefly.
- Explain in detail.

### History Questions
- Who was Napoleon?
- What caused World War II?
""")

    st.header("⚙️ Features")

    st.markdown("""
✅ Conversational Memory

✅ Question Rewriting

✅ Confusion Detection

✅ Adaptive Teaching

✅ Educational Dialogue

✅ Context Awareness
""")


# ───────────────────────────────────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ───────────────────────────────────────────────────────────────────────────
for msg in st.session_state.chat:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])


# ───────────────────────────────────────────────────────────────────────────
# USER INPUT
# ───────────────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask a history question..."
)


# ───────────────────────────────────────────────────────────────────────────
# PROCESS INPUT
# ───────────────────────────────────────────────────────────────────────────
if user_input:

    st.session_state.chat.append({
        "role": "user",
        "content": user_input
    })

    result = pipeline(
        st.session_state.history,
        user_input
    )

    response = result["response"]

    st.session_state.chat.append({
        "role": "assistant",
        "content": response
    })

    st.session_state.history.append(user_input)

    st.session_state.history.append(response)

    with st.chat_message("user"):

        st.write(user_input)

    with st.chat_message("assistant"):

        chips = []

        if result["method"] != "passthrough":

            chips.append(
                f"✍️ Rewrite: {result['method']}"
            )

        if result["confused"]:

            chips.append("🧠 Confusion Detected")

        chips.append(
            f"📚 Mode: {result['mode']}"
        )

        st.markdown(
            " ".join(
                f'<span class="chip">{c}</span>'
                for c in chips
            ),
            unsafe_allow_html=True
        )

        if result["confused"]:

            st.markdown(
                '<div class="conf">⚠️ Confusion detected — switching to analogy mode.</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f'<span class="mode {result["mode"]}">{result["mode"].capitalize()} Mode</span>',
            unsafe_allow_html=True
        )

        st.write(response)

        with st.expander("🧠 Tutor Reasoning"):

            st.markdown("### Rewritten Question")
            st.write(result["rewritten"])

            st.markdown("### Topic")
            st.write(result["topic"])

            st.markdown("### Teaching Mode")
            st.write(result["mode"])

            st.markdown("### Retrieved Knowledge")

            for d in result["docs"]:

                st.write(
                    f"- [{d['topic']}] {d['content']}"
                )


# ───────────────────────────────────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────────────────────────────────
st.markdown("---")

st.markdown(
    "<center><small>🏛️ Adaptive History Tutor · Conversational Educational AI · Pure Python TF-IDF</small></center>",
    unsafe_allow_html=True
)
