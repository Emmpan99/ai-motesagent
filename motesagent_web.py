# motesagent_web.py
import io
import json
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="AI Mötesagent", page_icon="🤖")

st.title("🤖 AI Mötesagent – känslofokus")
st.caption("Ladda upp .txt / .docx / .pdf → AI tolkar känslor (med evidens), ger stressfokuserade tidsrekommendationer och strukturerar anteckningar.")

# ---------------- Hjälpfunktioner för filinläsning ----------------
def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")

def read_docx(uploaded_file) -> str:
    import docx  # python-docx
    doc = docx.Document(uploaded_file)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(file_bytes: bytes) -> str:
    import pdfplumber
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks).strip()

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".txt"):
        return read_txt(data)
    elif name.endswith(".docx"):
        return read_docx(io.BytesIO(data))
    elif name.endswith(".pdf"):
        return read_pdf(data)
    else:
        return ""

# ---------------- Regelbaserad fallback-analys (känslofokuserad) ----------------
STRESS_WORDS = {"stress", "stressad", "orolig", "oro", "bråttom", "hinner inte", "press", "överbelastad"}
NEG_WORDS    = {"problem", "risk", "osäker", "tveksam", "oklar", "försening", "svårt", "frustrerad"}
POS_WORDS    = {"bra", "klart", "lugn", "fungerar", "positiv", "nöjd", "trygg", "stabil", "framsteg"}

def rules_analyze(text: str):
    t = text.lower()
    stress_score = sum(1 for w in STRESS_WORDS if w in t)
    neg_score    = sum(1 for w in NEG_WORDS if w in t)
    pos_score    = sum(1 for w in POS_WORDS if w in t)

    if stress_score > 0 and stress_score >= (pos_score + neg_score):
        emotion = "Stressad"
    elif (neg_score > pos_score) and neg_score > 0:
        emotion = "Frustrerad/Negativ"
    elif pos_score > (neg_score + stress_score) and pos_score > 0:
        emotion = "Positiv/Lugn"
    else:
        emotion = "Neutral/Osäker"

    # Enkla evidens (plocka ut korta matchande fraser)
    evidence = []
    for w in list(STRESS_WORDS | NEG_WORDS | POS_WORDS):
        if w in t and len(evidence) < 5:
            evidence.append(w)

    recs = []
    if stress_score > 0:
        recs.append("Lägg till mer tid för den mest tidskritiska uppgiften för att minska stress.")
        recs.append("Dela upp arbetet i mindre delmål med kortare avstämningar.")
    if neg_score > 0:
        recs.append("Boka ett kort klarläggande möte för att minska osäkerhet.")
    if not recs:
        recs.append("Fortsätt enligt plan.")

    return {
        "summary": "",
        "decisions": [],
        "action_items": [],
        "risks": [],  # lämnas kvar men visas längre ned
        "open_questions": [],
        "topics": [],
        "emotions": {"overall": emotion, "confidence": "låg–medel (regelbaserad)", "evidence": evidence, "by_speaker": []},
        "time_recommendations": recs[:],
        "scores": {"stress": stress_score, "neg": neg_score, "pos": pos_score},
        # Flatten för UI:
        "emotion": emotion,
        "confidence": "låg–medel (regelbaserad)",
        "recommendations": recs,
        "on_time_probability": None,  # sätts senare via heuristik
        "on_time_rationale": "",
    }

# ---------------- GPT-analys (ny OpenAI SDK) – Emotion-first prompt ----------------
def gpt_analyze(text: str, api_key: str, model: str = "gpt-4o-mini"):
    client = OpenAI(api_key=api_key)

    system_msg = (
        "Du är en AI-mötesassistent med känslofokus. Returnera ENDAST giltig JSON enligt specifikationen."
    )

    user_prompt = f"""
Du är en AI-mötesassistent med fokus på känsloklimat och arbetsbelastning. DITT PRIO:
1) Identifiera känslor (stress/oro/frustration/positiv/lugn) med evidens.
2) Härled konkreta tidsrekommendationer som minskar stress (t.ex. lägga till X tid, dela upp uppgift, flytta deadline).
3) Strukturera anteckningar (sammanfattning, action items, beslut) — men endast sekundärt.

Returnera ENDAST giltig JSON enligt specifikationen nedan. Om något saknas i texten, använd tomma listor eller null.

KRAV PÅ UTDATA (JSON):
{{
  "summary": "Kort övergripande sammanfattning (2–4 meningar, håll känslofokus).",
  "emotions": {{
    "overall": "Stressad|Frustrerad/Negativ|Positiv/Lugn|Neutral/Osäker",
    "confidence": "låg|medel|hög",
    "evidence": ["korta citat/fraser som stödjer bedömningen (max 5)"],
    "by_speaker": []
  }},
  "time_recommendations": [
    "Konkreta, genomförbara förslag som direkt minskar stress (lägg till X timmar/dagar, dela upp Y, flytta deadline Z, omfördela resurser)."
  ],
  "scores": {{"stress": 0, "neg": 0, "pos": 0}},

  "decisions": [
    {{"title": "Kort beslutsrubrik", "details": "Vad beslutades och varför", "timestamp": "HH:MM eller null"}}
  ],
  "action_items": [
    {{"title": "Åtgärd", "owner": "Namn eller 'okänd'", "due_date": "YYYY-MM-DD eller null", "priority": "hög|medel|låg", "notes": "Kort kontext"}}
  ],
  "risks": [],
  "open_questions": [
    {{"question": "Fråga som behöver svar", "owner": "Namn eller 'okänd'"}}
  ],
  "topics": ["kort ämne 1", "kort ämne 2"],

  "on_time_probability": 0,  <-- ändra till
  "on_time_probability": 0,   # heltal mellan 0 och 100 (procent)
  "on_time_rationale": "Kort motivering kopplad till känsloläge/stress"
}}

REGLER:
- PRIORITERA känslor och stressreducerande tidsrekommendationer före planeringspunkter.
- Om stress/oro antyds: ge MINST 1 tydlig tidsrekommendation kopplad till stressen (”för att minska stress…”).
- Max 5 evidenscitat. Listor max 5 punkter om inte texten motiverar fler.
- Returnera ENBART JSON (inga förklaringar, inga kodblock).
- on_time_probability ska vara ett heltal mellan 0 och 100, där 0 = ingen chans och 100 = garanterat i tid.

TEXT ATT ANALYSERA:
\"\"\"{text}\"\"\"
"""

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(raw[start:end+1])
        else:
            raise ValueError("Kunde inte tolka modellens JSON-svar:\n" + raw)

    # Defaults
    data.setdefault("summary", "")
    data.setdefault("decisions", [])
    data.setdefault("action_items", [])
    data.setdefault("risks", [])
    data.setdefault("open_questions", [])
    data.setdefault("topics", [])
    data.setdefault("emotions", {"overall":"Neutral/Osäker","confidence":"medel","evidence":[], "by_speaker":[]})
    data.setdefault("time_recommendations", [])
    data.setdefault("scores", {"stress": 0, "neg": 0, "pos": 0})
    data.setdefault("on_time_probability", None)
    data.setdefault("on_time_rationale", "")

    # Flatten för UI
    data.setdefault("emotion", data.get("emotions", {}).get("overall", "Neutral/Osäker"))
    data.setdefault("confidence", data.get("emotions", {}).get("confidence", "medel"))
    data.setdefault("recommendations", data.get("time_recommendations", []) or ["Fortsätt enligt plan."])

    return data

# ---------------- Normalisering ----------------
def normalize_result(result: dict) -> dict:
    if not isinstance(result, dict):
        return {
            "summary": "",
            "decisions": [],
            "action_items": [],
            "risks": [],
            "open_questions": [],
            "topics": [],
            "emotions": {"overall":"Neutral/Osäker","confidence":"medel","evidence":[], "by_speaker":[]},
            "time_recommendations": [],
            "scores": {"stress": 0, "neg": 0, "pos": 0},
            "emotion": "Neutral/Osäker",
            "confidence": "medel",
            "recommendations": ["Fortsätt enligt plan."],
            "on_time_probability": None,
            "on_time_rationale": "",
        }

    emo = result.get("emotions", {})
    if "emotion" not in result and isinstance(emo, dict) and emo.get("overall"):
        result["emotion"] = emo["overall"]
    if "confidence" not in result and isinstance(emo, dict) and emo.get("confidence"):
        result["confidence"] = emo["confidence"]

    result.setdefault("emotion", "Neutral/Osäker")
    result.setdefault("confidence", "medel")

    scores = result.get("scores") or {}
    result["scores"] = {
        "stress": scores.get("stress", 0),
        "neg": scores.get("neg", 0),
        "pos": scores.get("pos", 0),
    }
    result.setdefault("recommendations", result.get("time_recommendations", []) or ["Fortsätt enligt plan."])
    result.setdefault("summary", result.get("summary", ""))

    result.setdefault("decisions", [])
    result.setdefault("action_items", [])
    result.setdefault("risks", [])
    result.setdefault("open_questions", [])
    result.setdefault("topics", [])
    result.setdefault("emotions", emo if isinstance(emo, dict) else {"overall":"Neutral/Osäker","confidence":"medel","evidence":[], "by_speaker":[]})
    result.setdefault("time_recommendations", result.get("time_recommendations", []))
    result.setdefault("on_time_probability", result.get("on_time_probability", None))
    result.setdefault("on_time_rationale", result.get("on_time_rationale", ""))

    return result

# ---------------- Heuristik & procent-mappningar ----------------
def on_time_probability_heuristic(text: str, result: dict) -> dict:
    """
    Enkel heuristik: stress/negativt drar ner, beslut/action items/positivt drar upp.
    Returnerar {"prob": int 0-100, "rationale": str}
    """
    t = text.lower()
    scores = result.get("scores", {})
    stress = scores.get("stress", 0) or 0
    neg    = scores.get("neg", 0) or 0
    pos    = scores.get("pos", 0) or 0
    decisions = len(result.get("decisions", []))
    actions   = len(result.get("action_items", []))

    prob = 70
    prob -= 15 * min(stress, 3)
    prob -= 10 * min(neg, 3)
    prob +=  8 * min(pos, 3)
    prob +=  6 * min(decisions, 3)
    prob +=  6 * min(actions, 3)

    trecs = " ".join(result.get("time_recommendations", [])).lower()
    if any(k in trecs for k in ["lägg till", "mer tid", "skjuta", "förläng"]):
        prob -= 10

    prob = max(0, min(100, prob))
    rationale = []
    if stress or neg: rationale.append("Stress/negativa signaler upptäcktes.")
    if pos:           rationale.append("Flera positiva signaler.")
    if decisions:     rationale.append(f"{decisions} beslut identifierade.")
    if actions:       rationale.append(f"{actions} action items identifierade.")
    if not rationale: rationale.append("Neutrala indikationer.")
    return {"prob": prob, "rationale": " ".join(rationale)}

def to_percent_from_scale3(value) -> int:
    try:
        return max(0, min(100, round((float(value) / 3.0) * 100)))
    except Exception:
        return 0

def confidence_to_percent(conf: str) -> int:
    if not isinstance(conf, str):
        return 50
    c = conf.lower()
    if "hög" in c: return 85
    if "medel" in c: return 65
    if "låg–medel" in c: return 55
    if "låg" in c: return 40
    return 50

# ---------------- UI ----------------
with st.sidebar:
    st.header("⚙️ Inställningar")
    engine = st.radio("AI-hjärna", ["GPT (OpenAI)", "Regelbaserad"], index=0)
    model = st.selectbox("GPT-modell", ["gpt-4o-mini", "gpt-3.5-turbo-0125"], index=0)
    api_key = st.secrets["OPENAI_API_KEY"]

uploaded = st.file_uploader("Ladda upp mötesanteckningar", type=["txt", "docx", "pdf"])

if uploaded:
    with st.spinner("Läser in filen…"):
        text = extract_text(uploaded)

    if not text.strip():
        st.error("Kunde inte läsa ut någon text från filen. Kontrollera filformatet eller innehållet.")
    else:
        st.subheader("📝 Utdragen text (förhandsvisning)")
        preview = text[:1500] + (" …" if len(text) > 1500 else "")
        st.write(preview)
        st.caption(f"Totalt antal tecken: {len(text)}")

        # --------- Analys ---------
        try:
            if engine == "GPT (OpenAI)":
                if not api_key:
                    st.warning("Ange din OpenAI-nyckel i sidopanelen eller växla till Regelbaserad.")
                    result = rules_analyze(text)
                else:
                    with st.spinner("Analyserar med GPT…"):
                        try:
                            result = gpt_analyze(text, api_key=api_key, model=model)
                        except Exception as e:
                            st.error(f"Fel vid GPT-analys ({e}), använder regelbaserad istället.")
                            result = rules_analyze(text)
            else:
                result = rules_analyze(text)

            # Normalisera för enhetlig UI
            result = normalize_result(result)

            # Tidsprognos: använd GPT:s värde om finns, annars heuristik
            if result.get("on_time_probability") is None:
                otp = on_time_probability_heuristic(text, result)
                result["on_time_probability"] = otp["prob"]
                result["on_time_rationale"] = otp["rationale"]

            # Procent för UI
            stress_pct = to_percent_from_scale3(result["scores"].get("stress", 0))
            pos_pct    = to_percent_from_scale3(result["scores"].get("pos", 0))
            hit_rate   = confidence_to_percent(result.get("confidence", "medel"))

            # --------- UI: Analys (känslofokus) ---------
            st.subheader("📊 Analys")
            c1, c2, c3 = st.columns(3)
            c1.metric("Känsla", result.get("emotion", "—"))
            c2.metric("Träffsäkerhet", f"{hit_rate}%")
            c3.metric("Tidsprognos, i tid", f"{result.get('on_time_probability', 0)}%")

            st.markdown("**Stressnivå**")
            st.progress(stress_pct)
            st.markdown("**Positiv ton**")
            st.progress(pos_pct)

            # 💡 Lyft stressreducerande rekommendationer direkt
            if result.get("time_recommendations"):
                st.subheader("💡 Rekommendationer (för att minska stress)")
                for r in result["time_recommendations"]:
                    st.write("• " + r)

            # Visa evidens för känsla
            emo = result.get("emotions", {})
            if emo.get("evidence"):
                st.caption("Evidens för känslobedömning:")
                for ev in emo["evidence"][:5]:
                    st.write(f"— “{ev}”")

            if result.get("on_time_rationale"):
                st.caption(f"Motivering tidsprognos: {result['on_time_rationale']}")

            # --------- UI: Strukturerat innehåll (sekundärt) ---------
            if result.get("summary"):
                st.subheader("🧭 Sammanfattning")
                st.write(result["summary"])

            if result.get("decisions"):
                st.subheader("✅ Beslut")
                for d in result["decisions"]:
                    title = d.get("title","(utan titel)")
                    details = d.get("details","")
                    ts = d.get("timestamp")
                    ts_str = f" — {ts}" if ts else ""
                    st.markdown(f"- **{title}**{ts_str}: {details}")

            if result.get("action_items"):
                st.subheader("🧩 Action items")
                for a in result["action_items"]:
                    title = a.get("title","(utan titel)")
                    owner = a.get("owner","okänd")
                    due = a.get("due_date")
                    prio = a.get("priority","medel")
                    notes = a.get("notes","")
                    due_str = f" — deadline: {due}" if due else ""
                    st.markdown(f"- **{title}** ({owner}, prioritet: {prio}{due_str})  \n  _{notes}_")

            if result.get("open_questions"):
                st.subheader("❓ Öppna frågor")
                for q in result["open_questions"]:
                    question = q.get("question","")
                    owner = q.get("owner","okänd")
                    st.markdown(f"- {question} _(ansvar: {owner})_")

            # Risker visas längst ned (låg prioritet)
            if result.get("risks"):
                st.subheader("⚠️ Risker (sekundärt)")
                for r in result["risks"]:
                    risk = r.get("risk","")
                    mitig = r.get("mitigation","")
                    sev = r.get("severity","medel")
                    st.markdown(f"- **{risk}** (allvar: {sev}) — Åtgärd: {mitig}")

            if result.get("topics"):
                st.subheader("🏷️ Ämnen")
                st.write(", ".join(result["topics"]))

            # --------- Generera & ladda ner anteckning ---------
            notes = (
                f"**Sammanfattning:** {result.get('summary','[—]')}\n\n"
                f"**Känsla:** {result.get('emotion','[—]')} (träffsäkerhet: {hit_rate}%)\n"
                f"**Tidsprognos (i tid):** {result.get('on_time_probability','—')}%\n"
                f"_Motivering tidsprognos:_ {result.get('on_time_rationale','')}\n\n"
                "**Rekommendationer (för att minska stress):**\n" + "\n".join(
                    [f"- {t}" for t in result.get('time_recommendations', [])]
                ) + "\n\n"
                "**Beslut:**\n" + "\n".join(
                    [f"- {d.get('title','(utan titel)')}: {d.get('details','')}" for d in result.get('decisions', [])]
                ) + "\n\n"
                "**Action items:**\n" + "\n".join(
                    [f"- {a.get('title','(utan titel)')} ({a.get('owner','okänd')}, prio: {a.get('priority','medel')}, deadline: {a.get('due_date') or '—'})"
                     for a in result.get('action_items', [])]
                ) + "\n\n"
                "**Öppna frågor:**\n" + "\n".join(
                    [f"- {q.get('question','')} (ansvar: {q.get('owner','okänd')})"
                     for q in result.get('open_questions', [])]
                ) + "\n\n"
                "**Risker (sekundärt):**\n" + "\n".join(
                    [f"- {r.get('risk','')}: Åtgärd {r.get('mitigation','')}, allvar: {r.get('severity','medel')}"
                     for r in result.get('risks', [])]
                )
            )

            st.download_button(
                "⬇️ Ladda ner anteckning som .txt",
                data=notes,
                file_name="motesanteckning_ai.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Något gick fel med analysen: {e}")
else:
    st.info("Välj en fil (.txt, .docx eller .pdf) för att börja.")

