"""
AI Interview Analyzer — app.py
MLA: TF-IDF + Logistic Regression quality classifier
NLP: Sentiment, Confidence, Communication scoring
"""

import re
import streamlit as st
import plotly.graph_objects as go

from utils.preprocessor         import preprocess_text
from utils.sentiment_analyzer   import analyze_sentiment
from utils.confidence_scorer    import compute_confidence_score
from utils.communication_scorer import compute_communication_score
from utils.feedback_generator   import generate_feedback
from model.predictor            import predict_quality, is_model_trained

try:
    from utils.voice_input import record_to_file, transcribe_file, check_dependencies, is_microphone_available
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

st.set_page_config(page_title="Interview Analyzer", page_icon="", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif;background:#F7F5F2;color:#1a1a1a;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2.5rem 3rem!important;max-width:1300px;}
.site-header{border-bottom:1px solid #E0DDD8;padding-bottom:1.75rem;margin-bottom:2rem;}
.site-title{font-family:'DM Serif Display',serif;font-size:2.2rem;color:#1a1a1a;letter-spacing:-0.5px;}
.site-title em{font-style:italic;color:#4A6FA5;}
.site-desc{margin-top:0.35rem;font-size:0.88rem;color:#7a7670;}
.field-label{font-size:0.68rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#9a9690;margin-bottom:0.4rem;}
.question-display{background:#fff;border:1px solid #E0DDD8;border-left:3px solid #4A6FA5;border-radius:6px;padding:0.9rem 1.15rem;font-size:0.95rem;color:#2a2a2a;line-height:1.6;margin-bottom:1.25rem;}
.stTextArea textarea{background:#ffffff!important;border:1px solid #D8D4CE!important;border-radius:6px!important;color:#1a1a1a!important;font-family:'DM Sans',sans-serif!important;font-size:0.94rem!important;line-height:1.65!important;padding:0.85rem 1rem!important;caret-color:#4A6FA5!important;resize:vertical!important;}
.stTextArea textarea::placeholder{color:#b0aba4!important;}
.stTextArea textarea:focus{border-color:#4A6FA5!important;box-shadow:0 0 0 3px rgba(74,111,165,0.1)!important;}
.word-count{font-size:0.73rem;color:#b0aba4;text-align:right;margin-top:0.25rem;}
.stButton>button{background:#1a1a1a!important;color:#F7F5F2!important;border:none!important;border-radius:6px!important;padding:0.65rem 1.5rem!important;font-family:'DM Sans',sans-serif!important;font-weight:500!important;font-size:0.88rem!important;letter-spacing:0.03em!important;width:100%!important;transition:background 0.2s!important;}
.stButton>button:hover{background:#4A6FA5!important;}
.stButton>button:disabled{background:#ccc!important;}
.stSelectbox>div>div{background:#ffffff!important;border:1px solid #D8D4CE!important;border-radius:6px!important;color:#1a1a1a!important;}
.transcript-box{background:#fff;border:1px solid #D8D4CE;border-radius:6px;padding:1rem 1.1rem;font-size:0.94rem;color:#1a1a1a;line-height:1.65;min-height:90px;margin-top:0.5rem;}
.transcript-empty{color:#b0aba4;}
.s-recording{background:#FEF3F3;border:1px solid #F5C6C6;border-radius:6px;padding:0.65rem 1rem;font-size:0.83rem;color:#B03030;margin-top:0.6rem;}
.s-ok{background:#F2F7F2;border:1px solid #B8D4B8;border-radius:6px;padding:0.65rem 1rem;font-size:0.83rem;color:#2E6B2E;margin-top:0.6rem;}
.s-err{background:#FDF4EE;border:1px solid #E8C8A8;border-radius:6px;padding:0.65rem 1rem;font-size:0.83rem;color:#8B4513;margin-top:0.6rem;white-space:pre-line;}
.rec-dot{display:inline-block;width:8px;height:8px;background:#C0392B;border-radius:50%;margin-right:5px;animation:pdot 1.2s ease-in-out infinite;}
@keyframes pdot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}}
.section-heading{font-family:'DM Serif Display',serif;font-size:1.1rem;color:#1a1a1a;margin:1.75rem 0 0.85rem 0;padding-bottom:0.5rem;border-bottom:1px solid #E0DDD8;}
.score-main{background:#fff;border:1px solid #E0DDD8;border-radius:8px;padding:1.75rem 2rem;text-align:center;}
.score-numeral{font-family:'DM Serif Display',serif;font-size:4.2rem;line-height:1;color:#1a1a1a;}
.score-denom{font-size:0.75rem;color:#b0aba4;letter-spacing:0.1em;margin-top:0.15rem;}
.score-grade{display:inline-block;margin-top:0.6rem;font-size:0.75rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;padding:0.2rem 0.8rem;border-radius:4px;}
.metric-card{background:#fff;border:1px solid #E0DDD8;border-radius:8px;padding:1.15rem 1.25rem;}
.metric-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.5rem;}
.metric-name{font-size:0.82rem;font-weight:500;color:#2a2a2a;}
.metric-weight{font-size:0.67rem;color:#b0aba4;letter-spacing:0.06em;}
.metric-score{font-family:'DM Serif Display',serif;font-size:1.35rem;color:#1a1a1a;}
.metric-bar-track{background:#F0EDE8;border-radius:2px;height:3px;margin-top:0.4rem;overflow:hidden;}
.metric-bar-fill{height:100%;border-radius:2px;background:#4A6FA5;}
.metric-detail{font-size:0.75rem;color:#9a9690;margin-top:0.4rem;}
.ml-card{background:#fff;border:1px solid #E0DDD8;border-radius:8px;padding:1.5rem 1.75rem;margin-bottom:1rem;}
.ml-badge{display:inline-block;font-size:0.8rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;padding:0.3rem 1rem;border-radius:4px;margin-bottom:0.75rem;}
.ml-algo{font-size:0.72rem;font-family:monospace;color:#9a9690;margin-bottom:0.75rem;background:#F7F5F2;padding:0.25rem 0.5rem;border-radius:4px;display:inline-block;}
.ml-reasoning{font-size:0.83rem;color:#5a5a6a;line-height:1.6;margin-top:0.5rem;}
.prob-row{display:flex;align-items:center;gap:0.75rem;margin:0.4rem 0;}
.prob-label{font-size:0.78rem;color:#4a4a4a;width:70px;flex-shrink:0;}
.prob-track{flex:1;background:#F0EDE8;border-radius:2px;height:6px;overflow:hidden;}
.prob-fill{height:100%;border-radius:2px;}
.prob-pct{font-size:0.75rem;color:#9a9690;width:38px;text-align:right;flex-shrink:0;}
.fb-positive{background:#F9F9F7;border-left:2px solid #4A6FA5;padding:0.6rem 0.95rem;border-radius:0 4px 4px 0;margin:0.35rem 0;font-size:0.86rem;color:#2a2a2a;line-height:1.55;}
.fb-suggestion{background:#F9F9F7;border-left:2px solid #C8A96E;padding:0.6rem 0.95rem;border-radius:0 4px 4px 0;margin:0.35rem 0;font-size:0.86rem;color:#2a2a2a;line-height:1.55;}
.stat-box{background:#fff;border:1px solid #E0DDD8;border-radius:6px;padding:0.9rem;text-align:center;}
.stat-number{font-family:'DM Serif Display',serif;font-size:1.65rem;color:#1a1a1a;line-height:1;}
.stat-label{font-size:0.67rem;color:#9a9690;letter-spacing:0.08em;text-transform:uppercase;margin-top:0.25rem;}
.placeholder-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:220px;gap:0.65rem;border:1px dashed #D8D4CE;border-radius:8px;}
.placeholder-line{width:36px;height:1px;background:#D8D4CE;}
.placeholder-text{font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:#c8c4be;}
.model-warn{background:#FEF9EE;border:1px solid #F0D080;border-radius:8px;padding:1rem 1.25rem;font-size:0.85rem;color:#7a5c00;line-height:1.6;}
hr{border-color:#E0DDD8!important;}
</style>
""", unsafe_allow_html=True)

QUESTIONS = [
    "Tell me about yourself and your background.",
    "What are your greatest strengths and weaknesses?",
    "Where do you see yourself in 5 years?",
    "Why do you want to work for our company?",
    "Describe a challenging situation you faced and how you overcame it.",
    "What motivates you in your work?",
    "How do you handle stress and pressure?",
    "Tell me about a time you demonstrated leadership.",
    "Why are you leaving your current job?",
    "What makes you the best candidate for this position?"
]
LANG_OPTIONS = {"English (India)":"en-IN","English (US)":"en-US","English (UK)":"en-GB","Hindi":"hi-IN"}
PROB_COLORS  = {"Poor":"#E57373","Average":"#FFB74D","Good":"#81C784","Excellent":"#4A6FA5"}

DEFAULTS = {"voice_transcript":"","voice_status":"","voice_status_type":"","input_mode":"type","audio_filepath":None,"phase":"idle"}
for k,v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k]=v

def clean(t):
    return re.sub(r'[^\x00-\x7F\u00C0-\u024F\s\w.,!?\-\'\"()]','',t).strip()

# Header
st.markdown("""
<div class="site-header">
    <div class="site-title">Interview <em>Analyzer</em></div>
    <div class="site-desc">NLP scoring &nbsp;&middot;&nbsp; TF-IDF &nbsp;&middot;&nbsp; Logistic Regression classifier &nbsp;&middot;&nbsp; MLA Portfolio Project</div>
</div>""", unsafe_allow_html=True)

if not is_model_trained():
    st.markdown('<div class="model-warn"><strong>ML model not trained yet.</strong> Run this once in your terminal, then refresh:<br><br><code>python model/train_model.py</code><br><br>NLP analysis still works without it.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Input section
col_q, col_ans = st.columns([1, 1.3], gap="large")
with col_q:
    st.markdown('<div class="field-label">Question</div>', unsafe_allow_html=True)
    selected_q = st.selectbox("", options=QUESTIONS, label_visibility="collapsed")
    st.markdown(f'<div class="question-display">{selected_q}</div>', unsafe_allow_html=True)
    st.markdown('<div class="field-label">Input method</div>', unsafe_allow_html=True)
    m1,m2 = st.columns(2)
    with m1:
        if st.button("Type",key="btn_type"): st.session_state.input_mode="type"; st.rerun()
    with m2:
        if st.button("Speak",key="btn_speak"): st.session_state.input_mode="voice"; st.rerun()
    with st.expander("Tips"):
        st.markdown("Use **STAR** — Situation, Task, Action, Result.\nSay *I led*, *I built*, *I achieved*.\nAim for 80–150 words.")

with col_ans:
    if st.session_state.input_mode == "type":
        st.markdown('<div class="field-label">Your answer</div>', unsafe_allow_html=True)
        user_answer = st.text_area("", height=200, placeholder="Write your answer here...", label_visibility="collapsed", key="typed_answer")
        wc = len(user_answer.split()) if user_answer.strip() else 0
        st.markdown(f'<div class="word-count">{wc} words</div>', unsafe_allow_html=True)
    else:
        if not VOICE_AVAILABLE:
            st.warning("Run: pip install sounddevice soundfile faster-whisper  then restart.")
            user_answer = ""
        else:
            missing = check_dependencies()
            if missing:
                st.markdown(f'<div class="s-err">Missing: {", ".join(missing)}\nRun: pip install {" ".join(missing)}</div>', unsafe_allow_html=True)
                user_answer = ""
            else:
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="field-label">Language</div>', unsafe_allow_html=True)
                    lang_label = st.selectbox("", list(LANG_OPTIONS.keys()), label_visibility="collapsed", key="vlang")
                    lang_code = LANG_OPTIONS[lang_label]
                with c2:
                    st.markdown('<div class="field-label">Duration (seconds)</div>', unsafe_allow_html=True)
                    duration = st.select_slider("", [10,15,20,30,45,60], value=15, label_visibility="collapsed", key="vdur")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.session_state.voice_transcript:
                    st.markdown(f'<div class="transcript-box">{st.session_state.voice_transcript}</div>', unsafe_allow_html=True)
                    wc_v = len(st.session_state.voice_transcript.split())
                    st.markdown(f'<div class="word-count">{wc_v} words captured</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="transcript-box transcript-empty">Your transcribed speech will appear here after recording.</div>', unsafe_allow_html=True)
                if st.session_state.voice_status:
                    css = {"recording":"s-recording","transcribing":"s-recording","ok":"s-ok","err":"s-err"}.get(st.session_state.voice_status_type,"s-err")
                    prefix = '<span class="rec-dot"></span>' if st.session_state.voice_status_type in ("recording","transcribing") else ""
                    st.markdown(f'<div class="{css}">{prefix}{st.session_state.voice_status}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.session_state.phase == "idle":
                    b1,b2 = st.columns([3,1])
                    with b1:
                        if st.button("Record answer", key="rec_btn", use_container_width=True):
                            st.session_state.voice_status = f"Recording {duration}s — speak now..."
                            st.session_state.voice_status_type = "recording"
                            with st.spinner(f"Recording {duration}s..."):
                                filepath, err = record_to_file(duration)
                            if err:
                                st.session_state.voice_status=err; st.session_state.voice_status_type="err"; st.session_state.audio_filepath=None; st.session_state.phase="idle"
                            else:
                                st.session_state.audio_filepath=filepath; st.session_state.voice_status="Recording done. Click Transcribe."; st.session_state.voice_status_type="ok"; st.session_state.phase="recorded"
                            st.rerun()
                    with b2:
                        if st.button("Clear",key="clr_btn"):
                            for k in ["voice_transcript","voice_status","voice_status_type"]: st.session_state[k]=""
                            st.session_state.audio_filepath=None; st.session_state.phase="idle"; st.rerun()
                elif st.session_state.phase == "recorded":
                    b1,b2,b3 = st.columns([2,2,1])
                    with b1:
                        if st.button("Transcribe",key="tx_btn",use_container_width=True):
                            st.session_state.voice_status="Transcribing..."; st.session_state.voice_status_type="transcribing"
                            with st.spinner("Transcribing..."):
                                text, err = transcribe_file(st.session_state.audio_filepath, lang_code)
                            if err:
                                st.session_state.voice_status=err; st.session_state.voice_status_type="err"; st.session_state.phase="idle"
                            else:
                                existing=st.session_state.voice_transcript.strip()
                                st.session_state.voice_transcript=(existing+" "+text if existing else text)
                                st.session_state.voice_status=f"Done — {len(text.split())} words captured."; st.session_state.voice_status_type="ok"; st.session_state.audio_filepath=None; st.session_state.phase="done"
                            st.rerun()
                    with b2:
                        if st.button("Record again",key="rec2_btn",use_container_width=True):
                            st.session_state.audio_filepath=None; st.session_state.voice_status=""; st.session_state.phase="idle"; st.rerun()
                    with b3:
                        if st.button("Clear",key="clr2_btn"):
                            for k in ["voice_transcript","voice_status","voice_status_type"]: st.session_state[k]=""
                            st.session_state.audio_filepath=None; st.session_state.phase="idle"; st.rerun()
                elif st.session_state.phase == "done":
                    b1,b2 = st.columns([3,1])
                    with b1:
                        if st.button("Record more",key="rec3_btn",use_container_width=True):
                            st.session_state.voice_status=""; st.session_state.phase="idle"; st.rerun()
                    with b2:
                        if st.button("Clear",key="clr3_btn"):
                            for k in ["voice_transcript","voice_status","voice_status_type"]: st.session_state[k]=""
                            st.session_state.audio_filepath=None; st.session_state.phase="idle"; st.rerun()
                if st.session_state.voice_transcript:
                    with st.expander("Edit transcript"):
                        edited=st.text_area("",value=st.session_state.voice_transcript,height=100,label_visibility="collapsed",key="edit_tx")
                        if st.button("Save edits",key="save_edit"): st.session_state.voice_transcript=edited; st.rerun()
                user_answer = st.session_state.voice_transcript

# Analyze button
st.markdown("<br>", unsafe_allow_html=True)
_,btn_col,_ = st.columns([2,1,2])
with btn_col:
    analyze_btn = st.button("Analyze answer", key="analyze_btn")
st.markdown('<hr style="margin:1.5rem 0;">', unsafe_allow_html=True)

# Results
if analyze_btn:
    final_text = (user_answer or "").strip()
    word_count = len(final_text.split()) if final_text else 0
    if not final_text:
        st.warning("Please provide an answer before analyzing.")
    elif word_count < 10:
        st.warning("Answer too short — write or speak at least 10 words.")
    else:
        with st.spinner("Analyzing..."):
            tokens    = preprocess_text(final_text)
            sent_r    = analyze_sentiment(final_text)
            conf_r    = compute_confidence_score(final_text)
            comm_r    = compute_communication_score(final_text)
            nlp_score = round(sent_r["score"]*0.30 + conf_r["score"]*0.40 + comm_r["score"]*0.30)
            feedback  = generate_feedback(sent_r, conf_r, comm_r, nlp_score)
            ml_result = predict_quality(final_text) if is_model_trained() else None
            final_score = round(nlp_score*0.55 + ml_result["ml_score"]*0.45) if ml_result else nlp_score

        if final_score>=75:   grade,gbg,gcol="Excellent","#EEF3EC","#3A7D44"
        elif final_score>=55: grade,gbg,gcol="Good","#F5F0E8","#8B6914"
        elif final_score>=35: grade,gbg,gcol="Fair","#FDF4EE","#B05D2A"
        else:                 grade,gbg,gcol="Needs Work","#FCEEED","#A83232"

        # NLP row
        st.markdown('<div class="section-heading">NLP Analysis</div>', unsafe_allow_html=True)
        r1,r2,r3,r4 = st.columns(4)
        with r1:
            st.markdown(f'<div class="score-main"><div class="field-label">Combined Score</div><div class="score-numeral">{final_score}</div><div class="score-denom">out of 100</div><span class="score-grade" style="background:{gbg};color:{gcol};">{grade}</span></div>', unsafe_allow_html=True)
        for col,name,weight,score,detail in [(r2,"Sentiment","30%",sent_r["score"],sent_r["label"]),(r3,"Confidence","40%",conf_r["score"],conf_r["level"]),(r4,"Communication","30%",comm_r["score"],comm_r["rating"])]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-top"><div><div class="metric-name">{name}</div><div class="metric-weight">{weight} weight</div></div><div class="metric-score">{score}</div></div><div class="metric-bar-track"><div class="metric-bar-fill" style="width:{score}%;"></div></div><div class="metric-detail">{clean(detail)}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ML row
        st.markdown('<div class="section-heading">ML Prediction — Logistic Regression + TF-IDF</div>', unsafe_allow_html=True)
        if ml_result:
            ml_col1, ml_col2 = st.columns([1, 1.4], gap="large")
            pred_cls   = ml_result["predicted_class"]
            pred_color = PROB_COLORS.get(pred_cls,"#4A6FA5")
            badge_bg   = {"Poor":"#FCEEED","Average":"#FDF4EE","Good":"#EEF3EC","Excellent":"#EEF3EC"}.get(pred_cls,"#F0F0F0")
            with ml_col1:
                prob_bars = ""
                for cls in ["Poor","Average","Good","Excellent"]:
                    prob=ml_result["probabilities"].get(cls,0.0); pct=int(prob*100); color=PROB_COLORS[cls]
                    bold="font-weight:600;" if cls==pred_cls else ""
                    prob_bars += f'<div class="prob-row"><div class="prob-label" style="{bold}">{cls}</div><div class="prob-track"><div class="prob-fill" style="width:{pct}%;background:{color};"></div></div><div class="prob-pct">{pct}%</div></div>'
                st.markdown(f'<div class="ml-card"><span class="ml-badge" style="background:{badge_bg};color:{pred_color};">{pred_cls}</span><div class="ml-label">Predicted Quality Class</div><div class="ml-algo">TfidfVectorizer(ngram=(1,2)) + LogisticRegression(ovr)</div><div style="margin:0.75rem 0 0.5rem;font-size:0.78rem;font-weight:600;letter-spacing:0.08em;color:#9a9690;text-transform:uppercase;">Class Probabilities</div>{prob_bars}<div class="ml-reasoning">{ml_result["reasoning"]}</div></div>', unsafe_allow_html=True)
            with ml_col2:
                classes=["Poor","Average","Good","Excellent"]
                probs=[ml_result["probabilities"].get(c,0)*100 for c in classes]
                fig=go.Figure(go.Bar(x=classes,y=probs,marker_color=[PROB_COLORS[c] for c in classes],text=[f"{p:.1f}%" for p in probs],textposition="outside",textfont=dict(size=11,family="DM Sans",color="#4a4a4a")))
                fig.update_layout(title=dict(text="Class Probability Distribution",font=dict(family="DM Serif Display",size=14,color="#1a1a1a"),x=0),yaxis=dict(range=[0,115],showgrid=True,gridcolor="#F0EDE8",ticksuffix="%",tickfont=dict(size=10,color="#9a9690"),title=""),xaxis=dict(tickfont=dict(size=12,family="DM Sans",color="#4a4a4a"),title=""),plot_bgcolor="#F9F9F7",paper_bgcolor="rgba(247,245,242,0)",height=280,margin=dict(t=40,b=20,l=10,r=10),showlegend=False,font=dict(family="DM Sans"))
                st.plotly_chart(fig, use_container_width=True)
                conf_pct=int(ml_result["confidence"]*100)
                st.markdown(f'<div style="background:#fff;border:1px solid #E0DDD8;border-radius:8px;padding:1rem 1.25rem;"><div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;"><span style="font-size:0.78rem;color:#4a4a4a;font-weight:500;">Model Confidence</span><span style="font-family:\'DM Serif Display\',serif;font-size:1.2rem;">{conf_pct}%</span></div><div style="background:#F0EDE8;border-radius:2px;height:4px;"><div style="width:{conf_pct}%;height:100%;border-radius:2px;background:{pred_color};"></div></div><div style="font-size:0.72rem;color:#9a9690;margin-top:0.35rem;">Trained on 120 labelled samples &nbsp;&middot;&nbsp; 5-fold cross-validated</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="model-warn">Run <code>python model/train_model.py</code> to enable ML prediction.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feedback + radar
        fb_col, right_col = st.columns([1.3, 1], gap="large")
        with fb_col:
            st.markdown('<div class="section-heading">What you did well</div>', unsafe_allow_html=True)
            for p in feedback["positives"]: st.markdown(f'<div class="fb-positive">{clean(p)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">Areas to improve</div>', unsafe_allow_html=True)
            for s in feedback["suggestions"]: st.markdown(f'<div class="fb-suggestion">{clean(s)}</div>', unsafe_allow_html=True)
        with right_col:
            st.markdown('<div class="section-heading">Score breakdown</div>', unsafe_allow_html=True)
            radar=go.Figure()
            radar.add_trace(go.Scatterpolar(r=[sent_r["score"],conf_r["score"],comm_r["score"],sent_r["score"]],theta=["Sentiment","Confidence","Communication","Sentiment"],fill='toself',fillcolor='rgba(74,111,165,0.09)',line=dict(color='#4A6FA5',width=1.5)))
            radar.add_trace(go.Scatterpolar(r=[70,70,70,70],theta=["Sentiment","Confidence","Communication","Sentiment"],fill='toself',fillcolor='rgba(0,0,0,0)',line=dict(color='#D8D4CE',width=1,dash='dot')))
            radar.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)',radialaxis=dict(visible=True,range=[0,100],color='#c8c4be',gridcolor='#E8E4DE',tickfont=dict(size=9,color='#c8c4be')),angularaxis=dict(color='#7a7670',gridcolor='#E8E4DE',tickfont=dict(size=11,color='#4a4a4a',family='DM Sans'))),showlegend=False,paper_bgcolor='rgba(247,245,242,0)',height=280,margin=dict(t=10,b=10,l=50,r=50),font=dict(family='DM Sans',color='#7a7670'))
            st.plotly_chart(radar, use_container_width=True)
            st.markdown('<div class="section-heading">Answer statistics</div>', unsafe_allow_html=True)
            sentences=max(len([x for x in final_text.split('.') if x.strip()]),1)
            unique_w=len(set(final_text.lower().split()))
            fillers=sum(final_text.lower().count(w) for w in ["um","uh","like","you know","basically","literally"])
            s1,s2,s3,s4=st.columns(4)
            for col,num,label in [(s1,word_count,"Words"),(s2,sentences,"Sentences"),(s3,unique_w,"Unique"),(s4,fillers,"Fillers")]:
                with col: st.markdown(f'<div class="stat-box"><div class="stat-number">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="placeholder-state"><div class="placeholder-line"></div><div class="placeholder-text">Results will appear here after analysis</div><div class="placeholder-line"></div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<hr><div style="text-align:center;font-size:0.7rem;color:#c8c4be;padding:0.6rem 0;letter-spacing:0.06em;">Interview Analyzer &nbsp;&middot;&nbsp; NLP + Logistic Regression &nbsp;&middot;&nbsp; MLA Portfolio Project</div>', unsafe_allow_html=True)
