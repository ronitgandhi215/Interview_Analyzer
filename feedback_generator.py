"""
utils/feedback_generator.py
-----------------------------
Large pool of varied, context-sensitive feedback.
Every analysis produces different suggestions based on actual answer content.
"""

import random


# ═══════════════════════════════════════════════════════════════════════════════
# POSITIVE FEEDBACK POOLS  (keyed by what was actually detected)
# ═══════════════════════════════════════════════════════════════════════════════

POOL_POSITIVE_TONE = [
    "Your answer carries a genuinely enthusiastic tone that most interviewers find refreshing and memorable.",
    "The positivity in your response comes across naturally, not forced — that is a harder thing to achieve than it sounds.",
    "You framed your experience in an optimistic light, which signals maturity and self-awareness to a hiring manager.",
    "Your answer radiates a sense of ownership and excitement — exactly what interviewers are hoping to hear.",
    "The confident, upbeat energy in your response will keep an interviewer engaged from start to finish.",
    "You managed to stay positive without sounding hollow or rehearsed — that balance is worth preserving.",
    "Your tone communicates that you genuinely care about the work, not just the job title.",
]

POOL_STRONG_KEYWORDS = [
    "You used concrete action verbs like 'led', 'built', and 'delivered' — these paint a vivid picture of what you actually did.",
    "Specific achievement words throughout your answer give the interviewer something tangible to remember you by.",
    "Words like 'improved', 'achieved', and 'created' signal a results-driven mindset that stands out.",
    "Your vocabulary reflects someone who takes initiative rather than just fulfilling a role — that distinction matters.",
    "Quantifiable language and action words are present in your answer, which is exactly what structured interviews look for.",
    "You chose words that imply ownership and impact rather than passive participation — a subtle but powerful difference.",
]

POOL_ASSERTIVE_LANGUAGE = [
    "Using 'I achieved' and 'I led' rather than 'we' or 'the team' makes your individual contribution crystal clear.",
    "First-person ownership language throughout the answer signals confidence without coming across as arrogant.",
    "You claimed your accomplishments directly — interviewers notice when candidates avoid taking personal credit.",
    "Assertive language like 'I built' and 'I managed' removes ambiguity about your personal role in outcomes.",
    "The way you spoke in the first person consistently tells the interviewer exactly what you bring to the table.",
    "Direct ownership of your work is rare and impressive — your language reflects a person who knows their own value.",
]

POOL_NO_FILLERS = [
    "Zero filler words — your answer reads like you have done this many times. That polish is noticed.",
    "The absence of fillers like 'um' or 'basically' makes your answer feel deliberate and well-considered.",
    "Clean, filler-free delivery communicates that you are comfortable with the question and confident in your answer.",
    "Not a single filler word — this is harder than it seems and gives your answer a professional quality.",
    "Your answer flows without verbal stumbles, which is a strong signal of preparation and composure.",
]

POOL_GOOD_LENGTH = [
    "The length of your answer hits a sweet spot — enough detail to be convincing, short enough to stay engaging.",
    "You gave the interviewer enough to evaluate you without rambling. That kind of restraint takes practice.",
    "Your answer is appropriately substantial. It shows you took the question seriously and prepared.",
    "Good depth here — you have said enough to make a strong impression without losing the thread.",
    "The answer is neither too short to seem dismissive nor too long to seem unfocused. Well-judged.",
]

POOL_GOOD_GRAMMAR = [
    "Your grammar is clean throughout, which keeps the focus on what you are saying rather than how you are saying it.",
    "Strong grammar signals attention to detail — a quality that will transfer well to written workplace communication.",
    "Well-constructed sentences throughout. The interviewer can follow your thinking without any friction.",
    "No grammatical distractions — your ideas come through clearly because the language supports them.",
]

POOL_RICH_VOCABULARY = [
    "You used varied vocabulary rather than repeating the same words — that is a sign of genuine fluency.",
    "The range of words in your answer demonstrates that you can express nuanced ideas clearly.",
    "Vocabulary diversity here is strong. It gives your answer texture and prevents it from feeling repetitive.",
    "You avoided the trap of using the same three adjectives throughout — your word choice keeps the answer interesting.",
]

POOL_GOOD_STRUCTURE = [
    "Transition phrases like 'firstly' and 'as a result' give your answer a logical shape that is easy to follow.",
    "Your answer has a clear beginning, middle, and end — that structure reflects well-organised thinking.",
    "The connective language in your response signals to the interviewer that you can communicate complex ideas step by step.",
    "Structured answers are far easier for interviewers to evaluate fairly — yours works well on that front.",
    "Your use of transitions shows you are guiding the listener through your thinking, not just talking at them.",
]

POOL_HIGH_CONFIDENCE = [
    "You come across as composed and self-assured — two qualities that interviewers consistently rank highly.",
    "The confidence in this answer is earned-sounding rather than performative. That distinction matters.",
    "Your answer projects quiet confidence — you are not overselling, just stating what you know and have done.",
    "Strong, grounded confidence throughout. This kind of answer makes interviewers feel they can rely on you.",
]

POOL_GOOD_COMMUNICATION = [
    "Your communication quality is high — the ideas are clear, structured, and easy to absorb.",
    "You expressed a complex thought in a way that is immediately understandable. That is a genuine skill.",
    "The clarity and flow of your answer reflects someone who can communicate well under pressure.",
    "Strong communicator. Your answer is direct, coherent, and leaves no room for misinterpretation.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SUGGESTION POOLS  (keyed by what needs improvement)
# ═══════════════════════════════════════════════════════════════════════════════

POOL_NEGATIVE_TONE = [
    "Reframe difficulties as growth moments. Instead of 'it was awful', try 'it taught me to approach X differently'.",
    "Interviewers expect challenges — what they are evaluating is how you responded. Shift the focus there.",
    "Even when describing failure, end on what you learned or changed. That pivot is what impresses interviewers.",
    "Negative framing makes you memorable for the wrong reasons. Find the constructive angle in the same story.",
    "Consider replacing phrases like 'I struggled' with 'I worked through' — same fact, entirely different impression.",
    "Your story may be honest, but the tone risks making an interviewer worry about how you handle adversity. Reframe it.",
    "Every negative experience has a useful lesson buried in it. Surface that lesson and lead with it instead.",
]

POOL_LOW_SENTIMENT = [
    "Add at least one line about what you enjoyed, valued, or gained from the experience you are describing.",
    "Positive energy is contagious in interviews. Even a single sentence about genuine enthusiasm shifts the tone.",
    "Try opening with a brief statement of what motivates you about this type of work before diving into the detail.",
    "The content is there — the tone just needs lifting. Ask yourself: what is the most energising part of this story?",
    "Enthusiasm about the work itself (not just the outcome) is something interviewers specifically listen for.",
    "Even a neutral answer can feel flat. Add one line that shows you were genuinely engaged, not just present.",
]

POOL_FILLER_WORDS = [
    "Every 'um' or 'like' costs you a small amount of credibility. Replace them with a single beat of silence.",
    "Practise pausing instead of filling. A confident pause says far more than 'basically' or 'you know'.",
    "Record yourself answering this question aloud and count the fillers. You will be surprised — and it will fix itself.",
    "Fillers are usually a sign that your next thought is not ready. Slow down slightly and the fillers will drop out.",
    "Try this: speak at 80 percent of your normal pace. Fillers appear when we rush to keep up with our own thoughts.",
    "Filler words dilute strong content. Your ideas are good — let them land without the verbal padding.",
    "Preparation is the best cure for fillers. The more you know what you want to say, the less you reach for placeholders.",
]

POOL_HEDGE_WORDS = [
    "Replace 'I think I could' with 'I will' or 'I have'. Hedging makes real capability sound uncertain.",
    "Every 'maybe' or 'probably' is an invitation for the interviewer to doubt you. Own your statements fully.",
    "You have the experience — your language just is not reflecting it yet. Cut the qualifiers and state it directly.",
    "Hedging phrases signal self-doubt even when none exists. Read back your answer and remove every 'I guess'.",
    "There is a meaningful difference between being humble and being hesitant. Your hedges tip into hesitant territory.",
    "Try rewriting your answer replacing every 'I think' with 'I know' and every 'maybe' with 'I will'. See how it feels.",
    "'I believe I can' is weaker than 'I have done this before'. Ground your claims in evidence, not tentative belief.",
]

POOL_PASSIVE_VOICE = [
    "Swap passive constructions for active ones. 'The project was delivered' becomes 'I delivered the project'.",
    "Passive voice hides your agency. If you did it, say you did it — the interviewer needs to know.",
    "Active voice is shorter, clearer, and more confident. Audit your answer for 'was done by' or 'was managed by'.",
    "Every passive sentence is a missed opportunity to claim credit. Rewrite them with 'I' as the subject.",
    "Passive voice often slips in when we are being modest. In interviews, modesty of that kind works against you.",
]

POOL_TOO_SHORT = [
    "Use the STAR method to expand: Situation (what was the context), Task (your role), Action (what you did), Result (what changed).",
    "A short answer rarely gives the interviewer enough to evaluate you. Add one concrete example to anchor the claim.",
    "Aim for 80 to 150 words. You have space to add a specific outcome, a number, or a brief story.",
    "Interviewers need evidence, not just assertions. Add one sentence of proof to every claim you make.",
    "Your answer makes a point — now back it up. What happened, specifically, that demonstrates what you are saying?",
    "Think of the follow-up question an interviewer would ask after your current answer. Then answer it now, preemptively.",
    "Length signals effort and preparation. A very short answer can read as either uninterested or unprepared.",
]

POOL_TOO_LONG = [
    "Identify the single most important point in your answer and cut everything that does not support it.",
    "If your answer takes more than 90 seconds to say aloud, an interviewer's attention will begin to drift.",
    "Try the one-sentence summary test: what is the one thing you want them to remember? Build toward that.",
    "Rambling often comes from wanting to cover everything. Pick your best example and go deep on it instead of broad.",
    "Editing an answer down is harder than expanding it — but it is also what separates good communicators from great ones.",
    "Every sentence should earn its place. Read back and cut any line that repeats or delays your core point.",
]

POOL_POOR_GRAMMAR = [
    "Read your answer aloud slowly — your ear will catch grammatical errors that your eye skips over.",
    "Subject-verb agreement is the most common issue in spoken answers. Check every sentence starts cleanly.",
    "Strong ideas lose impact when grammar undermines them. A single proofread pass makes a noticeable difference.",
    "Write your answer out and run it through a tool like Grammarly before your interview to catch recurring patterns.",
    "Grammar errors in an interview answer can distract the listener from what you are actually saying.",
]

POOL_POOR_VOCABULARY = [
    "Replace vague words like 'good', 'nice', or 'things' with precise alternatives that say exactly what you mean.",
    "Using the same word three times in one answer draws attention to itself. Build a short list of synonyms for your key terms.",
    "Specific language is more persuasive than general language. 'Reduced load time by 40 percent' beats 'made it faster'.",
    "Vocabulary variety makes an answer more engaging to listen to. Try replacing the most repeated word in your answer.",
    "Read more in your field. The language you absorb becomes the language you reach for under interview pressure.",
    "Avoid corporate filler phrases like 'synergy', 'leverage', or 'moving the needle' — they add words without meaning.",
]

POOL_POOR_STRUCTURE = [
    "Add one transition phrase — 'as a result', 'which led to', 'to address that' — and your answer will feel twice as organised.",
    "Structure is invisible when it works well. Try: one sentence of context, two of what you did, one of the outcome.",
    "The STAR method (Situation, Task, Action, Result) is a reliable skeleton for almost any interview question.",
    "Your answer contains good content but it is hard to follow. Add a brief signpost at the start: 'There are two things I'd highlight here'.",
    "Interviewers score structured answers higher, not because of the structure itself but because it signals clear thinking.",
    "Try ending your answer with a one-sentence summary of the key takeaway. It makes the whole response feel complete.",
    "Practise saying your answer in exactly three sentences first. Once you have that tight version, you can add detail without losing shape.",
]

POOL_NO_ASSERTIVE = [
    "Claim your contributions in the first person. If you built it, say 'I built it' — not 'it was built' or 'we built it'.",
    "Interviewers are hiring you, not your team. Make your individual role unmistakably clear.",
    "Avoid hiding behind 'we' when the action was yours. Be specific: 'I was responsible for' or 'I personally led'.",
    "Own your achievements without apology. Stating what you did is not arrogance — it is what the interview requires.",
    "Passive attribution of your own work is one of the most common and costly interview mistakes. Name yourself as the actor.",
]

POOL_STAR_METHOD = [
    "The STAR method keeps answers focused and complete: Situation, Task, Action, Result — one paragraph each.",
    "STAR is effective because it forces you to show evidence, not just make claims. Practise mapping this answer to it.",
    "Without a clear structure, good answers often ramble. STAR gives the interviewer a shape they recognise and score well.",
    "Most behavioural questions are best answered with STAR. Your answer has the content — it just needs the framework.",
    "Try writing out your STAR version of this answer. You will almost always find it is cleaner and more persuasive.",
]

POOL_GENERAL_IMPROVEMENT = [
    "Record yourself answering this question on video. Watching it back once will teach you more than ten practice sessions in your head.",
    "Practise with a friend who will push back and ask follow-up questions. Comfort under pressure is the real skill.",
    "Write the answer out longhand first, then memorise the structure (not the words). That way it sounds natural, not scripted.",
    "Preparation transforms interview anxiety into interview confidence. The more versions of this answer you practise, the better any one version becomes.",
    "Read two or three strong sample answers to this question online — not to copy them but to understand what good looks like.",
    "Ask yourself: if I could only say one sentence in answer to this question, what would it be? Build outward from that.",
    "Time yourself. Most good answers land in 60 to 90 seconds. If yours is shorter or longer, that is useful to know.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def _pick(pool: list, n: int = 1) -> list:
    """Randomly pick n unique items from a pool."""
    return random.sample(pool, min(n, len(pool)))


def generate_feedback(
    sentiment_result:     dict,
    confidence_result:    dict,
    communication_result: dict,
    final_score:          int
) -> dict:
    """
    Generate varied, context-sensitive feedback by drawing from large pools.

    Returns:
        {
            "positives":   List[str],
            "suggestions": List[str],
        }
    """
    positives   = []
    suggestions = []

    s  = sentiment_result["score"]
    c  = confidence_result["score"]
    m  = communication_result["score"]

    filler_count    = confidence_result.get("filler_count", 0)
    hedge_count     = confidence_result.get("hedge_count", 0)
    assertive_count = confidence_result.get("assertive_count", 0)
    passive_count   = confidence_result.get("passive_count", 0)
    word_count      = confidence_result.get("word_count", 0)
    pos_hits        = sentiment_result.get("pos_hits", 0)
    neg_hits        = sentiment_result.get("neg_hits", 0)

    grammar_s   = communication_result.get("grammar", 0)
    vocab_s     = communication_result.get("vocabulary", 0)
    structure_s = communication_result.get("structure", 0)

    # ── POSITIVES ─────────────────────────────────────────────────────────────

    if s >= 65:
        positives += _pick(POOL_POSITIVE_TONE, 1)

    if pos_hits >= 3:
        positives += _pick(POOL_STRONG_KEYWORDS, 1)
    elif pos_hits >= 1:
        positives += _pick(POOL_STRONG_KEYWORDS, 1)

    if assertive_count >= 2:
        positives += _pick(POOL_ASSERTIVE_LANGUAGE, 1)

    if filler_count == 0:
        positives += _pick(POOL_NO_FILLERS, 1)

    if 80 <= word_count <= 250:
        positives += _pick(POOL_GOOD_LENGTH, 1)

    if grammar_s >= 78:
        positives += _pick(POOL_GOOD_GRAMMAR, 1)

    if vocab_s >= 60:
        positives += _pick(POOL_RICH_VOCABULARY, 1)

    if structure_s >= 65:
        positives += _pick(POOL_GOOD_STRUCTURE, 1)

    if c >= 70:
        positives += _pick(POOL_HIGH_CONFIDENCE, 1)

    if m >= 72:
        positives += _pick(POOL_GOOD_COMMUNICATION, 1)

    # Guarantee at least one positive
    if not positives:
        positives.append(
            "You attempted a direct answer to the question — that starting point is something to build from."
        )

    # Cap at 4 positives — quality over quantity
    positives = positives[:4]

    # ── SUGGESTIONS ───────────────────────────────────────────────────────────

    # Sentiment issues
    if neg_hits >= 2:
        suggestions += _pick(POOL_NEGATIVE_TONE, 1)
    elif s < 50:
        suggestions += _pick(POOL_LOW_SENTIMENT, 1)

    # Confidence issues
    if filler_count >= 3:
        suggestions += _pick(POOL_FILLER_WORDS, 1)
    elif filler_count >= 1:
        suggestions += _pick(POOL_FILLER_WORDS, 1)

    if hedge_count >= 3:
        suggestions += _pick(POOL_HEDGE_WORDS, 1)
    elif hedge_count >= 1:
        suggestions += _pick(POOL_HEDGE_WORDS, 1)

    if passive_count >= 2:
        suggestions += _pick(POOL_PASSIVE_VOICE, 1)

    if assertive_count == 0:
        suggestions += _pick(POOL_NO_ASSERTIVE, 1)

    # Length issues
    if word_count < 50:
        suggestions += _pick(POOL_TOO_SHORT, 2)
    elif word_count > 300:
        suggestions += _pick(POOL_TOO_LONG, 1)

    # Communication issues
    if grammar_s < 60:
        suggestions += _pick(POOL_POOR_GRAMMAR, 1)

    if vocab_s < 45:
        suggestions += _pick(POOL_POOR_VOCABULARY, 1)
    elif vocab_s < 60:
        suggestions += _pick(POOL_POOR_VOCABULARY, 1)

    if structure_s < 55:
        suggestions += _pick(POOL_POOR_STRUCTURE, 1)
    elif structure_s < 70:
        suggestions += _pick(POOL_POOR_STRUCTURE, 1)

    # STAR method tip for mid-range answers
    if 30 <= final_score <= 72:
        suggestions += _pick(POOL_STAR_METHOD, 1)

    # Always add one general practice tip
    suggestions += _pick(POOL_GENERAL_IMPROVEMENT, 1)

    # Deduplicate while preserving order
    seen = set()
    unique_suggestions = []
    for item in suggestions:
        if item not in seen:
            seen.add(item)
            unique_suggestions.append(item)

    # Cap at 5 suggestions
    unique_suggestions = unique_suggestions[:5]

    # Guarantee at least 2 suggestions
    while len(unique_suggestions) < 2:
        candidate = random.choice(POOL_GENERAL_IMPROVEMENT)
        if candidate not in unique_suggestions:
            unique_suggestions.append(candidate)

    return {
        "positives":   positives,
        "suggestions": unique_suggestions,
    }