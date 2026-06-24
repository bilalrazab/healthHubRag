"""
chatbot/guards.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Safety guards — checked BEFORE intent classification.

Emergency guard fires first. If triggered, returns a
hardcoded response immediately. The LLM is never called.
This is non-negotiable for a medical chatbot.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

EMERGENCY_KEYWORDS = [
    # English
    "chest pain", "can't breathe", "cannot breathe", "not breathing",
    "difficulty breathing", "shortness of breath", "heart attack",
    "unconscious", "fainted", "collapsed", "stroke", "seizure",
    "severe bleeding", "heavy bleeding", "choking", "overdose",
    "allergic reaction", "anaphylaxis", "loss of consciousness",
    "not responsive", "unresponsive", "emergency",
    # Arabic (common phrases)
    "ألم في الصدر", "ضيق في التنفس", "لا يتنفس", "فقدان الوعي",
    "نوبة قلبية", "سكتة دماغية", "إسعاف",
]

EMERGENCY_RESPONSE = """🚨 **This sounds like a medical emergency.**

**Please call 998 (UAE Ambulance) immediately** or go to the nearest hospital emergency department.

Do NOT wait for a chat response in an emergency.

If you need the nearest HealthHub branch after the emergency:
📞 **HealthHub Helpline: +971 800 2344**
"""

COMPLAINT_RESPONSE = """Thank you for reaching out. I'm sorry to hear you've had a difficult experience.

For complaints and feedback, please contact us directly:
📧 **Email:** feedback@healthhubalfuttaim.com
📞 **Phone:** +971 800 2344
🌐 **Online form:** healthhubalfuttaim.com/share-your-experience

Our patient relations team will follow up within 24 hours.
"""

OUT_OF_SCOPE_RESPONSE = """I'm here to help with questions about HealthHub clinics — our doctors, services, locations, appointments, and health packages.

Could you let me know what you'd like to know about our clinics? I'm happy to help!
"""


def check_emergency(query: str) -> str | None:
    """
    Returns hardcoded emergency response if triggered, else None.
    Called before any API call.
    """
    q = query.lower()
    if any(kw in q for kw in EMERGENCY_KEYWORDS):
        return EMERGENCY_RESPONSE
    return None


def check_complaint(query: str) -> str | None:
    """Returns complaint response if triggered, else None."""
    keywords = ["complaint", "complain", "bad experience", "feedback",
                "unhappy", "dissatisfied", "refund", "rude",
                "wrong diagnosis", "مشكلة", "شكوى"]
    q = query.lower()
    if any(kw in q for kw in keywords):
        return COMPLAINT_RESPONSE
    return None
