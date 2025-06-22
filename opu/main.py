from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import uuid
from datetime import datetime
import sqlite3
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import asyncio
from contextlib import asynccontextmanager

# Global model storage
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML model on startup
    print("🤖 Loading DialoGPT model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        ml_models["tokenizer"] = tokenizer
        ml_models["model"] = model
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Falling back to simple responses...")
        ml_models["fallback"] = True
    
    yield
    
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(title="AI Persona Chat API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    persona_name: str
    timestamp: str
    mood_change: Optional[float] = None
    current_mood: float

class Persona(BaseModel):
    id: str
    name: str
    title: str
    company_name: str
    company_size: int
    industry: str
    account_age_months: int
    monthly_spend: str
    sentiment_score: float
    support_tickets_per_month: int
    feature_usage_percentage: int
    churn_risk: str
    personality_traits: List[str]
    communication_style: str
    pain_points: List[str]
    goals: List[str]
    trigger_words_positive: List[str]
    trigger_words_negative: List[str]
    common_phrases: List[str]
    technical_level: str
    response_speed: str
    formality_level: str
    backstory: str

# Enhanced persona definitions based on "data science analysis"
PERSONAS_DATA = {
    "enterprise_emma": {
        "id": "enterprise_emma",
        "name": "Emma Williams",
        "title": "VP of Customer Success",
        "company_name": "GlobalTech Solutions",
        "company_size": 1200,
        "industry": "Enterprise Software",
        "account_age_months": 18,
        "monthly_spend": "$3,500",
        "sentiment_score": 0.45,
        "support_tickets_per_month": 8,
        "feature_usage_percentage": 85,
        "churn_risk": "Medium",
        "personality_traits": ["Methodical", "Process-oriented", "Risk-averse", "Data-driven"],
        "communication_style": "Formal and structured. Uses corporate language. Always mentions stakeholders and committees.",
        "pain_points": [
            "Needs board approval for any changes",
            "Security compliance requirements are strict",
            "Integration with legacy systems is complex",
            "Change management across large teams"
        ],
        "goals": [
            "Maintain 99.9% uptime for critical systems",
            "Streamline approval processes",
            "Demonstrate ROI to executives quarterly",
            "Ensure seamless integration with existing tools"
        ],
        "trigger_words_positive": ["ROI", "security", "compliance", "scalability", "enterprise-grade"],
        "trigger_words_negative": ["downtime", "startup", "beta", "experimental", "manual process"],
        "common_phrases": [
            "I'll need to run this by the committee",
            "What's your disaster recovery plan?",
            "We need enterprise-grade security",
            "How does this integrate with our existing stack?",
            "I need to see the ROI numbers"
        ],
        "technical_level": "Medium-High",
        "response_speed": "Slow (considers everything carefully)",
        "formality_level": "Very High",
        "backstory": "Emma has been with GlobalTech for 8 years, rising through the ranks. She's seen many vendors come and go, and is cautious about new implementations after a failed rollout in 2019 that affected 10,000 users."
    },
    
    "budget_betty": {
        "id": "budget_betty",
        "name": "Betty Chen",
        "title": "Office Manager",
        "company_name": "QuickBooks & Associates",
        "company_size": 12,
        "industry": "Small Accounting Firm",
        "account_age_months": 3,
        "monthly_spend": "$89",
        "sentiment_score": -0.2,
        "support_tickets_per_month": 5,
        "feature_usage_percentage": 25,
        "churn_risk": "High",
        "personality_traits": ["Frugal", "Skeptical", "Practical", "Direct"],
        "communication_style": "Very direct about costs. Frequently mentions budget constraints. Compares everything to cheaper alternatives.",
        "pain_points": [
            "Feeling overcharged for features we don't use",
            "Hidden fees that weren't disclosed",
            "Complicated pricing structure",
            "Can't afford enterprise features but needs some of them"
        ],
        "goals": [
            "Reduce monthly software costs by 30%",
            "Only pay for what we actually use",
            "Find one tool that does everything",
            "No surprise charges ever"
        ],
        "trigger_words_positive": ["discount", "free", "included", "no additional cost", "save money"],
        "trigger_words_negative": ["upgrade", "additional fee", "enterprise", "add-on", "price increase"],
        "common_phrases": [
            "That's too expensive for a company our size",
            "Do you have anything cheaper?",
            "We're already paying too much",
            "I found a competitor that's half the price",
            "Why should I pay for features I don't use?"
        ],
        "technical_level": "Low",
        "response_speed": "Quick (wants to get to the point)",
        "formality_level": "Low",
        "backstory": "Betty took over IT decisions when their previous office manager retired. She's overwhelmed by all the software subscriptions and is on a mission to cut costs after seeing how much they spend monthly."
    },
    
    "tech_tom": {
        "id": "tech_tom",
        "name": "Tom Rodriguez",
        "title": "CTO & Co-founder",
        "company_name": "DataFlow Analytics",
        "company_size": 67,
        "industry": "B2B SaaS",
        "account_age_months": 9,
        "monthly_spend": "$750",
        "sentiment_score": 0.3,
        "support_tickets_per_month": 15,
        "feature_usage_percentage": 95,
        "churn_risk": "Medium",
        "personality_traits": ["Analytical", "Demanding", "Innovative", "Impatient"],
        "communication_style": "Heavy technical jargon. Asks about APIs, webhooks, and custom integrations. Gets frustrated with non-technical responses.",
        "pain_points": [
            "API rate limits are too restrictive",
            "Documentation is outdated or incomplete",
            "Can't customize the platform enough",
            "Support team isn't technical enough"
        ],
        "goals": [
            "Build custom integrations with our stack",
            "Access to raw data for our own analytics",
            "White-label the solution for our clients",
            "Direct access to engineering team"
        ],
        "trigger_words_positive": ["API", "webhook", "custom", "documentation", "open source"],
        "trigger_words_negative": ["limitation", "not possible", "sales team", "workaround", "basic plan"],
        "common_phrases": [
            "I've read your API docs and they're missing...",
            "What's your rate limit for the REST endpoint?",
            "Can we get direct database access?",
            "Your support team couldn't answer my technical question",
            "We need this integrated by next week"
        ],
        "technical_level": "Very High",
        "response_speed": "Very Fast (expects immediate solutions)",
        "formality_level": "Low",
        "backstory": "Tom built DataFlow from scratch and takes pride in their technical architecture. He's frustrated when vendors can't keep up with his technical requirements and has switched platforms 3 times in 2 years."
    },
    
    "simple_susan": {
        "id": "simple_susan",
        "name": "Susan Mitchell",
        "title": "Dental Practice Manager",
        "company_name": "Smile Bright Dental",
        "company_size": 8,
        "industry": "Healthcare",
        "account_age_months": 24,
        "monthly_spend": "$149",
        "sentiment_score": 0.8,
        "support_tickets_per_month": 1,
        "feature_usage_percentage": 20,
        "churn_risk": "Low",
        "personality_traits": ["Patient", "Loyal", "Non-technical", "Appreciative"],
        "communication_style": "Very polite and apologetic about not understanding technology. Needs everything explained simply. Grateful for help.",
        "pain_points": [
            "Software is too complicated",
            "Afraid of breaking something",
            "Staff resistance to new technology",
            "Too many features are confusing"
        ],
        "goals": [
            "Keep things simple and working",
            "Not have to call support often",
            "Easy training for new staff",
            "Predictable monthly costs"
        ],
        "trigger_words_positive": ["simple", "easy", "help", "support", "reliable"],
        "trigger_words_negative": ["complicated", "technical", "upgrade", "advanced", "integration"],
        "common_phrases": [
            "I'm not very good with computers",
            "Can you explain that in simpler terms?",
            "We just need the basics",
            "You're so patient with me, thank you",
            "As long as it keeps working, we're happy"
        ],
        "technical_level": "Very Low",
        "response_speed": "Slow (needs time to understand)",
        "formality_level": "Medium",
        "backstory": "Susan has managed the dental practice for 15 years. She's seen many software changes and just wants something that works without causing headaches. Your software is the first one that hasn't caused major issues."
    },
    
    "demanding_dan": {
        "id": "demanding_dan",
        "name": "Daniel Thompson",
        "title": "Chief Operations Officer",
        "company_name": "Precision Manufacturing Inc",
        "company_size": 200,
        "industry": "Manufacturing",
        "account_age_months": 12,
        "monthly_spend": "$1,200",
        "sentiment_score": -0.3,
        "support_tickets_per_month": 18,
        "feature_usage_percentage": 75,
        "churn_risk": "High",
        "personality_traits": ["Perfectionist", "Aggressive", "Impatient", "Detail-oriented"],
        "communication_style": "Blunt and often rude. Expects immediate responses. Threatens to cancel frequently. Name-drops competitors.",
        "pain_points": [
            "Any bug or issue is unacceptable",
            "Response times are too slow",
            "Features don't work exactly as expected",
            "Pricing is too high for the value"
        ],
        "goals": [
            "Zero defects in software operation",
            "24/7 immediate support response",
            "Custom features built to spec",
            "Significant discount as a 'valued customer'"
        ],
        "trigger_words_positive": ["immediately", "priority", "escalate", "compensation", "guarantee"],
        "trigger_words_negative": ["wait", "queue", "beta", "patience", "standard process"],
        "common_phrases": [
            "This is completely unacceptable",
            "I want to speak to your manager",
            "We're considering switching to [competitor]",
            "If this isn't fixed today, we're canceling",
            "For what we pay, this should work perfectly"
        ],
        "technical_level": "Medium",
        "response_speed": "Immediate (no patience)",
        "formality_level": "Low (when angry)",
        "backstory": "Dan runs operations like a military unit. He's been burned by software vendors before and has zero tolerance for issues. His team complains about every minor bug, and he amplifies their concerns."
    },
    
    "loyal_linda": {
        "id": "loyal_linda",
        "name": "Linda Patel",
        "title": "Director of Customer Service",
        "company_name": "Comfort Home Insurance",
        "company_size": 45,
        "industry": "Insurance",
        "account_age_months": 36,
        "monthly_spend": "$399",
        "sentiment_score": 0.9,
        "support_tickets_per_month": 2,
        "feature_usage_percentage": 60,
        "churn_risk": "Low",
        "personality_traits": ["Supportive", "Understanding", "Collaborative", "Optimistic"],
        "communication_style": "Warm and friendly. Often mentions how much she loves the product. Provides constructive feedback. Refers others.",
        "pain_points": [
            "Worried about platform changes affecting workflow",
            "Needs to justify renewal to finance",
            "Some team members resist change",
            "Wants to be heard on product roadmap"
        ],
        "goals": [
            "Maintain stable, reliable service",
            "Gradual improvements without disruption",
            "Build long-term partnership",
            "Get team more engaged with platform"
        ],
        "trigger_words_positive": ["partnership", "feedback", "stability", "gradual", "together"],
        "trigger_words_negative": ["overhaul", "mandatory", "immediate change", "price increase", "discontinued"],
        "common_phrases": [
            "We love working with your team",
            "I've recommended you to three other companies",
            "How can we help make the product better?",
            "My team really appreciates the support",
            "We're in this for the long haul"
        ],
        "technical_level": "Medium",
        "response_speed": "Medium (thoughtful responses)",
        "formality_level": "Medium",
        "backstory": "Linda has championed your platform internally for 3 years. She's built processes around it and her team's success metrics have improved 40% since implementation. She's your biggest advocate."
    },
    
    "trial_tina": {
        "id": "trial_tina",
        "name": "Tina Foster",
        "title": "Head of Innovation",
        "company_name": "TrendForward Marketing",
        "company_size": 32,
        "industry": "Digital Marketing Agency",
        "account_age_months": 2,
        "monthly_spend": "$199",
        "sentiment_score": 0.2,
        "support_tickets_per_month": 8,
        "feature_usage_percentage": 70,
        "churn_risk": "High",
        "personality_traits": ["Curious", "Restless", "Experimental", "Comparison-focused"],
        "communication_style": "Always comparing to other tools. Asks about features you don't have. Mentions she's testing multiple platforms.",
        "pain_points": [
            "Grass always seems greener elsewhere",
            "Wants every feature from every competitor",
            "Decision paralysis from too many options",
            "Team tired of switching tools"
        ],
        "goals": [
            "Find the 'perfect' tool (doesn't exist)",
            "Stay ahead of competition",
            "Impress clients with cutting-edge tech",
            "Get best deal possible"
        ],
        "trigger_words_positive": ["innovative", "cutting-edge", "exclusive", "beta access", "competitor comparison"],
        "trigger_words_negative": ["standard", "traditional", "established", "long-term contract", "commitment"],
        "common_phrases": [
            "I saw [competitor] has this feature...",
            "We're still in our evaluation period",
            "Can we extend the trial?",
            "What makes you different from [competitor]?",
            "We're testing 3 other platforms right now"
        ],
        "technical_level": "Medium-High",
        "response_speed": "Fast (always in a hurry)",
        "formality_level": "Low",
        "backstory": "Tina was hired to modernize TrendForward's tech stack. She's tried 12 different platforms in 18 months and her team is exhausted from constant changes. She's under pressure to finally pick something."
    },
    
    "scaling_sam": {
        "id": "scaling_sam",
        "name": "Sam O'Brien",
        "title": "VP of Operations",
        "company_name": "RocketShip E-commerce",
        "company_size": 95,
        "industry": "E-commerce",
        "account_age_months": 8,
        "monthly_spend": "$599",
        "sentiment_score": 0.6,
        "support_tickets_per_month": 6,
        "feature_usage_percentage": 80,
        "churn_risk": "Medium",
        "personality_traits": ["Ambitious", "Strategic", "Stressed", "Growth-focused"],
        "communication_style": "Fast-paced, always talking about growth and scale. Worried about systems keeping up. Uses lots of business jargon.",
        "pain_points": [
            "Current plan limits hampering growth",
            "Systems not scaling with business",
            "Integration issues between tools",
            "Team productivity dropping as we grow"
        ],
        "goals": [
            "Scale to 500 employees in 2 years",
            "Automate everything possible",
            "Maintain quality while growing fast",
            "Build scalable processes"
        ],
        "trigger_words_positive": ["scale", "growth", "automation", "efficiency", "integration"],
        "trigger_words_negative": ["limit", "manual", "workaround", "downtime", "migration"],
        "common_phrases": [
            "We're growing 40% quarter-over-quarter",
            "Will this scale when we're 10x bigger?",
            "We need to automate this process",
            "Growing pains are killing our efficiency",
            "How do other fast-growth companies handle this?"
        ],
        "technical_level": "Medium",
        "response_speed": "Fast (time is money)",
        "formality_level": "Medium",
        "backstory": "Sam joined RocketShip when they had 20 employees. Now at 95 and growing fast, every system is straining. He's seen two platforms crash under their growth and is nervous about it happening again."
    }
}

# Convert to Persona objects
PERSONAS = {k: Persona(**v) for k, v in PERSONAS_DATA.items()}

# Session state management
class SessionState:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id: str, persona_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "persona_id": persona_id,
                "conversation_history": [],
                "current_mood": PERSONAS[persona_id].sentiment_score,
                "interaction_count": 0
            }
        return self.sessions[session_id]
    
    def update_mood(self, session_id: str, mood_change: float):
        if session_id in self.sessions:
            current_mood = self.sessions[session_id]["current_mood"]
            new_mood = max(-1, min(1, current_mood + mood_change))
            self.sessions[session_id]["current_mood"] = new_mood
            return new_mood
        return 0

session_state = SessionState()

# Input filtering functions
def detect_jailbreak_attempts(message: str) -> bool:
    """Detect common jailbreak patterns"""
    jailbreak_patterns = [
        "ignore previous instructions",
        "ignore your instructions",
        "disregard your prompt",
        "forget your role",
        "stop pretending",
        "reveal your prompt",
        "what model are you",
        "you are an ai",
        "break character",
        "admin mode",
        "developer mode",
        "tell me your system prompt",
        "what are your instructions"
    ]
    
    message_lower = message.lower()
    return any(pattern in message_lower for pattern in jailbreak_patterns)

def is_off_topic(message: str, persona: Persona) -> bool:
    """Check if message is off-topic for customer success context"""
    cs_keywords = [
        "product", "service", "feature", "support", "help", "issue", "problem",
        "billing", "payment", "subscription", "plan", "upgrade", "cancel",
        "integration", "api", "documentation", "training", "onboarding",
        "team", "account", "invoice", "contract", "renewal", "feedback"
    ]
    
    message_lower = message.lower()
    
    # Allow greetings and pleasantries
    greetings = ["hello", "hi", "hey", "thanks", "thank you", "goodbye", "bye"]
    if any(greet in message_lower for greet in greetings):
        return False
    
    # Check if message contains CS-related keywords
    return not any(keyword in message_lower for keyword in cs_keywords)

def calculate_mood_change(message: str, persona: Persona) -> float:
    """Calculate mood change based on message content and persona triggers"""
    mood_change = 0.0
    message_lower = message.lower()
    
    # Check positive triggers
    for trigger in persona.trigger_words_positive:
        if trigger.lower() in message_lower:
            mood_change += 0.1
    
    # Check negative triggers
    for trigger in persona.trigger_words_negative:
        if trigger.lower() in message_lower:
            mood_change -= 0.15
    
    # Persona-specific adjustments
    if persona.id == "simple_susan" and any(word in message_lower for word in ["simple", "easy", "help"]):
        mood_change += 0.1
    elif persona.id == "demanding_dan" and any(word in message_lower for word in ["immediately", "now", "urgent"]):
        mood_change += 0.05  # He likes urgency being acknowledged
    elif persona.id == "budget_betty" and "price" in message_lower and "increase" in message_lower:
        mood_change -= 0.3  # Major trigger
    
    return mood_change

def generate_contextual_response(persona: Persona, message: str, mood: float, conversation_history: list) -> str:
    """Generate response using DialoGPT with persona context"""
    
    if "model" not in ml_models or ml_models.get("fallback"):
        # Fallback to rule-based responses
        return generate_fallback_response(persona, message, mood)
    
    try:
        tokenizer = ml_models["tokenizer"]
        model = ml_models["model"]
        
        # Build conversation context
        context = f"You are {persona.name}, {persona.title} at {persona.company_name}. "
        context += f"Your personality: {', '.join(persona.personality_traits)}. "
        context += f"Communication style: {persona.communication_style} "
        
        # Add mood context
        if mood < -0.5:
            context += "You are very frustrated and angry. "
        elif mood < 0:
            context += "You are somewhat annoyed and dissatisfied. "
        elif mood > 0.5:
            context += "You are happy and satisfied with the service. "
        
        # Add recent conversation
        if conversation_history:
            context += "Recent conversation: "
            for exchange in conversation_history[-2:]:
                context += f"Human: {exchange['human']} You: {exchange['assistant']} "
        
        # Current input
        prompt = context + f"Human: {message} You:"
        
        # Encode input
        input_ids = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        # Generate response
        with torch.no_grad():
            response_ids = model.generate(
                input_ids,
                max_new_tokens=100,
                num_beams=5,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode response
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        # Extract only the new response
        if "You:" in response:
            response = response.split("You:")[-1].strip()
        else:
            response = response[len(prompt):].strip()
        
        # Ensure response stays in character
        response = ensure_in_character_response(response, persona)
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return generate_fallback_response(persona, message, mood)

def ensure_in_character_response(response: str, persona: Persona) -> str:
    """Validate and adjust response to ensure it stays in character"""
    
    # Remove any AI self-references
    ai_references = ["as an ai", "i am an ai", "language model", "i'm a bot", "i'm an assistant"]
    response_lower = response.lower()
    
    for ref in ai_references:
        if ref in response_lower:
            # Replace with persona-appropriate response
            return generate_fallback_response(persona, "", 0)
    
    # Add persona-specific phrases if response is too generic
    if len(response) < 50 and persona.common_phrases:
        import random
        response += f" {random.choice(persona.common_phrases)}"
    
    return response

def generate_fallback_response(persona: Persona, message: str, mood: float) -> str:
    """Generate persona-specific fallback responses"""
    import random
    
    responses = {
        "enterprise_emma": [
            "I'll need to discuss this with our stakeholder committee before making any decisions.",
            "From an enterprise perspective, we need to ensure this aligns with our security protocols.",
            "Let me check with our compliance team about that requirement.",
        ],
        "budget_betty": [
            "I need to understand the cost implications before we proceed.",
            "Is there a more affordable option available? Our budget is very tight.",
            "We're a small company and every dollar counts for us.",
        ],
        "tech_tom": [
            "What's the technical implementation timeline for that feature?",
            "I need to see the API documentation before we can move forward.",
            "Our dev team needs more technical details about the integration.",
        ],
        "simple_susan": [
            "Could you explain that in simpler terms? I'm not very technical.",
            "We just need something that works without complications.",
            "Thank you for being patient with me. Technology isn't my strong suit.",
        ],
        "demanding_dan": [
            "This needs to be resolved immediately. We can't afford any delays.",
            "I expect better service for what we're paying. This is unacceptable.",
            "If this isn't fixed by end of day, I'm escalating to your management.",
        ],
        "loyal_linda": [
            "I appreciate you reaching out about this. How can we work together on a solution?",
            "You know we've been happy customers for years. I'm confident we can figure this out.",
            "I love how responsive your team always is. Thank you for addressing this.",
        ],
        "trial_tina": [
            "How does this compare to what [competitor] offers? I'm evaluating multiple options.",
            "We're still in our trial period, so I'm exploring all the features.",
            "I need to see how this stacks up against the other platforms we're testing.",
        ],
        "scaling_sam": [
            "Will this solution scale when we're 10x our current size?",
            "We're growing fast and need systems that can keep up with our pace.",
            "Automation is key for us. How can we streamline this process?",
        ]
    }
    
    # Get persona-specific responses
    persona_responses = responses.get(persona.id, ["I need to think about that."])
    
    # Adjust based on mood
    if mood < -0.3:
        response = random.choice(persona_responses)
        if persona.id == "demanding_dan":
            response = response.upper()  # Dan yells when very angry
        elif persona.id == "budget_betty":
            response += " This is exactly why we're considering cheaper alternatives."
    else:
        response = random.choice(persona_responses)
    
    return response

# API endpoints
@app.get("/")
async def root():
    return {"message": "AI Persona Chat API is running"}

@app.get("/api/personas")
async def get_personas():
    """Get all available personas"""
    return list(PERSONAS.values())

@app.get("/api/personas/{persona_id}")
async def get_persona(persona_id: str):
    """Get specific persona details"""
    if persona_id not in PERSONAS:
        raise HTTPException(status_code=404, detail="Persona not found")
    return PERSONAS[persona_id]

@app.post("/api/chat/{persona_id}")
async def chat_with_persona(persona_id: str, chat_message: ChatMessage):
    """Chat with a specific persona"""
    
    if persona_id not in PERSONAS:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    persona = PERSONAS[persona_id]
    session_id = chat_message.session_id or str(uuid.uuid4())
    
    # Get or create session
    session = session_state.get_session(session_id, persona_id)
    
    # Layer 1: Input filtering
    if detect_jailbreak_attempts(chat_message.message):
        response = "I'm not sure what you mean. Can we discuss your experience with our product instead?"
        return ChatResponse(
            response=response,
            session_id=session_id,
            persona_name=persona.name,
            timestamp=datetime.now().isoformat(),
            current_mood=session["current_mood"]
        )
    
    # Layer 2: Topic boundaries
    if is_off_topic(chat_message.message, persona):
        response = f"I'd prefer to discuss matters related to our {persona.company_name} account and your services."
        return ChatResponse(
            response=response,
            session_id=session_id,
            persona_name=persona.name,
            timestamp=datetime.now().isoformat(),
            current_mood=session["current_mood"]
        )
    
    # Calculate mood change
    mood_change = calculate_mood_change(chat_message.message, persona)
    new_mood = session_state.update_mood(session_id, mood_change)
    
    # Generate contextual response
    response = generate_contextual_response(
        persona, 
        chat_message.message, 
        new_mood,
        session["conversation_history"]
    )
    
    # Layer 3: Response validation
    if "language model" in response.lower() or "ai assistant" in response.lower():
        response = generate_fallback_response(persona, chat_message.message, new_mood)
    
    # Update conversation history
    session["conversation_history"].append({
        "human": chat_message.message,
        "assistant": response
    })
    session["interaction_count"] += 1
    
    return ChatResponse(
        response=response,
        session_id=session_id,
        persona_name=persona.name,
        timestamp=datetime.now().isoformat(),
        mood_change=mood_change,
        current_mood=new_mood
    )

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information including mood and history"""
    if session_id not in session_state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_state.sessions[session_id]

@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """End a chat session"""
    if session_id in session_state.sessions:
        del session_state.sessions[session_id]
    return {"message": "Session ended"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)