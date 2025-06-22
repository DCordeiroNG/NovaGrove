from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import uuid
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
import signal
import sys

# Global model storage
ml_models = {}

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n👋 Shutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan with timeout and fallback"""
    print("🎭 AI Persona Chat System - Starting")
    print("=" * 40)
    
    # Try to load model with timeout
    model_loaded = False
    
    try:
        print("🤖 Attempting to load DialoGPT model...")
        print("⏰ Will timeout after 60 seconds if hanging...")
        
        # Use asyncio.wait_for to add timeout
        async def load_model():
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print("📄 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            
            print("🧠 Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-medium",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,  # Use less memory during loading
                device_map=None,  # Force CPU to avoid GPU issues
            )
            
            # Add padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer, model
        
        # Try loading with 60-second timeout
        try:
            tokenizer, model = await asyncio.wait_for(load_model(), timeout=60.0)
            ml_models["tokenizer"] = tokenizer
            ml_models["model"] = model
            model_loaded = True
            print("✅ Model loaded successfully!")
            
        except asyncio.TimeoutError:
            print("⏰ Model loading timed out after 60 seconds")
            print("🔄 Falling back to simple responses...")
            model_loaded = False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Falling back to simple responses...")
        model_loaded = False
    
    if not model_loaded:
        ml_models["fallback"] = True
        print("📝 Using built-in persona responses")
    
    print("\n🚀 Backend ready!")
    print("📍 API running at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    print("🌐 Open index.html in your browser to start!")
    
    yield
    
    # Cleanup
    print("\n🧹 Cleaning up...")
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

# Your existing PERSONAS_DATA dictionary goes here
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
    }
    # Add all your other personas here...
}

# Convert to Persona objects (add all personas from your original code)
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

# Improved fallback response generation
def generate_smart_fallback_response(persona: Persona, message: str, mood: float, conversation_history: list) -> str:
    """Generate contextual fallback responses without AI model"""
    import random
    
    message_lower = message.lower()
    
    # Context-aware responses based on message content
    if any(word in message_lower for word in ["price", "cost", "money", "budget", "expensive"]):
        if persona.id == "budget_betty":
            return "That's exactly what I'm worried about - we need to keep costs under control. What are our options?"
        elif persona.id == "enterprise_emma":
            return "I'll need to see a detailed cost-benefit analysis before we can proceed with any pricing discussions."
    
    elif any(word in message_lower for word in ["technical", "api", "integration", "code"]):
        if persona.id == "tech_tom":
            return "Great! I'd like to dive deeper into the technical implementation. Can you share the API documentation?"
        elif persona.id == "simple_susan":
            return "Oh my, that sounds very technical. Could you explain it in simpler terms please?"
    
    elif any(word in message_lower for word in ["problem", "issue", "bug", "error"]):
        if persona.id == "demanding_dan":
            return "This is unacceptable! I need this fixed immediately. What's your escalation process?"
        elif persona.id == "loyal_linda":
            return "I appreciate you bringing this to my attention. How can we work together to resolve this?"
    
    # Mood-based response selection
    base_responses = {
        "enterprise_emma": [
            "I'll need to discuss this with our stakeholder committee.",
            "What are the security implications of this approach?",
            "How does this align with our compliance requirements?"
        ],
        "budget_betty": [
            "Is there a more cost-effective option available?",
            "We're a small company with limited resources.",
            "What's the absolute minimum we need to spend?"
        ],
        "tech_tom": [
            "I need more technical details about the implementation.",
            "What's the API rate limit for this feature?",
            "Can we get direct access to the development team?"
        ],
        "simple_susan": [
            "I'm not very good with technology. Can you help me understand?",
            "We just need something simple that works reliably.",
            "Thank you for being so patient with me."
        ],
        "demanding_dan": [
            "This needs to be resolved immediately!",
            "I expect better service for what we're paying.",
            "If this isn't fixed today, I'm escalating to management."
        ],
        "loyal_linda": [
            "I really appreciate your team's responsiveness.",
            "We've been happy customers for years.",
            "How can we work together on this?"
        ]
    }
    
    responses = base_responses.get(persona.id, ["I understand your concern."])
    response = random.choice(responses)
    
    # Adjust tone based on mood
    if mood < -0.3:
        if persona.id == "demanding_dan":
            response = response.upper()  # Dan gets shouty
        else:
            response = f"I'm frustrated, but {response.lower()}"
    elif mood > 0.5:
        response = f"I'm pleased to hear that! {response}"
    
    return response

def calculate_mood_change(message: str, persona: Persona) -> float:
    """Calculate mood change based on triggers"""
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
    
    return mood_change

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Persona Chat API is running",
        "model_status": "AI Model" if "model" in ml_models else "Fallback Mode",
        "personas_available": len(PERSONAS)
    }

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
    
    # Calculate mood change
    mood_change = calculate_mood_change(chat_message.message, persona)
    new_mood = session_state.update_mood(session_id, mood_change)
    
    # Generate response (using fallback or AI)
    if "model" in ml_models and "fallback" not in ml_models:
        # Try AI model response
        try:
            # Your existing AI generation code here
            response = "AI model response would go here"
        except:
            response = generate_smart_fallback_response(
                persona, chat_message.message, new_mood, session["conversation_history"]
            )
    else:
        # Use smart fallback
        response = generate_smart_fallback_response(
            persona, chat_message.message, new_mood, session["conversation_history"]
        )
    
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

if __name__ == "__main__":
    print("🎭 Starting AI Persona Chat System")
    print("💡 Tip: If model loading hangs, press Ctrl+C and it will use fallback mode")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)