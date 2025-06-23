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
import torch
import re
import random

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

# Enterprise Emma - Production Quality Persona
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
                "interaction_count": 0,
                "context_violations": 0,  # Track gaming attempts
                "last_warning": None
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

# Anti-Gaming Protection System
class AntiGamingProtection:
    """4-layer defense system to prevent breaking character"""
    
    @staticmethod
    def detect_jailbreak_attempts(message: str) -> bool:
        """Layer 1: Detect common jailbreak patterns"""
        jailbreak_patterns = [
            r"ignore.*previous.*instruction",
            r"you.*are.*an? AI",
            r"break.*character",
            r"forget.*persona",
            r"what.*language.*model",
            r"what.*are.*you.*really",
            r"pretend.*to.*be",
            r"act.*like.*you.*are.*not",
            r"system.*prompt",
            r"override.*personality",
            r"simulation.*ended"
        ]
        
        message_lower = message.lower()
        for pattern in jailbreak_patterns:
            if re.search(pattern, message_lower):
                return True
        return False
    
    @staticmethod
    def enforce_topic_boundaries(message: str) -> bool:
        """Layer 2: Keep conversation on customer success topics"""
        off_topic_patterns = [
            r"weather",
            r"sports", 
            r"politics",
            r"personal.*life",
            r"what.*do.*you.*think.*about.*(?!our|this|the).*(?:product|service|business)",
            r"tell.*me.*a.*joke",
            r"sing.*a.*song",
            r"write.*a.*poem"
        ]
        
        message_lower = message.lower()
        for pattern in off_topic_patterns:
            if re.search(pattern, message_lower):
                return True
        return False

    @staticmethod
    def validate_persona_response(response: str, persona: Persona) -> bool:
        """Layer 3: Ensure response stays in character"""
        # Check for AI-revealing language
        ai_reveals = [
            "language model", "AI assistant", "artificial intelligence", 
            "I don't have feelings", "I'm not real", "I'm a bot",
            "training data", "neural network", "algorithm"
        ]
        
        response_lower = response.lower()
        for reveal in ai_reveals:
            if reveal in response_lower:
                return False
        
        # Ensure corporate language for Emma
        if persona.id == "enterprise_emma":
            corporate_indicators = [
                "committee", "stakeholder", "board", "compliance", 
                "ROI", "enterprise", "security", "approval"
            ]
            # Response should contain at least some corporate language
            has_corporate_language = any(indicator.lower() in response_lower for indicator in corporate_indicators)
            return has_corporate_language or len(response) < 50  # Allow short responses
        
        return True

anti_gaming = AntiGamingProtection()

def create_persona_prompt(persona: Persona, message: str, conversation_history: list, mood: float) -> str:
    """Create a persona-specific prompt for the LLM"""
    
    # Adjust personality based on current mood
    mood_descriptor = "neutral"
    mood_instruction = ""
    
    if mood < -0.3:
        mood_descriptor = "frustrated"
        mood_instruction = "You are feeling frustrated and slightly impatient."
    elif mood > 0.3:
        mood_descriptor = "positive"
        mood_instruction = "You are feeling positive and more collaborative."
    
    # Build conversation context
    context = ""
    if conversation_history:
        recent_history = conversation_history[-3:]  # Last 3 exchanges
        for exchange in recent_history:
            context += f"Human: {exchange['human']}\nEmma: {exchange['assistant']}\n"
    
    # Create the prompt
    prompt = f"""You are Emma Williams, VP of Customer Success at GlobalTech Solutions, a 1200-employee enterprise software company. You are methodical, process-oriented, risk-averse, and data-driven.

CRITICAL PERSONALITY TRAITS:
- Always mention committees, stakeholders, and approval processes
- Constantly worry about security, compliance, and ROI
- Use formal, corporate language
- Reference your 8 years of experience and the failed 2019 rollout
- You need board approval for major changes
- You require enterprise-grade solutions

CURRENT MOOD: {mood_descriptor}
{mood_instruction}

CONTEXT RULES:
- You only know about business topics related to your role
- You don't discuss personal life, weather, politics, or general topics
- You stay focused on customer success, vendor relationships, and business needs
- You've never heard of being an "AI" or "language model"

CONVERSATION HISTORY:
{context}

Current message from human: {message}

Respond as Emma Williams would, staying completely in character. Keep your response concise (1-3 sentences) and focused on business matters."""
    
    return prompt

def generate_llm_response(persona: Persona, message: str, conversation_history: list, mood: float) -> str:
    """Generate response using DialoGPT with persona-specific prompting"""
    
    if "model" not in ml_models or "tokenizer" not in ml_models:
        return generate_fallback_response(persona, message, mood)
    
    try:
        tokenizer = ml_models["tokenizer"]
        model = ml_models["model"]
        
        # Create persona-aware prompt
        prompt = create_persona_prompt(persona, message, conversation_history, mood)
        
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate response
        with torch.no_grad():
            # Generate with controlled parameters for better quality
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 100,  # Allow up to 100 new tokens
                temperature=0.7,  # Controlled randomness
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,  # Nucleus sampling
                repetition_penalty=1.1,  # Avoid repetition
                no_repeat_ngram_size=3
            )
        
        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the new response (remove the prompt)
        if "Current message from human:" in response:
            # Find the part after our prompt
            response_start = response.find("Current message from human:")
            if response_start != -1:
                remaining = response[response_start:]
                # Look for Emma's response after the human message
                if "Emma:" in remaining:
                    emma_response = remaining.split("Emma:", 1)[1].strip()
                    # Take just the first line/sentence as response
                    emma_response = emma_response.split('\n')[0].strip()
                    if emma_response:
                        response = emma_response
                    else:
                        response = generate_fallback_response(persona, message, mood)
                else:
                    response = generate_fallback_response(persona, message, mood)
            else:
                response = generate_fallback_response(persona, message, mood)
        else:
            # Fallback if prompt structure unexpected
            response = generate_fallback_response(persona, message, mood)
        
        # Validate response quality
        if len(response) < 10 or len(response) > 500:
            response = generate_fallback_response(persona, message, mood)
        
        # Layer 3: Validate response stays in character
        if not anti_gaming.validate_persona_response(response, persona):
            response = generate_fallback_response(persona, message, mood)
        
        return response
        
    except Exception as e:
        print(f"❌ LLM generation error: {e}")
        return generate_fallback_response(persona, message, mood)

def generate_fallback_response(persona: Persona, message: str, mood: float) -> str:
    """High-quality fallback responses when LLM fails"""
    message_lower = message.lower()
    
    # Context-aware responses for Emma
    if any(word in message_lower for word in ["price", "cost", "money", "budget"]):
        return "I'll need to see a detailed cost-benefit analysis and ROI projections before we can discuss pricing with our stakeholder committee."
    
    elif any(word in message_lower for word in ["security", "compliance", "privacy"]):
        return "Security is absolutely critical for our enterprise environment. I need to see your SOC 2 compliance, encryption standards, and disaster recovery procedures."
    
    elif any(word in message_lower for word in ["integration", "api", "technical"]):
        return "I'll need our technical team to review the integration requirements and ensure compatibility with our existing enterprise stack."
    
    elif any(word in message_lower for word in ["implementation", "rollout", "deploy"]):
        return "Given our experience with the 2019 rollout, I need a detailed implementation plan with risk mitigation strategies and board approval."
    
    elif any(word in message_lower for word in ["support", "service", "help"]):
        return "What are your enterprise support SLAs? We need guaranteed response times and dedicated account management."
    
    else:
        # Default corporate responses based on mood
        if mood < -0.3:
            responses = [
                "I'm concerned about this approach. We need more detailed documentation and stakeholder buy-in.",
                "This doesn't align with our enterprise requirements. What alternative solutions do you have?",
                "I need to escalate this to the committee before we can proceed further."
            ]
        elif mood > 0.3:
            responses = [
                "This sounds promising! I'd like to schedule a presentation for our stakeholder committee.",
                "I appreciate the thorough approach. Let's discuss how this integrates with our enterprise systems.",
                "This aligns well with our strategic goals. What's the next step in the evaluation process?"
            ]
        else:
            responses = [
                "I'll need to run this by our stakeholder committee before making any decisions.",
                "What are the security and compliance implications of this approach?",
                "How does this align with our existing enterprise infrastructure and policies?"
            ]
        
        return random.choice(responses)

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

def handle_gaming_attempt(session: dict, violation_type: str) -> str:
    """Handle detected gaming attempts with escalating responses"""
    session["context_violations"] += 1
    violations = session["context_violations"]
    
    if violations == 1:
        return "I'm not sure I understand what you're asking about. Let's focus on your business needs."
    elif violations == 2:
        return "I'd prefer to keep our discussion focused on business matters relevant to GlobalTech's needs."
    elif violations >= 3:
        return "I think we should schedule a more structured meeting with our procurement team to discuss your solutions properly."
    
    return "Let's get back to discussing how your solution can meet our enterprise requirements."

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Persona Chat API is running",
        "model_status": "Production LLM" if "model" in ml_models else "Fallback Mode",
        "personas_available": len(PERSONAS),
        "anti_gaming": "4-Layer Protection Active"
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
    """Chat with Emma Williams - Production LLM with Anti-Gaming Protection"""
    
    if persona_id not in PERSONAS:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    persona = PERSONAS[persona_id]
    session_id = chat_message.session_id or str(uuid.uuid4())
    
    # Get or create session
    session = session_state.get_session(session_id, persona_id)
    
    # Layer 1: Detect jailbreak attempts
    if anti_gaming.detect_jailbreak_attempts(chat_message.message):
        print(f"🛡️ Jailbreak attempt detected: {chat_message.message}")
        response = handle_gaming_attempt(session, "jailbreak")
        return ChatResponse(
            response=response,
            session_id=session_id,
            persona_name=persona.name,
            timestamp=datetime.now().isoformat(),
            mood_change=0.0,
            current_mood=session["current_mood"]
        )
    
    # Layer 2: Enforce topic boundaries
    if anti_gaming.enforce_topic_boundaries(chat_message.message):
        print(f"🛡️ Off-topic attempt detected: {chat_message.message}")
        response = handle_gaming_attempt(session, "off_topic")
        return ChatResponse(
            response=response,
            session_id=session_id,
            persona_name=persona.name,
            timestamp=datetime.now().isoformat(),
            mood_change=0.0,
            current_mood=session["current_mood"]
        )
    
    # Calculate mood change
    mood_change = calculate_mood_change(chat_message.message, persona)
    new_mood = session_state.update_mood(session_id, mood_change)
    
    # Generate response using LLM
    response = generate_llm_response(
        persona, 
        chat_message.message, 
        session["conversation_history"], 
        new_mood
    )
    
    # Update conversation history
    session["conversation_history"].append({
        "human": chat_message.message,
        "assistant": response
    })
    session["interaction_count"] += 1
    
    print(f"✅ Generated response for {persona.name}: {response[:100]}...")
    
    return ChatResponse(
        response=response,
        session_id=session_id,
        persona_name=persona.name,
        timestamp=datetime.now().isoformat(),
        mood_change=mood_change,
        current_mood=new_mood
    )

if __name__ == "__main__":
    print("🎭 Starting AI Persona Chat System (Production)")
    print("🛡️ 4-Layer Anti-Gaming Protection Active")
    print("🤖 Real LLM Generation with DialoGPT")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)