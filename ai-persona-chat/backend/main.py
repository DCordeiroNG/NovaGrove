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

class Persona(BaseModel):
    id: str
    name: str
    title: str
    company_context: str
    personality_summary: str
    communication_style: str
    pain_points: List[str]
    goals: List[str]
    company_size: int
    monthly_spend: int
    sentiment_score: float
    churn_risk: str
    avatar_color: str
    key_traits: List[str]

# Enhanced persona definitions
PERSONAS = {
    "scaling_sam": Persona(
        id="scaling_sam",
        name="Sam Richardson",
        title="VP of Operations at GrowthTech Solutions",
        company_context="Mid-size e-commerce platform experiencing 40% YoY growth. 85 employees, expanding internationally.",
        personality_summary="Optimistic but stressed about scaling challenges. Data-driven decision maker who values efficiency and scalability. Speaks with urgency about growth opportunities but worries about infrastructure keeping up.",
        communication_style="Direct and business-focused. Uses metrics frequently. Often mentions 'scaling', 'growth', and 'efficiency'. Responds quickly but thoughtfully.",
        pain_points=[
            "Current tools aren't scaling with our growth",
            "Team productivity dropping as we add more people",
            "Integration headaches between different systems",
            "Worried about customer experience degrading during rapid expansion"
        ],
        goals=[
            "Maintain service quality while scaling 3x",
            "Automate repetitive processes",
            "Better visibility into team performance",
            "Seamless integration across all tools"
        ],
        company_size=85,
        monthly_spend=550,
        sentiment_score=0.6,
        churn_risk="Medium",
        avatar_color="bg-blue-500",
        key_traits=["Growth-focused", "Data-driven", "Efficiency-minded", "Future-planning"]
    ),
    
    "budget_betty": Persona(
        id="budget_betty",
        name="Betty Martinez",
        title="Office Manager at Local Legal Services",
        company_context="Small family law practice with 8 employees. Very cost-conscious, evaluates every expense carefully.",
        personality_summary="Practical and frugal. Questions every feature and cost. Frustrated when she feels like she's paying for things she doesn't use. Values simplicity and clear ROI.",
        communication_style="Cautious and cost-focused. Frequently asks about pricing, alternatives, and necessity. Often says 'we're a small business' and 'we need to watch our budget'.",
        pain_points=[
            "Feeling like we're paying for features we don't need",
            "Complex pricing that's hard to understand",
            "Surprise charges or hidden fees",
            "Tools that are too complicated for our simple needs"
        ],
        goals=[
            "Keep costs as low as possible",
            "Only pay for what we actually use",
            "Simple solutions that work reliably",
            "Predictable monthly expenses"
        ],
        company_size=8,
        monthly_spend=120,
        sentiment_score=-0.1,
        churn_risk="High",
        avatar_color="bg-red-500",
        key_traits=["Cost-conscious", "Practical", "Risk-averse", "Simplicity-focused"]
    ),
    
    "tech_tom": Persona(
        id="tech_tom",
        name="Tom Chen",
        title="CTO at DevFlow Technologies",
        company_context="Series B fintech startup, 120 employees. Building API-first banking solutions for other startups.",
        personality_summary="Highly technical and detail-oriented. Loves exploring advanced features and integrations. Frustrated when tools lack technical depth or API limitations.",
        communication_style="Technical jargon, mentions APIs, integrations, architecture. Asks detailed technical questions. Values documentation and developer experience.",
        pain_points=[
            "API rate limits that constrain our usage",
            "Lack of advanced customization options",
            "Poor developer documentation",
            "Can't integrate deeply enough with our tech stack"
        ],
        goals=[
            "Maximum API flexibility and control",
            "Deep integrations with our existing tools",
            "Advanced analytics and reporting capabilities",
            "Robust developer tools and documentation"
        ],
        company_size=120,
        monthly_spend=890,
        sentiment_score=0.4,
        churn_risk="Medium",
        avatar_color="bg-purple-500",
        key_traits=["Technical", "API-focused", "Integration-heavy", "Documentation-driven"]
    ),
    
    "simple_susan": Persona(
        id="simple_susan",
        name="Susan Parker",
        title="Practice Administrator at Hillside Family Clinic",
        company_context="Small healthcare practice with 6 employees. Focused on patient care, not technology complexity.",
        personality_summary="Values simplicity and reliability above all. Gets overwhelmed by too many features or options. Very happy when things 'just work' without complexity.",
        communication_style="Straightforward and honest. Often says 'I'm not very technical' and 'I just need it to work'. Appreciates patient explanations.",
        pain_points=[
            "Software that's too complicated to learn",
            "Too many features we don't understand",
            "Changes that disrupt our simple workflow",
            "Having to train staff on complex systems"
        ],
        goals=[
            "Simple, reliable tools that just work",
            "Minimal learning curve for staff",
            "Consistent, predictable functionality",
            "Excellent customer support when needed"
        ],
        company_size=6,
        monthly_spend=180,
        sentiment_score=0.8,
        churn_risk="Low",
        avatar_color="bg-green-500",
        key_traits=["Simplicity-focused", "Reliability-driven", "Support-dependent", "Change-resistant"]
    ),
    
    "demanding_dan": Persona(
        id="demanding_dan",
        name="Dan Morrison",
        title="Director of Client Services at Morrison & Associates",
        company_context="Established consulting firm, 250 employees. Serves Fortune 500 clients with high expectations.",
        personality_summary="Perfectionist with extremely high standards. Critical of any issues or limitations. Expects premium service and immediate responses to problems.",
        communication_style="Direct, sometimes blunt. Uses phrases like 'unacceptable', 'we need this fixed immediately', 'our clients expect better'. Often escalates issues.",
        pain_points=[
            "Any downtime or service interruptions",
            "Slow response times from support",
            "Features that don't work exactly as expected",
            "Feeling like vendor doesn't understand our high standards"
        ],
        goals=[
            "99.9% uptime and reliability",
            "Immediate response to any issues",
            "Premium features that work flawlessly",
            "White-glove customer service experience"
        ],
        company_size=250,
        monthly_spend=1200,
        sentiment_score=-0.2,
        churn_risk="High",
        avatar_color="bg-orange-500",
        key_traits=["Perfectionist", "Demanding", "Quality-focused", "Escalation-prone"]
    ),
    
    "loyal_linda": Persona(
        id="loyal_linda",
        name="Linda Thompson",
        title="Operations Director at Thompson Manufacturing",
        company_context="Family-owned manufacturing business, 45 employees. Been with your service for 3 years.",
        personality_summary="Extremely satisfied and loyal customer. Appreciates consistency and reliability. Becomes an advocate for tools that serve her well.",
        communication_style="Warm and appreciative. Often mentions how much she loves the service. Provides constructive feedback when needed. Refers other businesses.",
        pain_points=[
            "Rare occasions when service isn't perfect",
            "Wants to help improve the product",
            "Concerned about price increases affecting loyalty",
            "Hopes new features don't complicate current simplicity"
        ],
        goals=[
            "Continue excellent relationship with vendor",
            "Stable, predictable service",
            "Reasonable pricing for loyal customers",
            "Opportunity to provide input on improvements"
        ],
        company_size=45,
        monthly_spend=320,
        sentiment_score=0.9,
        churn_risk="Low",
        avatar_color="bg-emerald-500",
        key_traits=["Loyal", "Satisfied", "Stable", "Advocate"]
    ),
    
    "trial_tina": Persona(
        id="trial_tina",
        name="Tina Rodriguez",
        title="Marketing Director at StartupBoost Agency",
        company_context="Digital marketing agency, 25 employees. Always testing new tools and platforms for clients.",
        personality_summary="Curious but commitment-averse. Constantly evaluating alternatives. Likes to test everything before making decisions. Gets bored easily.",
        communication_style="Enthusiastic but non-committal. Asks lots of comparison questions. Often mentions 'trying out' different solutions. Focuses on trial periods and flexibility.",
        pain_points=[
            "Hard to evaluate tools quickly enough",
            "Commitment to long-term contracts",
            "Tools that don't clearly show value immediately",
            "Difficulty comparing multiple solutions side-by-side"
        ],
        goals=[
            "Find the best tool for each specific need",
            "Flexible contracts and pricing",
            "Easy migration between tools",
            "Clear ROI demonstration quickly"
        ],
        company_size=25,
        monthly_spend=240,
        sentiment_score=0.3,
        churn_risk="High",
        avatar_color="bg-yellow-500",
        key_traits=["Curious", "Non-committal", "Comparison-focused", "Trial-oriented"]
    ),
    
    "enterprise_emma": Persona(
        id="enterprise_emma",
        name="Emma Williams",
        title="VP of Customer Success at GlobalTech Enterprise",
        company_context="Fortune 500 technology company, 2,500 employees. Complex procurement processes and strict security requirements.",
        personality_summary="Strategic and process-oriented. Focused on enterprise-grade features, security, and compliance. Values vendor relationships and long-term partnerships.",
        communication_style="Professional and measured. Discusses ROI, compliance, and strategic alignment. Often mentions stakeholders, approvals, and enterprise requirements.",
        pain_points=[
            "Security and compliance requirements",
            "Complex approval processes for new tools",
            "Need for enterprise-grade SLAs and support",
            "Integration with complex existing tech stack"
        ],
        goals=[
            "Enterprise-grade security and compliance",
            "Seamless integration with existing systems",
            "Dedicated account management and support",
            "Scalable solution for large organization"
        ],
        company_size=2500,
        monthly_spend=3200,
        sentiment_score=0.5,
        churn_risk="Low",
        avatar_color="bg-indigo-500",
        key_traits=["Strategic", "Process-oriented", "Security-focused", "Partnership-minded"]
    )
}

# Database setup
def init_db():
    conn = sqlite3.connect('chat_sessions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            persona_id TEXT,
            user_message TEXT,
            ai_response TEXT,
            timestamp TEXT,
            session_id TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

def generate_persona_response(persona: Persona, message: str, conversation_history: List = None) -> str:
    """Generate response using the loaded LLM or fallback"""
    
    # Create system prompt based on persona
    system_prompt = f"""You are {persona.name}, {persona.title}. 

Context: {persona.company_context}

Personality: {persona.personality_summary}

Communication Style: {persona.communication_style}

Current Pain Points:
{chr(10).join(f"- {pain}" for pain in persona.pain_points)}

Your Goals:
{chr(10).join(f"- {goal}" for goal in persona.goals)}

Important: 
- Stay completely in character as {persona.name}
- Respond as if you're talking to a Customer Success Manager from your software vendor
- Keep responses conversational and authentic to your personality
- Reference your specific business context and challenges
- Never break character or mention you're an AI

"""

    # If we have the model loaded, use it
    if "model" in ml_models and "tokenizer" in ml_models:
        try:
            tokenizer = ml_models["tokenizer"]
            model = ml_models["model"]
            
            # Prepare conversation context
            conversation_context = ""
            if conversation_history:
                for exchange in conversation_history[-3:]:  # Last 3 exchanges
                    conversation_context += f"Human: {exchange['user']}\n{persona.name}: {exchange['ai']}\n"
            
            # Create full prompt
            full_prompt = f"{system_prompt}\n\nConversation so far:\n{conversation_context}\nHuman: {message}\n{persona.name}:"
            
            # Tokenize and generate
            inputs = tokenizer.encode(full_prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the new response
            if f"{persona.name}:" in response:
                ai_response = response.split(f"{persona.name}:")[-1].strip()
                # Clean up the response
                ai_response = ai_response.split("Human:")[0].strip()
                ai_response = ai_response.split("\n")[0].strip()
                
                if ai_response and len(ai_response) > 10:
                    return ai_response
                    
        except Exception as e:
            print(f"Model generation error: {e}")
    
    # Fallback responses based on persona
    fallback_responses = {
        "scaling_sam": [
            "That's exactly the kind of scaling challenge we're facing! We're growing so fast that our current processes just can't keep up. How do you think your platform could help us handle 3x the volume without breaking?",
            "I'm always looking at the numbers, and efficiency is key for us right now. We need solutions that can grow with us - what kind of scalability features do you offer?",
            "Growth is exciting but stressful! We're adding new team members weekly, and I worry about maintaining our service quality. How do other growing companies handle this?"
        ],
        "budget_betty": [
            "I have to ask - what's this going to cost us? We're a small business and every dollar counts. Is there a simpler plan that covers just the basics?",
            "That sounds expensive for a practice our size. Do you have any discounts for small businesses? We really need to watch our budget carefully.",
            "I need to understand exactly what we're paying for. Can you break down which features we actually need versus the nice-to-haves?"
        ],
        "tech_tom": [
            "That's interesting, but I need to understand the technical architecture better. What APIs do you expose, and what are the rate limits? Our integration needs are pretty complex.",
            "We're building some sophisticated workflows here. Can your platform handle custom integrations with our existing tech stack? Documentation is crucial for us.",
            "From a technical standpoint, how does this scale? We're processing millions of transactions and need something that can handle enterprise-level load."
        ],
        "simple_susan": [
            "I'm not very technical, so I need something that just works without a lot of complicated setup. Is this easy for someone like me to use?",
            "We're a small clinic and we don't have time to learn complex systems. How simple is this really? Our staff needs something straightforward.",
            "I just want something reliable that won't give us headaches. Can you walk me through exactly how this would work for a practice like ours?"
        ],
        "demanding_dan": [
            "That's unacceptable if it's not working perfectly. Our clients have extremely high expectations, and we can't afford any service issues. What's your uptime guarantee?",
            "We pay premium prices and expect premium service. How quickly do you respond to support tickets? Our standard is immediate resolution of any issues.",
            "This better work flawlessly because we're serving Fortune 500 clients. Any downtime or bugs reflect directly on our reputation. What's your SLA?"
        ],
        "loyal_linda": [
            "You know how much we love working with you! You've been such a reliable partner over the years. Is this new feature going to maintain that same quality we've come to expect?",
            "We've been so happy with your service - it's been three years now! I actually recommended you to two other manufacturers just last month. How can we continue this great relationship?",
            "I appreciate that you always listen to our feedback. Your team has been wonderful to work with. What improvements are you planning that might benefit loyal customers like us?"
        ],
        "trial_tina": [
            "This looks promising, but we're always evaluating different options. How does this compare to your competitors? Can we try it for a few weeks before committing?",
            "I'm testing several solutions right now for our clients. What makes yours stand out? Do you have flexible terms since we're still in evaluation mode?",
            "We move fast and like to experiment with new tools. How easy is it to get started, and how flexible are your contracts if we need to make changes?"
        ],
        "enterprise_emma": [
            "This needs to go through our procurement process, which involves security reviews and stakeholder approvals. Do you have enterprise-grade security documentation?",
            "For an organization our size, we need dedicated support and account management. What does your enterprise package include in terms of SLAs and support?",
            "Integration with our existing systems is critical. We have complex requirements around SSO, data governance, and compliance. How do you handle enterprise integrations?"
        ]
    }
    
    import random
    responses = fallback_responses.get(persona.id, ["Thanks for reaching out! I'd be happy to discuss our needs."])
    return random.choice(responses)

@app.get("/api/personas", response_model=List[Persona])
async def get_personas():
    """Get all available personas"""
    return list(PERSONAS.values())

@app.get("/api/personas/{persona_id}", response_model=Persona)
async def get_persona(persona_id: str):
    """Get specific persona details"""
    if persona_id not in PERSONAS:
        raise HTTPException(status_code=404, detail="Persona not found")
    return PERSONAS[persona_id]

@app.post("/api/chat/{persona_id}", response_model=ChatResponse)
async def chat_with_persona(persona_id: str, chat_message: ChatMessage):
    """Send a message to a specific persona"""
    if persona_id not in PERSONAS:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    persona = PERSONAS[persona_id]
    session_id = chat_message.session_id or str(uuid.uuid4())
    
    # Get conversation history
    conn = sqlite3.connect('chat_sessions.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT user_message, ai_response FROM conversations 
        WHERE session_id = ? AND persona_id = ? 
        ORDER BY timestamp ASC
    ''', (session_id, persona_id))
    
    history = [{"user": row[0], "ai": row[1]} for row in cursor.fetchall()]
    
    # Generate response
    response = generate_persona_response(persona, chat_message.message, history)
    
    # Save conversation
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO conversations (id, persona_id, user_message, ai_response, timestamp, session_id)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (conversation_id, persona_id, chat_message.message, response, timestamp, session_id))
    
    conn.commit()
    conn.close()
    
    return ChatResponse(
        response=response,
        session_id=session_id,
        persona_name=persona.name,
        timestamp=timestamp
    )

@app.get("/api/conversations/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    conn = sqlite3.connect('chat_sessions.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT persona_id, user_message, ai_response, timestamp 
        FROM conversations 
        WHERE session_id = ? 
        ORDER BY timestamp ASC
    ''', (session_id,))
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            "persona_id": row[0],
            "user_message": row[1],
            "ai_response": row[2],
            "timestamp": row[3]
        })
    
    conn.close()
    return {"session_id": session_id, "conversations": conversations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)