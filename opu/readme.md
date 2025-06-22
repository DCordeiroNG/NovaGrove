# AI Persona Chat System

A sophisticated customer success training tool that allows CSMs to practice conversations with 8 distinct AI-powered customer personas.

## 🎯 Features

- **8 Rich Personas**: From Enterprise Emma to Budget Betty, each with unique personalities
- **Real-time Mood System**: Personas react to your approach with dynamic mood changes
- **Anti-Gaming Protection**: 4-layer defense system prevents breaking character
- **Conversation Memory**: Personas remember context within each session
- **Free LLM**: Uses Microsoft DialoGPT (no API keys needed!)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js (optional, for serving frontend)

### Installation

1. **Clone/Create project directory**
```bash
mkdir ai-persona-chat
cd ai-persona-chat
```

2. **Save the files**
- Save `main.py` as your backend
- Save `index.html` as your frontend
- Save `requirements.txt` for dependencies

3. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the System

1. **Start the backend**
```bash
python main.py
```
The backend will:
- Download DialoGPT model on first run (~1.5GB)
- Start FastAPI server on http://localhost:8000

2. **Open the frontend**
- Simply open `index.html` in your browser
- Or serve it with Python: `python -m http.server 8080`

## 🎭 The 8 Personas

### 1. **Enterprise Emma** 🏢
- VP at GlobalTech (1200 employees)
- Process-oriented, needs board approval
- Triggers: ROI, security, compliance (+)
- Avoids: Downtime, beta features (-)

### 2. **Budget Betty** 💰
- Small accounting firm (12 employees)
- Very cost-conscious, skeptical
- Triggers: Discounts, free features (+)
- Avoids: Price increases, add-ons (-)

### 3. **Tech-Savvy Tom** 💻
- CTO at DataFlow Analytics
- Wants API access, technical depth
- Triggers: API, webhooks, custom (+)
- Avoids: Limitations, basic plans (-)

### 4. **Simple Susan** 😊
- Dental practice manager
- Non-technical, values simplicity
- Triggers: Easy, simple, support (+)
- Avoids: Complicated, technical (-)

### 5. **Demanding Dan** 😤
- Manufacturing COO
- Perfectionist, impatient
- Triggers: Immediate, priority (+)
- Avoids: Wait times, bugs (-)

### 6. **Loyal Linda** 💚
- Insurance company director
- 3-year customer, very satisfied
- Triggers: Partnership, stability (+)
- Avoids: Major changes (-)

### 7. **Trial-and-Error Tina** 🔄
- Marketing agency innovator
- Always comparing options
- Triggers: Innovation, trials (+)
- Avoids: Commitments, contracts (-)

### 8. **Scaling Sam** 📈
- E-commerce VP of Operations
- Growing 40% QoQ, stressed
- Triggers: Scale, automation (+)
- Avoids: Limits, manual work (-)

## 🛡️ Anti-Gaming Features

The system includes 4 layers of protection:

1. **Input Filtering**: Detects jailbreak attempts
2. **Topic Boundaries**: Keeps conversation on customer success
3. **Response Validation**: Ensures personas stay in character
4. **Context Limits**: Personas only know their business context

## 📊 Mood System

Each persona has a dynamic mood that changes based on:
- Positive trigger words → Mood improves
- Negative trigger words → Mood decreases
- Persona-specific reactions (e.g., Betty hates price talk)

## 🔧 Customization

### Adding New Personas

Edit `PERSONAS_DATA` in `main.py`:

```python
"new_persona": {
    "id": "new_persona",
    "name": "New Person",
    "title": "Job Title",
    "company_name": "Company Inc",
    # ... add all required fields
}
```

### Adjusting Mood Sensitivity

Modify `calculate_mood_change()` function:
```python
# Increase/decrease these values
mood_change += 0.1  # for positive triggers
mood_change -= 0.15  # for negative triggers
```

## 🐛 Troubleshooting

### "Model download stuck"
- First run downloads ~1.5GB DialoGPT model
- Check internet connection
- Delete `~/.cache/huggingface/` and retry

### "Personas not loading"
- Ensure backend is running on port 8000
- Check browser console for CORS errors
- Try opening frontend from `http://localhost:8080`

### "Responses are generic"
- Model needs time to warm up
- If DialoGPT fails, system uses fallback responses
- Check console for model loading errors

## 📈 Next Steps

1. **Collect Training Data**: Use generated conversations to improve personas
2. **Upgrade LLM**: Move to GPT-4 or Claude for better responses
3. **Add Analytics**: Track which approaches work best
4. **Expand Personas**: Add industry-specific variants

## 🤝 Contributing

Feel free to:
- Add new personas
- Improve conversation patterns
- Enhance the UI/UX
- Add analytics features

## 📄 License

MIT License - Use freely for your customer success training!