import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Enhanced personas with realistic behavioral patterns
personas = {
    "Scaling Sam": {
        "company_size": (20, 150),
        "monthly_spend": (200, 800),
        "base_sentiment": 0.55,
        "support_tickets": (2, 8),
        "feature_usage": (0.6, 0.9),
        "industries": ["E-commerce", "Marketing Agency", "Real Estate", "Insurance"],
        "account_age_preference": (6, 24),
        "price_sensitivity": 0.3,
        "loyalty_factor": 0.7
    },
    "Budget Betty": {
        "company_size": (5, 30),
        "monthly_spend": (50, 250),
        "base_sentiment": 0.1,
        "support_tickets": (1, 5),
        "feature_usage": (0.3, 0.6),
        "industries": ["Small Business Services", "Local Retail", "Consulting", "Non-profit"],
        "account_age_preference": (1, 18),
        "price_sensitivity": 0.8,
        "loyalty_factor": 0.3
    },
    "Tech-Savvy Tom": {
        "company_size": (30, 200),
        "monthly_spend": (300, 1200),
        "base_sentiment": 0.45,
        "support_tickets": (3, 12),
        "feature_usage": (0.7, 1.0),
        "industries": ["Software", "Fintech", "SaaS", "Tech Startup"],
        "account_age_preference": (3, 30),
        "price_sensitivity": 0.4,
        "loyalty_factor": 0.6
    },
    "Simple Susan": {
        "company_size": (3, 25),
        "monthly_spend": (80, 300),
        "base_sentiment": 0.65,
        "support_tickets": (0, 3),
        "feature_usage": (0.2, 0.5),
        "industries": ["Healthcare", "Education", "Local Services", "Family Business"],
        "account_age_preference": (6, 36),
        "price_sensitivity": 0.5,
        "loyalty_factor": 0.8
    },
    "Demanding Dan": {
        "company_size": (50, 300),
        "monthly_spend": (400, 1500),
        "base_sentiment": -0.1,
        "support_tickets": (5, 20),
        "feature_usage": (0.6, 0.9),
        "industries": ["Legal", "Financial Services", "Enterprise", "Government"],
        "account_age_preference": (3, 24),
        "price_sensitivity": 0.2,
        "loyalty_factor": 0.9
    },
    "Loyal Linda": {
        "company_size": (15, 100),
        "monthly_spend": (150, 600),
        "base_sentiment": 0.75,
        "support_tickets": (0, 2),
        "feature_usage": (0.5, 0.8),
        "industries": ["Manufacturing", "Logistics", "Healthcare", "Professional Services"],
        "account_age_preference": (12, 36),
        "price_sensitivity": 0.3,
        "loyalty_factor": 0.95
    },
    "Trial-and-Error Tina": {
        "company_size": (10, 80),
        "monthly_spend": (100, 400),
        "base_sentiment": 0.25,
        "support_tickets": (2, 10),
        "feature_usage": (0.4, 0.8),
        "industries": ["Startup", "Digital Marketing", "E-commerce", "Creative Agency"],
        "account_age_preference": (1, 12),
        "price_sensitivity": 0.7,
        "loyalty_factor": 0.2
    },
    "Enterprise Emma": {
        "company_size": (200, 2000),
        "monthly_spend": (1000, 5000),
        "base_sentiment": 0.5,
        "support_tickets": (3, 15),
        "feature_usage": (0.7, 0.95),
        "industries": ["Enterprise", "Fortune 500", "Banking", "Telecommunications"],
        "account_age_preference": (6, 36),
        "price_sensitivity": 0.1,
        "loyalty_factor": 0.8
    }
}

def calculate_realistic_sentiment(base_sentiment, account_age, support_tickets, feature_usage, persona_data):
    """Calculate realistic sentiment based on multiple factors"""
    sentiment = base_sentiment
    
    # Account age effect (honeymoon period, then reality, then loyalty)
    if account_age <= 3:
        sentiment += 0.2  # Honeymoon period
    elif account_age <= 12:
        sentiment -= 0.15  # Reality check period
    else:
        sentiment += 0.1 * persona_data['loyalty_factor']  # Loyalty builds over time
    
    # Feature usage effect (more usage = more satisfaction)
    if feature_usage > 0.7:
        sentiment += 0.2
    elif feature_usage < 0.3:
        sentiment -= 0.2
    
    # Support tickets effect (too many = frustration, but good resolution helps)
    if support_tickets > 10:
        sentiment -= 0.3  # Too many tickets = frustrated
    elif support_tickets > 5:
        sentiment -= 0.1
    elif support_tickets == 0:
        sentiment += 0.1  # No issues = happy
    
    # Keep sentiment in reasonable bounds
    return max(-1, min(1, sentiment))

def determine_churn_risk(sentiment, account_age, persona_data):
    """Determine churn risk based on multiple factors"""
    risk_score = 0
    
    # Sentiment impact
    if sentiment < -0.2:
        risk_score += 3
    elif sentiment < 0.2:
        risk_score += 2
    elif sentiment < 0.5:
        risk_score += 1
    
    # Account age impact (new customers and very old dissatisfied ones at risk)
    if account_age <= 6:
        risk_score += 1  # New customers might churn
    elif account_age >= 24 and sentiment < 0.3:
        risk_score += 2  # Long-term but unhappy
    
    # Persona-specific factors
    if persona_data['loyalty_factor'] < 0.4:
        risk_score += 1  # Low loyalty personas
    
    # Convert to categories
    if risk_score >= 4:
        return "High"
    elif risk_score >= 2:
        return "Medium"
    else:
        return "Low"

def generate_customer_data():
    """Generate 50 customers with VERY OBVIOUS persona patterns for high AI accuracy"""
    
    customers = []
    customer_id = 1

    # Distribute 50 customers across personas (6-7 each)
    persona_distribution = [7, 6, 7, 6, 6, 7, 6, 5]  # Adds up to 50
    persona_names = list(personas.keys())

    for i, persona_name in enumerate(persona_names):
        persona_data = personas[persona_name]
        num_customers = persona_distribution[i]
        
        for _ in range(num_customers):
            # MAKE PATTERNS VERY OBVIOUS - use extremes of ranges
            
            if persona_name == "Enterprise Emma":
                # Make Enterprise Emma VERY obviously enterprise
                company_size = random.randint(500, 2000)  # Much larger
                monthly_spend = random.randint(2000, 5000)  # Much higher spend
                sentiment = round(random.uniform(0.3, 0.6), 2)  # Moderate (enterprises are cautious)
                support_tickets = random.randint(5, 15)  # Many tickets (complex needs)
                feature_usage = random.uniform(0.8, 0.95)  # High usage
                account_age_months = random.randint(12, 36)  # Established
                
            elif persona_name == "Budget Betty":
                # Make Budget Betty VERY obviously budget-conscious
                company_size = random.randint(3, 15)  # Very small
                monthly_spend = random.randint(50, 150)  # Very low spend
                sentiment = round(random.uniform(-0.3, 0.2), 2)  # Low sentiment (price complaints)
                support_tickets = random.randint(3, 8)  # Complains about costs
                feature_usage = random.uniform(0.2, 0.4)  # Low usage (basic plan)
                account_age_months = random.randint(1, 12)  # Shops around frequently
                
            elif persona_name == "Tech-Savvy Tom":
                # Make Tech Tom VERY obviously technical
                company_size = random.randint(50, 200)  # Tech company size
                monthly_spend = random.randint(500, 1200)  # Pays for features
                sentiment = round(random.uniform(0.2, 0.7), 2)  # Mixed (wants more features)
                support_tickets = random.randint(8, 20)  # Lots of technical questions
                feature_usage = random.uniform(0.85, 1.0)  # VERY high usage
                account_age_months = random.randint(3, 24)
                
            elif persona_name == "Simple Susan":
                # Make Simple Susan VERY obviously simple
                company_size = random.randint(2, 12)  # Very small
                monthly_spend = random.randint(80, 200)  # Low-medium spend
                sentiment = round(random.uniform(0.6, 0.9), 2)  # Happy (simple works)
                support_tickets = random.randint(0, 2)  # Almost no tickets
                feature_usage = random.uniform(0.1, 0.3)  # VERY low usage
                account_age_months = random.randint(12, 36)  # Stable, long-term
                
            elif persona_name == "Demanding Dan":
                # Make Demanding Dan VERY obviously demanding
                company_size = random.randint(100, 500)  # Large enough to be demanding
                monthly_spend = random.randint(800, 2000)  # Pays a lot, expects a lot
                sentiment = round(random.uniform(-0.5, 0.1), 2)  # Very negative sentiment
                support_tickets = random.randint(15, 25)  # LOTS of complaints
                feature_usage = random.uniform(0.7, 0.9)  # High usage but still complains
                account_age_months = random.randint(6, 24)
                
            elif persona_name == "Loyal Linda":
                # Make Loyal Linda VERY obviously loyal
                company_size = random.randint(20, 80)  # Stable mid-size
                monthly_spend = random.randint(200, 500)  # Steady spend
                sentiment = round(random.uniform(0.7, 0.95), 2)  # VERY positive
                support_tickets = random.randint(0, 1)  # Almost never contacts support
                feature_usage = random.uniform(0.5, 0.7)  # Moderate, consistent usage
                account_age_months = random.randint(18, 36)  # Long-term customer
                
            elif persona_name == "Trial-and-Error Tina":
                # Make Trial Tina VERY obviously a trial customer
                company_size = random.randint(8, 50)  # Startup size
                monthly_spend = random.randint(100, 300)  # Trying different price points
                sentiment = round(random.uniform(0.0, 0.4), 2)  # Neutral/slightly negative
                support_tickets = random.randint(5, 12)  # Lots of questions
                feature_usage = random.uniform(0.4, 0.8)  # Exploring features
                account_age_months = random.randint(1, 6)  # VERY new customer
                
            else:  # Scaling Sam
                # Make Scaling Sam VERY obviously scaling
                company_size = random.randint(25, 120)  # Growing size
                monthly_spend = random.randint(300, 700)  # Increasing spend
                sentiment = round(random.uniform(0.4, 0.8), 2)  # Positive but growth concerns
                support_tickets = random.randint(3, 8)  # Questions about scaling
                feature_usage = random.uniform(0.6, 0.85)  # High usage, needs more
                account_age_months = random.randint(6, 18)  # Established but growing
            
            # Rest of customer generation logic remains the same...
            # (company names, contact details, etc.)
            
            # Calculate realistic sentiment and churn risk
            sentiment = calculate_realistic_sentiment(
                persona_data["base_sentiment"], 
                account_age_months, 
                support_tickets, 
                feature_usage, 
                persona_data
            )
            sentiment = round(sentiment, 2)
            
            churn_risk = determine_churn_risk(sentiment, account_age_months, persona_data)
            
            # Generate company details
            company_name = f"{random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Omega', 'Prime', 'Core', 'Peak', 'Elite', 'Pro'])} {random.choice(['Solutions', 'Systems', 'Corp', 'Inc', 'Group', 'Services', 'Partners', 'Enterprises'])}"
            
            signup_date = datetime.now() - timedelta(days=account_age_months * 30)
            
            first_names = ['John', 'Sarah', 'Mike', 'Emma', 'David', 'Lisa', 'Chris', 'Amy', 'Robert', 'Jessica']
            last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
            contact_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            job_titles = ['Operations Manager', 'Customer Service Director', 'IT Manager', 'CEO', 'VP Operations', 'Contact Center Manager', 'Head of Customer Success', 'Business Owner']
            job_title = random.choice(job_titles)
            
            # Industry selection based on persona
            if persona_name == "Tech-Savvy Tom":
                industry = random.choice(["Software", "Fintech", "SaaS", "Tech Startup"])
            elif persona_name == "Enterprise Emma":
                industry = random.choice(["Enterprise", "Fortune 500", "Banking", "Government"])
            elif persona_name == "Budget Betty":
                industry = random.choice(["Small Business", "Local Retail", "Non-profit"])
            elif persona_name == "Simple Susan":
                industry = random.choice(["Healthcare", "Education", "Family Business"])
            else:
                industry = random.choice(persona_data["industries"])
            
            locations = ['Sydney, Australia', 'Melbourne, Australia', 'New York, USA', 'London, UK', 'Toronto, Canada']
            location = random.choice(locations)
            
            # Last login based on sentiment
            if sentiment > 0.6:
                last_login_days = random.randint(0, 3)
            elif sentiment > 0.2:
                last_login_days = random.randint(1, 7)
            else:
                last_login_days = random.randint(7, 21)
            
            customer = {
                'customer_id': f"CUST_{customer_id:03d}",
                'company_name': company_name,
                'contact_name': contact_name,
                'job_title': job_title,
                'industry': industry,
                'location': location,
                'company_size': company_size,
                'monthly_spend_usd': monthly_spend,
                'account_age_months': account_age_months,
                'signup_date': signup_date.strftime('%Y-%m-%d'),
                'sentiment_score': sentiment,
                'support_tickets_last_30_days': support_tickets,
                'feature_usage_percentage': int(feature_usage * 100),
                'last_login_days_ago': last_login_days,
                'churn_risk': churn_risk,
                'persona': persona_name
            }
            
            customers.append(customer)
            customer_id += 1

    return customers

if __name__ == "__main__":
    print("🚀 GENERATING SYNTHETIC CUSTOMER DATA...")
    
    # Generate customer data
    customers = generate_customer_data()
    
    # Create DataFrame
    df = pd.DataFrame(customers)
    
    # Save to CSV
    filename = 'synthetic_customers.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Generated {len(df)} customers")
    print(f"📁 Saved to: {filename}")
    
    # Show summary
    print(f"\n📊 PERSONA DISTRIBUTION:")
    print(df['persona'].value_counts().to_string())
    
    print(f"\n📈 SAMPLE DATA:")
    print(df[['customer_id', 'company_name', 'persona', 'company_size', 'monthly_spend_usd', 'sentiment_score']].head(10).to_string(index=False))
    
    print(f"\n🎯 Next step: Run 'offline_persona_classifier.py' to test LLM classification!")