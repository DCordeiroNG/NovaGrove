import pygame
import random
import math
import threading
import requests
import json
import sys

# Try to import tkinter
try:
    import tkinter as tk
    from tkinter import scrolledtext, ttk
    TKINTER_AVAILABLE = True
except ImportError:
    print("Warning: tkinter not available. Chat functionality will be disabled.")
    print("To enable chat, install tkinter:")
    print("  - Ubuntu/Debian: sudo apt-get install python3-tk")
    print("  - macOS: tkinter should be included with Python")
    print("  - Windows: tkinter should be included with Python")
    TKINTER_AVAILABLE = False

# Initialize Pygame
pygame.init()

# Initialize Tkinter (needed for chat windows)
root = None
if TKINTER_AVAILABLE:
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
    except Exception as e:
        print(f"Warning: Could not initialize tkinter: {e}")
        TKINTER_AVAILABLE = False

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
SPRITE_SIZE = 64  # Even bigger size
TILE_SIZE = 32

# Colors
GRASS_GREEN = (88, 205, 96)
GRASS_DARK = (72, 185, 76)
SKIN_TONE = (255, 220, 177)
BLACK = (0, 0, 0)
BROWN = (101, 67, 33)
DARK_BROWN = (61, 37, 13)
WHITE = (255, 255, 255)

# Hugging Face API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
# Note: In production, you should use environment variables for API keys
# For now, users can get a free API token from https://huggingface.co/settings/tokens
HF_API_TOKEN = "YOUR_HF_TOKEN_HERE"  # Replace with your token or set as environment variable

if TKINTER_AVAILABLE:
    class ChatWindow:
    """A separate chat window for conversing with AI personas"""
    
    def __init__(self, character):
        self.character = character
        self.conversation_history = []
        
        # Create window
        self.window = tk.Toplevel(root)  # Use the root window we created
        self.window.title(f"Chat with {character.name} - {character.persona_type}")
        self.window.geometry("500x600")
        
        # Make window appear on top
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        
        # Header with persona info
        header_frame = tk.Frame(self.window, bg="#4a90e2", pady=10)
        header_frame.pack(fill=tk.X)
        
        persona_info = f"{character.name} ({character.age_range})\n{character.occupation}\n{character.personality}"
        header_label = tk.Label(header_frame, text=persona_info, bg="#4a90e2", fg="white", font=("Arial", 12, "bold"))
        header_label.pack()
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(self.window, wrap=tk.WORD, height=20, width=60)
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Input frame
        input_frame = tk.Frame(self.window)
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.input_field = tk.Entry(input_frame, font=("Arial", 12))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_field.bind("<Return>", lambda e: self.send_message())
        
        send_button = tk.Button(input_frame, text="Send", command=self.send_message, bg="#4a90e2", fg="white")
        send_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Add initial greeting
        self.add_message(f"Hi! I'm {character.name}. {character.greeting}", "assistant")
        
        # Add note about API token if not set
        if HF_API_TOKEN == "YOUR_HF_TOKEN_HERE":
            self.add_message("(Note: Set your Hugging Face API token to enable AI responses)", "system")
        
        # Focus on input
        self.input_field.focus()
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        print(f"Chat window created for {character.name}")
    
    def on_close(self):
        """Handle window close event"""
        self.character.in_conversation = False
        self.window.destroy()
    
    def add_message(self, message, sender):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        if sender == "user":
            self.chat_display.insert(tk.END, f"\nYou: {message}\n", "user")
            self.chat_display.tag_config("user", foreground="#0066cc")
        elif sender == "system":
            self.chat_display.insert(tk.END, f"\n[{message}]\n", "system")
            self.chat_display.tag_config("system", foreground="#666666", font=("Arial", 10, "italic"))
        else:
            self.chat_display.insert(tk.END, f"\n{self.character.name}: {message}\n", "assistant")
            self.chat_display.tag_config("assistant", foreground="#009900")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def send_message(self):
        """Send user message and get AI response"""
        user_message = self.input_field.get().strip()
        if not user_message:
            return
        
        # Clear input
        self.input_field.delete(0, tk.END)
        
        # Add user message
        self.add_message(user_message, "user")
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Show typing indicator
        self.add_message("typing...", "assistant")
        
        # Get AI response in background thread
        thread = threading.Thread(target=self.get_ai_response, args=(user_message,))
        thread.daemon = True
        thread.start()
    
    def get_ai_response(self, user_message):
        """Get response from Hugging Face API"""
        try:
            # Check if API token is set
            if HF_API_TOKEN == "YOUR_HF_TOKEN_HERE":
                ai_response = "Please set your Hugging Face API token in the code to enable AI responses. Get a free token at huggingface.co/settings/tokens"
            else:
                # Build context with persona info
                context = f"You are {self.character.name}, a {self.character.age_range} {self.character.occupation}. "
                context += f"Your personality is {self.character.personality}. "
                context += f"Interests: {', '.join(self.character.interests)}. "
                context += "Respond in character to: " + user_message
                
                headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
                
                # For DialoGPT, we need to format the conversation
                # Build conversation string
                past_messages = "\n".join([f"{msg['content']}" for msg in self.conversation_history[-3:]])
                
                payload = {
                    "inputs": f"{past_messages}\n{user_message}",
                    "parameters": {
                        "max_length": 200,
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "do_sample": True
                    }
                }
                
                response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract the generated text
                    if isinstance(result, list) and len(result) > 0:
                        ai_response = result[0].get('generated_text', '').split('\n')[-1].strip()
                        if not ai_response or ai_response == user_message:
                            ai_response = f"As a {self.character.occupation}, I find that interesting! Tell me more about your thoughts on this."
                    else:
                        ai_response = "I'm not sure how to respond to that. Can you tell me more?"
                elif response.status_code == 503:
                    ai_response = "The AI model is loading. Please try again in a few seconds..."
                else:
                    ai_response = f"API Error (Status {response.status_code}). Please check your token and try again."
                    print(f"API Error: {response.text}")
                    
        except requests.exceptions.Timeout:
            ai_response = "The response is taking too long. Please try again."
        except Exception as e:
            ai_response = f"An error occurred. Please make sure your API token is valid."
            print(f"Error in get_ai_response: {e}")
            import traceback
            traceback.print_exc()
        
        # Remove typing indicator and add actual response
        self.window.after(0, self.update_response, ai_response)
    
    def update_response(self, response):
        """Update the chat with AI response"""
        # Remove typing indicator
        self.chat_display.config(state=tk.NORMAL)
        content = self.chat_display.get("1.0", tk.END)
        lines = content.strip().split('\n')
        if lines[-1].endswith("typing..."):
            # Remove last two lines (the "Name: typing..." and empty line)
            new_content = '\n'.join(lines[:-2])
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.insert("1.0", new_content)
        self.chat_display.config(state=tk.DISABLED)
        
        # Add actual response
        self.add_message(response, "assistant")
        self.conversation_history.append({"role": "assistant", "content": response})

class PersonaGenerator:
    """Generates diverse customer personas"""
    
    @staticmethod
    def generate_persona():
        """Generate a random customer persona with attributes"""
        persona_types = [
            {"type": "Early Adopter", "age": "25-35", "personality": "enthusiastic and tech-savvy"},
            {"type": "Enterprise Buyer", "age": "35-50", "personality": "analytical and risk-averse"},
            {"type": "Small Business Owner", "age": "30-55", "personality": "practical and budget-conscious"},
            {"type": "Creative Professional", "age": "22-40", "personality": "innovative and aesthetic-focused"},
            {"type": "Student", "age": "18-25", "personality": "curious and price-sensitive"},
            {"type": "Parent", "age": "30-45", "personality": "cautious and value-oriented"},
            {"type": "Retiree", "age": "60+", "personality": "patient and traditional"},
            {"type": "Freelancer", "age": "25-40", "personality": "independent and efficiency-focused"},
            {"type": "Startup Founder", "age": "28-45", "personality": "ambitious and growth-oriented"},
            {"type": "IT Manager", "age": "30-50", "personality": "systematic and security-conscious"}
        ]
        
        occupations = [
            "Software Developer", "Marketing Manager", "Teacher", "Designer",
            "Sales Executive", "Consultant", "Engineer", "Writer",
            "Product Manager", "Data Analyst", "Entrepreneur", "Researcher"
        ]
        
        interests = [
            "Technology", "Gaming", "Reading", "Fitness", "Travel", "Cooking",
            "Photography", "Music", "Art", "Sports", "Movies", "Gardening",
            "Fashion", "Investing", "Podcasts", "Social Media", "DIY Projects"
        ]
        
        greetings = [
            "Nice to meet you! What would you like to know?",
            "Hey there! I'm happy to chat about anything.",
            "Hello! Feel free to ask me questions.",
            "Hi! I love meeting new people. What's on your mind?",
            "Greetings! I'm here to share my perspective with you."
        ]
        
        persona = random.choice(persona_types)
        
        return {
            "persona_type": persona["type"],
            "age_range": persona["age"],
            "personality": persona["personality"],
            "occupation": random.choice(occupations),
            "interests": random.sample(interests, random.randint(3, 5)),
            "greeting": random.choice(greetings)
        }

class SpriteGenerator:
    """Generates chibi-style character sprites programmatically"""
    
    @staticmethod
    def generate_character_sprites(shirt_color, hair_color, skin_tone, has_glasses=False):
        """Generate 4-directional walk cycle sprites for a character"""
        sprites = {
            'down': [],
            'up': [],
            'left': [],
            'right': []
        }
        
        # Generate sprites for each direction and animation frame
        for direction in sprites.keys():
            for frame in range(2):  # 2 frames for simple walk cycle
                sprite = pygame.Surface((SPRITE_SIZE, SPRITE_SIZE), pygame.SRCALPHA)
                sprite.fill((0, 0, 0, 0))  # Transparent background
                
                # Draw character based on direction
                if direction == 'down':
                    SpriteGenerator._draw_character_down(sprite, shirt_color, hair_color, skin_tone, frame, has_glasses)
                elif direction == 'up':
                    SpriteGenerator._draw_character_up(sprite, shirt_color, hair_color, skin_tone, frame, has_glasses)
                elif direction == 'left':
                    SpriteGenerator._draw_character_left(sprite, shirt_color, hair_color, skin_tone, frame, has_glasses)
                elif direction == 'right':
                    SpriteGenerator._draw_character_right(sprite, shirt_color, hair_color, skin_tone, frame, has_glasses)
                
                sprites[direction].append(sprite)
        
        return sprites
    
    @staticmethod
    def _draw_character_down(surface, shirt_color, hair_color, skin_tone, frame, has_glasses):
        """Draw character facing down"""
        cx, cy = SPRITE_SIZE // 2, SPRITE_SIZE // 2
        
        # Shadow
        pygame.draw.ellipse(surface, (0, 0, 0, 50), (cx - 12, cy + 16, 24, 8))
        
        # Legs
        leg_offset = 3 if frame == 1 else -3
        pygame.draw.rect(surface, DARK_BROWN, (cx - 6 + leg_offset, cy + 6, 4, 12))
        pygame.draw.rect(surface, DARK_BROWN, (cx + 2 - leg_offset, cy + 6, 4, 12))
        
        # Body
        pygame.draw.rect(surface, shirt_color, (cx - 8, cy - 4, 16, 14))
        
        # Arms
        pygame.draw.rect(surface, skin_tone, (cx - 11, cy - 2, 4, 8))
        pygame.draw.rect(surface, skin_tone, (cx + 7, cy - 2, 4, 8))
        
        # Head
        pygame.draw.circle(surface, skin_tone, (cx, cy - 10), 10)
        
        # Hair
        pygame.draw.arc(surface, hair_color, (cx - 10, cy - 20, 20, 20), 0, math.pi, 10)
        
        # Face features
        pygame.draw.circle(surface, BLACK, (cx - 4, cy - 8), 2)
        pygame.draw.circle(surface, BLACK, (cx + 4, cy - 8), 2)
        
        # Glasses if applicable
        if has_glasses:
            pygame.draw.circle(surface, BLACK, (cx - 4, cy - 8), 4, 1)
            pygame.draw.circle(surface, BLACK, (cx + 4, cy - 8), 4, 1)
            pygame.draw.line(surface, BLACK, (cx - 1, cy - 8), (cx + 1, cy - 8), 1)
    
    @staticmethod
    def _draw_character_up(surface, shirt_color, hair_color, skin_tone, frame, has_glasses):
        """Draw character facing up"""
        cx, cy = SPRITE_SIZE // 2, SPRITE_SIZE // 2
        
        # Shadow
        pygame.draw.ellipse(surface, (0, 0, 0, 50), (cx - 12, cy + 16, 24, 8))
        
        # Legs
        leg_offset = 3 if frame == 1 else -3
        pygame.draw.rect(surface, DARK_BROWN, (cx - 6 + leg_offset, cy + 6, 4, 12))
        pygame.draw.rect(surface, DARK_BROWN, (cx + 2 - leg_offset, cy + 6, 4, 12))
        
        # Body
        pygame.draw.rect(surface, shirt_color, (cx - 8, cy - 4, 16, 14))
        
        # Arms
        pygame.draw.rect(surface, skin_tone, (cx - 11, cy - 2, 4, 8))
        pygame.draw.rect(surface, skin_tone, (cx + 7, cy - 2, 4, 8))
        
        # Head
        pygame.draw.circle(surface, hair_color, (cx, cy - 10), 10)
    
    @staticmethod
    def _draw_character_left(surface, shirt_color, hair_color, skin_tone, frame, has_glasses):
        """Draw character facing left"""
        cx, cy = SPRITE_SIZE // 2, SPRITE_SIZE // 2
        
        # Shadow
        pygame.draw.ellipse(surface, (0, 0, 0, 50), (cx - 12, cy + 16, 24, 8))
        
        # Legs
        leg_offset = 3 if frame == 1 else -3
        pygame.draw.rect(surface, DARK_BROWN, (cx - 3, cy + 6 + leg_offset, 4, 12))
        pygame.draw.rect(surface, DARK_BROWN, (cx - 3, cy + 6 - leg_offset, 4, 12))
        
        # Body
        pygame.draw.rect(surface, shirt_color, (cx - 6, cy - 4, 10, 14))
        
        # Arm (visible)
        pygame.draw.rect(surface, skin_tone, (cx - 3, cy - 2, 4, 8))
        
        # Head
        pygame.draw.circle(surface, skin_tone, (cx - 3, cy - 10), 10)
        
        # Hair
        pygame.draw.arc(surface, hair_color, (cx - 13, cy - 20, 20, 20), 0, math.pi, 10)
        
        # Face feature
        pygame.draw.circle(surface, BLACK, (cx - 7, cy - 8), 2)
        
        # Glasses if applicable
        if has_glasses:
            pygame.draw.circle(surface, BLACK, (cx - 7, cy - 8), 4, 1)
    
    @staticmethod
    def _draw_character_right(surface, shirt_color, hair_color, skin_tone, frame, has_glasses):
        """Draw character facing right"""
        cx, cy = SPRITE_SIZE // 2, SPRITE_SIZE // 2
        
        # Shadow
        pygame.draw.ellipse(surface, (0, 0, 0, 50), (cx - 12, cy + 16, 24, 8))
        
        # Legs
        leg_offset = 3 if frame == 1 else -3
        pygame.draw.rect(surface, DARK_BROWN, (cx - 1, cy + 6 + leg_offset, 4, 12))
        pygame.draw.rect(surface, DARK_BROWN, (cx - 1, cy + 6 - leg_offset, 4, 12))
        
        # Body
        pygame.draw.rect(surface, shirt_color, (cx - 4, cy - 4, 10, 14))
        
        # Arm (visible)
        pygame.draw.rect(surface, skin_tone, (cx - 1, cy - 2, 4, 8))
        
        # Head
        pygame.draw.circle(surface, skin_tone, (cx + 3, cy - 10), 10)
        
        # Hair
        pygame.draw.arc(surface, hair_color, (cx - 7, cy - 20, 20, 20), 0, math.pi, 10)
        
        # Face feature
        pygame.draw.circle(surface, BLACK, (cx + 7, cy - 8), 2)
        
        # Glasses if applicable
        if has_glasses:
            pygame.draw.circle(surface, BLACK, (cx + 7, cy - 8), 4, 1)

class GrassPattern:
    """Generates a Pokémon-style grass tile pattern"""
    
    @staticmethod
    def generate_grass_tile():
        """Generate a single grass tile"""
        tile = pygame.Surface((TILE_SIZE, TILE_SIZE))
        tile.fill(GRASS_GREEN)
        
        # Add grass texture
        for _ in range(8):
            x = random.randint(0, TILE_SIZE - 4)
            y = random.randint(0, TILE_SIZE - 4)
            pygame.draw.line(tile, GRASS_DARK, (x, y), (x + 2, y - 4), 1)
            pygame.draw.line(tile, GRASS_DARK, (x + 2, y), (x + 4, y - 4), 1)
        
        # Add some dots for variation
        for _ in range(4):
            x = random.randint(2, TILE_SIZE - 2)
            y = random.randint(2, TILE_SIZE - 2)
            pygame.draw.circle(tile, GRASS_DARK, (x, y), 1)
        
        return tile

class Character:
    """Represents a walking character with AI behavior"""
    
    def __init__(self, x, y, sprites, persona_data):
        self.x = x
        self.y = y
        self.sprites = sprites
        self.direction = random.choice(['down', 'up', 'left', 'right'])
        self.speed = 0.8  # Slower speed
        self.animation_frame = 0
        self.animation_timer = 0
        self.movement_timer = random.randint(120, 300)  # 2-5 seconds at 60 FPS
        self.pause_timer = 0
        self.is_paused = False
        self.rect = pygame.Rect(x - SPRITE_SIZE // 2, y - SPRITE_SIZE // 2, SPRITE_SIZE, SPRITE_SIZE)
        self.in_conversation = False
        
        # Persona attributes
        self.name = self._generate_name()
        self.persona_type = persona_data["persona_type"]
        self.age_range = persona_data["age_range"]
        self.personality = persona_data["personality"]
        self.occupation = persona_data["occupation"]
        self.interests = persona_data["interests"]
        self.greeting = persona_data["greeting"]
        
        # Font for label
        self.label_font = pygame.font.Font(None, 16)
    
    def _generate_name(self):
        """Generate a random Pokémon-style name"""
        first_names = [
            "Ash", "Misty", "Brock", "Dawn", "May", "Max", "Iris", "Cilan",
            "Serena", "Clemont", "Bonnie", "Gary", "Paul", "Barry", "Kenny",
            "Zoey", "Nando", "Conway", "Ursula", "Jessie", "James", "Butch",
            "Cassidy", "Todd", "Ritchie", "Casey", "Sakura", "Lily", "Daisy",
            "Violet", "Erika", "Sabrina", "Koga", "Bruno", "Karen", "Will",
            "Jasmine", "Whitney", "Morty", "Chuck", "Pryce", "Clair", "Falkner"
        ]
        return random.choice(first_names)
    
    def update(self):
        """Update character movement and animation"""
        # Don't move if in conversation
        if self.in_conversation:
            return
            
        # Update animation
        self.animation_timer += 1
        if self.animation_timer >= 10:  # Change frame every 10 ticks
            self.animation_timer = 0
            self.animation_frame = (self.animation_frame + 1) % 2
        
        # Handle pause state
        if self.is_paused:
            self.pause_timer -= 1
            if self.pause_timer <= 0:
                self.is_paused = False
                self.movement_timer = random.randint(120, 300)
                self.direction = random.choice(['down', 'up', 'left', 'right'])
            return
        
        # Update movement
        self.movement_timer -= 1
        if self.movement_timer <= 0:
            self.is_paused = True
            self.pause_timer = random.randint(30, 60)  # 0.5-1 second pause
            return
        
        # Move based on direction
        dx, dy = 0, 0
        if self.direction == 'down':
            dy = self.speed
        elif self.direction == 'up':
            dy = -self.speed
        elif self.direction == 'left':
            dx = -self.speed
        elif self.direction == 'right':
            dx = self.speed
        
        # Check boundaries and move
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Boundary collision
        if new_x < 32 or new_x > SCREEN_WIDTH - 32:
            self.direction = 'left' if new_x > SCREEN_WIDTH // 2 else 'right'
            self.movement_timer = random.randint(120, 300)
        elif new_y < 32 or new_y > SCREEN_HEIGHT - 32:
            self.direction = 'up' if new_y > SCREEN_HEIGHT // 2 else 'down'
            self.movement_timer = random.randint(120, 300)
        else:
            self.x = new_x
            self.y = new_y
            # Update rect position
            self.rect.x = int(self.x - SPRITE_SIZE // 2)
            self.rect.y = int(self.y - SPRITE_SIZE // 2)
    
    def draw(self, screen):
        """Draw the character on the screen"""
        # Draw character sprite
        sprite = self.sprites[self.direction][self.animation_frame]
        screen.blit(sprite, (int(self.x - SPRITE_SIZE // 2), int(self.y - SPRITE_SIZE // 2)))
        
        # Draw floating label with persona type
        label_text = f"{self.name} - {self.persona_type}"
        label = self.label_font.render(label_text, True, WHITE)
        label_bg = pygame.Surface((label.get_width() + 6, label.get_height() + 4))
        label_bg.fill((0, 0, 0))
        label_bg.set_alpha(180)
        
        label_x = int(self.x - label.get_width() // 2)
        label_y = int(self.y - SPRITE_SIZE // 2 - 25)
        
        screen.blit(label_bg, (label_x - 3, label_y - 2))
        screen.blit(label, (label_x, label_y))

class SpeechBubble:
    """Represents a speech bubble that appears next to a character"""
    
    def __init__(self, character):
        self.character = character
        self.active = True
        self.showing_options = True
        self.timer = 0
        
        # Font
        self.font = pygame.font.Font(None, 20)
        
        # Position bubble to the right of character by default
        self.bubble_width = 180
        self.bubble_height = 80
        
        # Determine best position for bubble
        if self.character.x > SCREEN_WIDTH - 200:
            # Place bubble to the left if character is on right side
            self.x = self.character.x - self.bubble_width - 40
        else:
            # Place bubble to the right
            self.x = self.character.x + 40
        
        self.y = self.character.y - 40
        
        # Buttons for options
        self.name_button_rect = pygame.Rect(self.x + 30, self.y + 15, 120, 25)
        self.chat_button_rect = pygame.Rect(self.x + 30, self.y + 45, 120, 25)
    
    def handle_click(self, pos):
        """Handle mouse clicks on the speech bubble"""
        if self.showing_options:
            if self.name_button_rect.collidepoint(pos):
                # Show name briefly then close
                self.showing_options = False
                self.timer = 120  # Show for 2 seconds
                return "name"
            elif self.chat_button_rect.collidepoint(pos):
                # Open chat window
                if not TKINTER_AVAILABLE:
                    print("Chat functionality is not available - tkinter not installed")
                    return None
                try:
                    print(f"Opening chat window for {self.character.name}")
                    ChatWindow(self.character)
                    self.active = False
                    self.character.in_conversation = False
                    return "chat"
                except Exception as e:
                    print(f"Error opening chat window: {e}")
                    import traceback
                    traceback.print_exc()
        return None
    
    def update(self):
        """Update the speech bubble"""
        if not self.showing_options and self.timer > 0:
            self.timer -= 1
            if self.timer <= 0:
                self.active = False
                self.character.in_conversation = False
    
    def draw(self, screen):
        """Draw the speech bubble"""
        # Draw bubble body
        pygame.draw.ellipse(screen, (255, 255, 255), (self.x, self.y, self.bubble_width, self.bubble_height))
        pygame.draw.ellipse(screen, (0, 0, 0), (self.x, self.y, self.bubble_width, self.bubble_height), 2)
        
        # Draw bubble tail
        tail_points = []
        if self.x < self.character.x:  # Bubble is on the left
            tail_points = [
                (self.x + self.bubble_width - 20, self.y + self.bubble_height - 10),
                (self.x + self.bubble_width - 10, self.y + self.bubble_height - 10),
                (self.character.x - 10, self.character.y - 10)
            ]
        else:  # Bubble is on the right
            tail_points = [
                (self.x + 20, self.y + self.bubble_height - 10),
                (self.x + 10, self.y + self.bubble_height - 10),
                (self.character.x + 10, self.character.y - 10)
            ]
        pygame.draw.polygon(screen, (255, 255, 255), tail_points)
        pygame.draw.lines(screen, (0, 0, 0), False, tail_points, 2)
        
        if self.showing_options:
            # Draw buttons
            pygame.draw.rect(screen, (200, 200, 255), self.name_button_rect)
            pygame.draw.rect(screen, (0, 0, 0), self.name_button_rect, 2)
            name_text = self.font.render("Ask Name", True, (0, 0, 0))
            name_text_rect = name_text.get_rect(center=self.name_button_rect.center)
            screen.blit(name_text, name_text_rect)
            
            pygame.draw.rect(screen, (100, 255, 100), self.chat_button_rect)
            pygame.draw.rect(screen, (0, 0, 0), self.chat_button_rect, 2)
            chat_text = self.font.render("Chat", True, (0, 0, 0))
            chat_text_rect = chat_text.get_rect(center=self.chat_button_rect.center)
            screen.blit(chat_text, chat_text_rect)
        else:
            # Draw name
            name_text = f"I'm {self.character.name}!"
            text = self.font.render(name_text, True, (0, 0, 0))
            text_rect = text.get_rect(center=(self.x + self.bubble_width // 2, self.y + self.bubble_height // 2))
            screen.blit(text, text_rect)

class PokemonWalkingSimulation:
    """Main simulation class"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Customer Persona World")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Speech bubble
        self.speech_bubble = None
        
        # Generate grass background
        self.grass_tile = GrassPattern.generate_grass_tile()
        self.background = self._create_tiled_background()
        
        # Create characters with random colors
        self.characters = []
        self._create_characters()
    
    def _create_tiled_background(self):
        """Create a tiled grass background"""
        background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        for x in range(0, SCREEN_WIDTH, TILE_SIZE):
            for y in range(0, SCREEN_HEIGHT, TILE_SIZE):
                background.blit(self.grass_tile, (x, y))
        return background
    
    def _create_characters(self):
        """Create 10-15 characters with random appearances and personas"""
        num_characters = random.randint(10, 15)
        
        # Define color palettes
        shirt_colors = [
            (255, 92, 92),   # Red
            (92, 172, 255),  # Blue
            (255, 206, 84),  # Yellow
            (147, 255, 147), # Green
            (255, 147, 255), # Pink
            (255, 184, 108), # Orange
            (184, 147, 255), # Purple
            (147, 255, 255), # Cyan
        ]
        
        hair_colors = [
            (61, 37, 13),    # Dark brown
            (255, 206, 84),  # Blonde
            (0, 0, 0),       # Black
            (184, 61, 0),    # Red
            (101, 67, 33),   # Brown
            (128, 128, 128), # Gray
        ]
        
        skin_tones = [
            (255, 220, 177), # Light
            (234, 192, 134), # Medium light
            (198, 134, 66),  # Medium
            (141, 85, 36),   # Medium dark
            (88, 57, 39),    # Dark
        ]
        
        for _ in range(num_characters):
            x = random.randint(50, SCREEN_WIDTH - 50)
            y = random.randint(50, SCREEN_HEIGHT - 50)
            shirt_color = random.choice(shirt_colors)
            hair_color = random.choice(hair_colors)
            skin_tone = random.choice(skin_tones)
            has_glasses = random.choice([True, False, False])  # 1/3 chance of glasses
            
            persona_data = PersonaGenerator.generate_persona()
            sprites = SpriteGenerator.generate_character_sprites(shirt_color, hair_color, skin_tone, has_glasses)
            character = Character(x, y, sprites, persona_data)
            self.characters.append(character)
    
    def run(self):
        """Main game loop"""
        # Show initial instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "Welcome to AI Customer Persona World!",
            "Click on any character to interact with them.",
            "Choose 'Chat' to have a conversation with their AI persona.",
            "",
            "Note: Set your Hugging Face API token in the code to enable AI chat.",
            "(Get a free token at huggingface.co/settings/tokens)"
        ]
        
        showing_instructions = True
        instruction_timer = 300  # Show for 5 seconds
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        showing_instructions = False
                        pos = pygame.mouse.get_pos()
                        
                        # Check if clicking on speech bubble
                        if self.speech_bubble:
                            action = self.speech_bubble.handle_click(pos)
                            if action == "chat":
                                self.speech_bubble = None
                        else:
                            # Check if clicking on a character
                            for character in reversed(self.characters):  # Check from top to bottom
                                if character.rect.collidepoint(pos):
                                    self.speech_bubble = SpeechBubble(character)
                                    character.in_conversation = True
                                    break
            
            # Update speech bubble
            if self.speech_bubble:
                self.speech_bubble.update()
                if not self.speech_bubble.active:
                    self.speech_bubble = None
            
            # Update all characters (they check their own conversation status)
            for character in self.characters:
                character.update()
            
            # Sort characters by Y position for depth effect
            self.characters.sort(key=lambda c: c.y)
            
            # Draw everything
            self.screen.blit(self.background, (0, 0))
            for character in self.characters:
                character.draw(self.screen)
            
            # Draw speech bubble if active
            if self.speech_bubble:
                self.speech_bubble.draw(self.screen)
            
            # Draw instructions if showing
            if showing_instructions:
                instruction_timer -= 1
                if instruction_timer <= 0:
                    showing_instructions = False
                
                # Draw semi-transparent overlay
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                overlay.set_alpha(200)
                overlay.fill((0, 0, 0))
                self.screen.blit(overlay, (0, 0))
                
                # Draw instructions
                y_offset = SCREEN_HEIGHT // 2 - len(instructions) * 15
                for i, line in enumerate(instructions):
                    text = font.render(line, True, WHITE)
                    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, y_offset + i * 30))
                    self.screen.blit(text, text_rect)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
            
            # Process tkinter events if available
            if TKINTER_AVAILABLE and root:
                try:
                    root.update_idletasks()
                    root.update()
                except tk.TclError:
                    pass  # Window was closed
        
        pygame.quit()
        if TKINTER_AVAILABLE and root:
            try:
                root.quit()  # Clean up tkinter
            except:
                pass

# Run the simulation
if __name__ == "__main__":
    try:
        sim = PokemonWalkingSimulation()
        sim.run()
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if TKINTER_AVAILABLE and root:
            try:
                root.quit()
            except:
                pass