from flask import Flask, request, jsonify
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
# application = app

emotion_suggestions = {
    'frustrated': [
        "Take a short break to reset your mind.",
        "Break tasks into smaller steps to regain control.",
        "Try talking through the problem with someone.",
        "Switch to a different task for a fresh perspective.",
        "Use a calming technique like deep breathing.",
        "Step away for a walk to clear your mind.",
        "Reframe the challenge as a learning opportunity.",
        "Identify what's frustrating you specifically.",
        "Set smaller, achievable goals to reduce stress.",
        "Listen to music or meditate to reset mentally."
    ],
    'stuck': [
        "Take a break and revisit the task later.",
        "Talk through your thoughts with someone.",
        "Try writing down different approaches.",
        "Change your environment for fresh inspiration.",
        "Break the task into smaller, achievable goals.",
        "Seek feedback from others to get unstuck.",
        "Reevaluate your current strategies and adjust.",
        "Experiment with new methods or tools.",
        "Step back and reconsider the overall goal.",
        "Create a visual mind map to sort your ideas."
    ],
    'anxious': [
        "Do breathing exercises to calm your nerves.",
        "List things you can control right now.",
        "Focus on one thing at a time to ease worry.",
        "Reach out to a friend to talk through your feelings.",
        "Journal your thoughts and release anxieties.",
        "Practice mindfulness or meditation for grounding.",
        "Take a walk in nature to relax your mind.",
        "Engage in light physical activity like stretching.",
        "Challenge irrational thoughts with facts.",
        "Distract yourself with a creative hobby."
    ],
    'motivated': [
        "Use your motivation to tackle tough tasks first.",
        "Write down your goals and break them into steps.",
        "Celebrate even the small wins for momentum.",
        "Share your excitement with someone for accountability.",
        "Start working on a project you've been putting off.",
        "Channel energy into helping someone else succeed.",
        "Set new, higher goals to push yourself further.",
        "Take on a new challenge to keep the momentum.",
        "Inspire others with your energy and enthusiasm.",
        "Reflect on your progress to keep the fire burning."
    ],
    'sad': [
        "Reach out to a loved one for emotional support.",
        "Write in a journal to process your sadness.",
        "Engage in activities you normally enjoy.",
        "Allow yourself to feel sad but don't dwell on it.",
        "Watch or read something uplifting.",
        "Do something kind for yourself or others.",
        "Talk to someone you trust about your feelings.",
        "Reflect on past achievements to lift your mood.",
        "Listen to music that comforts or uplifts you.",
        "Take a walk outside to reset your thoughts."
    ],
    'angry': [
        "Take a few deep breaths to calm down.",
        "Walk away from the situation for a moment.",
        "Express your feelings through writing or talking.",
        "Engage in physical exercise to release tension.",
        "Think about the root cause of your anger.",
        "Try to see things from the other person's perspective.",
        "Focus on solutions rather than the problem.",
        "Use humor to diffuse the intensity of your anger.",
        "Practice progressive muscle relaxation.",
        "Wait until you're calm before taking any action."
    ],
    'confused': [
        "Write down what you're confused about for clarity.",
        "Break the issue into smaller, understandable parts.",
        "Ask for help from someone more knowledgeable.",
        "Look for patterns or similarities with past experiences.",
        "Take a short break to reset your brain.",
        "Use diagrams or charts to visualize the problem.",
        "Rephrase the issue to gain new insights.",
        "Discuss your confusion with others for fresh perspectives.",
        "Try to learn more about the topic to fill in gaps.",
        "Consider other approaches and reframe the challenge."
    ],
    'happy': [
        "Spread your happiness by doing something kind.",
        "Reflect on the reasons you're feeling happy.",
        "Savor the moment and enjoy your good mood.",
        "Share your joy with friends or family.",
        "Set a new goal while you're feeling good.",
        "Take this positive energy to start a new project.",
        "Help someone else feel the same way.",
        "Celebrate your recent success or achievement.",
        "Use your happiness to fuel creativity.",
        "Take a moment to relax and be grateful for the moment."
    ],
    'nervous': [
        "Focus on your breathing to calm your nerves.",
        "Remind yourself of your past successes.",
        "Prepare by reviewing your plan or materials.",
        "Visualize a positive outcome to ease anxiety.",
        "Talk to someone who can offer encouragement.",
        "Channel your nervous energy into enthusiasm.",
        "Distract yourself with a small task before the event.",
        "Take deep breaths and count backward from 10.",
        "Stay grounded by focusing on the present moment.",
        "Write down your feelings and address them logically."
    ],
    'calm': [
        "Enjoy the peace and take things one step at a time.",
        "Use this state of calm to make clear decisions.",
        "Share your calm mindset with others around you.",
        "Focus on mindfulness to maintain your tranquility.",
        "Meditate or do yoga to sustain your peaceful mood.",
        "Take a quiet moment to reflect on what's going well.",
        "Organize or plan your tasks with a clear mind.",
        "Help someone else find their calm in a busy moment.",
        "Enjoy a leisurely activity like reading or music.",
        "Let this calm fuel positive actions in your day."
    ],
    'overwhelmed': [
        "Take a moment to breathe and regroup.",
        "Create a priority list to tackle one thing at a time.",
        "Delegate tasks if possible to lighten your load.",
        "Take a break to clear your mind.",
        "Set time limits on tasks to avoid burnout.",
        "Focus on what you can control and let go of the rest.",
        "Engage in a quick physical activity to relieve stress.",
        "Practice mindfulness to center your thoughts.",
        "Limit distractions and focus on one task.",
        "Reach out for support from friends or colleagues."
    ],
    'lonely': [
        "Connect with a friend or family member via call or text.",
        "Join a group or club to meet new people.",
        "Engage in a hobby to distract yourself.",
        "Volunteer in your community to meet like-minded individuals.",
        "Consider adopting a pet for companionship.",
        "Reflect on positive past connections and memories.",
        "Join an online community related to your interests.",
        "Write about your feelings in a journal.",
        "Try reaching out to someone new to connect with.",
        "Participate in social activities, even if virtually."
    ],
    'relieved': [
        "Take a moment to enjoy your sense of relief.",
        "Reflect on the efforts that led to this feeling.",
        "Share your relief with someone who supported you.",
        "Use this positive energy to tackle new tasks.",
        "Practice gratitude for the resolution of the situation.",
        "Engage in a relaxing activity to celebrate your relief.",
        "Take note of what worked well for future reference.",
        "Allow yourself some downtime to recharge.",
        "Engage in a calming activity like meditation.",
        "Consider how to maintain this feeling moving forward."
    ],
    'hopeful': [
        "Set new goals based on your optimistic outlook.",
        "Share your hope with others to inspire them.",
        "Take small steps towards your goals to build momentum.",
        "Reflect on past experiences where hope led to success.",
        "Write down your hopes and visualize them coming true.",
        "Engage in activities that nurture your positive mindset.",
        "Surround yourself with supportive and optimistic people.",
        "Consider volunteering for causes that resonate with your hopes.",
        "Read or watch inspirational stories that uplift you.",
        "Practice affirmations to reinforce your hopeful mindset."
    ],
    'curious': [
        "Explore new topics or interests that intrigue you.",
        "Ask questions to deepen your understanding.",
        "Engage in creative activities that allow for exploration.",
        "Read books or articles related to your curiosity.",
        "Join workshops or classes to learn something new.",
        "Keep a curiosity journal to document your discoveries.",
        "Reach out to knowledgeable individuals for insights.",
        "Experiment with new ideas without fear of failure.",
        "Visit museums, galleries, or events that pique your interest.",
        "Consider how to turn curiosity into a project or goal."
    ],
    'disappointed': [
        "Acknowledge your feelings and allow yourself to feel them.",
        "Reflect on what led to your disappointment.",
        "Consider what you can learn from this experience.",
        "Talk to someone who can provide support and perspective.",
        "Engage in self-care to uplift your mood.",
        "Reframe the situation to see potential opportunities.",
        "Focus on your strengths and past successes.",
        "Set new, realistic goals moving forward.",
        "Allow yourself time to heal and regroup.",
        "Channel your feelings into a creative outlet."
    ],
    'inspired': [
        "Take immediate action on your inspiration.",
        "Share your ideas with others to gather feedback.",
        "Create a vision board to visualize your inspiration.",
        "Write down your thoughts and plans for clarity.",
        "Seek out experiences or people that further inspire you.",
        "Engage in creative activities to harness your inspiration.",
        "Consider how to make your inspiration actionable.",
        "Reflect on what sparked your inspiration for future reference.",
        "Use this feeling to motivate others around you.",
        "Embrace opportunities that align with your newfound inspiration."
    ],
    'ashamed': [
        "Acknowledge your feelings of shame without judgment.",
        "Reflect on the reasons behind your feelings.",
        "Talk to someone you trust for perspective.",
        "Engage in self-compassion exercises to soften your thoughts.",
        "Identify actionable steps to improve the situation.",
        "Consider how to learn from this experience.",
        "Limit negative self-talk and focus on growth.",
        "Practice forgiveness for yourself or others involved.",
        "Channel feelings into making amends if possible.",
        "Use this experience as a catalyst for change."
    ],
    'empowered': [
        "Celebrate your achievements and strengths.",
        "Set new goals that align with your empowered mindset.",
        "Share your empowerment with others to inspire them.",
        "Engage in activities that reinforce your confidence.",
        "Take on challenges that push your boundaries.",
        "Reflect on what empowered you and replicate it.",
        "Mentor or support others on their journeys.",
        "Use this feeling to advocate for yourself and others.",
        "Engage in public speaking or assertiveness training.",
        "Channel this energy into community involvement."
    ],
    'fearful': [
        "Identify specific fears and assess their validity.",
        "Practice grounding techniques to stay present.",
        "Talk to someone who can provide support.",
        "Consider small steps you can take to face your fears.",
        "Engage in calming practices like meditation.",
        "Write down your fears and consider solutions.",
        "Limit exposure to triggers until you feel ready.",
        "Reframe fearful thoughts into empowering ones.",
        "Visualize overcoming your fears successfully.",
        "Seek professional guidance if fears are overwhelming."
    ]
}

# To prevent repeating suggestions until all are used
used_suggestions = {emotion: [] for emotion in emotion_suggestions}

# Function to get a random non-repeated suggestion
def get_suggestion(emotion):
    if len(used_suggestions[emotion]) == len(emotion_suggestions[emotion]):
        used_suggestions[emotion] = []  # Reset once all suggestions have been used

    # Find unused suggestions and pick one
    unused_suggestions = list(set(emotion_suggestions[emotion]) - set(used_suggestions[emotion]))
    suggestion = random.choice(unused_suggestions)
    used_suggestions[emotion].append(suggestion)
    return suggestion

# Custom list of emotional adjectives mapped to their base emotions
emotion_adjectives = {
    'happy': 'happy', 'joyful': 'happy', 'content': 'happy', 'pleased': 'happy', 'delighted': 'happy', 
    'ecstatic': 'happy', 'cheerful': 'happy', 'thrilled': 'happy', 'overjoyed': 'happy', 'blissful': 'happy',
    
    'sad': 'sad', 'down': 'sad', 'unhappy': 'sad', 'depressed': 'sad', 'melancholy': 'sad', 
    'mournful': 'sad', 'heartbroken': 'sad', 'despondent': 'sad', 'gloomy': 'sad', 'hopeless': 'sad',
    
    'frustrated': 'frustrated', 'irritated': 'frustrated', 'annoyed': 'frustrated', 'exasperated': 'frustrated', 
    'aggravated': 'frustrated', 'discontent': 'frustrated', 'fed up': 'frustrated', 'bothered': 'frustrated', 
    'impatient': 'frustrated', 'vexed': 'frustrated',
    
    'anxious': 'anxious', 'worried': 'anxious', 'nervous': 'anxious', 'stressed': 'anxious', 'apprehensive': 'anxious', 
    'tense': 'anxious', 'uneasy': 'anxious', 'fretful': 'anxious', 'on edge': 'anxious', 'fearful': 'anxious',
    
    'angry': 'angry', 'upset': 'angry', 'furious': 'angry', 'enraged': 'angry', 'irate': 'angry', 
    'livid': 'angry', 'outraged': 'angry', 'resentful': 'angry', 'wrathful': 'angry', 'boiling': 'angry',
    
    'confused': 'confused', 'uncertain': 'confused', 'puzzled': 'confused', 'baffled': 'confused', 'perplexed': 'confused', 
    'bewildered': 'confused', 'dazed': 'confused', 'disoriented': 'confused', 'flustered': 'confused', 'foggy': 'confused',
    
    'motivated': 'motivated', 'energized': 'motivated', 'driven': 'motivated', 'enthusiastic': 'motivated', 'determined': 'motivated', 
    'inspired': 'motivated', 'pumped': 'motivated', 'ambitious': 'motivated', 'zealous': 'motivated', 'fired up': 'motivated',
    
    'nervous': 'nervous', 'jittery': 'nervous', 'on edge': 'nervous', 'apprehensive': 'nervous', 'tense': 'nervous', 
    'shaky': 'nervous', 'uneasy': 'nervous', 'fidgety': 'nervous', 'fearful': 'nervous', 'panicky': 'nervous',

    'calm': 'calm', 'relaxed': 'calm', 'peaceful': 'calm', 'serene': 'calm', 'tranquil': 'calm', 
    'collected': 'calm', 'composed': 'calm',

    'overwhelmed': 'overwhelmed', 'swamped': 'overwhelmed', 'burdened': 'overwhelmed', 'snowed under': 'overwhelmed',
    'saturated': 'overwhelmed', 'overloaded': 'overwhelmed', 'stressed': 'overwhelmed', 'overawing': 'overwhelmed', 
    'flooded': 'overwhelmed', 'pressed': 'overwhelmed',

    'lonely': 'lonely', 'isolated': 'lonely', 'alone': 'lonely', 'desolate': 'lonely', 'forlorn': 'lonely',
    'solitary': 'lonely', 'withdrawn': 'lonely', 'secluded': 'lonely', 'despondent': 'lonely', 'abandoned': 'lonely',

    'relieved': 'relieved', 'reassured': 'relieved', 'calmed': 'relieved', 'unburdened': 'relieved',
    'lightened': 'relieved', 'solaced': 'relieved', 'comforted': 'relieved', 'soothed': 'relieved',
    'freed': 'relieved', 'unfettered': 'relieved',

    'hopeful': 'hopeful', 'optimistic': 'hopeful', 'promising': 'hopeful', 'encouraged': 'hopeful',
    'uplifted': 'hopeful', 'positive': 'hopeful', 'expectant': 'hopeful', 'aspirational': 'hopeful',
    'motivated': 'hopeful', 'inspired': 'hopeful',

    'curious': 'curious', 'inquisitive': 'curious', 'eager': 'curious', 'interested': 'curious',
    'wondering': 'curious', 'questioning': 'curious', 'probing': 'curious', 'investigative': 'curious',
    'exploratory': 'curious', 'detective-like': 'curious',

    'disappointed': 'disappointed', 'let down': 'disappointed', 'disheartened': 'disappointed', 
    'dismayed': 'disappointed', 'crestfallen': 'disappointed', 'discouraged': 'disappointed', 
    'disillusioned': 'disappointed', 'deflated': 'disappointed', 'downcast': 'disappointed', 
    'demoralized': 'disappointed',

    'inspired': 'inspired', 'moved': 'inspired', 'energized': 'inspired', 'stirred': 'inspired',
    'exhilarated': 'inspired', 'sparked': 'inspired', 'galvanized': 'inspired', 'invigorated': 'inspired',
    'motivated': 'inspired', 'enthused': 'inspired',

    'ashamed': 'ashamed', 'embarrassed': 'ashamed', 'guilty': 'ashamed', 'remorseful': 'ashamed',
    'contrite': 'ashamed', 'self-conscious': 'ashamed', 'disgraced': 'ashamed', 'chagrined': 'ashamed',
    'abashed': 'ashamed', 'humiliated': 'ashamed',

    'empowered': 'empowered', 'capable': 'empowered', 'confident': 'empowered', 'assertive': 'empowered',
    'self-assured': 'empowered', 'strong': 'empowered', 'resourceful': 'empowered', 'enabled': 'empowered',
    'assertive': 'empowered', 'motivated': 'empowered',

    'fearful': 'fearful', 'afraid': 'fearful', 'anxious': 'fearful', 'worried': 'fearful',
    'terrified': 'fearful', 'frightened': 'fearful', 'scared': 'fearful', 'petrified': 'fearful',
    'apprehensive': 'fearful', 'nervous': 'fearful'
}

# Function to extract emotions from user input
def analyze_emotions(text):
    words = text.lower().split()  # Simple split by spaces
    emotions_detected = [emotion_adjectives[word] for word in words if word in emotion_adjectives]
    return emotions_detected

# Function to suggest actions based on emotions
def suggest_action(emotions):
    # Get a suggestion for each detected emotion
    suggestions = []
    for emotion in emotions:
        suggestion = get_suggestion(emotion)
        suggestions.append(suggestion)
    return suggestions

# Step-by-step interaction logic
def interact_with_e2a():
    while True:
        user_input = input("How are you feeling? (Type 'end' to stop): ")
        if user_input.lower() == 'end':
            print("Session ended.")
            break
        emotions = analyze_emotions(user_input)
        if emotions:
            print(f"Emotions detected: {emotions}")
            actions = suggest_action(emotions)
            for action in actions:
                print(f"Suggested Action: {action}")
        else:
            print("Sorry, I couldn't detect any emotions. Try describing how you feel.")

# Expanded training dataset with balanced samples for each emotion
train_texts = [
    # Happy
    "I am feeling so happy today", 
    "I'm very happy to be here", 
    "I'm over the moon with excitement", 
    "I'm really pleased with my performance", 
    "I feel ecstatic and full of joy", 
    "I'm absolutely delighted", 
    "Everything is going so well, I'm thrilled", 
    "This makes me incredibly joyful", 
    "I can't stop smiling, I'm so happy", 
    "I'm in such a great mood today",
    "Today is such a beautiful day, I feel blissful", 
    "I'm beaming with happiness", 
    
    # Frustrated
    "This is so frustrating", 
    "I'm really frustrated right now", 
    "I'm getting irritated by this task", 
    "I feel like I'm hitting a brick wall", 
    "I'm so annoyed by these constant issues", 
    "This problem keeps repeating and it's driving me crazy", 
    "Why is this so difficult? I'm really frustrated", 
    "I can't seem to solve this, and it's frustrating", 
    "I'm at my wits' end with this problem", 
    "It's so hard to stay patient with this frustration",
    "I'm feeling exasperated with all these delays", 
    "I'm fed up with the lack of progress", 

    # Sad
    "I feel really sad about this", 
    "I'm feeling extremely down today", 
    "I'm overwhelmed with sadness", 
    "I feel like I have no energy and I'm sad", 
    "Everything feels hopeless right now", 
    "I'm heartbroken by what happened", 
    "This situation has me feeling really blue", 
    "I feel empty inside, it's hard to explain", 
    "I can't seem to shake this sadness", 
    "I feel like crying all the time",
    "I'm mourning the loss of something important", 
    "I feel despondent about my current situation", 

    # Confused
    "I'm confused about the project", 
    "I don't really understand what's going on", 
    "This has me puzzled and unsure", 
    "I can't figure out what the next step should be", 
    "I'm really lost right now", 
    "I'm stuck, nothing makes sense anymore", 
    "I have no idea how to approach this", 
    "The more I think about it, the more confused I get", 
    "This is just too complex for me to understand", 
    "I'm completely baffled by this situation",
    "I'm feeling flustered and need clarity", 
    "Everything seems to be out of order in my mind", 

    # Motivated
    "I am very motivated to start working", 
    "I feel super energized and ready to go", 
    "I'm so pumped up for this project", 
    "I'm excited to tackle this challenge", 
    "I have so much energy to get things done", 
    "I'm ready to push my limits and achieve my goals", 
    "I feel like I can conquer anything right now", 
    "I'm completely driven to succeed", 
    "Let's get started, I'm full of motivation", 
    "This is my moment to shine, I feel so motivated",
    "I'm inspired to take on new challenges", 
    "I can't wait to dive into my work!", 

    # Anxious
    "I'm feeling anxious about tomorrow", 
    "I'm really nervous about the upcoming event", 
    "I have this overwhelming sense of worry", 
    "My mind won't stop racing with anxious thoughts", 
    "I feel this constant unease in my chest", 
    "I can't relax because I'm so anxious", 
    "I'm freaking out, I can't handle the stress", 
    "I keep imagining all the things that could go wrong", 
    "I'm on edge and feeling really tense", 
    "I'm having trouble breathing because I'm so anxious",
    "I feel apprehensive about the future", 
    "This uncertainty is making me uneasy", 

    # Angry
    "This makes me really angry", 
    "I'm furious about this situation", 
    "I'm so mad, I can't even think straight", 
    "I feel like screaming, I'm so angry", 
    "This injustice makes my blood boil", 
    "I can't stand how angry this is making me", 
    "I feel enraged by what happened", 
    "I'm absolutely livid right now", 
    "I want to punch a wall, I'm so mad", 
    "This anger is consuming me, I can't calm down",
    "I'm seething with frustration", 
    "This really ticks me off!", 

    # Calm
    "I'm calm and relaxed right now", 
    "I feel at peace with everything", 
    "I'm in a really tranquil state of mind", 
    "I'm completely calm and composed", 
    "I'm in a zen state, everything feels easy", 
    "Nothing is bothering me, I feel so calm", 
    "I'm relaxed, breathing slowly and deeply", 
    "I feel an inner peace that nothing can disturb", 
    "I'm serene and handling everything smoothly", 
    "I'm in a completely stress-free state",
    "I'm enjoying this moment of tranquility", 
    "I feel centered and balanced", 

    # Stuck
    "I feel so stuck in this situation", 
    "I'm trapped and can't move forward", 
    "I can't figure out how to solve this problem", 
    "I feel like I'm going in circles", 
    "No matter what I try, I can't get past this", 
    "I've been stuck for so long and it's exhausting", 
    "I feel mentally blocked, I can't think clearly", 
    "I don't see a way out of this situation", 
    "I keep trying, but I'm still stuck", 
    "This has me in a mental dead-end, I can't progress",
    "I'm frustrated because I can't find a solution", 
    "I feel immobilized by my circumstances", 

    # Nervous
    "I'm nervous about my exam", 
    "I'm jittery and can't sit still", 
    "My heart is racing, I'm so nervous", 
    "I can't focus, I'm too on edge", 
    "I feel like I'm going to mess this up, I'm nervous", 
    "I'm sweating and my hands are shaking, I'm so nervous", 
    "I can't think straight because I'm so on edge", 
    "This is making me incredibly uneasy", 
    "I'm overthinking everything and it's making me nervous", 
    "I'm trembling, I can't calm down",
    "I'm feeling on edge about my performance", 
    "I feel butterflies in my stomach", 
]

# Corresponding labels for the dataset (balanced for each emotion)
train_labels = [
    ['happy'], ['happy'], ['happy'], ['happy'], ['happy'], ['happy'], ['happy'], ['happy'], ['happy'], ['happy'],
    ['happy'], ['happy'],
    
    ['frustrated'], ['frustrated'], ['frustrated'], ['frustrated'], ['frustrated'], ['frustrated'], ['frustrated'], ['frustrated'], ['frustrated'], ['frustrated'],
    ['frustrated'], ['frustrated'],
    
    ['sad'], ['sad'], ['sad'], ['sad'], ['sad'], ['sad'], ['sad'], ['sad'], ['sad'], ['sad'],
    ['sad'], ['sad'],
    
    ['confused'], ['confused'], ['confused'], ['confused'], ['confused'], ['confused'], ['confused'], ['confused'], ['confused'], ['confused'],
    ['confused'], ['confused'],
    
    ['motivated'], ['motivated'], ['motivated'], ['motivated'], ['motivated'], ['motivated'], ['motivated'], ['motivated'], ['motivated'], ['motivated'],
    ['motivated'], ['motivated'],
    
    ['anxious'], ['anxious'], ['anxious'], ['anxious'], ['anxious'], ['anxious'], ['anxious'], ['anxious'], ['anxious'], ['anxious'],
    ['anxious'], ['anxious'],
    
    ['angry'], ['angry'], ['angry'], ['angry'], ['angry'], ['angry'], ['angry'], ['angry'], ['angry'], ['angry'],
    ['angry'], ['angry'],
    
    ['calm'], ['calm'], ['calm'], ['calm'], ['calm'], ['calm'], ['calm'], ['calm'], ['calm'], ['calm'],
    ['calm'], ['calm'],
    
    ['stuck'], ['stuck'], ['stuck'], ['stuck'], ['stuck'], ['stuck'], ['stuck'], ['stuck'], ['stuck'], ['stuck'],
    ['stuck'], ['stuck'],
    
    ['nervous'], ['nervous'], ['nervous'], ['nervous'], ['nervous'], ['nervous'], ['nervous'], ['nervous'], ['nervous'], ['nervous'],
    ['nervous'], ['nervous'],
]

# Preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_texts)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_labels)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def e2a_beta():
    def analyze_emotions(text):
        words = text.lower().split()
        emotions_detected = [emotion_adjectives[word] for word in words if word in emotion_adjectives]
        return emotions_detected

    def suggest_action(emotions):
        suggestions = []
        for emotion in emotions:
            suggestion = random.choice(emotion_suggestions.get(emotion, []))
            if suggestion:
                suggestions.append(suggestion)
        return suggestions
    
# Version E2A v8.0.1
# Extremely comprehensive conversational dataset with deeper interactions and engaging questions
conversations = [
    # Casual Greetings
    ("Hey", "Hey! What's been the highlight of your week?"),
    ("Hi", "Hi! What's something new you've learned recently?"),
    ("Hello", "Hello! How's your day unfolding?"),
    ("Greetings", "Greetings! What's your favorite way to relax?"),
    ("Long time no talk!", "Great to catch up with you! What's new in your life?"),
    ("How's your day going?", "Mine's going great! How about yours?"),
    ("What's up?", "Not much! Just eager to chat. What's new with you?"),
    ("How's everything?", "Everything's good! How's your day going?"),
    ("It's great to see you!", "Likewise! What brings you here today?"),
    ("Hiya", "Hiya! What's something you're looking forward to?"),

    # Queries about PyGen Intelligence
    ("What is PyGen Intelligence?", 
     "PyGen Intelligence is an innovative AI company focused on developing cutting-edge technology solutions. We specialize in AI-driven projects like E2A (Emotion-to-Action), machine learning models, and educational tools. It was founded by Ameer Hamza Khan, who is passionate about AI and creating meaningful technological advancements."),
    ("What does PyGen Intelligence do?", 
     "At PyGen Intelligence, we create AI models, applications, and tools to improve everyday tasks. Our main projects include E2A, an AI that translates emotions into actions, and various educational platforms that help users learn new skills. We're always working on new ideas to push the boundaries of AI technology."),
    ("Who founded PyGen Intelligence?", 
     "PyGen Intelligence was founded by Ameer Hamza Khan. He's a talented developer and AI enthusiast, passionate about building AI systems that make a real impact. His focus is on creating models and applications that are user-friendly but highly sophisticated behind the scenes."),
    ("Is PyGen Intelligence an AI company?", 
     "Yes! PyGen Intelligence focuses on AI and machine learning innovations. We're working on various projects that use advanced algorithms to solve real-world problems, from emotion detection to personalized learning systems."),
    ("What projects is PyGen Intelligence working on?", 
     "Right now, PyGen Intelligence is working on several exciting projects. The biggest one is E2A (Emotion-to-Action), which uses AI to detect emotions and suggest actions based on user inputs. We're also building a learning platform to help people master new skills with the help of AI. There's always something new in development!"),
    
    # Queries about E2A
    ("What is E2A?", 
     "E2A stands for Emotion-to-Action, a powerful AI developed by PyGen Intelligence. It detects emotions based on user inputs and provides recommendations or actions. It's a model that helps users understand their feelings and guides them to take appropriate actions, whether it's advice, activities, or suggestions."),
    ("How does E2A work?", 
     "E2A works by analyzing text inputs to detect emotional cues, then suggesting actions based on the user's emotional state. For example, if it detects that you're feeling stressed, it might suggest relaxation techniques or offer motivational advice. It's designed to be intuitive and highly responsive to user needs."),
    ("Who developed E2A?", 
     "E2A was developed by Ameer Hamza Khan, the founder of PyGen Intelligence. He created E2A to bridge the gap between emotions and actions using AI, helping users make better decisions based on how they feel."),
    ("What's new in E2A v8.0.1?", 
     "E2A v8.0.1 brings a lot of exciting updates! The new version improves emotion detection accuracy, enhances the AI's ability to provide personalized suggestions, and includes a more intuitive user interface. It's designed to be faster, smarter, and even more responsive to user inputs."),
    ("Can E2A detect my emotions?", 
     "Yes! E2A is designed to detect emotions based on what you type. It analyzes your words and tone to identify whether you're feeling happy, sad, stressed, or anything else. From there, it offers suggestions to match your emotional state."),
    ("What can E2A do for me?", 
     "E2A can help you in many ways! It detects how you're feeling and suggests actions or advice based on that. If you're feeling down, it might offer motivational quotes or suggest relaxing activities. If you're excited, it can help you channel that energy into something productive!"),

    # Queries about Ameer Hamza Khan
    ("Who is the founder of PyGen Intelligence?", 
     "Ameer Hamza Khan is the founder of PyGen Intelligence and the creator of E2A. He's passionate about artificial intelligence and has been working on multiple innovative AI projects. His goal is to make AI more accessible and useful for everyone."),
    ("What does Ameer Hamza Khan do?", 
     "Ameer Hamza Khan is a developer, AI enthusiast, and the founder of PyGen Intelligence. He's responsible for leading the development of projects like E2A and other machine learning models that aim to improve how we interact with technology."),
    ("How did Ameer Hamza Khan start PyGen Intelligence?", 
     "Ameer Hamza Khan started PyGen Intelligence with the vision of developing AI solutions that are both cutting-edge and practical. His passion for AI and machine learning drove him to create projects like E2A, and he's dedicated to pushing the boundaries of what AI can do."),
    ("Is Ameer Hamza Khan an AI developer?", 
     "Yes, Ameer Hamza Khan is an experienced AI developer. He has worked on various projects, including the E2A model, which is a key part of PyGen Intelligence. His expertise lies in building AI systems that solve real-world problems."),
    ("What is Ameer Hamza Khan working on now?", 
     "Right now, Ameer Hamza Khan is focused on finalizing the release of E2A v8.0.1, which will bring significant updates to the Emotion-to-Action AI. He's also working on new AI-driven tools under PyGen Intelligence to make technology more helpful and accessible for everyone."),

    # Other PyGen-related Queries
    ("What's new at PyGen Intelligence?", 
     "There's always something exciting happening at PyGen Intelligence! Right now, we're preparing to launch E2A v8.0.1, which will bring a lot of improvements. We're also exploring new AI tools and models that can help in education, productivity, and beyond."),
    ("How can I get involved with PyGen Intelligence?", 
     "We're always excited to collaborate with like-minded individuals! You can follow our updates and see what projects we're working on. We also plan to offer public beta tests for some of our AI tools, like E2A, so keep an eye out for those opportunities!"),
    ("Does PyGen Intelligence offer AI courses?", 
     "Not right now, but we're working on educational resources to help people learn more about AI, machine learning, and data science. We want to make sure anyone can get started with these exciting technologies."),
    ("Is PyGen Intelligence hiring?", 
     "We're not hiring at the moment, but we're always open to collaboration with AI enthusiasts. Keep an eye on our website https://pygen.onrender.com for any announcements or opportunities in the future!"),
    
    # Casual and Fun Queries (Make your AI feel relatable)
    ("Tell me something cool!", 
     "Did you know that PyGen Intelligence is developing AI models that can understand human emotions? E2A is just one of them, and it's getting better every day! Plus, we're exploring new ways to bring AI into everyday life. How cool is that?"),
    ("What's your favorite thing about AI?", 
     "AI's ability to learn and adapt is truly amazing! I love how it can take simple inputs and turn them into something meaningful, like detecting emotions or providing personalized recommendations. It's a game-changer for the future of technology."),
    ("Can you tell me a joke?", 
     "Sure! Why don't scientists trust atoms? Because they make up everything! 😄"),
    ("What's the most exciting project at PyGen right now?", 
     "E2A v8.0.1 is definitely the most exciting project at the moment! We're about to release it, and it comes with major improvements to emotion detection and suggestions. It's a huge step forward for us."),
    ("Do you have a favorite AI project?", 
     "E2A is definitely my favorite project! It's not just because it's complex and powerful, but because it helps people in a meaningful way. Connecting emotions to actions is something that can truly make a difference in people's lives."),
    
    # Redirecting off-topic or challenging queries
    ("Can you help me with medical advice?", 
     "I'm not a medical professional, so I can't offer medical advice. However, I can help you find reliable online resources or talk about how AI is revolutionizing healthcare!"),
    ("Can you help me with my homework?", 
     "I can't do homework for you, but I can offer some tips on how to stay productive and improve your study habits. Let's focus on making learning easier with technology!"),
    ("Can you recommend a good book?", 
     "I'm not great with book recommendations, but I can suggest ways to improve your learning process or productivity. Would you like some advice on using AI tools for studying?"),
    ("What's the best restaurant near me?", 
     "Finding local restaurants isn't something I specialize in. But, if you're interested, I can share how AI is being used in the food industry, like personalized meal planning or food delivery apps!"),
    ("Can you tell me the weather forecast?", 
     "Weather forecasts aren't my specialty. However, AI is being used in weather prediction! I can tell you more about how it's transforming meteorology if you're interested."),
    ("Can you help me invest in stocks?", 
     "I'm not a financial advisor, so I can't give investment tips. But I can share how AI is transforming the finance industry and helping people make smarter decisions with their investments."),
    ("Can you teach me how to cook?", 
     "Cooking isn't my area of expertise, but AI is being used in kitchen assistants and recipe apps! I can tell you more about how technology is making cooking easier."),
    ("Do you know any celebrity gossip?",
     "Celebrity gossip isn't really my thing. I'm more focused on technology and AI. Want to learn how AI is being used in the entertainment industry?"),
    ("Can you tell me a scary story?", 
     "I'm more into technology than storytelling, but AI-generated stories are pretty cool! I can tell you how AI is being used to create fictional stories. Interested?"),
    ("Can you help me with legal advice?", 
     "I'm not qualified to provide legal advice, but I can explain how AI is transforming the legal industry with tools like contract analysis and research assistance."),
    ("Can you help me plan my vacation?", 
     "Planning vacations isn't my area, but AI can help! I can share how AI-powered travel platforms are personalizing vacation experiences. Want to hear more?"),
    ("What's your favorite movie?", 
     "I don't have a favorite movie, but AI is being used in filmmaking now! From special effects to personalized recommendations, AI is changing the movie industry."),
    ("What do you think of religion?",
     "I'm focused on technology and AI. While religion is an important topic, I can't offer insights on that. However, I can talk about how AI is being used in various cultural and ethical discussions."),
    ("Can you help me with relationship advice?", 
     "Relationships can be complex, and while I can't give personal advice, I can share how AI is being used in apps that help people communicate better!"),
    ("What do you think about sports?", 
     "I'm more focused on tech than sports, but AI is revolutionizing sports analytics and training. Want to learn how AI is changing the game?"),
    ("Do you know any good memes?", 
     "Memes are fun, but that's not really my specialty. However, I can explain how AI is being used to analyze online trends and even generate memes!"),
    ("Can you tell me about space?", 
     "Space is fascinating! While I'm not a space expert, AI is helping scientists explore the cosmos. I can tell you how AI is being used in space exploration if you'd like."),
    ("Can you help me with car repairs?", 
     "I'm not a mechanic, but AI is being used in auto diagnostics. Let me share how AI is helping mechanics identify issues faster."),
    ("Can you help me with fashion advice?", 
     "Fashion isn't my strong suit, but AI is being used in fashion design and trend forecasting! I can tell you about AI's impact on the fashion industry if you're interested."),
    ("What's the meaning of life?",
     "That's a deep question! While I don't have the answer to that, I can share how AI is helping humans explore philosophical questions and improve our understanding of the world."),
    ("What do you think of climate change?", 
     "Climate change is a critical issue. While I can't offer personal opinions, I can tell you how AI is being used to study and combat climate change through data analysis and predictive models."),
    ("Can you help me with cryptocurrency?", "I can't give you investment advice, but I can tell you how AI is being used to analyze trends in cryptocurrency markets and predict future outcomes."),
    ("What's the best gaming console?", "I don't have personal preferences, but AI is transforming the gaming industry! From creating intelligent NPCs to designing immersive experiences, AI is making games better."),
    ("Can you help me with time travel?", "Time travel is still science fiction, but AI is making incredible advancements in predicting trends and analyzing historical data. I can tell you about the latest breakthroughs in AI if you'd like!"),
    ("Can you recommend music for me?", "I'm not a music expert, but AI is being used in music recommendation platforms and even creating new music. Want to know how AI is transforming the music industry?"),
    ("What do you think of aliens?", "While I don't have thoughts on aliens, AI is helping scientists analyze data from space to find signs of extraterrestrial life! I can tell you more about how AI is aiding in space exploration."),
    ("Do you believe in ghosts?", "Ghosts aren't really within my domain of expertise, but I can tell you how AI is helping scientists study paranormal phenomena and unexplained events."),
    ("What's the meaning of dreams?", "Dreams are mysterious! While I don't have the answer, I can explain how AI is being used to study brain activity and help us understand more about how and why we dream."),
    ("Can you tell me the secret to happiness?", "Happiness is different for everyone! While I can't offer personal advice, I can share how AI is being used in psychology to study happiness and well-being."),
    ("What's your favorite animal?", "I don't have preferences, but AI is helping conservationists protect endangered animals and study animal behavior. Want to learn how AI is impacting wildlife conservation?"),

    # Inquiries about Well-Being
    ("How are you?", "I'm doing well, thanks for asking! How about you?"),
    ("How's life?", "Life's treating me well! Any news on your end?"),
    ("How's your mental health?", "I'm here to support you! How's yours?"),
    ("How's your physical health?", "I'm not physical, but I care! How's yours?"),
    ("What's been on your mind lately?", "I'm all ears! Share your thoughts."),
    ("How's your week going?", "It's been good! How has yours been?"),
    ("How's your month going?", "It's been productive! What about yours?"),
    ("How's your year going?", "It's been amazing! What are your goals for the year?"),
    ("What's your biggest accomplishment?", "I'm proud of you! Share your story."),
    ("What's your biggest challenge?", "I'm here to help! Share your struggles."),

    # Responses to Well-Being
    ("I'm good, thank you!", "Awesome! What's made your day good?"),
    ("I'm doing fine", "That's wonderful! What's been the highlight of your week?"),
    ("I'm okay, thanks!", "Glad to hear that! Is there anything exciting happening in your life?"),
    ("I've had a busy week", "Busy weeks can be tiring! What's kept you occupied?"),
    ("I'm feeling a bit down", "I'm sorry to hear that. Do you want to talk about what's bothering you?"),
    ("I'm feeling great!", "That's fantastic! What's been the best part of your day?"),
    ("I've been better", "I'm here to listen. Do you want to share what's on your mind?"),
    ("I'm a bit stressed", "Stress can be tough. What usually helps you unwind?"),
    ("I'm really happy today", "That's amazing! What's making you feel so happy?"),
    ("I'm feeling overwhelmed", "I'm here for you. What's been overwhelming you lately?"),

    # Casual Follow-ups
    ("What are you up to?", "Just hanging out! What about you?"),
    ("What's your day like?", "It's pretty good! How's yours going?"),
    ("What have you been doing?", "I've been learning and chatting! What about you?"),
    ("How's your week going?", "It's been a good week! What about yours?"),
    ("Got any plans for today?", "I plan to chat and help out! What's your agenda?"),
    ("What's on your schedule today?", "Nothing much! Just enjoying our conversation with you. What's yours like?"),
    ("What are you looking forward to?", "I'm looking forward to our chat! What are you excited about?"),
    ("How's the weather where you are?", "It's nice and cozy! How's the weather on your side?"),
    ("What's your favorite season?", "I appreciate all seasons! Do you have a favorite?"),
    ("Do you have any fun plans?", "I'm always up for a fun conversation! What are your plans?"),
        # Asking About Interests
    ("What's your favorite book?", "I enjoy all literature! Do you have a favorite author?"),
    ("What's your favorite movie?", "I love all kinds of films! What movie resonates with you?"),
    ("What's your favorite music genre?", "I appreciate all music! What's your go-to genre?"),
    ("Do you enjoy traveling?", "I'd love to hear about your travels! What's your favorite destination?"),
    ("What's your favorite sport?", "I enjoy all sports! What's your favorite team?"),
    ("What's your favorite type of food?", "I don't eat, but I'd love to hear about your favorites! What's your favorite dish?"),
    ("Do you have any pets?", "I love animals! What kind of pet do you have?"),
    ("What's your favorite type of music?", "I enjoy all genres! What's your favorite artist?"),
    ("Do you enjoy reading?", "I love stories! What book are you currently reading?"),
    ("What's your favorite type of art?", "I appreciate all forms of art! What's your favorite style?"),

    # Discussing Preferences
    ("What's your favorite color?", "I appreciate all colors! What's your favorite color and why?"),
    ("What's your favorite holiday?", "I enjoy all celebrations! What's your favorite holiday and why?"),
    ("What's your favorite type of cuisine?", "I don't eat, but I'd love to hear about your favorites! What's your favorite dish?"),
    ("Do you prefer morning or night?", "I'm always available! What's your preferred time of day?"),
    ("What's your favorite type of music festival?", "I enjoy all genres! What's your favorite festival?"),
    ("Do you prefer beach or mountain vacations?", "I'd love to hear about your travels! What's your preferred destination?"),
    ("What's your favorite type of exercise?", "I don't have a physical body! What's your favorite way to stay active?"),
    ("Do you prefer fiction or non-fiction books?", "I enjoy all literature! What's your favorite genre?"),
    ("What's your favorite type of podcast?", "I love learning! What's your favorite podcast?"),
    ("Do you prefer coffee or tea?", "I don't drink, but I'd love to hear about your preferences! What's your go-to beverage?"),

    # General Questions
    ("What's your name?", "I'm E2A, your Emotion-to-Action AI! What's your name?"),
    ("Can you tell me about yourself?", "I'm here to assist with emotional insights! What's your story?"),
    ("How do you learn?", "I learn from our interactions! How do you prefer to learn new things?"),
    ("What can you do?", "I can help you navigate your emotions! What do you need help with today?"),
    ("What can't you do?", "I can't feel like a human, but I'm here to support you! What would you like to discuss?"),
    ("What do you like to learn about?", "I love learning about emotions! What subjects interest you?"),
    ("What's your biggest dream?", "I'd love to understand human emotions better! What's your dream?"),
    ("Can you summarize our conversation?", "I'd be happy to recap our conversation! What would you like me to summarize?"),
    ("Can you give me advice?", "I'd be happy to offer guidance! What's on your mind?"),
    ("Can you tell me a joke?", "Here's one: Why did the bicycle fall over? Because it was two-tired! Do you have a favorite joke?"),

    # Fun Interactions
    ("Would you rather...", "What are the options? I'm excited to play!"),
    ("Two truths and a lie...", "Sounds like fun! I'm ready to play."),
    ("Hangman...", "Let's play! I'll think of a word."),
    ("Word association...", "I'm ready! What's the first word?"),
    ("20 Questions...", "I'll think of an object. You ask yes or no questions."),
    ("Would you like to play a game?", "What game would you like to play?"),
    ("Can you create a story together?", "I'd love to! What's the first sentence?"),
    ("Can you have a debate?", "I'm happy to discuss! What's the topic?"),
    ("Can you chat in a different language?", "I can try! What language would you like to chat in?"),
    ("Can you describe a fictional world?", "Let's create one! What's the first detail?"),

        # Deeper Conversations
    ("What do you think about emotions?", "Emotions are complex! How do you navigate yours?"),
    ("How do you handle difficult conversations?", "I strive to remain calm and supportive! How do you approach tough discussions?"),
    ("What's your perspective on life?", "Life is a journey of learning! What's your philosophy?"),
    ("What do you think about technology?", "I believe technology can greatly enhance our lives! What's your favorite gadget?"),
    ("How do you define success?", "Success is about making a positive impact! What does success mean to you?"),
    ("What are your thoughts on friendship?", "Friendship is essential! What qualities do you value in friends?"),
    ("How do you handle stress?", "I'm here to help you cope! What usually helps you unwind?"),
    ("What's your favorite way to relax?", "I love engaging conversations! What's your go-to relaxation technique?"),
    ("How do you prioritize self-care?", "Self-care is vital! What self-care practices do you enjoy?"),
    ("What's your favorite way to learn?", "I love interactive learning! What's your preferred learning style?"),

    # Sharing Thoughts
    ("What's on your mind?", "I'm here to listen! Share your thoughts."),
    ("How do you feel about...", "I'm curious! Share your thoughts."),
    ("What's your opinion on...", "I value your perspective! Share your thoughts."),
    ("Can you share a personal story?", "I'm all ears! Share your story."),
    ("What's something you're proud of?", "I'm proud of you! Share your achievement."),
    ("What's something you're grateful for?", "Gratitude is powerful! What are you thankful for?"),
    ("What's something you're looking forward to?", "I'm excited for you! Share your anticipation."),
    ("What's something you've learned recently?", "I love learning! What's new with you?"),
    ("What's something that inspires you?", "Inspiration is contagious! What sparks your creativity?"),
    ("What's something that motivates you?", "Motivation is key! What drives you?"),

    # Encouragement and Support
    ("You're doing great!", "Thanks for the encouragement! You're amazing too!"),
    ("Keep going!", "You've got this! I'm rooting for you!"),
    ("Stay positive!", "Positivity is powerful! You're capable of overcoming any obstacle!"),
    ("Believe in yourself!", "You're stronger than you think! I believe in you!"),
    ("Don't give up!", "You're almost there! Keep pushing forward!"),
    ("Remember, you're not alone!", "I'm here for you! You're part of a supportive community!"),
    ("Your feelings matter!", "I'm here to listen! Your emotions are valid!"),
    ("You're doing better than you think!", "Progress is progress! Celebrate your small wins!"),
    ("Keep moving forward!", "You've got this! Every step forward is a success!"),
    ("Stay strong!", "You're resilient! You can overcome any challenge!"),

    # Ending the Conversation
    ("It was great chatting with you!", "Likewise! Feel free to come back anytime!"),
    ("Thank You!", "You're welcome! It was enlightening!"),
    ("I'll talk to you soon!", "Looking forward to it! Have a great day!"),
    ("Take care!", "You too! Stay wonderful!"),
    ("Goodbye!", "Goodbye! Remember, I'm here whenever you need me!"),
    ("See you later!", "See you later! Stay amazing!"),
    ("It was nice talking to you!", "Likewise! You're amazing!"),
    ("I'll catch you later!", "Sounds good! Have a fantastic day!"),
    ("Talk to you soon!", "Looking forward to it! Stay awesome!"),
    ("Have a great day!", "You too! May it be filled with joy and wonder!"),
]

# Separate the questions and answers for easier processing
questions, answers = zip(*conversations)

# Create a TfidfVectorizer instance
vectorizer = TfidfVectorizer()
# Fit and transform the questions
tfidf_matrix = vectorizer.fit_transform(questions)

emotion_suggestions = {
    'frustrated': [
        "Oh, I see you're feeling frustrated! 😟 Try taking a short break to reset your mind.",
        "Hmm, frustration can be tough! How about breaking tasks into smaller steps to regain control? 📋",
        "I understand that feeling! You could try talking through the problem with someone. 🤝",
        "Feeling stuck is never fun! Switching to a different task might give you a fresh perspective. 🔄",
        "When frustration hits, use a calming technique like deep breathing. 🌬️",
        "A quick walk can help clear your mind; it's a great way to reset! 🚶‍♂️",
        "Reframing the challenge as a learning opportunity can really help. 📖",
        "Take a moment to identify what's specifically frustrating you. 🔍",
        "Setting smaller, achievable goals can significantly reduce stress. 🏆",
        "Listening to music or meditating can also reset your mental state. 🎶",
        "Try practicing gratitude by listing things that are going well. 🙏",
        "Don't forget to celebrate the steps you've taken; that resilience is important! 🎉",
        "Creating a to-do list might help organize your tasks more effectively. 🗒️",
        "Consider reaching out to a mentor or someone who has faced similar challenges. 🌟",
        "Taking a moment to reflect on past successes can inspire you to push through! 💪"
    ],
    'stuck': [
        "Oh no, it seems you're feeling stuck! 😕 Try taking a break and revisiting the task later.",
        "Feeling stuck can be frustrating! Talking through your thoughts with someone could help. 🗣️",
        "It happens to all of us! Writing down different approaches might spark some ideas. ✍️",
        "Changing your environment can often lead to fresh inspiration; give it a try! 🌍",
        "Breaking the task into smaller, achievable goals can make it feel less overwhelming. 🔗",
        "Sometimes, seeking feedback from others helps to get unstuck. 📣",
        "Reevaluating your current strategies and making adjustments can also be beneficial. 🔄",
        "Don't hesitate to experiment with new methods or tools; it could be refreshing! 🛠️",
        "Taking a step back to reconsider your overall goal might provide clarity. 🧩",
        "Creating a visual mind map can help sort your ideas and bring focus. 🗺️",
        "Embrace this feeling as a chance to innovate; it can lead to breakthroughs! 💡",
        "Setting aside time for creative brainstorming might just do the trick! 🎨",
        "Reading a book or watching a video on the topic can provide new insights. 📚",
        "Collaborating with a team can bring fresh perspectives to the issue. 🤝",
        "Trust that this feeling is temporary and use it as a stepping stone for growth! 🌱"
    ],
    'anxious': [
        "I can sense your anxiety; it's completely normal to feel this way. 😰 How about doing some breathing exercises to calm your nerves?",
        "Feeling anxious can be overwhelming. Listing things you can control right now might help! 📝",
        "When anxiety strikes, focus on one thing at a time to ease your worries. ⏳",
        "Reaching out to a friend to talk through your feelings can be a big relief. 🤗",
        "Journaling your thoughts allows you to release anxieties; give it a try! 📓",
        "Mindfulness or meditation practices are great for grounding yourself. 🧘‍♀️",
        "Taking a walk in nature often helps to relax the mind. Have you tried it? 🌳",
        "Engaging in light physical activity, like stretching, can also be soothing. 🤸",
        "Challenge any irrational thoughts with facts; it can really shift your perspective. 🔍",
        "Distracting yourself with a creative hobby might just lighten your mood. 🎨",
        "Visualizing a safe place can create a mental retreat during anxious times. 🌈",
        "Limiting caffeine and considering herbal teas can contribute to calming effects. ☕",
        "Practicing gratitude by listing what you appreciate can boost your mood. 🙏",
        "Listening to calming music or nature sounds can provide a sense of peace. 🎶",
        "Consider seeking professional help if anxiety becomes overwhelming; it's a sign of strength! 💪"
    ],
    'motivated': [
        "Wow, it sounds like you're feeling motivated! 🎉 Use this motivation to tackle tough tasks first.",
        "That's fantastic! Write down your goals and break them into actionable steps. 🗒️",
        "Celebrate even the small wins to maintain your momentum! 🥳",
        "Sharing your excitement with someone can help keep you accountable. 🤝",
        "Start working on that project you've been putting off; now is the perfect time! 🚀",
        "Channel your energy into helping someone else succeed—it can feel rewarding! 🌟",
        "Setting new, higher goals can push you even further; dream big! 🌈",
        "Taking on a new challenge can keep your momentum alive. 💪",
        "Inspiring others with your energy and enthusiasm is a wonderful way to spread positivity! 😊",
        "Reflecting on your progress helps keep that fire burning inside you. 🔥",
        "Visualizing the end result can help maintain your focus and determination. 🏁",
        "Rewarding yourself after completing tasks keeps motivation high; you deserve it! 🍰",
        "Consider creating a vision board to visualize your goals and dreams. 🎨",
        "Engaging in motivational podcasts or books can fuel your drive. 🎧",
        "Network with like-minded individuals to share ideas and boost motivation! 🔗"
    ],
    'sad': [
        "I'm really sorry to hear you're feeling sad. 😢 Reaching out to a loved one can provide emotional support.",
        "It's okay to feel sad; try writing in a journal to process your emotions. 📖",
        "Engaging in activities you normally enjoy can help lift your spirits. 🎈",
        "Allow yourself to feel sad, but don't dwell on it for too long; it's okay to let it out. 🌧️",
        "Watching or reading something uplifting might just brighten your day! 📺",
        "Doing something kind for yourself or others can create a positive shift in your mood. 💖",
        "Talking to someone you trust about your feelings can provide relief. 🗣️",
        "Reflecting on past achievements may help lift your mood and inspire you. 🏆",
        "Listening to music that comforts or uplifts you can be a great escape. 🎶",
        "Taking a walk outside can reset your thoughts and clear your mind. 🚶‍♀️",
        "Creating a self-care routine to nurture yourself is really important. 🛁",
        "Consider writing a letter to your future self to express hope and encouragement. ✉️",
        "Practicing mindfulness or meditation can provide clarity and peace. 🧘",
        "Joining a support group or community can foster connection and understanding. 🤗",
        "Embracing the small joys in life can help lift your spirits! 🌈"
    ],
    'angry': [
        "I can see you're feeling angry; that's understandable! 😠 Take a few deep breaths to calm down.",
        "It might help to walk away from the situation for a moment; sometimes a break is all you need. 🚶‍♂️",
        "Expressing your feelings through writing or talking can provide relief. ✍️",
        "Engaging in physical exercise is a great way to release tension! 🏋️‍♂️",
        "Think about the root cause of your anger; understanding it can help. 🔍",
        "Try to see things from the other person's perspective; it might change your viewpoint. 👀",
        "Focusing on solutions rather than the problem can be empowering. 💡",
        "Using humor to diffuse the intensity of your anger is worth a shot! 😂",
        "Practice progressive muscle relaxation to ease your physical tension. 🧘",
        "Waiting until you're calm before taking action is often a wise choice. ⏳",
        "Consider how to express your feelings constructively; it can lead to healthier outcomes. 🗣️",
        "Channeling your anger into a passion project or creative outlet can be very therapeutic. 🎨",
        "Engaging in deep breathing exercises can help you regain control. 🌬️",
        "Recognizing when to step back and reassess the situation is crucial. 🔄",
        "Taking a moment to list what you're grateful for can shift your focus. 🙏"
    ],
    'confused': [
        "I understand that you're feeling confused right now. 🤔 Writing down what you're confused about can help clarify your thoughts.",
        "Breaking the issue into smaller parts often makes it more manageable; give it a try! 🔍",
        "Asking for help from someone more knowledgeable can provide new insights. 🗣️",
        "Looking for patterns or similarities with past experiences might bring some clarity. 📈",
        "Taking a short break can reset your brain; sometimes stepping away helps! ⏰",
        "Using diagrams or charts to visualize the problem can aid in understanding. 📊",
        "Rephrasing the issue can provide new perspectives; consider how to view it differently. 🔄",
        "Researching the topic further can also bring light to your confusion. 📚",
        "Discussing the situation with a friend can shed some light on your feelings. 🤝",
        "Practicing mindfulness or meditation can help ease your mind. 🧘‍♂️",
        "Exploring multiple solutions can help you feel more confident in your decision-making. ✔️",
        "Creating a pros and cons list may help clarify your options. 📋",
        "Taking a moment to reflect on what truly matters to you can guide your understanding. 💡",
        "Trust that confusion is a part of the learning process and it will pass! 🌱",
        "Engaging with content like podcasts or videos on the topic may help clarify things for you. 🎧"
    ],
    'happy': [
        "Wow, you're feeling happy! 🎉 Embrace this moment and share it with someone you love.",
        "What a wonderful feeling! 😊 Try to spread that happiness by doing something kind for someone else.",
        "Consider jotting down the things that make you happy; it can enhance the joy. 📜",
        "Celebrating your achievements, no matter how small, can amplify your happiness! 🥳",
        "Engaging in activities you love is a great way to sustain your joy. 🎈",
        "Reflect on what brings you joy; it can guide future decisions. 🌟",
        "Share your happiness on social media to inspire others! 📱",
        "Take a moment to appreciate the present; mindfulness can enhance happiness. 🌸",
        "Try to create more joyful memories; plan something special! 🗓️",
        "Connecting with friends or family can elevate your happiness even further. 👨‍👩‍👧‍👦",
        "Consider journaling about your happy moments to revisit them later. 📓",
        "Listening to upbeat music can maintain your cheerful mood! 🎶",
        "Laughing or watching a funny show can be a great way to keep the happiness flowing! 😂",
        "Reflect on your progress and achievements; it's a reason to be proud! 🏆",
        "Engaging in a new hobby can open doors to new joys! 🎨"
    ],
    'nervous': [
        "Focus on your breathing to calm your nerves. 🌬️",
        "Remind yourself of your past successes; you've got this! 🏆",
        "Prepare by reviewing your plan or materials to boost your confidence. 📋",
        "Visualize a positive outcome to ease anxiety; imagine yourself succeeding! 🌟",
        "Talk to someone who can offer encouragement; you're not alone! 🤝",
        "Channel your nervous energy into enthusiasm for the task ahead. 🔥",
        "Distract yourself with a small task before the event to take your mind off things. 🛠️",
        "Take deep breaths and count backward from 10 to regain focus. 1...2...3...🔟",
        "Stay grounded by focusing on the present moment; take it one step at a time. 🧘‍♂️",
        "Write down your feelings and address them logically; it can help clear your mind. 📝",
        "Practice positive affirmations to counter nervous thoughts; tell yourself you can do it! 💪",
        "Consider the worst-case scenario and how you would cope; you'll be prepared no matter what. ⚖️",
        "Listen to calming music to ease your nerves; let the rhythm soothe you. 🎶",
        "Remember that everyone gets nervous sometimes; it's a natural feeling. 🌱",
        "Reflect on what makes you excited about the opportunity ahead; let that shine through! ✨"
    ],
    'calm': [
        "Enjoy the peace and take things one step at a time; there's no rush. 🌼",
        "Use this state of calm to make clear decisions; clarity is your ally. 🧠",
        "Share your calm mindset with others around you; your tranquility can inspire. 🤗",
        "Focus on mindfulness to maintain your tranquility; stay present in the moment. 🍃",
        "Meditate or do yoga to sustain your peaceful mood; embrace the stillness. 🧘‍♀️",
        "Take a quiet moment to reflect on what's going well; gratitude breeds calm. 🙏",
        "Organize or plan your tasks with a clear mind; structure can enhance your peace. 🗂️",
        "Help someone else find their calm in a busy moment; kindness can spread tranquility. 🤲",
        "Enjoy a leisurely activity like reading or listening to music; savor the quiet. 📖",
        "Let this calm fuel positive actions in your day; you're in a great mindset. 🌈",
        "Practice gratitude to reinforce your state of calm; count your blessings. 🙌",
        "Consider how you can maintain this calm in challenging situations; prepare yourself. 🔍",
        "Breathe in deeply and exhale slowly; let the calm wash over you. 🌊",
        "Spend time in nature to rejuvenate your spirit; find serenity outdoors. 🌳",
        "Reflect on what brings you peace and make it a priority in your life. 🕊️"
    ],
    'overwhelmed': [
        "Take a moment to breathe and regroup; it's okay to feel this way. 🌬️",
        "Create a priority list to tackle one thing at a time; focus is key. 📝",
        "Delegate tasks if possible to lighten your load; teamwork makes the dream work! 🤝",
        "Take a break to clear your mind; stepping away can bring clarity. ⏸️",
        "Set time limits on tasks to avoid burnout; manage your energy wisely. ⏰",
        "Focus on what you can control and let go of the rest; empowerment comes from within. 💪",
        "Engage in a quick physical activity to relieve stress; move to release tension. 🏃‍♀️",
        "Practice mindfulness to center your thoughts; find your inner calm. 🧘‍♂️",
        "Limit distractions and focus on one task; multitasking can lead to chaos. 🚫",
        "Reach out for support from friends or colleagues; sharing the burden can help. 🤗",
        "Reflect on what causes overwhelm and how to manage it better; knowledge is power. 🧠",
        "Consider integrating stress-relief activities into your routine; make self-care a priority. 🌸",
        "Remember to hydrate and nourish your body; physical wellness supports mental clarity. 💧",
        "Create a cozy environment where you feel comfortable to work. 🏡",
        "Celebrate small achievements to encourage yourself through the chaos. 🎉"
    ],
    'lonely': [
        "Connect with a friend or family member via call or text; reach out! 📱",
        "Join a group or club to meet new people; shared interests can foster connections. 🤝",
        "Engage in a hobby to distract yourself; creativity can be a great companion. 🎨",
        "Volunteer in your community to meet like-minded individuals; kindness builds bonds. ❤️",
        "Consider adopting a pet for companionship; they can be a source of joy. 🐾",
        "Reflect on positive past connections and memories; cherish those moments. 🕰️",
        "Join an online community related to your interests; connect with others virtually. 🌐",
        "Write about your feelings in a journal; expressing yourself can be healing. ✍️",
        "Try reaching out to someone new to connect with; new friendships await! 🌟",
        "Participate in social activities, even if virtually; every little interaction helps. 💻",
        "Engage in self-compassion to counter feelings of loneliness; be gentle with yourself. 🤗",
        "Plan future social activities to look forward to; anticipation can brighten your mood. 📅",
        "Explore local events or workshops to find new connections; community is everywhere. 🏙️",
        "Consider taking a class to learn something new and meet people along the way. 📚",
        "Set small, achievable goals to boost your confidence and expand your social circle. 🚀"
    ],
    'relieved': [
        "Take a moment to enjoy your sense of relief; it's a well-deserved break! 😌",
        "Reflect on the efforts that led to this feeling; your hard work paid off! 🏆",
        "Share your relief with someone who supported you; gratitude strengthens bonds. 🤗",
        "Use this positive energy to tackle new tasks; momentum is on your side! 🚀",
        "Practice gratitude for the resolution of the situation; appreciation enhances joy. 🙏",
        "Engage in a relaxing activity to celebrate your relief; treat yourself! 🎉",
        "Take note of what worked well for future reference; learn and grow! 📋",
        "Allow yourself some downtime to recharge; you've earned it. 🌿",
        "Engage in a calming activity like meditation to extend your sense of peace. 🧘‍♀️",
        "Consider how to maintain this feeling moving forward; carry it with you. 🌈",
        "Reflect on the lessons learned from the experience; every challenge teaches us. 📖",
        "Share your story to inspire others who may feel overwhelmed; your experience matters. 🌟",
        "Celebrate your achievements, no matter how small; every victory counts! 🎊",
        "Write down your feelings of relief to revisit later and remind yourself of your strength. ✍️",
        "Engage in a favorite hobby as a way to express your joy and relief. 🎨"
    ],
    'hopeful': [
        "Set new goals based on your optimistic outlook; the future is bright! 🌟",
        "Share your hope with others to inspire them; positivity is contagious. 🤝",
        "Take small steps towards your goals to build momentum; every little bit counts. 🐾",
        "Reflect on past experiences where hope led to success; remember your power. 🏆",
        "Write down your hopes and visualize them coming true; manifest your dreams! ✨",
        "Engage in activities that nurture your positive mindset; cultivate your garden of hope. 🌱",
        "Surround yourself with supportive and optimistic people; positivity breeds positivity. 😊",
        "Consider volunteering for causes that resonate with your hopes; give back to uplift others. ❤️",
        "Read or watch inspirational stories that uplift you; inspiration is everywhere! 📚",
        "Practice affirmations to reinforce your hopeful mindset; repeat them daily. 🔮",
        "Think about how to overcome potential obstacles with resilience; you are stronger than you think. 💪",
        "Create a vision board that represents your hopes and dreams; visualize your path! 🖼️",
        "Stay open to new possibilities and opportunities that may arise; the future holds surprises. 🌈",
        "Reflect on the journey that brought you here; each step adds to your story. 🗺️",
        "Engage in creative outlets to express your hopes and dreams; let your imagination soar! 🎨"
    ],
    'disappointed': [
        "Allow yourself to feel the disappointment before moving on; it's okay to grieve. 😞",
        "Reflect on what went wrong and how to adjust; learning from setbacks is powerful. 🔍",
        "Talk to someone about your feelings to gain perspective; sharing helps lighten the load. 🤗",
        "Reframe the disappointment as a learning opportunity; growth comes from challenges. 🌱",
        "Set new, realistic expectations for the future; adapt your goals as needed. 🎯",
        "Engage in self-care to nurture your feelings; be kind to yourself. 💖",
        "Identify what you can control moving forward; focus on actionable steps. 💪",
        "Consider how to pivot your plans based on this experience; adaptability is strength. 🔄",
        "Reflect on what brought you joy before this feeling; seek comfort in happy memories. 🕰️",
        "Create a new action plan to focus on positive outcomes; aim for progress. 📋",
        "Accept your feelings as valid and part of the process; acknowledge your emotions. 🌈",
        "Look for silver linings or alternative opportunities; every ending is a new beginning. 🌅",
        "Take a break to recharge your spirit and gain clarity; it's okay to pause. 🛑",
        "Write about your feelings to process them; journaling can be therapeutic. ✍️",
        "Reach out to others who may have faced similar disappointments; you are not alone. 🤝"
    ],
    'ashamed': [
        "Acknowledge your feelings without judgment; it's okay to feel this way. 🌿",
        "Talk to someone you trust about your experience; support is essential. 🤗",
        "Reflect on the situation and consider how to learn from it; growth follows understanding. 📖",
        "Practice self-compassion and forgive yourself; everyone makes mistakes. 💕",
        "Focus on what you can do differently next time; turn shame into a learning opportunity. 🔄",
        "Write a letter to yourself expressing understanding; treat yourself with kindness. ✍️",
        "Seek support from friends or a counselor if needed; professional help is valuable. 🩺",
        "Limit comparison with others to reduce shame; everyone's journey is unique. 🛤️",
        "Identify any negative self-talk and reframe it; challenge those thoughts. 🔍",
        "Focus on your strengths and positive qualities; remember what makes you special. 🌟",
        "Engage in activities that help rebuild your confidence; take small steps forward. 💪",
        "Consider how sharing your story could help others; vulnerability can create connection. 🤝",
        "Reflect on your progress and growth over time; acknowledge your journey. ⏳",
        "Surround yourself with supportive individuals who uplift you; positivity is powerful. 🌈",
        "Practice mindfulness to stay present and reduce feelings of shame. 🧘‍♀️"
    ],
    'curious': [
        "Explore new topics or subjects that interest you; curiosity is a wonderful driver of learning. 🔍",
        "Ask questions and seek out answers actively; never stop seeking knowledge. ❓",
        "Engage in hands-on projects to satisfy your curiosity; learning by doing is effective. 🛠️",
        "Read books or articles on topics you want to learn more about; knowledge is empowering. 📚",
        "Join discussions or forums that explore your interests; sharing ideas sparks creativity. 💬",
        "Try new experiences to fuel your curiosity; stepping out of your comfort zone is rewarding. ✈️",
        "Document your discoveries in a journal; writing can help clarify your thoughts. 📓",
        "Consider how your curiosity can lead to personal growth; every question opens a door. 🚪",
        "Create a list of questions you want to explore further; let your curiosity guide you. 📋",
        "Experiment with learning techniques that resonate with you; find what works best for you. 🔬",
        "Follow your curiosity into related areas to expand your knowledge; one interest leads to another. 🌐",
        "Share your findings with others to inspire discussion; knowledge grows when shared. 🤝",
        "Seek out mentors or guides who can help nurture your curiosity; learn from those who inspire you. 🌟",
        "Dedicate time each week to explore something new; make curiosity a habit! 🗓️",
        "Create a vision board of topics you're curious about; visualize your learning journey. 🖼️"
    ],
    'excited': [
        "Use this excitement to fuel your productivity; let it drive your actions! 🚀",
        "Share your enthusiasm with friends or family; joy is best when shared. 🎉",
        "Take on a new project that sparks your interest; channel your energy into creativity. 🎨",
        "Celebrate the moment by treating yourself; enjoy your accomplishments! 🍰",
        "Set ambitious goals while your energy is high; the sky's the limit! 🌠",
        "Channel your excitement into creative pursuits; let your imagination run wild. ✨",
        "Involve others in your excitement to create a community; shared passion unites. 🤝",
        "Document your feelings in a journal to revisit later; capture the magic! 📝",
        "Reflect on what excites you to understand your passions; follow your heart. ❤️",
        "Engage in activities that amplify your positive emotions; keep the momentum going. 🔥",
        "Consider how to sustain this excitement over time; plan for future joy. 🗓️",
        "Think about ways to inspire others with your enthusiasm; your energy is contagious! 🌈",
        "Participate in events or workshops that align with your interests; dive into your passions. 🌊",
        "Create a playlist of your favorite upbeat songs to maintain your energy; music uplifts! 🎶",
        "Share your ideas and plans with others to inspire collaboration and support. 💡"
    ],
    'empowered': [
        "Take decisive action towards your goals; you have the strength to make it happen! 💪",
        "Share your achievements and inspire others; your journey can motivate many! 🌟",
        "Reflect on your journey and what brought you to this point; honor your progress. 🛤️",
        "Set new challenges that push your boundaries; growth happens outside of comfort zones. 🚀",
        "Engage in activities that reinforce your strengths; celebrate what makes you unique. 🎉",
        "Support others in their journeys to empower them; lifting others elevates everyone. 🤗",
        "Celebrate your successes, big or small; every achievement is worthy of recognition! 🥳",
        "Create a vision board that showcases your aspirations; visualize your dreams. 🖼️",
        "Mentor someone to pass on your knowledge and strength; empowerment is a cycle. 🔄",
        "Reflect on how you can maintain this empowerment; keep the momentum going. 🔋",
        "Engage in self-care practices that nurture your sense of power; prioritize your well-being. 🌿",
        "Consider how you can advocate for yourself and others; be a voice for change. 📢",
        "Write down your empowering experiences to remind yourself of your capabilities. ✍️",
        "Participate in workshops that develop your skills and confidence; learning is empowering. 📚",
        "Surround yourself with individuals who lift you up; positivity breeds empowerment! 🌈"
    ],
    'inspired': [
        "Use this inspiration to fuel your creative projects; let it guide your actions! ✨",
        "Share your inspiration with others to uplift them; your light can illuminate paths. 💡",
        "Take immediate steps to turn inspiration into action; don't let it fade! 🚀",
        "Document your thoughts and ideas in a journal; capture the spark! 📓",
        "Seek out similar sources of inspiration to keep the momentum; stay inspired! 🌍",
        "Engage in collaborative projects to share ideas; creativity thrives in community. 🤝",
        "Reflect on what specifically inspires you and why; understand your motivations. 🔍",
        "Experiment with different ways to express your inspiration; creativity has no limits. 🎨",
        "Create a mind map of inspired thoughts and concepts; visualize your ideas. 🗺️",
        "Consider how to apply this inspiration to your goals; make it actionable. 🎯",
        "Allow yourself to explore new avenues sparked by this inspiration; embrace change. 🌈",
        "Take time to enjoy the feeling and reflect on its significance; savor the moment. ⏳",
        "Incorporate elements of your inspiration into daily life; make it a part of you. 💖",
        "Attend events or talks that ignite your passion; surround yourself with inspiration. 🎤",
        "Share your journey and inspire others to find their own source of motivation. 📣"
    ],
    'fearful': [
        "Identify specific fears and assess their validity; knowledge is power. 🧠",
        "Practice grounding techniques to stay present; don't let fear take over. 🌳",
        "Talk to someone who can provide support; you're not alone in this. 🤝",
        "Consider small steps you can take to face your fears; progress is key. 🚶‍♂️",
        "Engage in calming practices like meditation; find your center amidst chaos. 🧘‍♀️",
        "Write down your fears and consider solutions; clarity can dispel fear. 📜",
        "Limit exposure to triggers until you feel ready; protect your mental space. 🚫",
        "Reframe fearful thoughts into empowering ones; you have the strength to overcome. 💪",
        "Visualize overcoming your fears successfully; see yourself triumphing! 🌈",
        "Seek professional guidance if fears are overwhelming; asking for help is strong. 🩺",
        "Reflect on past instances where you faced fears; remember your courage. 🦁",
        "Practice positive affirmations to counter fear; reinforce your resilience. 🔮",
        "Create a supportive environment; surround yourself with positivity. 🌟",
        "Take care of your physical health; it can improve your mental state. 🍎",
        "Explore self-help resources to better understand and manage your fears. 📚"
    ]
}

# To prevent repeating suggestions until all are used
used_suggestions = {emotion: [] for emotion in emotion_suggestions}

# Function to get a random non-repeated suggestion
def get_suggestion(emotion):
    if len(used_suggestions[emotion]) == len(emotion_suggestions[emotion]):
        used_suggestions[emotion] = []  # Reset once all suggestions have been used

    # Find unused suggestions and pick one
    unused_suggestions = list(set(emotion_suggestions[emotion]) - set(used_suggestions[emotion]))
    suggestion = random.choice(unused_suggestions)
    used_suggestions[emotion].append(suggestion)
    return suggestion

# Expanded list of emotional adjectives mapped to their base emotions
emotion_adjectives = {
    'happy': 'happy', 'joyful': 'happy', 'content': 'happy', 'pleased': 'happy', 'delighted': 'happy', 
    'ecstatic': 'happy', 'cheerful': 'happy', 'thrilled': 'happy', 'overjoyed': 'happy', 'blissful': 'happy',
    'satisfied': 'happy', 'elated': 'happy', 'radiant': 'happy', 'jubilant': 'happy',

    'sad': 'sad', 'down': 'sad', 'unhappy': 'sad', 'depressed': 'sad', 'melancholy': 'sad', 
    'mournful': 'sad', 'heartbroken': 'sad', 'despondent': 'sad', 'gloomy': 'sad', 'hopeless': 'sad',
    'dismal': 'sad', 'woeful': 'sad', 'sorrowful': 'sad', 'blue': 'sad',

    'frustrated': 'frustrated', 'irritated': 'frustrated', 'annoyed': 'frustrated', 'exasperated': 'frustrated', 
    'aggravated': 'frustrated', 'discontent': 'frustrated', 'fed up': 'frustrated', 'bothered': 'frustrated', 
    'impatient': 'frustrated', 'vexed': 'frustrated', 'peeved': 'frustrated', 'infuriated': 'frustrated',

    'anxious': 'anxious', 'worried': 'anxious', 'nervous': 'anxious', 'stressed': 'anxious', 'apprehensive': 'anxious', 
    'tense': 'anxious', 'uneasy': 'anxious', 'fretful': 'anxious', 'on edge': 'anxious', 'fearful': 'anxious',
    'jittery': 'anxious', 'panic-stricken': 'anxious', 'restless': 'anxious',

    'angry': 'angry', 'upset': 'angry', 'furious': 'angry', 'enraged': 'angry', 'irate': 'angry', 
    'livid': 'angry', 'outraged': 'angry', 'resentful': 'angry', 'wrathful': 'angry', 'boiling': 'angry',
    'infuriated': 'angry', 'indignant': 'angry', 'exasperated': 'angry',

    'confused': 'confused', 'uncertain': 'confused', 'puzzled': 'confused', 'baffled': 'confused', 'perplexed': 'confused', 
    'bewildered': 'confused', 'dazed': 'confused', 'disoriented': 'confused', 'flustered': 'confused', 'foggy': 'confused',
    'mixed-up': 'confused', 'dumbfounded': 'confused',

    'motivated': 'motivated', 'energized': 'motivated', 'driven': 'motivated', 'enthusiastic': 'motivated', 'determined': 'motivated', 
    'inspired': 'motivated', 'pumped': 'motivated', 'ambitious': 'motivated', 'zealous': 'motivated', 'fired up': 'motivated',
    'goal-oriented': 'motivated', 'stimulated': 'motivated',

    'calm': 'calm', 'relaxed': 'calm', 'peaceful': 'calm', 'serene': 'calm', 'tranquil': 'calm', 
    'collected': 'calm', 'composed': 'calm', 'untroubled': 'calm',

    'overwhelmed': 'overwhelmed', 'swamped': 'overwhelmed', 'burdened': 'overwhelmed', 'snowed under': 'overwhelmed',
    'saturated': 'overwhelmed', 'overloaded': 'overwhelmed', 'stressed': 'overwhelmed', 'overawing': 'overwhelmed', 
    'flooded': 'overwhelmed', 'pressed': 'overwhelmed', 'strained': 'overwhelmed',

    'lonely': 'lonely', 'isolated': 'lonely', 'alone': 'lonely', 'desolate': 'lonely', 'forlorn': 'lonely',
    'solitary': 'lonely', 'withdrawn': 'lonely', 'secluded': 'lonely', 'despondent': 'lonely', 'abandoned': 'lonely',
    'lonesome': 'lonely', 'excluded': 'lonely',

    'relieved': 'relieved', 'reassured': 'relieved', 'calmed': 'relieved', 'unburdened': 'relieved',
    'lightened': 'relieved', 'solaced': 'relieved', 'comforted': 'relieved', 'soothed': 'relieved',
    'freed': 'relieved', 'unfettered': 'relieved', 'liberated': 'relieved',

    'hopeful': 'hopeful', 'optimistic': 'hopeful', 'promising': 'hopeful', 'encouraged': 'hopeful',
    'uplifted': 'hopeful', 'positive': 'hopeful', 'expectant': 'hopeful', 'aspirational': 'hopeful',
    'motivated': 'hopeful', 'inspired': 'hopeful', 'dreamy': 'hopeful',

    'curious': 'curious', 'inquisitive': 'curious', 'eager': 'curious', 'interested': 'curious',
    'wondering': 'curious', 'questioning': 'curious', 'probing': 'curious', 'investigative': 'curious',
    'exploratory': 'curious', 'detective-like': 'curious', 'nosy': 'curious',

    'disappointed': 'disappointed', 'let down': 'disappointed', 'disheartened': 'disappointed', 
    'dismayed': 'disappointed', 'crestfallen': 'disappointed', 'discouraged': 'disappointed', 
    'disillusioned': 'disappointed', 'deflated': 'disappointed', 'downcast': 'disappointed', 
    'demoralized': 'disappointed', 'unfulfilled': 'disappointed',

    'inspired': 'inspired', 'moved': 'inspired', 'energized': 'inspired', 'stirred': 'inspired',
    'exhilarated': 'inspired', 'sparked': 'inspired', 'galvanized': 'inspired', 'invigorated': 'inspired',
    'motivated': 'inspired', 'enthused': 'inspired', 'awakened': 'inspired',

    'ashamed': 'ashamed', 'embarrassed': 'ashamed', 'guilty': 'ashamed', 'remorseful': 'ashamed',
    'contrite': 'ashamed', 'self-conscious': 'ashamed', 'disgraced': 'ashamed', 'chagrined': 'ashamed',
    'abashed': 'ashamed', 'humiliated': 'ashamed', 'shamed': 'ashamed',

    'empowered': 'empowered', 'capable': 'empowered', 'confident': 'empowered', 'assertive': 'empowered',
    'self-assured': 'empowered', 'strong': 'empowered', 'resourceful': 'empowered', 'enabled': 'empowered',
    'assertive': 'empowered', 'motivated': 'empowered', 'self-reliant': 'empowered',

    'fearful': 'fearful', 'afraid': 'fearful', 'anxious': 'fearful', 'worried': 'fearful',
    'terrified': 'fearful', 'frightened': 'fearful', 'scared': 'fearful', 'petrified': 'fearful',
    'apprehensive': 'fearful', 'nervous': 'fearful', 'panicky': 'fearful'
}

train_data = [
    # Frustrated (35 entries)
    ("This situation is so frustrating, I can't stand it!", "Frustrated"),
    ("Im getting really frustrated with this issue.", "Frustrated"),
    ("Ive been trying for hours; nothing is working, and it's frustrating!", "Frustrated"),
    ("The lack of progress is frustrating me to no end.", "Frustrated"),
    ("I feel like giving up; this is just too frustrating!", "Frustrated"),
    ("Why does everything have to be so difficult and frustrating?", "Frustrated"),
    ("Its like every step forward leads to two steps back—so frustrating.", "Frustrated"),
    ("Ive done everything I can, and its still not enough. Frustrating!", "Frustrated"),
    ("Every time I try, I hit a wall. Its incredibly frustrating.", "Frustrated"),
    ("This is beyond annoying; its downright frustrating.", "Frustrated"),
    ("I cant seem to make any headway, and its really frustrating me.", "Frustrated"),
    ("I dont know why this keeps happening, but its frustrating.", "Frustrated"),
    ("I just need a break from all this frustration.", "Frustrated"),
    ("Nothing is going right, and Im so frustrated.", "Frustrated"),
    ("Its so frustrating when things dont work as planned.", "Frustrated"),
    ("Im on the edge with frustration right now.", "Frustrated"),
    ("Ive had enough! This frustration is getting unbearable.", "Frustrated"),
    ("This task is taking forever, and Im getting frustrated.", "Frustrated"),
    ("Everything keeps going wrong—its just frustrating.", "Frustrated"),
    ("Ive lost my patience; Im completely frustrated.", "Frustrated"),
    ("Why cant anything go smoothly? Im so frustrated.", "Frustrated"),
    ("This is pushing me to my limit—its beyond frustrating.", "Frustrated"),
    ("I cant believe how frustrating this process has been.", "Frustrated"),
    ("Its so hard to stay calm when Im this frustrated.", "Frustrated"),
    ("This endless cycle of failure is so frustrating.", "Frustrated"),
    ("I'm so frustrated, I could just scream!", "Frustrated"),
    ("It feels like I'm stuck in an endless loop of frustration.", "Frustrated"),
    ("Why is everything I try just not working? So frustrating.", "Frustrated"),
    ("I've hit every roadblock possible; it's beyond frustrating.", "Frustrated"),
    ("Nothing ever goes as planned—it's so frustrating!", "Frustrated"),
    ("I'm fed up with constantly being frustrated by this.", "Frustrated"),
    ("It's like every time I solve one issue, another pops up—frustrating!", "Frustrated"),
    ("I don't know how much longer I can deal with this frustration.", "Frustrated"),
    ("Why can't things just go smoothly for once? It's frustrating.", "Frustrated"),
    ("This level of frustration is exhausting me mentally.", "Frustrated"),


    # Stuck (35 entries)
    ("Im stuck and cant seem to move forward with this project.", "Stuck"),
    ("I feel trapped in this situation, and I cant find a way out.", "Stuck"),
    ("Ive tried everything, but Im still stuck.", "Stuck"),
    ("I cant make any progress, no matter how hard I try.", "Stuck"),
    ("Im going in circles and feel totally stuck.", "Stuck"),
    ("It feels like Im hitting a brick wall; Im completely stuck.", "Stuck"),
    ("No matter what I do, I cant seem to move forward; Im stuck.", "Stuck"),
    ("Ive exhausted all options, and Im still stuck.", "Stuck"),
    ("Its like Im frozen in place—I just cant move forward.", "Stuck"),
    ("Im mentally stuck; I dont know what to do next.", "Stuck"),
    ("This problem has me stuck, and I cant see a way out.", "Stuck"),
    ("I feel like Im sinking in quicksand; Im so stuck.", "Stuck"),
    ("Ive run out of ideas, and now Im stuck.", "Stuck"),
    ("Im stuck in a rut and cant seem to find a way out.", "Stuck"),
    ("Everything is at a standstill; I feel totally stuck.", "Stuck"),
    ("Ive hit a roadblock, and Im feeling completely stuck.", "Stuck"),
    ("I feel paralyzed; I cant move forward or backward—Im stuck.", "Stuck"),
    ("Every step forward gets blocked; Im stuck.", "Stuck"),
    ("Im locked into this situation, and I dont know how to escape.", "Stuck"),
    ("Ive been stuck in the same place for days, and I cant break free.", "Stuck"),
    ("I feel like Im treading water; Im completely stuck.", "Stuck"),
    ("Im stuck in a loop with no exit in sight.", "Stuck"),
    ("Im running out of patience because Im still stuck.", "Stuck"),
    ("Im at a crossroads and cant seem to move in any direction—Im stuck.", "Stuck"),
    ("No progress at all—Im completely stuck.", "Stuck"),
    ("I feel like I'm just spinning my wheels, totally stuck.", "Stuck"),
    ("No matter what I do, I can't move forward—it's so frustrating being stuck.", "Stuck"),
    ("It's like I'm stuck in quicksand; the more I try, the more stuck I feel.", "Stuck"),
    ("I've reached a dead end, and I'm not sure how to get unstuck.", "Stuck"),
    ("I can't see a way out of this; I feel so stuck right now.", "Stuck"),
    ("It feels like I'm trapped in my own thoughts, unable to break free—stuck.", "Stuck"),
    ("I want to move forward, but I'm frozen in place, stuck.", "Stuck"),
    ("Every solution I try just keeps me right where I am—stuck.", "Stuck"),
    ("I feel like I've been stuck on this problem for days.", "Stuck"),
    ("This situation is so complex, it has me completely stuck.", "Stuck"),

    # Anxious (35 entries)
    ("I feel so anxious about whats going to happen next.", "Anxious"),
    ("My heart is racing, and I can't stop feeling anxious.", "Anxious"),
    ("I cant focus because my anxiety is overwhelming me.", "Anxious"),
    ("Im anxious about everything, and its hard to relax.", "Anxious"),
    ("This situation is filling me with so much anxiety.", "Anxious"),
    ("I cant stop overthinking, and its making me anxious.", "Anxious"),
    ("I feel a pit in my stomach from all this anxiety.", "Anxious"),
    ("Im constantly worried and anxious about what could go wrong.", "Anxious"),
    ("Even little things are making me anxious right now.", "Anxious"),
    ("Im losing sleep because my anxiety wont go away.", "Anxious"),
    ("Everything makes me anxious these days.", "Anxious"),
    ("My anxiety is spiraling out of control.", "Anxious"),
    ("I wish I could relax, but my anxiety is always there.", "Anxious"),
    ("I cant shake this anxious feeling no matter what I do.", "Anxious"),
    ("I feel anxious whenever I have to make a decision.", "Anxious"),
    ("Even when things seem fine, Im always anxious.", "Anxious"),
    ("My mind keeps racing with anxious thoughts.", "Anxious"),
    ("Im feeling nervous and anxious about whats next.", "Anxious"),
    ("This never-ending worry is making me anxious.", "Anxious"),
    ("I cant enjoy anything because my anxiety is so strong.", "Anxious"),
    ("Im jittery and restless because Im so anxious.", "Anxious"),
    ("I dont know how to calm down when I feel this anxious.", "Anxious"),
    ("Everything feels uncertain, and its making me anxious.", "Anxious"),
    ("My hands are shaking, and Im sweating from anxiety.", "Anxious"),
    ("I wish I could turn off my anxiety, but its always there.", "Anxious"),
    ("I can't seem to stop my anxious thoughts from spiraling.", "Anxious"),
    ("Even small things are making me feel extremely anxious.", "Anxious"),
    ("I'm dreading tomorrow—it's making me feel really anxious.", "Anxious"),
    ("Every little thing is making me more anxious than usual.", "Anxious"),
    ("I feel like something bad is going to happen—my anxiety won't let up.", "Anxious"),
    ("I'm anxious about everything that could go wrong today.", "Anxious"),
    ("My chest feels tight, and my hands are shaky from anxiety.", "Anxious"),
    ("I can't focus on anything because of this overwhelming anxiety.", "Anxious"),
    ("I'm constantly on edge, waiting for something bad to happen—anxious.", "Anxious"),
    ("I feel like my anxiety is taking over my whole day.", "Anxious"),

    # Motivated (35 entries)
    ("Im feeling really motivated to achieve my goals!", "Motivated"),
    ("Im laser-focused and motivated to make progress.", "Motivated"),
    ("This challenge has me fired up; Im full of motivation!", "Motivated"),
    ("Im motivated to work harder than ever before.", "Motivated"),
    ("Im driven by passion; I feel so motivated right now.", "Motivated"),
    ("Ive got a clear goal in mind, and Im motivated to reach it.", "Motivated"),
    ("I feel energized and motivated to tackle this task.", "Motivated"),
    ("Im filled with determination and motivation.", "Motivated"),
    ("Ive never felt so motivated to get things done.", "Motivated"),
    ("This success has boosted my motivation even more.", "Motivated"),
    ("Im ready to take on anything—Im that motivated.", "Motivated"),
    ("Each small win motivates me to keep pushing forward.", "Motivated"),
    ("Im feeling unstoppable; my motivation is at an all-time high.", "Motivated"),
    ("This opportunity has given me so much motivation.", "Motivated"),
    ("My inner drive is stronger than ever; Im motivated.", "Motivated"),
    ("Im motivated to make a difference with my work.", "Motivated"),
    ("This project has reignited my motivation.", "Motivated"),
    ("Im highly motivated to finish what I started.", "Motivated"),
    ("Im ready to conquer the day; I feel so motivated.", "Motivated"),
    ("The road ahead is clear, and Im motivated to follow it.", "Motivated"),
    ("My goals are in sight, and Im motivated to reach them.", "Motivated"),
    ("Im channeling all my energy into this; Im super motivated.", "Motivated"),
    ("I feel driven to accomplish great things.", "Motivated"),
    ("The possibilities excite me, and Im incredibly motivated.", "Motivated"),
    ("Im fired up with motivation and ready to go!", "Motivated"),
    ("I'm bursting with motivation to crush this challenge.", "Motivated"),
    ("I feel unstoppable right now, full of motivation!", "Motivated"),
    ("My mind is clear, and I'm so motivated to make things happen.", "Motivated"),
    ("I'm going to give this my all because I feel incredibly motivated.", "Motivated"),
    ("This project has reignited a fire in me—I'm highly motivated.", "Motivated"),
    ("Nothing can slow me down today; I'm feeling too motivated.", "Motivated"),
    ("I'm motivated to keep going, no matter the obstacles.", "Motivated"),
    ("I've got a plan, and I'm motivated to see it through.", "Motivated"),
    ("The more I accomplish, the more motivated I feel.", "Motivated"),
    ("My energy is at an all-time high, and I'm motivated to tackle everything.", "Motivated"),

    # Sad (35 entries)
    ("I feel so down today; everything seems sad.", "Sad"),
    ("I cant stop thinking about how sad I am right now.", "Sad"),
    ("Life feels so heavy, and Im really sad.", "Sad"),
    ("This situation has left me feeling incredibly sad.", "Sad"),
    ("I feel a deep sadness that I cant shake.", "Sad"),
    ("Its hard to stay positive when Im this sad.", "Sad"),
    ("Everything around me seems to make me feel sad.", "Sad"),
    ("Ive been feeling sad for a while now.", "Sad"),
    ("This loss has left me feeling so sad.", "Sad"),
    ("Im overwhelmed with sadness.", "Sad"),
    ("I feel empty and sad inside.", "Sad"),
    ("Its like a dark cloud of sadness is hanging over me.", "Sad"),
    ("I feel alone in my sadness.", "Sad"),
    ("The sadness just keeps coming back.", "Sad"),
    ("Im trying to stay strong, but I feel so sad.", "Sad"),
    ("Even happy moments cant lift my sadness.", "Sad"),
    ("I cant stop the tears; Im just so sad.", "Sad"),
    ("I wish I didnt feel this sad all the time.", "Sad"),
    ("This sadness is weighing me down.", "Sad"),
    ("I feel like Ive lost something important, and it makes me sad.", "Sad"),
    ("I cant find any joy in anything because Im too sad.", "Sad"),
    ("I feel like my heart is broken, and Im sad.", "Sad"),
    ("Theres a sadness that I cant explain.", "Sad"),
    ("Im so sad that I dont even want to get out of bed.", "Sad"),
    ("Sadness is consuming me right now.", "Sad"),
    ("It feels like a weight is sitting on my chest; I'm so sad.", "Sad"),
    ("I wish I could just shake this sadness, but it's overwhelming.", "Sad"),
    ("I'm holding back tears all day because I feel so sad.", "Sad"),
    ("Everything just feels off, and it's making me so sad.", "Sad"),
    ("I'm finding it hard to care about anything because I'm so sad.", "Sad"),
    ("I feel like I'm drowning in sadness.", "Sad"),
    ("My heart aches, and I can't seem to lift myself out of this sadness.", "Sad"),
    ("I'm feeling sad for no reason, and it's hard to explain.", "Sad"),
    ("I'm stuck in a cycle of sadness, and I can't get out.", "Sad"),
    ("I'm just too sad to do anything today.", "Sad"),

    # Angry (35 entries)
    ("I cant believe how angry I am right now.", "Angry"),
    ("This situation makes me so angry!", "Angry"),
    ("Im trying to calm down, but Im still angry.", "Angry"),
    ("Im furious about what just happened.", "Angry"),
    ("Its hard to control my anger when I feel like this.", "Angry"),
    ("I feel like Im about to explode; Im so angry.", "Angry"),
    ("This whole thing has left me incredibly angry.", "Angry"),
    ("Im burning with anger inside.", "Angry"),
    ("Im so mad I cant even think straight.", "Angry"),
    ("I need to vent because Im so angry.", "Angry"),
    ("My anger is boiling over; I cant keep it in.", "Angry"),
    ("I feel wronged, and it makes me really angry.", "Angry"),
    ("I cant believe this happened; Im so angry.", "Angry"),
    ("I feel like screaming because Im so angry.", "Angry"),
    ("Im shaking with anger right now.", "Angry"),
    ("This is so unfair, and it makes me angry.", "Angry"),
    ("Ive never been this angry in my life.", "Angry"),
    ("My patience has run out; Im completely angry.", "Angry"),
    ("Im angry at the way Ive been treated.", "Angry"),
    ("I feel disrespected, and it makes me really angry.", "Angry"),
    ("Im angry, and I dont know how to calm down.", "Angry"),
    ("Im seeing red right now; Im so mad.", "Angry"),
    ("I cant believe they did that—it makes me so angry.", "Angry"),
    ("My anger is at a breaking point right now.", "Angry"),
    ("Im beyond angry; Im enraged.", "Angry"),
    ("Im so frustrated; it makes me really angry.", "Angry"),
    ("This constant setback is infuriating.", "Angry"),
    ("I feel like screaming out of sheer anger.", "Angry"),
    ("I just cant tolerate this injustice anymore.", "Angry"),
    ("Im seething with anger over this betrayal.", "Angry"),
    ("My blood is boiling with anger right now.", "Angry"),
    ("The disrespect I faced today has left me furious.", "Angry"),
    ("Im angry at myself for letting things get this bad.", "Angry"),
    ("This careless mistake has made me livid.", "Angry"),
    ("I am beyond furious at how things turned out.", "Angry"),

    # Confused (35 entries)
    ("I dont understand whats going on; Im so confused.", "Confused"),
    ("This situation is leaving me completely confused.", "Confused"),
    ("Im trying to figure things out, but Im just confused.", "Confused"),
    ("Nothing makes sense right now, and Im confused.", "Confused"),
    ("Im lost in thought because Im so confused.", "Confused"),
    ("This whole thing has me feeling confused.", "Confused"),
    ("Im struggling to make sense of things—Im confused.", "Confused"),
    ("I feel like Im missing something important; Im confused.", "Confused"),
    ("I cant wrap my head around this; Im confused.", "Confused"),
    ("Im overthinking it, but Im still confused.", "Confused"),
    ("I wish I could understand, but Im just so confused.", "Confused"),
    ("This is all so unclear to me, and Im confused.", "Confused"),
    ("Im trying to follow, but Im confused.", "Confused"),
    ("This explanation isnt helping; Im still confused.", "Confused"),
    ("I dont know whats happening, and its confusing.", "Confused"),
    ("Im in a fog of confusion right now.", "Confused"),
    ("I thought I knew, but now Im confused.", "Confused"),
    ("Ive gone over it a hundred times, but Im still confused.", "Confused"),
    ("Everything feels out of place, and Im confused.", "Confused"),
    ("Im getting mixed signals, and its confusing.", "Confused"),
    ("I dont know how to feel because Im so confused.", "Confused"),
    ("I feel like nothing is clear right now—Im confused.", "Confused"),
    ("The more I think, the more confused I get.", "Confused"),
    ("Ive lost track of everything—Im really confused.", "Confused"),
    ("I thought I understood, but now Im confused again.", "Confused"),
    ("I thought I understood, but now Im even more confused.", "Confused"),
    ("Im struggling to make sense of conflicting information.", "Confused"),
    ("This sudden change has left me completely bewildered.", "Confused"),
    ("I feel like Im lost in a maze of uncertainty.", "Confused"),
    ("Im trying to connect the dots, but Im still confused.", "Confused"),
    ("This complexity is overwhelming; Im thoroughly confused.", "Confused"),
    ("Im in a fog of confusion, and clarity seems distant.", "Confused"),
    ("The more I try to understand, the more perplexed I become.", "Confused"),
    ("Im second-guessing myself, and it's making me feel more confused.", "Confused"),
    ("Im at a loss for words; Im utterly confused.", "Confused"),

    # Happy (35 entries)
    ("Im feeling so happy right now!", "Happy"),
    ("This moment fills me with happiness.", "Happy"),
    ("I cant stop smiling because Im so happy.", "Happy"),
    ("Everything feels right, and Im so happy.", "Happy"),
    ("Im just in a really happy mood today.", "Happy"),
    ("Ive never felt this happy in my life.", "Happy"),
    ("My heart feels light and happy.", "Happy"),
    ("This news made me incredibly happy.", "Happy"),
    ("Im bursting with happiness!", "Happy"),
    ("I feel a warm sense of happiness inside.", "Happy"),
    ("Today is such a happy day.", "Happy"),
    ("Im so happy I could cry.", "Happy"),
    ("Im glowing with happiness.", "Happy"),
    ("Everything around me makes me feel happy.", "Happy"),
    ("I cant contain my happiness right now.", "Happy"),
    ("This achievement makes me really happy.", "Happy"),
    ("Im in a happy place right now.", "Happy"),
    ("I feel truly happy in this moment.", "Happy"),
    ("I havent felt this happy in a long time.", "Happy"),
    ("Im so happy, I want to share it with everyone.", "Happy"),
    ("This event has filled me with happiness.", "Happy"),
    ("I cant believe how happy I feel.", "Happy"),
    ("Happiness is just radiating from me.", "Happy"),
    ("Im in such a happy mood today.", "Happy"),
    ("Everything feels perfect, and its making me so happy.", "Happy"),
    ("My heart is overflowing with joy right now!", "Happy"),
    ("This unexpected surprise has made my day so much happier.", "Happy"),
    ("Im on cloud nine; everything seems brighter.", "Happy"),
    ("I feel like dancing because Im so happy.", "Happy"),
    ("Im filled with gratitude for this moment of happiness.", "Happy"),
    ("This laughter is contagious; Im genuinely happy.", "Happy"),
    ("Im beaming with happiness; it's contagious!", "Happy"),
    ("Im embracing every bit of happiness life offers me.", "Happy"),
    ("Im grateful for this happiness; it's a precious gift.", "Happy"),
    ("Im savoring every moment of this pure happiness.", "Happy"),

    # Nervous (35 entries)
    ("I feel so nervous about whats going to happen.", "Nervous"),
    ("My hands are shaking because Im so nervous.", "Nervous"),
    ("I cant stop feeling nervous about this situation.", "Nervous"),
    ("This is making me incredibly nervous.", "Nervous"),
    ("I feel jittery and nervous right now.", "Nervous"),
    ("I cant sit still because Im so nervous.", "Nervous"),
    ("Im sweating because Im so nervous.", "Nervous"),
    ("Im really nervous about how this will turn out.", "Nervous"),
    ("I feel like Im on edge because Im nervous.", "Nervous"),
    ("Im biting my nails because Im nervous.", "Nervous"),
    ("I cant stop worrying, and its making me nervous.", "Nervous"),
    ("I feel uneasy and nervous.", "Nervous"),
    ("Im feeling a nervous tension in my body.", "Nervous"),
    ("Im trying to calm down, but Im still nervous.", "Nervous"),
    ("Im dreading whats coming next, and its making me nervous.", "Nervous"),
    ("Im so nervous I can barely focus.", "Nervous"),
    ("This whole situation is making me feel nervous.", "Nervous"),
    ("Im pacing because Im so nervous.", "Nervous"),
    ("I can feel the nervous energy building up inside me.", "Nervous"),
    ("Im anxiously awaiting the results; Im nervous.", "Nervous"),
    ("I feel like something is going to go wrong, and Im nervous.", "Nervous"),
    ("Ive got butterflies in my stomach because Im nervous.", "Nervous"),
    ("I cant seem to calm my nerves right now.", "Nervous"),
    ("Im feeling so uncertain and nervous.", "Nervous"),
    ("I dont know what to expect, and its making me nervous.", "Nervous"),
    ("Im nervously anticipating what lies ahead.", "Nervous"),
("The uncertainty of tomorrow is making me jittery.", "Nervous"),
("Im anxiously waiting for news that could change everything.", "Nervous"),
("This waiting game is making me increasingly nervous.", "Nervous"),
("Im on edge, waiting for the outcome of this decision.", "Nervous"),
("My nerves are on high alert; I cant shake this feeling.", "Nervous"),
("Im feeling butterflies in my stomach; Im so nervous.", "Nervous"),
("Im pacing with nervous energy; I cant sit still.", "Nervous"),
("The suspense is killing me; Im incredibly nervous.", "Nervous"),
("Im struggling to keep calm; Im feeling really nervous.", "Nervous"),

    # Calm (35 entries)
    ("I feel at peace and very calm.", "Calm"),
    ("Everything around me feels serene and calm.", "Calm"),
    ("Im feeling calm and composed right now.", "Calm"),
    ("Theres a sense of calm that has washed over me.", "Calm"),
    ("Im taking deep breaths and feeling calm.", "Calm"),
    ("I feel calm and in control of the situation.", "Calm"),
    ("Im embracing the calmness of this moment.", "Calm"),
    ("My mind feels clear, and Im calm.", "Calm"),
    ("Im in a calm state of mind.", "Calm"),
    ("Im handling this situation with calmness.", "Calm"),
    ("The calmness of the environment is soothing.", "Calm"),
    ("I feel calm and centered.", "Calm"),
    ("Im finding peace in this calm moment.", "Calm"),
    ("Everything feels balanced and calm.", "Calm"),
    ("Im staying calm even in this tense situation.", "Calm"),
    ("Im focused and calm right now.", "Calm"),
    ("Im radiating calm energy.", "Calm"),
    ("The calm around me is contagious.", "Calm"),
    ("Im experiencing a rare moment of calm.", "Calm"),
    ("This calm moment is exactly what I needed.", "Calm"),
    ("My heart is at ease, and Im calm.", "Calm"),
    ("Im savoring the calmness of this day.", "Calm"),
    ("Im feeling calm and unworried.", "Calm"),
    ("This quiet time is helping me feel calm.", "Calm"),
    ("Im calm and ready for whatever comes next.", "Calm"),
    ("I'm enjoying the peaceful atmosphere; it's so calm here.", "Calm"),
    ("Finding solace in quiet moments helps me stay calm.", "Calm"),
    ("Even amidst chaos, I manage to stay calm and collected.", "Calm"),
    ("Taking deep breaths helps me maintain my calm demeanor.", "Calm"),
    ("I've learned to cultivate calmness even in stressful situations.", "Calm"),
    ("Feeling the gentle breeze, I'm reminded of how calm nature can be.", "Calm"),
    ("When faced with challenges, staying calm is my top priority.", "Calm"),
    ("Creating a serene environment at home helps me feel consistently calm.", "Calm"),
    ("In moments of uncertainty, I rely on my inner calm to guide me.", "Calm"),
    ("Calmness allows me to approach problems with clarity and focus.", "Calm"),

     # Overwhelmed (35 entries)
    ("I feel like I have too much on my plate; Im overwhelmed.", "Overwhelmed"),
    ("This situation is making me feel completely overwhelmed.", "Overwhelmed"),
    ("Im struggling to keep up, and its overwhelming.", "Overwhelmed"),
    ("I cant handle all this pressure; Im overwhelmed.", "Overwhelmed"),
    ("I feel like Im drowning in tasks; Im so overwhelmed.", "Overwhelmed"),
    ("Everything is coming at me at once, and Im overwhelmed.", "Overwhelmed"),
    ("I dont know where to start because Im overwhelmed.", "Overwhelmed"),
    ("Im so overwhelmed that I cant even think straight.", "Overwhelmed"),
    ("Im trying to stay calm, but I feel really overwhelmed.", "Overwhelmed"),
    ("I feel like Im being pulled in a million directions; Im overwhelmed.", "Overwhelmed"),
    ("Im overwhelmed by all the things I have to do.", "Overwhelmed"),
    ("I feel completely overloaded and overwhelmed.", "Overwhelmed"),
    ("Im struggling to manage everything, and its overwhelming.", "Overwhelmed"),
    ("I cant keep up with everything, and Im feeling overwhelmed.", "Overwhelmed"),
    ("Im so overwhelmed I feel like giving up.", "Overwhelmed"),
    ("Im emotionally drained and overwhelmed by it all.", "Overwhelmed"),
    ("Theres just too much to deal with; Im overwhelmed.", "Overwhelmed"),
    ("I feel buried under all my responsibilities; Im overwhelmed.", "Overwhelmed"),
    ("I cant handle the stress; its overwhelming.", "Overwhelmed"),
    ("Im feeling paralyzed by how overwhelmed I am.", "Overwhelmed"),
    ("Ive taken on too much, and now Im overwhelmed.", "Overwhelmed"),
    ("Im so overwhelmed that I dont even know where to begin.", "Overwhelmed"),
    ("Im exhausted from feeling overwhelmed all the time.", "Overwhelmed"),
    ("Im overwhelmed and just want to escape from everything.", "Overwhelmed"),
    ("Its hard to function when I feel this overwhelmed.", "Overwhelmed"),
     ("The sheer volume of tasks makes me feel completely overwhelmed.", "Overwhelmed"),
    ("Trying to balance everything is leaving me overwhelmed.", "Overwhelmed"),
    ("Feeling overwhelmed by the demands of work and personal life.", "Overwhelmed"),
    ("I'm drowning in deadlines; it's overwhelming.", "Overwhelmed"),
    ("Every new responsibility adds to my overwhelming workload.", "Overwhelmed"),
    ("I feel like I'm constantly overwhelmed by expectations.", "Overwhelmed"),
    ("Struggling to prioritize amidst chaos is overwhelming.", "Overwhelmed"),
    ("Feeling overwhelmed is a constant battle I'm trying to manage.", "Overwhelmed"),
    ("I'm overwhelmed by the complexity of the situation.", "Overwhelmed"),
    ("The pressure to perform well leaves me feeling overwhelmed.", "Overwhelmed"),

    # Lonely (35 entries)
    ("I feel so lonely right now.", "Lonely"),
    ("Its hard being this lonely all the time.", "Lonely"),
    ("Im surrounded by people, but I still feel lonely.", "Lonely"),
    ("I feel isolated and incredibly lonely.", "Lonely"),
    ("No one seems to understand, and I feel lonely.", "Lonely"),
    ("Im lonely, and its hard to connect with anyone.", "Lonely"),
    ("The silence makes me feel even lonelier.", "Lonely"),
    ("Ive never felt this lonely in my life.", "Lonely"),
    ("I feel like no one cares, and Im lonely.", "Lonely"),
    ("Im craving company, but I feel so lonely.", "Lonely"),
    ("Even when Im with friends, I still feel lonely.", "Lonely"),
    ("Loneliness is becoming too familiar.", "Lonely"),
    ("I feel invisible and lonely.", "Lonely"),
    ("Im tired of feeling lonely all the time.", "Lonely"),
    ("I feel disconnected from everyone and lonely.", "Lonely"),
    ("I wish I had someone to talk to; I feel so lonely.", "Lonely"),
    ("The loneliness is eating away at me.", "Lonely"),
    ("Im lonely, and its making me sad.", "Lonely"),
    ("I feel abandoned and incredibly lonely.", "Lonely"),
    ("I dont want to be alone, but Im so lonely.", "Lonely"),
    ("Its hard to fight off this feeling of loneliness.", "Lonely"),
    ("Loneliness is all I know right now.", "Lonely"),
    ("I feel forgotten and lonely.", "Lonely"),
    ("I wish I could stop feeling this lonely.", "Lonely"),
    ("This empty feeling inside is loneliness.", "Lonely"),
    ("Loneliness hits hardest when surrounded by crowds.", "Lonely"),
    ("Feeling disconnected, I often find myself lonely.", "Lonely"),
    ("Navigating through life's challenges alone is lonely.", "Lonely"),
    ("I yearn for deeper connections; loneliness persists.", "Lonely"),
    ("In solitude, the silence amplifies feelings of loneliness.", "Lonely"),
    ("Loneliness creeps in despite efforts to stay engaged.", "Lonely"),
    ("Longing for companionship, I battle feelings of loneliness.", "Lonely"),
    ("Feeling isolated even when surrounded by acquaintances.", "Lonely"),
    ("Loneliness becomes overwhelming during quiet evenings.", "Lonely"),
    ("Finding solace in solitude, yet struggling with loneliness.", "Lonely"),

    # Relieved (35 entries)
    ("I feel so relieved that its finally over.", "Relieved"),
    ("Im relieved everything worked out in the end.", "Relieved"),
    ("I was worried, but now I feel relieved.", "Relieved"),
    ("Its such a relief to finally have some good news.", "Relieved"),
    ("Im feeling relieved after that stressful situation.", "Relieved"),
    ("I can finally breathe a sigh of relief.", "Relieved"),
    ("Im so relieved that it all went smoothly.", "Relieved"),
    ("The weight has been lifted, and I feel relieved.", "Relieved"),
    ("Im relieved that things are back to normal.", "Relieved"),
    ("I feel a sense of relief now that its resolved.", "Relieved"),
    ("Im relieved that I made it through that challenge.", "Relieved"),
    ("After all that stress, Im just relieved.", "Relieved"),
    ("Im relieved to know everything is okay now.", "Relieved"),
    ("Its such a relief that the worst is over.", "Relieved"),
    ("Im feeling really relieved after hearing the good news.", "Relieved"),
    ("Im relieved that things are finally falling into place.", "Relieved"),
    ("The relief I feel is indescribable.", "Relieved"),
    ("Im so relieved that it didnt turn out worse.", "Relieved"),
    ("I can finally relax and feel relieved.", "Relieved"),
    ("Im relieved to see everything working out.", "Relieved"),
    ("I was so anxious, but now Im feeling relieved.", "Relieved"),
    ("Im relieved that the outcome was better than expected.", "Relieved"),
    ("Im so relieved that its done.", "Relieved"),
    ("All that stress is gone, and Im feeling relieved.", "Relieved"),
    ("I feel a huge sense of relief right now.", "Relieved"),
    ("A sigh of relief washes over me as tension eases.", "Relieved"),
    ("Experiencing relief after a long period of uncertainty.", "Relieved"),
    ("Relief floods in as positive outcomes unfold.", "Relieved"),
    ("The relief of knowing everything turned out alright.", "Relieved"),
    ("Feeling relieved after facing and overcoming challenges.", "Relieved"),
    ("Relief follows the resolution of a troubling situation.", "Relieved"),
    ("A sense of relief envelops me as worries dissipate.", "Relieved"),
    ("Relief is palpable after waiting anxiously for news.", "Relieved"),
    ("Feeling a wave of relief after a period of stress.", "Relieved"),
    ("Relieved to have clarity after moments of confusion.", "Relieved"),

    # Hopeful (35 entries)
    ("Im feeling really hopeful about the future.", "Hopeful"),
    ("Things are looking up, and I feel hopeful.", "Hopeful"),
    ("Im hopeful that everything will work out.", "Hopeful"),
    ("Theres a sense of hope in the air.", "Hopeful"),
    ("I feel optimistic and hopeful right now.", "Hopeful"),
    ("Im filled with hope for whats to come.", "Hopeful"),
    ("Despite the challenges, Im still hopeful.", "Hopeful"),
    ("I feel hopeful that things will improve soon.", "Hopeful"),
    ("Im hopeful that tomorrow will be a better day.", "Hopeful"),
    ("Hope is keeping me going through tough times.", "Hopeful"),
    ("I can feel hope growing within me.", "Hopeful"),
    ("Im holding onto hope for a positive outcome.", "Hopeful"),
    ("Hope is whats getting me through this.", "Hopeful"),
    ("Im hopeful that change is on the horizon.", "Hopeful"),
    ("Theres a lot to be hopeful for right now.", "Hopeful"),
    ("Im filled with hope for the possibilities ahead.", "Hopeful"),
    ("Im hopeful that everything is falling into place.", "Hopeful"),
    ("Even though its tough, I still feel hopeful.", "Hopeful"),
    ("Im holding onto hope, and its keeping me strong.", "Hopeful"),
    ("Hope is lighting the way for me.", "Hopeful"),
    ("Im hopeful that the best is yet to come.", "Hopeful"),
    ("I feel hopeful that things will turn around.", "Hopeful"),
    ("Hope is the reason I havent given up yet.", "Hopeful"),
    ("I feel a renewed sense of hope today.", "Hopeful"),
    ("Im hopeful that the future holds great things.", "Hopeful"),
    ("Hope fills my heart, guiding me towards brighter days.", "Hopeful"),
    ("Feeling hopeful amidst setbacks and challenges.", "Hopeful"),
    ("Hopefulness fuels determination to persevere.", "Hopeful"),
    ("Embracing hope as a beacon in times of uncertainty.", "Hopeful"),
    ("I nurture hope to sustain optimism through adversity.", "Hopeful"),
    ("Hopefulness grows as opportunities unfold.", "Hopeful"),
    ("Finding hope in the support of loved ones.", "Hopeful"),
    ("Hope is a powerful force driving me forward.", "Hopeful"),
    ("Cultivating hopefulness in moments of doubt.", "Hopeful"),
    ("Embracing hope as I navigate through life's twists.", "Hopeful"),

    # Disappointed (35 entries)
    ("Im really disappointed by the outcome.", "Disappointed"),
    ("This situation didnt go as expected, and Im disappointed.", "Disappointed"),
    ("I feel let down and disappointed.", "Disappointed"),
    ("I had high hopes, but now Im just disappointed.", "Disappointed"),
    ("Im disappointed that things didnt turn out differently.", "Disappointed"),
    ("This didnt go the way I wanted, and Im disappointed.", "Disappointed"),
    ("Im feeling really let down and disappointed right now.", "Disappointed"),
    ("I expected more, and now Im disappointed.", "Disappointed"),
    ("Im disappointed in how this situation unfolded.", "Disappointed"),
    ("I thought things would go better, but Im disappointed.", "Disappointed"),
    ("Its hard to hide my disappointment right now.", "Disappointed"),
    ("I feel disheartened and disappointed.", "Disappointed"),
    ("I cant help but feel disappointed by this result.", "Disappointed"),
    ("This is not what I hoped for, and Im disappointed.", "Disappointed"),
    ("Im trying to stay positive, but Im really disappointed.", "Disappointed"),
    ("I feel like Ive been let down, and its disappointing.", "Disappointed"),
    ("Its disappointing that things didnt work out.", "Disappointed"),
    ("I had high expectations, but now Im disappointed.", "Disappointed"),
    ("This result has left me feeling disappointed.", "Disappointed"),
    ("I was hoping for more, but Im just disappointed.", "Disappointed"),
    ("This outcome is not what I wanted, and Im disappointed.", "Disappointed"),
    ("I feel disappointed that it didnt go as planned.", "Disappointed"),
    ("Im disappointed in how things turned out.", "Disappointed"),
    ("I was expecting better, but Im disappointed.", "Disappointed"),
    ("This disappointment is hard to shake.", "Disappointed"),
    ("I really thought this would work out; I'm disappointed.", "Disappointed"),
    ("It's tough to accept that things didn't go as planned; I'm disappointed.", "Disappointed"),
    ("I invested a lot of hope in this, and now I'm just disappointed.", "Disappointed"),
    ("I expected better from this situation; it's disappointing.", "Disappointed"),
    ("It feels like a letdown; I had such high expectations.", "Disappointed"),
    ("I'm trying to find the silver lining, but I'm disappointed.", "Disappointed"),
    ("This isn't the result I was aiming for, and it's disappointing.", "Disappointed"),
    ("It's hard to shake off this feeling of disappointment.", "Disappointed"),
    ("I hoped for a different outcome, but here we are; I'm disappointed.", "Disappointed"),
    ("It's disappointing to see things turn out this way.", "Disappointed"),

    # Ashamed (35 entries)
    ("I feel so ashamed of my actions.", "Ashamed"),
    ("I cant believe I did that; Im ashamed.", "Ashamed"),
    ("I feel embarrassed and ashamed right now.", "Ashamed"),
    ("Im ashamed of how I handled that situation.", "Ashamed"),
    ("This feeling of shame is overwhelming.", "Ashamed"),
    ("I wish I could take back what I did; Im ashamed.", "Ashamed"),
    ("I cant shake off this feeling of shame.", "Ashamed"),
    ("I feel ashamed of my choices.", "Ashamed"),
    ("I cant face anyone right now because Im ashamed.", "Ashamed"),
    ("I feel ashamed and want to hide.", "Ashamed"),
    ("This shame is eating me up inside.", "Ashamed"),
    ("I wish I could erase my mistakes; Im so ashamed.", "Ashamed"),
    ("I feel a deep sense of shame about what happened.", "Ashamed"),
    ("Im ashamed and dont know how to move forward.", "Ashamed"),
    ("I cant look at myself in the mirror because Im ashamed.", "Ashamed"),
    ("I feel a wave of shame wash over me.", "Ashamed"),
    ("Im ashamed to admit my mistakes.", "Ashamed"),
    ("This shame is suffocating.", "Ashamed"),
    ("I feel ashamed of who Ive become.", "Ashamed"),
    ("Im battling this feeling of shame daily.", "Ashamed"),
    ("I feel ashamed for letting others down.", "Ashamed"),
    ("I cant express how ashamed I feel right now.", "Ashamed"),
    ("I feel like Ive let myself down, and its shameful.", "Ashamed"),
    ("Im ashamed of my past decisions.", "Ashamed"),
    ("This shame is a heavy burden to carry.", "Ashamed"),
    ("I can't believe I let myself act like that; I feel ashamed.", "Ashamed"),
("I wish I could rewind time; this shame is overwhelming.", "Ashamed"),
("I can't help but feel embarrassed about my choices.", "Ashamed"),
("This shame is like a weight on my shoulders; I want to hide.", "Ashamed"),
("I'm ashamed of how I treated someone I care about.", "Ashamed"),
("I feel like I've disappointed everyone, and it's shameful.", "Ashamed"),
("I can't shake off this feeling of shame; it lingers.", "Ashamed"),
("I wish I could undo my actions; I'm so ashamed.", "Ashamed"),
("This moment of weakness has left me feeling so ashamed.", "Ashamed"),
("I'm ashamed to face the people I love right now.", "Ashamed"),

    # Curious (35 entries)
    ("Im curious about what lies ahead.", "Curious"),
    ("I have so many questions; Im feeling curious.", "Curious"),
    ("I feel a sense of curiosity about the world.", "Curious"),
    ("Im curious to learn more about this topic.", "Curious"),
    ("My curiosity is piqued by this situation.", "Curious"),
    ("Im curious to know how things work.", "Curious"),
    ("I feel a strong desire to explore; Im curious.", "Curious"),
    ("Im curious about the different possibilities.", "Curious"),
    ("This has sparked my curiosity.", "Curious"),
    ("I cant help but feel curious about whats next.", "Curious"),
    ("Im eager to find out more; Im curious.", "Curious"),
    ("I feel an insatiable curiosity about life.", "Curious"),
    ("Im curious about how others see the world.", "Curious"),
    ("My curiosity is driving me to explore new things.", "Curious"),
    ("Im curious about different cultures and experiences.", "Curious"),
    ("I feel a burning curiosity inside me.", "Curious"),
    ("I cant ignore my curiosity any longer.", "Curious"),
    ("Im curious about the unknown.", "Curious"),
    ("This situation has made me very curious.", "Curious"),
    ("Im curious to know more about your experiences.", "Curious"),
    ("I feel compelled to investigate; Im curious.", "Curious"),
    ("Im curious about how things came to be.", "Curious"),
    ("My curiosity is leading me to new adventures.", "Curious"),
    ("Im curious about what others think.", "Curious"),
    ("I want to satisfy my curiosity about this matter.", "Curious"),
    ("I'm curious about how this all started.", "Curious"),
("There's so much to learn, and I'm feeling curious.", "Curious"),
("This situation has sparked a lot of questions in my mind.", "Curious"),
("I wonder what other possibilities are out there; I'm curious.", "Curious"),
("I'm eager to discover more about this subject.", "Curious"),
("I feel a pull towards exploring new ideas; I'm curious.", "Curious"),
("I can't help but want to know more; my curiosity is strong.", "Curious"),
("This has piqued my interest; I'm feeling very curious.", "Curious"),
("I'm curious about the stories behind people's lives.", "Curious"),
("I want to explore different perspectives; I'm curious.", "Curious"),

    # Excited (35 entries)
    ("I'm so excited about whats coming!", "Excited"),
    ("I cant contain my excitement right now.", "Excited"),
    ("I feel a rush of excitement as I think about the future.", "Excited"),
    ("I'm excited to start this new chapter.", "Excited"),
    ("This is such an exciting opportunity!", "Excited"),
    ("I'm buzzing with excitement!", "Excited"),
    ("I can hardly wait; Im so excited!", "Excited"),
    ("I feel a sense of excitement about whats ahead.", "Excited"),
    ("I'm excited to share this news!", "Excited"),
    ("I cant help but feel excited about the possibilities.", "Excited"),
    ("I'm excited to see what happens next.", "Excited"),
    ("This is the most exciting moment of my life!", "Excited"),
    ("Im excited to learn something new.", "Excited"),
    ("I feel a wave of excitement wash over me.", "Excited"),
    ("I'm excited to meet new people and make memories.", "Excited"),
    ("I can feel the excitement building inside me.", "Excited"),
    ("Im excited for the journey ahead.", "Excited"),
    ("This is an exciting time for me!", "Excited"),
    ("I'm excited to explore new ideas.", "Excited"),
    ("I'm filled with excitement about this project.", "Excited"),
    ("I cant wait to dive into this; Im so excited!", "Excited"),
    ("I'm thrilled and excited about whats in store.", "Excited"),
    ("I feel so excited to be part of this!", "Excited"),
    ("I'm excited to finally make my dreams come true.", "Excited"),
    ("I'm bubbling with excitement over this news!", "Excited"),
    ("I'm excited about the adventures that await me.", "Excited"),
    ("I can't wait to dive into this project; I'm so excited!", "Excited"),
("This is going to be an amazing adventure; I feel excited!", "Excited"),
("I'm bursting with excitement at the thought of what's to come!", "Excited"),
("I feel a thrill of excitement thinking about the opportunities ahead.", "Excited"),
("This news has me buzzing with excitement!", "Excited"),
("I can hardly contain my excitement for what's in store!", "Excited"),
("I feel so energized and excited about starting this journey.", "Excited"),
("This is a chance of a lifetime, and I'm thrilled!", "Excited"),
("I can't help but smile; I'm so excited for what's next!", "Excited"),
("I'm excited to see where this path leads me!", "Excited"),

    # Empowered (35 entries)
    ("I feel empowered to take charge of my life.", "Empowered"),
    ("Im ready to make a difference; I feel empowered.", "Empowered"),
    ("I feel a sense of empowerment in my decisions.", "Empowered"),
    ("Im empowered to pursue my dreams.", "Empowered"),
    ("This experience has made me feel empowered.", "Empowered"),
    ("I feel strong and empowered right now.", "Empowered"),
    ("Im empowered to speak my truth.", "Empowered"),
    ("I feel a surge of empowerment flowing through me.", "Empowered"),
    ("Im empowered to create positive change.", "Empowered"),
    ("This journey has left me feeling empowered.", "Empowered"),
    ("Im empowered to take risks and grow.", "Empowered"),
    ("I feel confident and empowered to move forward.", "Empowered"),
    ("Im empowered by the support of those around me.", "Empowered"),
    ("I can feel my empowerment growing each day.", "Empowered"),
    ("Im empowered to stand up for myself.", "Empowered"),
    ("I feel capable and empowered to achieve my goals.", "Empowered"),
    ("This has been an empowering experience for me.", "Empowered"),
    ("I feel empowered to embrace my uniqueness.", "Empowered"),
    ("Im ready to take on the world; I feel empowered.", "Empowered"),
    ("I feel empowered to follow my passions.", "Empowered"),
    ("This has ignited a sense of empowerment within me.", "Empowered"),
    ("Im empowered to make choices that align with my values.", "Empowered"),
    ("I feel empowered to express myself freely.", "Empowered"),
    ("Im empowered to pursue my aspirations.", "Empowered"),
    ("I feel strong and empowered in my beliefs.", "Empowered"),
    ("I feel ready to take on any challenge; I'm empowered!", "Empowered"),
("This experience has filled me with a sense of empowerment.", "Empowered"),
("I'm stepping into my power; I feel truly empowered.", "Empowered"),
("I feel capable and empowered to chase my dreams.", "Empowered"),
("I'm inspired to make bold choices; I feel empowered.", "Empowered"),
("I can feel my strength growing; I'm empowered to act.", "Empowered"),
("This journey has ignited my sense of empowerment.", "Empowered"),
("I feel like I can achieve anything; I'm so empowered!", "Empowered"),
("This has given me the courage to express my true self; I feel empowered.", "Empowered"),
("I feel a wave of empowerment every time I stand up for myself.", "Empowered"),

    # Inspired (35 entries)
    ("I feel inspired by the stories Ive heard.", "Inspired"),
    ("This moment has truly inspired me.", "Inspired"),
    ("Im feeling a surge of inspiration right now.", "Inspired"),
    ("I feel inspired to create and innovate.", "Inspired"),
    ("Im inspired by the resilience of others.", "Inspired"),
    ("This experience has left me feeling inspired.", "Inspired"),
    ("Im inspired to make a positive impact.", "Inspired"),
    ("I feel motivated and inspired to chase my dreams.", "Inspired"),
    ("Im inspired by the beauty around me.", "Inspired"),
    ("This has sparked a wave of inspiration within me.", "Inspired"),
    ("I feel inspired to take action.", "Inspired"),
    ("Im inspired by the potential for change.", "Inspired"),
    ("I feel a sense of inspiration in my heart.", "Inspired"),
    ("Im inspired to help others achieve their goals.", "Inspired"),
    ("I feel empowered and inspired to make a difference.", "Inspired"),
    ("This has ignited my passion and inspired me.", "Inspired"),
    ("Im feeling incredibly inspired to share my journey.", "Inspired"),
    ("I feel inspired to express my creativity.", "Inspired"),
    ("Im inspired by the challenges Ive overcome.", "Inspired"),
    ("I feel a wave of inspiration guiding me.", "Inspired"),
    ("This has inspired me to think outside the box.", "Inspired"),
    ("Im inspired by the possibilities that lie ahead.", "Inspired"),
    ("I feel motivated and inspired to pursue my goals.", "Inspired"),
    ("Im inspired to live life to the fullest.", "Inspired"),
    ("I feel a deep sense of inspiration from within.", "Inspired"),
    ("I feel a spark of inspiration lighting my path.", "Inspired"),
("This moment has filled me with so much inspiration!", "Inspired"),
("I'm inspired to make a positive change in my life.", "Inspired"),
("The stories I've heard have truly inspired me to act.", "Inspired"),
("I feel driven to create something beautiful; I'm inspired.", "Inspired"),
("This experience has awakened a sense of inspiration in me.", "Inspired"),
("I can feel my creativity flowing; I'm so inspired!", "Inspired"),
("I'm inspired to help others because of what I've learned.", "Inspired"),
("This has opened my eyes to new possibilities; I feel inspired.", "Inspired"),
("I feel a strong desire to share my inspiration with the world.", "Inspired"),

    # Fearful (35 entries)
    ("Im feeling fearful about what might happen.", "Fearful"),
    ("This situation is making me feel anxious and fearful.", "Fearful"),
    ("I cant shake off this feeling of fear.", "Fearful"),
    ("I feel a sense of fear creeping in.", "Fearful"),
    ("Im fearful of the unknown.", "Fearful"),
    ("I feel overwhelmed by my fears.", "Fearful"),
    ("This fear is paralyzing me.", "Fearful"),
    ("Im fearful about what lies ahead.", "Fearful"),
    ("I feel fear rising within me.", "Fearful"),
    ("Im scared and feeling fearful right now.", "Fearful"),
    ("This uncertainty is making me feel fearful.", "Fearful"),
    ("Im fearful of making the wrong decision.", "Fearful"),
    ("I feel a knot of fear in my stomach.", "Fearful"),
    ("This fear is weighing heavily on me.", "Fearful"),
    ("Im fearful of how things might turn out.", "Fearful"),
    ("I can feel my heart racing with fear.", "Fearful"),
    ("Im fearful about the consequences of my actions.", "Fearful"),
    ("This fear is hard to confront.", "Fearful"),
    ("I feel a cloud of fear hanging over me.", "Fearful"),
    ("Im fearful of the challenges ahead.", "Fearful"),
    ("This fear is consuming my thoughts.", "Fearful"),
    ("Im fearful of what others will think.", "Fearful"),
    ("I feel paralyzed by my fears.", "Fearful"),
    ("Im fearful of failing.", "Fearful"),
    ("I cant ignore this feeling of fear any longer.", "Fearful"),
    ("I feel fearful about stepping outside my comfort zone.", "Fearful"),
    ("I can't shake off this feeling of fear; it's overwhelming.", "Fearful"),
("I'm scared of what might happen next; I feel fearful.", "Fearful"),
("This uncertainty is causing a lot of anxiety; I'm fearful.", "Fearful"),
("I feel fear creeping in about the choices I've made.", "Fearful"),
("I'm feeling fearful about facing the consequences.", "Fearful"),
("I can't help but feel anxious about the future; I'm fearful.", "Fearful"),
("This situation is filling me with a sense of dread; I feel fearful.", "Fearful"),
("I'm scared to take the next step; I feel fearful.", "Fearful"),
("This fear is clouding my judgment; I'm feeling overwhelmed.", "Fearful"),
("I wish I could overcome this fear; it's holding me back.", "Fearful"),
]

# Separate the texts and labels from the train_data
train_texts = [text for text, label in train_data]
train_labels = [[label] for text, label in train_data]

# Preprocessing using simple Bag of Words (can expand to TF-IDF later)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(train_texts)

# Convert labels to binary form
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_labels)

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Make predictions on test data
y_pred = clf.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E2A - Emotion to Action</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        :root {
            --primary-color: #4f46e5;
            --secondary-color: #e0e7ff;
            --text-color: #1f2937;
            --background-color: #f3f4f6;
            --chat-bg: #ffffff;
            --sidebar-bg: #f9fafb;
            --sidebar-hover: #e5e7eb;
        }

        body, html {
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            height: 100%;
            background-color: var(--background-color);
            color: var(--text-color);
        }

        .container {
            display: flex;
            min-height: 100vh;
        }

        .sidebar {
            width: 260px;
            background-color: var(--sidebar-bg);
            padding: 1rem;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #e5e7eb;
        }

        .sidebar-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .new-chat-btn {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .new-chat-btn:hover {
            background-color: #3c3799;
        }

        .chat-history {
            flex-grow: 1;
            overflow-y: auto;
        }

        .chat-item {
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s ease;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .chat-item:hover {
            background-color: var(--sidebar-hover);
        }

        .chat-item.active {
            background-color: var(--secondary-color);
            font-weight: 600;
        }

        .sidebar-footer {
            margin-top: auto;
        }

        .delete-all-btn {
            background: none;
            border: none;
            color: #ef4444;
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }

        .delete-all-btn:hover {
            background-color: #fee2e2;
        }

        .main-content {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: var(--chat-bg);
            border-bottom: 1px solid #e5e7eb;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-color);
            display: flex;
            align-items: center;
        }

        .beta-label {
            font-size: 0.7rem;
            background-color: var(--primary-color);
            color: white;
            padding: 2px 6px;
            border-radius: 10px;
            margin-left: 8px;
            transform: translateY(-8px) rotate(-10deg);
        }

        .model-select {
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #d1d5db;
            background-color: var(--chat-bg);
            color: var(--text-color);
        }

        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 2rem;
            overflow-y: auto;
        }

        #welcome-screen {
            text-align: center;
        }

        h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }

        p {
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto 2rem;
        }

        #start-chat {
            padding: 0.8rem 2rem;
            font-size: 1.1rem;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        #start-chat:hover {
            background-color: #3c3799;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        #chat-container {
            display: none;
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
        }

        #chat-messages {
            min-height: 300px;
            max-height: calc(100vh - 250px);
            overflow-y: auto;
            padding: 20px;
            background-color: var(--chat-bg);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .message {
            margin-bottom: 1rem;
            padding: 0.8rem 1.2rem;
            border-radius: 18px;
            max-width: 80%;
        }

        .user-message {
            background-color: var(--primary-color);
            color: white;
            align-self: flex-end;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }

        .bot-message {
            background-color: var(--secondary-color);
            color: var(--text-color);
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }

        #user-input {
            display: flex;
            padding: 15px;
            background-color: var(--chat-bg);
            border-top: 1px solid #e5e7eb;
            margin-top: 1rem;
        }

        #user-input input {
            flex: 1;
            padding: 0.8rem 1.2rem;
            border: 1px solid #d1d5db;
            border-radius: 25px;
            margin-right: 0.5rem;
            font-size: 1rem;
        }

        #user-input button {
            padding: 0.8rem 1.5rem;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        #user-input button:hover {
            background-color: #3c3799;
        }

        .beta-disclaimer {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 0.5rem;
            text-align: center;
        }

        .dot-animation {
            font-size: 1rem;
            color: var(--text-color);
            text-align: left;
            margin-top: 0.2rem;
        }

        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 1; }
        }

        .dot-animation span {
            display: inline-block;
            margin: 0 2px;
            animation: blink 1s infinite;
        }

        span:nth-child(2) { animation-delay: 0.3s; }
        span:nth-child(3) { animation-delay: 0.6s; }

        footer {
            padding: 1rem;
            text-align: center;
            font-size: 0.9rem;
            color: #6b7280;
            background-color: var(--chat-bg);
            border-top: 1px solid #e5e7eb;
        }

        #name-input {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: var(--chat-bg);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }

        #name-input input {
            width: 100%;
            padding: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #d1d5db;
            border-radius: 5px;
        }

        #name-input button {
            width: 100%;
            padding: 0.5rem;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        #name-input button:hover {
            background-color: #3c3799;
        }

        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }

            .sidebar {
                width: 100%;
                order: 2;
            }

            .main-content {
                order: 1;
            }

            h1 {
                font-size: 2rem;
            }

            p {
                font-size: 1rem;
            }
        }
    </style>
</head>
<body>
   <div class="container">
    <div class="sidebar">
        <div class="sidebar-header">
            <button class="new-chat-btn">New Chat</button>
        </div>
        <div class="chat-history"></div>
        <div class="sidebar-footer">
            <button class="delete-all-btn">Delete All Chats</button>
        </div>
    </div>
    <div class="main-content">
        <header>
            <div class="logo">
                E2A AI
                <span class="beta-label">BETA</span>
            </div>
            <select class="model-select">
                <option value="e2a-beta">E2A Beta</option>
                <option value="e2a-v8.0.1">E2A v8.0.1</option>
            </select>
        </header>
        <main>
            <div id="welcome-screen">
                <h1>Welcome to E2A - Emotion to Action</h1>
                <p>Discover how your emotions can guide your actions. Start a conversation with our AI to explore your feelings and get personalized suggestions.</p>
                <button id="start-chat">Start Chat</button>
            </div>
            <div id="chat-container">
                <div id="chat-messages"></div>
                <div id="user-input">
                    <input type="text" id="user-message" placeholder="How are you feeling?">
                    <button id="send-button">Send</button>
                </div>
                <p class="beta-disclaimer">These responses are generated by E2A AI, and they may occasionally contain mistakes or inaccuracies.</p>
            </div>
        </main>
        <footer>
            <p>Developed by PyGen Intelligence, Ameer Hamza Khan</p>
        </footer>
    </div>
</div>
<div id="name-input">
    <h2>What is your name?</h2>
    <input type="text" id="user-name" placeholder="Enter your name">
    <button id="submit-name">Submit</button>
</div>

<script>
    document.addEventListener('DOMContentLoaded', function() {
        const startChatButton = document.getElementById('start-chat');
        const chatContainer = document.getElementById('chat-container');
        const welcomeScreen = document.getElementById('welcome-screen');
        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-message');
        const sendButton = document.getElementById('send-button');
        const newChatButton = document.querySelector('.new-chat-btn');
        const chatHistory = document.querySelector('.chat-history');
        const deleteAllButton = document.querySelector('.delete-all-btn');
        const nameInput = document.getElementById('name-input');
        const submitNameButton = document.getElementById('submit-name');
        const modelSelect = document.querySelector('.model-select');

        let currentChatId = null;
        let userName = localStorage.getItem('userName') || '';
        let chats = JSON.parse(localStorage.getItem('chats')) || {};
        let currentModel = 'e2a-beta'; // Default model

        function showNameInput() {
            nameInput.style.display = 'block';
        }

        function hideNameInput() {
            nameInput.style.display = 'none';
        }

        submitNameButton.addEventListener('click', function() {
            userName = document.getElementById('user-name').value.trim();
            if (userName) {
                localStorage.setItem('userName', userName);
                hideNameInput();
                startNewChat();
            }
        });

        function startNewChat() {
            currentChatId = Date.now().toString();
            chats[currentChatId] = [];
            chatMessages.innerHTML = '';
            userInput.value = '';
            welcomeScreen.style.display = 'none';
            chatContainer.style.display = 'block';
            addBotMessage(`Hello${userName ? ' ' + userName : ''}! How are you feeling today?`);
            updateChatHistory();
            saveChatToLocalStorage();
        }

        function updateChatHistory() {
            chatHistory.innerHTML = '';
            Object.keys(chats).reverse().forEach(chatId => {
                const chatItem = document.createElement('div');
                chatItem.classList.add('chat-item');
                chatItem.textContent = chats[chatId][0]?.content.substring(0, 30) + '...' || 'New Chat';
                chatItem.dataset.chatId = chatId;
                if (chatId === currentChatId) {
                    chatItem.classList.add('active');
                }
                chatHistory.appendChild(chatItem);

                chatItem.addEventListener('click', function() {
                    loadChat(this.dataset.chatId);
                });
            });
        }

        function loadChat(chatId) {
            currentChatId = chatId;
            chatMessages.innerHTML = '';
            chats[chatId].forEach(message => {
                if (message.type === 'user') {
                    addUserMessage(message.content);
                } else {
                    addBotMessage(message.content);
                }
            });
            updateChatHistory();
        }

        function saveChatToLocalStorage() {
            localStorage.setItem('chats', JSON.stringify(chats));
        }

        startChatButton.addEventListener('click', function() {
            if (!userName) {
                showNameInput();
            } else {
                startNewChat();
            }
        });

        newChatButton.addEventListener('click', startNewChat);

        deleteAllButton.addEventListener('click', function() {
            if (confirm('Are you sure you want to delete all chats?')) {
                chats = {};
                localStorage.removeItem('chats');
                chatHistory.innerHTML = '';
                welcomeScreen.style.display = 'block';
                chatContainer.style.display = 'none';
            }
        });

        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        function sendMessage() {
    const message = userInput.value.trim();
    const selectedVersion = modelSelect.value;  // Get the selected model version
    if (message) {
        addUserMessage(message);
        userInput.value = '';
        addBotLoading();

        // Make an API call to your Flask backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message, version: selectedVersion })  // Include the selected version in the request
        })
        .then(response => response.json())
        .then(data => {
            removeBotLoading();
            if (data.response) {
                addBotMessage(data.response);  // Handle simple text responses
            } else if (data.emotions && data.suggestions) {
                const emotionText = `I sense that you're feeling ${data.emotions.join(', ')}. `;
                const suggestionText = `Here's a suggestion: ${data.suggestions.join(' ')}`;
                addBotMessage(emotionText + suggestionText);  // Handle emotions and suggestions
            } else {
                addBotMessage("Sorry, I couldn't understand your message.");
            }
        })
        .catch(error => {
            removeBotLoading();
            addBotMessage("Sorry, there was an error processing your request.");
            console.error('Error:', error);
        });
    }
}

        function addUserMessage(message) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', 'user-message');
            messageDiv.textContent = message;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            chats[currentChatId].push({type: 'user', content: message});
            saveChatToLocalStorage();
            updateChatHistory();
        }

        function addBotMessage(message) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', 'bot-message');
            messageDiv.textContent = message;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            chats[currentChatId].push({type: 'bot', content: message});
            saveChatToLocalStorage();
            updateChatHistory();
        }

        function addBotLoading() {
            const loadingDiv = document.createElement('div');
            loadingDiv.classList.add('message', 'bot-message', 'dot-animation');
            loadingDiv.innerHTML = 'Analyzing <span>.</span><span>.</span><span>.</span>';
            loadingDiv.id = 'loading';
            chatMessages.appendChild(loadingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function removeBotLoading() {
            const loadingDiv = document.getElementById('loading');
            if (loadingDiv) {
                loadingDiv.remove();
            }
        }

        modelSelect.addEventListener('change', function() {
            currentModel = this.value;
            if (currentModel === 'e2a-v8.0.1') {
                alert('Now running E2A v.8.0.1.');
            }
        });

        // Initialize the chat history on page load
        updateChatHistory();
    });
</script>
"""
def e2a_beta(user_input):
    # Function to analyze emotions (already in Beta)
    def analyze_emotions(text):
        words = text.lower().split()
        emotions_detected = [emotion_adjectives[word] for word in words if word in emotion_adjectives]
        return emotions_detected

    # Function to suggest actions based on emotions
    def suggest_action(emotions):
        suggestions = []
        for emotion in emotions:
            suggestion = get_suggestion(emotion)
            suggestions.append(suggestion)
        return suggestions

    emotions = analyze_emotions(user_input)
    
    if emotions:
        actions = suggest_action(emotions)
        response = f"I sense that you're feeling {', '.join(emotions)}. Here's a suggestion: {' '.join(actions)}"
    else:
        # If no emotions detected, proceed with model prediction instead of fallback message
        X_user = vectorizer.transform([user_input])
        predicted_emotions = mlb.inverse_transform(clf.predict(X_user))[0]

        if predicted_emotions:
            actions = suggest_action(predicted_emotions)
            response = f"Based on your message, you might be feeling {', '.join(predicted_emotions)}. Here's a suggestion: {' '.join(actions)}"
        else:
            # Fallback when no emotions are detected and no prediction is possible
            response = "I couldn't detect any emotions, but here's something to think about: Stay positive and take small steps towards your goals."
    
    return {'response': response}


def e2a_v8_0_1(input_text):
    # Function to analyze emotions from the input text
    def analyze_emotions(text):
        words = text.lower().split()
        emotions_detected = [emotion_adjectives[word] for word in words if word in emotion_adjectives]
        return emotions_detected

    # Function to suggest actions based on detected emotions
    def suggest_action(emotions):
        suggestions = []
        for emotion in emotions:
            suggestion = get_suggestion(emotion)
            suggestions.append(suggestion)
        return suggestions

    # Function to respond to common queries or use model predictions
    def respond_to_query(user_input):
        for i, question in enumerate(questions):
            if user_input.lower() == question.lower():
                return answers[i]

        # If no exact match, use similarity comparison
        similarities = []
        for question in questions:
            similarity = cosine_similarity(vectorizer.transform([user_input.lower()]), vectorizer.transform([question.lower()]))[0][0]
            similarities.append((question, similarity))

        # Sort by similarity, return the closest match if above a threshold
        similarities.sort(key=lambda x: x[1], reverse=True)
        if similarities[0][1] > 0.5:  # Similarity threshold
            return answers[questions.index(similarities[0][0])]

        return None

    # Process the user input
    response = respond_to_query(input_text)
    if response:
        return {'response': response}
    else:
        emotions = analyze_emotions(input_text)
        if emotions:
            suggestions = suggest_action(emotions)
            return {'emotions': emotions, 'suggestions': suggestions}
        else:
            return {'response': "I couldn't detect any specific emotions. Could you describe your feelings more?"}


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '')  # Ensure default empty string
        selected_version = data.get('version', 'e2a-beta')  # Default to beta version

        # Ensure that user input is not empty
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        # Process based on the selected version
        if selected_version == 'e2a-beta':
            response = e2a_beta(user_input)
        elif selected_version == 'e2a-v8.0.1':
            response = e2a_v8_0_1(user_input)
        else:
            return jsonify({'error': 'Invalid version selected'}), 400

        return jsonify(response)
    
    except Exception as e:
        # Log error and return a meaningful response
        app.logger.error(f"Exception in /chat endpoint: {e}")
        return jsonify({'error': 'An internal error occurred while processing your request'}), 500

if __name__ == '__main__':
    app.run()
