from flask import Flask, request, jsonify
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

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
        "Identify what’s frustrating you specifically.",
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
        "Take a quiet moment to reflect on what’s going well.",
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
    "I’m mourning the loss of something important", 
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
    "I’m inspired to take on new challenges", 
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
    "I’m enjoying this moment of tranquility", 
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
    "I’m feeling on edge about my performance", 
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
                    <option value="e2a-v8.0.1" disabled>E2A v.8.0.1 (Coming Soon)</option>
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
                    <p class="beta-disclaimer">This is a beta version. Responses may not always be accurate.</p>
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
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            removeBotLoading();
            addBotMessage(data.response);
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
                if (this.value === 'e2a-v8.0.1') {
                    alert('This model is coming soon!');
                    this.value = 'e2a-beta';
                }
            });

            // Initialize the chat history on page load
            updateChatHistory();
        });
    </script>
</body>
</html>"""

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    emotions = analyze_emotions(user_input)
    
    if emotions:
        actions = suggest_action(emotions)
        response = f"I sense that you're feeling {', '.join(emotions)}. Here's a suggestion: {' '.join(actions)}"
    else:
        # If no emotions are detected, use the trained model to predict
        X_user = vectorizer.transform([user_input])
        predicted_emotions = mlb.inverse_transform(clf.predict(X_user))[0]
        
        if predicted_emotions:
            actions = suggest_action(predicted_emotions)
            response = f"Based on your message, you might be feeling {', '.join(predicted_emotions)}. Here's a suggestion: {' '.join(actions)}"
        else:
            response = "I'm not sure how you're feeling. Can you tell me more about your emotions?"

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run()
