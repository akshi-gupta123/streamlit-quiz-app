import streamlit as st
import pandas as pd
import time
from datetime import datetime
import random
import os

# Page configuration
st.set_page_config(
    page_title="AI Knowledge Quiz",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Main background and styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Question card styling */
    .question-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 0.8rem 0;
    }
    
    /* Timer styling */
    .timer {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #764ba2;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .timer.warning {
        color: #ff6b6b;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Progress bar styling */
    .progress-text {
        text-align: center;
        font-size: 1.2rem;
        color: white;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6E8CFB);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .stButton>button:disabled {
        background: #a0aec0;
        cursor: not-allowed;
        transform: none;
    }
    
    .stButton>button:disabled:hover {
        transform: none;
        box-shadow: none;
    }
    
    /* Radio button styling */
    .stRadio > label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Title styling */
    h1 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #2d3748;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Result card */
    .result-card {
        background: white;
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .score-display {
        font-size: 4rem;
        font-weight: bold;
        color: #667eea;
        margin: 2rem 0;
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* Email requirement text */
    .email-requirement {
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# Load quiz data
@st.cache_data
def load_quiz_data():
    data = {
        'Category': ['AI History', 'AI Concepts', 'AI Facts', 'AI in Finance', 'AI Concepts', 'AI History', 
                     'AI in Finance', 'Deep Learning', 'AI Concepts', 'AI History', 'Machine Learning', 
                     'AI Facts', 'AI Concepts', 'AI Facts', 'AI Facts', 'AI Facts', 'AI Usage', 'AI Facts', 
                     'AI Usage', 'AI Facts', 'AI Usage', 'AI Facts', 'AI Facts', 'AI Concepts', 'AI Concepts', 
                     'AI Concepts', 'AI History', 'AI History', 'AI History', 'AI History', 'AI Concepts', 
                     'AI History', 'AI History', 'AI Concepts', 'AI History', 'AI History', 'AI History', 
                     'AI History', 'AI History', 'AI History', 'AI Concepts', 'AI Concepts', 'AI Concepts'],
        'Question': [
            "Who coined the term 'Artificial Intelligence' at the Dartmouth Conference?",
            "What does 'hallucination' mean in the context of Large Language Models?",
            "How old is the term 'Artificial Intelligence'?",
            "What does 'alpha' refer to in quantitative finance and AI trading?",
            "What is 'overfitting' in machine learning models?",
            "What was the name of the first AI program to defeat a world chess champion?",
            "Which AI technique is most commonly used for fraud detection in financial transactions?",
            "What architecture revolutionized NLP and led to models like BERT and GPT?",
            "What is 'transfer learning' in machine learning?",
            "In what year did AlphaGo defeat Lee Sedol in the game of Go?",
            "What does 'RAG' stand for in the context of AI systems?",
            "How much electricity does it take to train a GPT-4 scale model?",
            "What is 'gradient descent' used for in neural networks?",
            "How many GPUs were approximately used to train GPT-3?",
            "What percentage of Fortune 500 companies were using AI in some capacity by 2024?",
            "How much data is ChatGPT estimated to have been trained on?",
            "By 2024, what percentage of customer service interactions were handled by AI chatbots?",
            "How long did it take ChatGPT to reach 100 million users?",
            "What percentage of code on GitHub is estimated to be AI-assisted by 2024?",
            "How many ChatGPT queries are processed daily (as of 2024)?",
            "What percentage of financial institutions use AI for fraud detection?",
            "How much is the global AI market projected to be worth by 2030?",
            "What was the estimated cost to train GPT-4?",
            "How do large language models like GPT learn to generate text?",
            "What does 'GPT' stand for in ChatGPT?",
            "What is 'fine-tuning' in the context of AI models?",
            "At which conference was the term 'Artificial Intelligence' first introduced?",
            "What was the primary reason for the first AI winter (1970s)?",
            "Which company created Watson, the AI that won on Jeopardy in 2011?",
            "What was the name of the first self-learning neural network, created in 1958?",
            "Which of the following tasks is still very hard for AI?",
            "Which AI approach dominated in the 1950s-1980s, focusing on logic and rules?",
            "Which AI system first used the concept of 'knowledge representation'?",
            "Which AI concept involves systems that can explain their reasoning in human terms?",
            "Which AI achievement is AlphaFold famous for?",
            "Which paper is considered the 'birth certificate' of AI?",
            "Which AI milestone was achieved by OpenAI in 2022?",
            "Which AI system by Google DeepMind defeated the world Go champion Lee Sedol in 2016?",
            "Which AI system defeated Garry Kasparov, the world chess champion, in 1997?",
            "Which AI program was the first to play chess?",
            "Which of the following is NOT a type of AI?",
            "Which AI concept involves mimicking natural selection to optimize solutions?",
            "Which AI concept powers self-driving cars?"
        ],
        'Option_A': ['Alan Turing', 'When the model processes visual data', "About 30 years old (1990s)", 
                     'The first version of an AI model', 'When a model performs too well on training data but poorly on new data',
                     'AlphaGo', 'Reinforcement Learning', 'Convolutional Neural Networks', 'Moving data between servers',
                     '2012', 'Rapid Algorithm Generation', 'Equivalent to powering 10 homes for a year',
                     'Reducing model size', 'Around 100 GPUs', 'Around 25%', '100 gigabytes', '15%', '2 weeks',
                     '10%', '10 million', '30%', '$100 billion', '$5 million', 'By memorizing entire books and articles',
                     'General Purpose Technology', 'Adjusting the model\'s speed', 'Dartmouth Conference',
                     'Lack of funding', 'Google', 'ELIZA', 'Playing chess', 'Neural Networks', 'SHRDLU',
                     'Expert Systems', 'Translating languages', 'Computing Machinery and Intelligence (Turing, 1950)',
                     'Self-driving cars', 'Deep Blue', 'Deep Thought', 'Mac Hack', 'Narrow AI',
                     'Reinforcement Learning', 'Computer Vision + Reinforcement Learning'],
        'Option_B': ['John McCarthy', 'When the model generates false or nonsensical information confidently',
                     'About 50 years old (1970s)', 'Excess returns above a benchmark', 
                     'When a model is too large to deploy', 'Deep Blue', 
                     'Anomaly Detection using unsupervised learning', 'Recurrent Neural Networks',
                     'Using a pre-trained model as a starting point for a new task', '2016',
                     'Retrieval-Augmented Generation', 'Equivalent to powering 100 homes for a year',
                     'Optimizing model parameters to minimize error', 'Around 1,000 GPUs', 'Around 50%',
                     '1 terabyte', '35%', '2 months', '25%', '100 million', '50%', '$500 billion',
                     '$25 million', 'By predicting the next word in a sequence', 'Generative Pre-trained Transformer',
                     'Training a pre-trained model on specific data for specialized tasks', 'Turing Symposium',
                     'Too much hype', 'IBM', 'Perceptron', 'Driving a car on highways', 'Symbolic AI',
                     'Watson', 'Neural Networks', 'Predicting protein structures',
                     'Perceptrons (Minsky & Papert, 1969)', 'AlphaFold', 'Watson', 'Watson', 'Deep Blue',
                     'General AI', 'Evolutionary Algorithms', 'NLP + Robotics'],
        'Option_C': ['Marvin Minsky', 'When the model becomes confused by ambiguous prompts',
                     'About 70 years old (1950s)', 'The learning rate in neural networks',
                     'When a model takes too long to train', 'Watson', 'Image Classification',
                     'Transformer', 'Converting models between programming languages', '2018',
                     'Random Access Gateway', 'Equivalent to powering 1,000+ homes for a year',
                     'Increasing processing speed', 'Around 10,000 GPUs', 'Around 75%', '45 terabytes',
                     '55%', '6 months', '40%', '500 million', '70%', '$1 trillion', '$100 million',
                     'By copying responses from a database', 'Global Processing Tool',
                     'Reducing the model\'s size', 'MIT AI Lab Meet', 'Ethical concerns', 'Microsoft',
                     'Hopfield Network', 'Understanding sarcasm and jokes in conversations',
                     'Genetic Algorithms', 'Deep Blue', 'Reinforcement Learning', 'Playing video games',
                     'Dartmouth Proposal (McCarthy, 1956)', 'GPT-3 release', 'AlphaGo', 'Deep Blue',
                     'Logic Theorist', 'Super AI', 'Expert Systems', 'Genetic Algorithms'],
        'Option_D': ['Herbert Simon', 'When the model requires more training data', 'About 90 years old (1930s)',
                     'The accuracy score of predictions', 'When a model uses too many features', 'Stockfish',
                     'Speech Recognition', 'Autoencoders', 'Sharing models between different companies',
                     '2020', 'Recurrent Attention Graph', 'Equivalent to powering an entire city for a year',
                     'Generating training data', 'Around 50,000 GPUs', 'Over 90%', '500 terabytes',
                     'Over 70%', '1 year', 'Over 50%', 'Over 1 billion', 'Over 85%', '$2 trillion+',
                     '$500 million+', 'By following pre-programmed grammar rules',
                     'Guided Prediction Training', 'Fixing bugs in the model\'s code', 'Bell Labs Workshop',
                     'Overuse of neural networks', 'Stanford', 'Backpropagation',
                     'Recognizing faces in photos', 'Reinforcement Learning', 'AlphaGo',
                     'Generative AI', 'Recognizing images', 'Logic Theorist (Newell & Simon, 1955)',
                     'ChatGPT release', 'LLaMA', 'Deep Thought', 'Mac Hack', 'Quantum AI',
                     'Bayesian Networks', 'Expert Systems'],
        'Correct_Answer': ['B', 'B', 'C', 'B', 'A', 'B', 'B', 'C', 'B', 'B', 'B', 'C', 'B', 'C', 'D',
                          'C', 'D', 'B', 'D', 'C', 'D', 'D', 'C', 'B', 'B', 'B', 'A', 'A', 'B', 'B',
                          'C', 'B', 'A', 'A', 'B', 'C', 'D', 'C', 'D', 'D', 'D', 'B', 'A'],
        'Explanation': [
            "John McCarthy coined the term 'Artificial Intelligence' in 1956 at the Dartmouth Conference, which is considered the birth of AI as a field.",
            "Hallucination refers to when AI models generate plausible-sounding but factually incorrect or nonsensical information with confidence.",
            "The term 'Artificial Intelligence' was coined in 1956 at the Dartmouth Conference, making it nearly 70 years old.",
            "Alpha represents the excess return of an investment relative to a benchmark index, a key metric for evaluating trading strategies.",
            "Overfitting occurs when a model learns the training data too well, including noise, leading to poor generalization on unseen data.",
            "IBM's Deep Blue defeated world chess champion Garry Kasparov in 1997, marking a historic milestone in AI development.",
            "Anomaly detection identifies unusual patterns that deviate from normal behavior, making it highly effective for detecting fraudulent transactions.",
            "The Transformer architecture, introduced in the 'Attention is All You Need' paper (2017), became the foundation for modern language models.",
            "Transfer learning leverages knowledge from a model trained on one task to improve learning on a related task, saving time and resources.",
            "DeepMind's AlphaGo defeated world champion Lee Sedol in 2016, demonstrating AI's ability to master complex strategic games.",
            "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation, allowing AI to access external knowledge bases for more accurate responses.",
            "Training large language models like GPT-4 requires massive computational resources. Estimates suggest it consumes electricity equivalent to powering over 1,000 homes for an entire year.",
            "Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize the loss function during training.",
            "Training GPT-3 required approximately 10,000 NVIDIA V100 GPUs running for several weeks.",
            "By 2024, over 90% of Fortune 500 companies had adopted AI in some form, demonstrating AI's critical role in modern business.",
            "ChatGPT was trained on approximately 45 terabytes of text data, equivalent to millions of books.",
            "By 2024, over 70% of customer service interactions were handled by AI chatbots and virtual assistants.",
            "ChatGPT reached 100 million users in just 2 months after launch, making it the fastest-growing consumer application in history.",
            "By 2024, over 50% of code pushed to GitHub was estimated to be AI-assisted, primarily through tools like GitHub Copilot.",
            "As of 2024, ChatGPT processes approximately 500 million queries daily, showcasing the massive scale and adoption of conversational AI.",
            "Over 85% of financial institutions use AI-powered fraud detection systems.",
            "The global AI market is projected to exceed $2 trillion by 2030, growing at over 30% annually.",
            "Training GPT-4 is estimated to have cost around $100 million, considering compute resources, electricity, and infrastructure.",
            "GPT models are trained by predicting the next word in billions of text sequences.",
            "GPT stands for Generative Pre-trained Transformer.",
            "Fine-tuning takes a pre-trained model and trains it further on specific data to make it expert in a particular domain.",
            "The term 'Artificial Intelligence' was first introduced at the Dartmouth Conference in 1956.",
            "The first AI winter in the 1970s was primarily caused by lack of funding as governments and institutions reduced investments.",
            "IBM created Watson, the question-answering AI system that defeated human champions on Jeopardy! in 2011.",
            "The Perceptron, invented by Frank Rosenblatt in 1958, was the first self-learning neural network.",
            "Understanding sarcasm and jokes requires deep contextual understanding that remains challenging for AI.",
            "Symbolic AI (also called 'Good Old-Fashioned AI') dominated early AI research, using explicit rules and logic.",
            "SHRDLU, developed by Terry Winograd in 1968-1970, was a pioneering natural language understanding program.",
            "Expert Systems are AI programs designed to mimic human expert decision-making and can explain their reasoning.",
            "AlphaFold revolutionized biology by accurately predicting 3D protein structures from amino acid sequences.",
            "The Dartmouth Proposal (1956) formally established AI as a field of study.",
            "OpenAI released ChatGPT in November 2022, bringing AI to mainstream attention.",
            "AlphaGo defeated world champion Lee Sedol 4-1 in 2016.",
            "IBM's Deep Blue defeated world chess champion Garry Kasparov in 1997.",
            "Mac Hack, developed by Richard Greenblatt at MIT in 1966, was the first chess program to compete in human tournaments.",
            "Quantum AI refers to using quantum computing for AI applications, not a type of AI itself.",
            "Evolutionary Algorithms use principles of natural selection to evolve solutions to optimization problems.",
            "Self-driving cars rely primarily on Computer Vision and Reinforcement Learning."
        ]
    }
    return pd.DataFrame(data)

# Initialize session state
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
    st.session_state.email = ""
    st.session_state.questions = []
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.answers = []
    st.session_state.start_time = None
    st.session_state.quiz_completed = False

# Functions for email tracking
def get_attempted_emails():
    """Retrieve list of emails that have attempted the quiz"""
    try:
        # Using session state to store attempted emails (persists during session)
        if 'attempted_emails' not in st.session_state:
            st.session_state.attempted_emails = set()
        return st.session_state.attempted_emails
    except:
        return set()

def store_attempted_email(email):
    """Store an email that has attempted the quiz"""
    try:
        if 'attempted_emails' not in st.session_state:
            st.session_state.attempted_emails = set()
        st.session_state.attempted_emails.add(email)
    except:
        pass

# For persistent storage across sessions (using a file)
def get_persistent_attempted_emails():
    """Retrieve emails from persistent storage"""
    try:
        # Create a simple file-based storage
        try:
            with open('attempted_emails.txt', 'r') as f:
                emails = set(line.strip().lower() for line in f if line.strip())
                return emails
        except FileNotFoundError:
            return set()
    except:
        return set()

def store_persistent_attempted_email(email):
    """Store email in persistent storage"""
    try:
        # Read existing emails
        existing_emails = get_persistent_attempted_emails()
        # Add new email
        existing_emails.add(email.lower())
        # Write back to file
        with open('attempted_emails.txt', 'w') as f:
            for email in existing_emails:
                f.write(email + '\n')
    except:
        pass

# Updated email validation function that checks both session and persistent storage
def has_email_attempted(email):
    """Check if email has already attempted the quiz"""
    email = email.lower()
    # Check session state (current run)
    if email in get_attempted_emails():
        return True
    # Check persistent storage (across sessions)
    if email in get_persistent_attempted_emails():
        return True
    return False

def store_email_attempt(email):
    """Store email in both session and persistent storage"""
    email = email.lower()
    store_attempted_email(email)
    store_persistent_attempted_email(email)

def select_random_questions(df, num_questions=6):
    """Select random questions from different categories"""
    categories = df['Category'].unique()
    selected_questions = []
    
    # Randomly select one question from each category until we have 6 questions
    random.shuffle(list(categories))
    
    for category in categories:
        if len(selected_questions) >= num_questions:
            break
        category_questions = df[df['Category'] == category].sample(n=1)
        selected_questions.append(category_questions.iloc[0])
    
    # If we still need more questions, randomly select from all
    while len(selected_questions) < num_questions:
        remaining = df.sample(n=1).iloc[0]
        if not any(q['Question'] == remaining['Question'] for q in selected_questions):
            selected_questions.append(remaining)
    
    random.shuffle(selected_questions)
    return selected_questions

def get_time_remaining():
    """Calculate remaining time for current question"""
    if st.session_state.start_time:
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, 20 - int(elapsed))
        return remaining
    return 20

# Main app
st.markdown("<h1>🧠 AI Knowledge Quiz</h1>", unsafe_allow_html=True)

# Email input screen
if not st.session_state.quiz_started and not st.session_state.quiz_completed:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="question-card">
                <h2 style="text-align: center; color: #667eea;">Welcome to the AI Quiz!</h2>
                <p style="text-align: center; font-size: 1.1rem; color: #4a5568; margin: 1rem 0;">
                    Test your knowledge of Artificial Intelligence.<br>
                    You'll have 20 seconds to answer each question.<br><br>
                    <strong>Ready to challenge yourself?</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        email = st.text_input("📧 Enter your email address:", placeholder="your.name@spglobal.com")
        
        if st.button("🚀 Start Quiz"):
            if email:
                email = email.strip()
                if '@spglobal.com' in email.lower():
                    # Check if email already attempted the quiz
                    if has_email_attempted(email):
                        st.error("🚫 This email has already attempted the quiz. Each user can only attempt once.")
                    else:
                        # Store the email in attempted emails
                        store_email_attempt(email)
                        st.session_state.email = email
                        df = load_quiz_data()
                        st.session_state.questions = select_random_questions(df, 6)
                        st.session_state.quiz_started = True
                        st.session_state.start_time = time.time()
                        st.rerun()
                else:
                    st.error("⚠️ Please use a valid @spglobal.com email address!")
            else:
                st.error("⚠️ Please enter your email address!")

# Quiz screen
elif st.session_state.quiz_started and not st.session_state.quiz_completed:
    current_q_idx = st.session_state.current_question
    
    if current_q_idx < len(st.session_state.questions):
        question_data = st.session_state.questions[current_q_idx]
        
        # Progress indicator
        progress = (current_q_idx) / len(st.session_state.questions)
        st.progress(progress)
        st.markdown(f"""
            <div class="progress-text">
                Question {current_q_idx + 1} of {len(st.session_state.questions)}
            </div>
        """, unsafe_allow_html=True)
        
        # Timer
        time_remaining = get_time_remaining()
        timer_class = "timer warning" if time_remaining <= 5 else "timer"
        timer_placeholder = st.empty()
        timer_placeholder.markdown(f'<div class="{timer_class}">⏱️ {time_remaining}s</div>', unsafe_allow_html=True)
        
        # Auto-submit if time runs out
        if time_remaining == 0:
            st.session_state.answers.append({
                'question': question_data['Question'],
                'selected': None,
                'correct': question_data['Correct_Answer'],
                'timed_out': True
            })
            st.session_state.current_question += 1
            if st.session_state.current_question < len(st.session_state.questions):
                st.session_state.start_time = time.time()
            st.rerun()
        
        # Question card
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown(f"""
                <div class="question-card">
                    <h2>{question_data['Question']}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Options
            options = {
                'A': question_data['Option_A'],
                'B': question_data['Option_B'],
                'C': question_data['Option_C'],
                'D': question_data['Option_D']
            }
            
            # No option pre-selected and disabled submit button when no selection
            selected = st.radio(
                "Select your answer:",
                options.keys(),
                format_func=lambda x: f"{x}. {options[x]}",
                key=f"q_{current_q_idx}",
                index=None  # This prevents any option from being pre-selected
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Disable submit button if no option selected
            if st.button("✅ Submit Answer", key=f"submit_{current_q_idx}", disabled=selected is None):
                is_correct = selected == question_data['Correct_Answer']
                if is_correct:
                    st.session_state.score += 1
                
                st.session_state.answers.append({
                    'question': question_data['Question'],
                    'selected': selected,
                    'correct': question_data['Correct_Answer'],
                    'is_correct': is_correct,
                    'explanation': question_data['Explanation'],
                    'timed_out': False
                })
                
                st.session_state.current_question += 1
                
                if st.session_state.current_question < len(st.session_state.questions):
                    st.session_state.start_time = time.time()
                else:
                    st.session_state.quiz_completed = True
                
                st.rerun()
        
        # Auto-refresh for timer
        time.sleep(1)
        st.rerun()
    
# Results screen
elif st.session_state.quiz_completed:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Calculate score based on attempted questions only
        attempted_questions = [ans for ans in st.session_state.answers if ans['selected'] is not None or ans.get('timed_out')]
        correct_answers = [ans for ans in attempted_questions if ans.get('is_correct', False)]
        
        total_attempted = len(attempted_questions)
        total_correct = len(correct_answers)
        
        # Calculate percentage based on attempted questions, not total questions
        if total_attempted > 0:
            percentage_score = int((total_correct / total_attempted) * 100)
        else:
            percentage_score = 0
        
        st.markdown(f"""
            <div class="result-card">
                <h1 style="color: #667eea;">🎉 Quiz Completed!</h1>
                <p style="font-size: 1.2rem; color: #4a5568; margin-bottom: 1rem;">
                    Thank you for participating, <strong>{st.session_state.email}</strong>!
                </p>
                <div class="score-display">
                    {total_correct} / {total_attempted}
                </div>
                <p style="font-size: 1.2rem; color: #4a5568;">
                    You scored {percentage_score}% on attempted questions
                </p>
                <p style="font-size: 1rem; color: #718096;">
                    Questions attempted: {total_attempted} out of {len(st.session_state.questions)}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Detailed results
        st.markdown("""
            <div class="question-card">
                <h2 style="color: #667eea; text-align: center;">📊 Detailed Results</h2>
            </div>
        """, unsafe_allow_html=True)
        
        for idx, answer in enumerate(st.session_state.answers, 1):
            if answer.get('timed_out'):
                status = "⏱️ Time's Up!"
                color = "#ff6b6b"
            elif answer.get('is_correct'):
                status = "✅ Correct"
                color = "#51cf66"
            elif answer['selected'] is None:
                status = "⏭️ Not Attempted"
                color = "#a0aec0"
            else:
                status = "❌ Incorrect"
                color = "#ff6b6b"
            
            st.markdown(f"""
                <div class="question-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="color: #2d3748; margin: 0;">Question {idx}</h3>
                        <span style="color: {color}; font-weight: bold; font-size: 1.2rem;">{status}</span>
                    </div>
                    <p style="font-size: 1.1rem; margin: 1rem 0;"><strong>{answer['question']}</strong></p>
                    <p style="color: #4a5568;">
                        <strong>Your answer:</strong> {answer.get('selected', 'No answer (timeout)')}<br>
                        <strong>Correct answer:</strong> {answer['correct']}
                    </p>
                    {f'<p style="color: #4a5568; margin-top: 1rem;"><em>{answer.get("explanation", "")}</em></p>' if answer.get('selected') is not None and not answer.get('timed_out') else ''}
                </div>
            """, unsafe_allow_html=True)