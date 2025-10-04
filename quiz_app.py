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
    
    /* Question text styling - SET TO BLACK FOR BETTER VISIBILITY */
    .question-text {
        color: #000000 !important;
        font-size: 1.5rem;
        font-weight: 600;
        line-height: 1.4;
        margin-bottom: 1rem;
    }
            
    /* New styles for important notes */
    .important-note {
        background: #fffaf0;
        border: 2px solid #d69e2e;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .goodies-notice {
        background: #f0fff4;
        border: 2px solid #38a169;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .logo-item {
        text-align: center;
        margin: 1rem;
        flex: 1;
        min-width: 200px;
    }
    
    .logo-image {
        max-width: 150px;
        max-height: 100px;
        margin-bottom: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .logo-input {
        width: 80% !important;
        margin: 0 auto;
    }
    
    /* Explanation card styling */
    .explanation-card {
        background: #f0fff4;
        border: 2px solid #9ae6b4;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .explanation-title {
        color: #22543d;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .explanation-text {
        color: #2d3748;
        font-size: 1.1rem;
        line-height: 1.5;
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
        background: linear-gradient(135deg, #6E8CFB, #764ba2);
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
        color: #000000 !important; /* Set to black for better visibility */
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
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
    
    /* Option text styling */
    .option-text {
        color: #2d3748;
        font-size: 1.1rem;
    }
    
    /* Ensure all text in question cards is visible */
    .question-card h2,
    .question-card h3,
    .question-card p,
    .question-card strong {
        color: #000000 !important;
    }
    
    /* Result details text */
    .result-details {
        color: #000000 !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# All available logos data
all_logos = [
    {'name': 'TensorFlow', 'image': 'logos/tensorflow.png', 'alt': ['tensorflow']},
    {'name': 'PyTorch', 'image': 'logos/pytorch.png', 'alt': ['pytorch']},
    {'name': 'OpenAI', 'image': 'logos/openai.png', 'alt': ['openai']},
    {'name': 'AWS Lambda', 'image': 'logos/aws-lambda.png', 'alt': ['lambda']},
    {'name': 'Amazon S3', 'image': 'logos/amazon-s3.png', 'alt': ['s3']},
    {'name': 'SageMaker', 'image': 'logos/sagemaker.png', 'alt': ['sagemaker']},
    {'name': 'Tableau', 'image': 'logos/tableau.png', 'alt': ['tableau']},
    {'name': 'Power BI', 'image': 'logos/powerbi.png', 'alt': ['powerbi', 'power bi']},
    {'name': 'Snowflake', 'image': 'logos/snowflake.png', 'alt': ['snowflake']},
    {'name': 'Python', 'image': 'logos/python.png', 'alt': ['python']},
    {'name': 'Java', 'image': 'logos/java.png', 'alt': ['java']},
    {'name': 'JavaScript', 'image': 'logos/javascript.png', 'alt': ['javascript', 'js']},
    {'name': 'Keras', 'image': 'logos/keras.png', 'alt': ['keras']},
    {'name': 'Hugging Face', 'image': 'logos/huggingface.png', 'alt': ['huggingface', 'hugging face']},
    {'name': 'NVIDIA', 'image': 'logos/nvidia.png', 'alt': ['nvidia']},
    {'name': 'DynamoDB', 'image': 'logos/dynamodb.png', 'alt': ['dynamodb']},
    {'name': 'Apache Spark', 'image': 'logos/apache-spark.png', 'alt': ['spark', 'apache spark']},
    {'name': 'Redshift', 'image': 'logos/redshift.png', 'alt': ['redshift']}
]

def check_logo_answer(user_answer, correct_name, alt_names):
    """Check if user answer matches correct name or alternative names (case insensitive)"""
    user_answer_clean = user_answer.strip().lower()
    correct_names = [correct_name.lower()] + [alt.lower() for alt in alt_names]
    return user_answer_clean in correct_names

def get_single_logo_question():
    """Get a single random logo for questions 7 and 8"""
    logo = random.choice(all_logos)
    return {
        'type': 'single_logo_quiz',
        'question': 'Identify the following logo:',
        'logo': logo,
        'correct_answer': logo['name']
    }

# Load quiz data
@st.cache_data
def load_quiz_data():
    data = {
        'Category': [
            'AI Concepts', 'AI Concepts', 'AI Concepts', 'AI Concepts', 'AI Concepts', 'AI Concepts',
            'AI Concepts', 'AI Concepts', 'AI Concepts', 'AI Concepts', 'AI Concepts', 'AI Concepts',
            'AI Facts', 'AI Facts', 'AI Facts', 'AI Facts', 'AI Facts', 'AI Facts', 'AI Facts',
            'AI Facts', 'AI Facts', 'AI History', 'AI History', 'AI History', 'AI History', 'AI History',
            'AI History', 'AI History', 'AI History', 'AI History', 'AI History', 'AI History', 'AI History',
            'AI History', 'AI History', 'AI History', 'AI in Finance', 'AI in Finance', 'AI in S&P Global',
            'AI in S&P Global', 'AI in S&P Global', 'AI in S&P Global', 'AI in S&P Global', 'AI in S&P Global',
            'AI in S&P Global', 'AI in S&P Global', 'AI in S&P Global', 'AI in S&P Global', 'AI Usage',
            'AI Usage', 'AI Usage', 'AI Usage', 'AI Usage', 'AI Usage', 'AI Usage', 'AI Usage', 'Deep Learning',
            'Deep Learning', 'Deep Learning', 'Deep Learning', 'Deep Learning', 'Deep Learning', 'Deep Learning',
            'Deep Learning', 'Machine Learning'
        ],
        'Question': [
            "What does 'hallucination' mean in the context of Large Language Models?",
            "What is 'overfitting' in machine learning models?",
            "What is 'transfer learning' in machine learning?",
            "What is 'gradient descent' used for in neural networks?",
            "How do large language models like GPT learn to generate text?",
            "What does 'GPT' stand for in ChatGPT?",
            "What is 'fine-tuning' in the context of AI models?",
            "Which of the following tasks is still very hard for AI?",
            "Which AI concept involves systems that can explain their reasoning in human terms?",
            "Which of the following is NOT a type of AI?",
            "Which AI concept involves mimicking natural selection to optimize solutions?",
            "Which AI concept powers self-driving cars?",
            "How old is the term 'Artificial Intelligence'?",
            "How much electricity does it take to train a GPT-4 scale model?",
            "How many GPUs were approximately used to train GPT-3?",
            "What percentage of Fortune 500 companies were using AI in some capacity by 2024?",
            "How much data is ChatGPT estimated to have been trained on?",
            "How long did it take ChatGPT to reach 100 million users?",
            "How many ChatGPT queries are processed daily (as of 2024)?",
            "How much is the global AI market projected to be worth by 2030?",
            "What was the estimated cost to train GPT-4?",
            "Who coined the term 'Artificial Intelligence' at the Dartmouth Conference?",
            "What was the name of the first AI program to defeat a world chess champion?",
            "In what year did AlphaGo defeat Lee Sedol in the game of Go?",
            "At which conference was the term 'Artificial Intelligence' first introduced?",
            "What was the primary reason for the first AI winter (1970s)?",
            "Which company created Watson, the AI that won on Jeopardy in 2011?",
            "What was the name of the first self-learning neural network, created in 1958?",
            "Which AI approach dominated in the 1950s-1980s, focusing on logic and rules?",
            "Which AI system first used the concept of 'knowledge representation'?",
            "Which AI achievement is AlphaFold famous for?",
            "Which paper is considered the 'birth certificate' of AI?",
            "Which AI milestone was achieved by OpenAI in 2022?",
            "Which AI system by Google DeepMind defeated the world Go champion Lee Sedol in 2016?",
            "Which AI system defeated Garry Kasparov, the world chess champion, in 1997?",
            "Which AI program was the first to play chess?",
            "What does 'alpha' refer to in quantitative finance and AI trading?",
            "Which AI technique is most commonly used for fraud detection in financial transactions?",
            "S&P Global Ratings issues credit scores that affect global borrowing. How can AI improve this process?",
            "Analysts at S&P Global need to spot early signs of company default. Which AI technique is most useful?",
            "S&P Global Market Intelligence deals with messy, unstructured data. Which AI tool is best for extracting insights?",
            "Bond markets are huge, and fraud is always a risk. Which AI method can detect unusual trading behavior?",
            "S&P Global Commodity Insights predicts energy market trends. Where does AI add the most value?",
            "Climate risk analysis is vital for financial markets. How can AI help S&P Global assess this?",
            "Real-time market data is one of S&P Global's strengths. Which AI capability is most important here?",
            "S&P Global provides ESG (Environmental, Social, Governance) scores. Which AI feature supports this best?",
            "S&P Global researchers aim to predict global financial crises. Which AI approach is most practical?",
            "Portfolio managers depend on S&P Global for risk analysis. Which AI application helps the most?",
            "By 2024, what percentage of customer service interactions were handled by AI chatbots?",
            "What percentage of code on GitHub is estimated to be AI-assisted by 2024?",
            "What percentage of financial institutions use AI for fraud detection?",
            "Your smart fridge wants to predict which food will expire soon. Which ML approach should it use?",
            "Netflix notices some users watch only horror movies. What type of ML task can help find similar users?",
            "An AI voice assistant learns your accent over time to improve recognition. This is an example of:",
            "You're building a handwriting recognition app. Which feature might surprisingly help?",
            "An AI artist generates memes based on trending topics. Which type of ML is it using?",
            "What architecture revolutionized NLP and led to models like BERT and GPT?",
            "In convolutional neural networks (CNNs), what does the term 'stride' refer to?",
            "What makes deep learning different from traditional machine learning?",
            "Which of these best describes how a convolutional neural network (CNN) 'sees' an image?",
            "What would happen if you remove all activation functions from a deep network?",
            "What is the role of an activation function in a neural network?",
            "How many layers do you think a deep learning network needs to beat humans at chess?",
            "Two models have similar accuracy. One is chosen over the other. Why?",
            "What does 'RAG' stand for in the context of AI systems?"
        ],
        'Option_A': [
            'When the model processes visual data',
            'When a model performs too well on training data but poorly on new data',
            'Moving data between servers',
            'Reducing model size',
            'By memorizing entire books and articles',
            'General Purpose Technology',
            'Adjusting the model\'s speed',
            'Playing chess',
            'Expert Systems',
            'Narrow AI',
            'Reinforcement Learning',
            'Computer Vision + Reinforcement Learning',
            'About 30 years old (1990s)',
            'Equivalent to powering 10 homes for a year',
            'Around 100 GPUs',
            'Around 25%',
            '100 gigabytes',
            '2 weeks',
            '10 million',
            '$100 billion',
            '$5 million',
            'Alan Turing',
            'AlphaGo',
            '2012',
            'Dartmouth Conference',
            'Lack of funding',
            'Google',
            'ELIZA',
            'Neural Networks',
            'SHRDLU',
            'Translating languages',
            'Computing Machinery and Intelligence (Turing, 1950)',
            'Self-driving cars',
            'Deep Blue',
            'Watson',
            'ELIZA',
            'The first version of an AI model',
            'Reinforcement Learning',
            'Using AR/VR to visualize company offices',
            'Chatbots for board meeting summaries',
            'Optical Character Recognition (OCR)',
            'Predictive emojis for market moods',
            'Automating customer billing at petrol pumps',
            'Building AI chatbots for weather jokes',
            'Using AI to predict company office layouts',
            'Sentiment analysis of sustainability news and reports',
            'VR headsets for risk management games',
            'Generating cartoons about stock markets',
            '15%',
            '10%',
            '30%',
            'Supervised Learning – teach it with past expiration dates',
            'Classification – label everyone as "horror lover" or "not"',
            'Supervised Learning – teacher corrects every word',
            'Stroke direction – humans have habits',
            'Generative AI – creates new content',
            'Convolutional Neural Networks',
            'Number of filters applied',
            'It uses bigger computers',
            'Like a puzzle, by focusing on small pieces (patches) and combining information to recognize the whole',
            'The network learns faster',
            'To turn neurons on or off, introducing non-linearity so the network can learn complex functions',
            '1',
            'It was trained using more recent data',
            'Rapid Algorithm Generation'
        ],
        'Option_B': [
            'When the model generates false or nonsensical information confidently',
            'When a model is too large to deploy',
            'Using a pre-trained model as a starting point for a new task',
            'Optimizing model parameters to minimize error',
            'By predicting the next word in a sequence',
            'Generative Pre-trained Transformer',
            'Training a pre-trained model on specific data for specialized tasks',
            'Driving a car on highways',
            'Neural Networks',
            'General AI',
            'Evolutionary Algorithms',
            'NLP + Robotics',
            'About 50 years old (1970s)',
            'Equivalent to powering 100 homes for a year',
            'Around 1,000 GPUs',
            'Around 50%',
            '1 terabyte',
            '2 months',
            '100 million',
            '$500 billion',
            '$25 million',
            'John McCarthy',
            'Deep Blue',
            '2016',
            'Turing Symposium',
            'Too much hype',
            'IBM',
            'Perceptron',
            'Symbolic AI',
            'Watson',
            'Predicting protein structures',
            'Perceptrons (Minsky & Papert, 1969)',
            'AlphaFold',
            'Watson',
            'AlphaGo',
            'Deep Blue',
            'Excess returns above a benchmark',
            'Anomaly Detection using unsupervised learning',
            'Reading millions of financial statements faster than analysts',
            'Predictive analytics on risk indicators',
            'Image recognition of company logos',
            'Anomaly detection algorithms',
            'Forecasting oil price movements using global supply-demand data',
            'Auto-generating carbon tax policies',
            'Processing high-frequency trading streams in milliseconds',
            'Machine learning to redesign corporate logos',
            'Systemic risk simulations using global datasets',
            'Deepfake simulations of CEO interviews',
            '35%',
            '25%',
            '50%',
            'Unsupervised Learning – let it randomly guess',
            'Clustering – group similar users together',
            'Online Learning – improves continuously with new data',
            'Pen pressure – light vs heavy pressure',
            'Supervised Learning – shown every meme ever',
            'Recurrent Neural Networks',
            'Size of the filter',
            'It automatically learns features from raw data instead of relying on manual feature engineering',
            'By reading every pixel as a separate letter',
            'The network behaves like a simple linear model, losing the ability to learn complex patterns',
            'To power the network with electricity',
            '5',
            'It has a simpler name',
            'Retrieval-Augmented Generation'
        ],
        'Option_C': [
            'When the model becomes confused by ambiguous prompts',
            'When a model takes too long to train',
            'Converting models between programming languages',
            'Increasing processing speed',
            'By copying responses from a database',
            'Global Processing Tool',
            'Reducing the model\'s size',
            'Understanding sarcasm and jokes in conversations',
            'Reinforcement Learning',
            'Super AI',
            'Expert Systems',
            'Genetic Algorithms',
            'About 70 years old (1950s)',
            'Equivalent to powering 1,000+ homes for a year',
            'Around 10,000 GPUs',
            'Around 75%',
            '45 terabytes',
            '6 months',
            '500 million',
            '$1 trillion',
            '$100 million',
            'Marvin Minsky',
            'Watson',
            '2018',
            'MIT AI Lab Meet',
            'Ethical concerns',
            'Microsoft',
            'Hopfield Network',
            'Genetic Algorithms',
            'Deep Blue',
            'Playing video games',
            'Dartmouth Proposal (McCarthy, 1956)',
            'GPT-3 release',
            'AlphaGo',
            'Deep Thought',
            'Logic Theorist',
            'The learning rate in neural networks',
            'Image Classification',
            'Replacing credit rating agencies with central banks',
            'Blockchain-based voting in shareholder meetings',
            'Natural Language Processing (NLP)',
            'Speech-to-text for investor calls',
            'Translating drilling reports into memes',
            'Running machine learning–based climate scenario models',
            'Auto-writing analyst commentary blogs',
            'ChatGPT-style bots to rewrite CSR reports',
            'Generative AI creating new financial products',
            'Machine learning for portfolio optimization & asset allocation',
            '55%',
            '40%',
            '70%',
            'Reinforcement Learning – reward fridge for keeping food fresh',
            'Regression – predict how scary a movie is',
            'Reinforcement Learning – gets rewards for understanding',
            'Writing speed – fast scribbles look messy',
            'Reinforcement Learning – rewarded for viral memes',
            'Transformer',
            'Step size with which the filter moves across the input',
            'It only works with images',
            'By randomly guessing what\'s in the picture',
            'The network becomes sentient',
            'To print out the result on a screen',
            'Hundreds or even thousands — depth helps model complex strategies',
            'It\'s more interpretable and easier to explain',
            'Random Access Gateway'
        ],
        'Option_D': [
            'When the model requires more training data',
            'When a model uses too many features',
            'Sharing models between different companies',
            'Generating training data',
            'By following pre-programmed grammar rules',
            'Guided Prediction Training',
            'Fixing bugs in the model\'s code',
            'Recognizing faces in photos',
            'Generative AI',
            'Quantum AI',
            'Bayesian Networks',
            'Expert Systems',
            'About 90 years old (1930s)',
            'Equivalent to powering an entire city for a year',
            'Around 50,000 GPUs',
            'Over 90%',
            '500 terabytes',
            '1 year',
            'Over 1 billion',
            '$2 trillion+',
            '$500 million+',
            'Herbert Simon',
            'Stockfish',
            '2020',
            'Bell Labs Workshop',
            'Overuse of neural networks',
            'Stanford',
            'Backpropagation',
            'Reinforcement Learning',
            'AlphaGo',
            'Recognizing images',
            'Logic Theorist (Newell & Simon, 1955)',
            'ChatGPT release',
            'LLaMA',
            'Deep Blue',
            'Mac Hack',
            'The accuracy score of predictions',
            'Speech Recognition',
            'Predicting weather patterns to set bond interest rates',
            'Computer vision on employee photos',
            'Virtual Reality text visualization',
            'Generative AI for fake bond data',
            'Generating 3D animations of refineries',
            'Using drones to monitor farms',
            'Generating interactive VR dashboards for traders',
            'Computer vision to inspect office buildings remotely',
            'Machine vision for counting stock certificates',
            'Predictive AI for choosing lucky stock tickers',
            'Over 70%',
            'Over 50%',
            'Over 85%',
            'Clustering – group food by color',
            'Dimensionality Reduction – shrink the users',
            'Unsupervised Learning – just listens quietly',
            'All of the above',
            'Clustering – groups memes by topic',
            'Autoencoders',
            'Padding added to the input',
            'It never makes mistakes',
            'By converting it into sound first',
            'The network starts talking',
            'To cool down the hardware',
            'None, humans are unbeatable',
            'It requires more memory',
            'Recurrent Attention Graph'
        ],
        'Correct_Answer': [
            'B', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'A', 'D', 'B', 'A', 'C', 'C', 'C', 'D',
            'C', 'B', 'C', 'D', 'C', 'B', 'B', 'B', 'A', 'A', 'B', 'B', 'B', 'A', 'B', 'C',
            'D', 'C', 'D', 'D', 'B', 'B', 'B', 'B', 'C', 'B', 'B', 'C', 'B', 'A', 'B', 'C',
            'D', 'D', 'D', 'A', 'B', 'B', 'D', 'A', 'C', 'C', 'B', 'A', 'B', 'A', 'C', 'C', 'B'
        ],
        'Explanation': [
            "Hallucination refers to when AI models generate plausible-sounding but factually incorrect or nonsensical information with confidence.",
            "Overfitting occurs when a model learns the training data too well, including noise, leading to poor generalization on unseen data.",
            "Transfer learning leverages knowledge from a model trained on one task to improve learning on a related task, saving time and resources.",
            "Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize the loss function during training.",
            "GPT models are trained by predicting the next word in billions of text sequences. This simple task, repeated trillions of times, enables them to learn grammar, facts, reasoning patterns, and even creativity.",
            "GPT stands for Generative Pre-trained Transformer. 'Generative' means it creates new text, 'Pre-trained' means it learned from vast data before fine-tuning, and 'Transformer' refers to its neural architecture.",
            "Fine-tuning takes a pre-trained model and trains it further on specific data (like financial documents or medical records) to make it expert in a particular domain while retaining its general knowledge.",
            "Understanding sarcasm and jokes requires deep contextual understanding, cultural knowledge, and emotional intelligence that remains challenging for AI, unlike tasks like chess, face recognition, or highway driving which AI handles well.",
            "Expert Systems are AI programs designed to mimic human expert decision-making and can explain their reasoning using if-then rules, making their logic transparent and understandable.",
            "Quantum AI refers to using quantum computing for AI applications, not a type of AI itself. The three types of AI are Narrow AI (specialized in specific tasks), General AI (human-level intelligence across domains), and Super AI (surpassing human intelligence).",
            "Evolutionary Algorithms use principles of natural selection, including mutation, crossover, and survival of the fittest, to evolve solutions to optimization problems over multiple generations.",
            "Self-driving cars rely primarily on Computer Vision to perceive their environment (detecting objects, lanes, signs) and Reinforcement Learning to make optimal driving decisions based on that perception.",
            "The term 'Artificial Intelligence' was coined in 1956 at the Dartmouth Conference, making it nearly 70 years old. Despite its age, AI has only recently achieved mainstream adoption.",
            "Training large language models like GPT-4 requires massive computational resources. Estimates suggest it consumes electricity equivalent to powering over 1,000 homes for an entire year, highlighting the importance of energy-efficient AI development.",
            "Training GPT-3 required approximately 10,000 NVIDIA V100 GPUs running for several weeks. The computational scale of modern AI training is staggering and continues to grow with each generation.",
            "By 2024, over 90% of Fortune 500 companies had adopted AI in some form, from customer service chatbots to advanced predictive analytics, demonstrating AI's critical role in modern business.",
            "ChatGPT was trained on approximately 45 terabytes of text data, equivalent to millions of books. This massive dataset enables it to understand and generate human-like text across countless topics.",
            "ChatGPT reached 100 million users in just 2 months after launch, making it the fastest-growing consumer application in history. For comparison, TikTok took 9 months and Instagram took 2.5 years.",
            "As of 2024, ChatGPT processes approximately 500 million queries daily, showcasing the massive scale and adoption of conversational AI in everyday tasks across the globe.",
            "The global AI market is projected to exceed $2 trillion by 2030, growing at over 30% annually. This massive growth reflects AI's transformation of virtually every industry worldwide.",
            "Training GPT-4 is estimated to have cost around $100 million, considering compute resources, electricity, and infrastructure. This highlights the massive investment required for cutting-edge AI development.",
            "John McCarthy coined the term 'Artificial Intelligence' in 1956 at the Dartmouth Conference, which is considered the birth of AI as a field.",
            "IBM's Deep Blue defeated world chess champion Garry Kasparov in 1997, marking a historic milestone in AI development.",
            "DeepMind's AlphaGo defeated world champion Lee Sedol in 2016, demonstrating AI's ability to master complex strategic games.",
            "The term 'Artificial Intelligence' was first introduced at the Dartmouth Conference in 1956, organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon.",
            "The first AI winter in the 1970s was primarily caused by lack of funding as governments and institutions reduced investments after AI failed to meet overly optimistic expectations.",
            "IBM created Watson, the question-answering AI system that defeated human champions on the quiz show Jeopardy! in 2011, demonstrating advanced natural language processing capabilities.",
            "The Perceptron, invented by Frank Rosenblatt in 1958, was the first self-learning neural network. It was designed to recognize patterns and could learn from experience.",
            "Symbolic AI (also called 'Good Old-Fashioned AI' or GOFAI) dominated early AI research, using explicit rules, logic, and symbol manipulation to represent and solve problems.",
            "SHRDLU, developed by Terry Winograd in 1968-1970, was a pioneering natural language understanding program that used knowledge representation to manipulate virtual blocks based on text commands.",
            "AlphaFold, developed by DeepMind, revolutionized biology by accurately predicting 3D protein structures from amino acid sequences, solving a 50-year-old grand challenge in science.",
            "The Dartmouth Proposal (1956) by John McCarthy and colleagues formally established AI as a field of study, proposing the famous Dartmouth Conference where the term 'Artificial Intelligence' was coined.",
            "OpenAI released ChatGPT in November 2022, which became the fastest-growing consumer application in history, reaching 100 million users in just two months and bringing AI to mainstream attention.",
            "AlphaGo defeated world champion Lee Sedol 4-1 in 2016, marking a historic achievement as Go was considered far more complex than chess due to its vast number of possible positions.",
            "IBM's Deep Blue defeated world chess champion Garry Kasparov in 1997 in a six-game match, becoming the first computer system to win a match against a reigning world champion under standard chess tournament conditions.",
            "Mac Hack, developed by Richard Greenblatt at MIT in 1966, was the first chess program to compete in human tournaments and achieved a respectable rating, pioneering computer chess.",
            "Alpha represents the excess return of an investment relative to a benchmark index, a key metric for evaluating trading strategies.",
            "Anomaly detection identifies unusual patterns that deviate from normal behavior, making it highly effective for detecting fraudulent transactions.",
            "AI's edge is processing huge datasets instantly. Central banks or VR don't help in actual credit scoring.",
            "Predictive analytics digs into subtle trends (like delayed payments, market shifts) before defaults happen.",
            "NLP interprets text (news, filings, reports). OCR just scans text, and VR/vision don't solve meaning extraction.",
            "Anomaly detection flags suspicious activity, unlike speech or emojis.",
            "AI forecasting models digest shipping, weather, and geopolitics to predict oil/gas movements.",
            "AI simulates floods, droughts, carbon taxes and calculates their effect on finance.",
            "The edge is speed — live market data processed faster than humans.",
            "Sentiment AI tracks ESG controversies and commitments across unstructured text.",
            "Simulation models help assess interconnected risks before they collapse markets.",
            "ML optimizes risk-return balance and creates smarter investment strategies.",
            "By 2024, over 70% of customer service interactions were handled by AI chatbots and virtual assistants, dramatically reducing response times and operational costs while improving 24/7 availability.",
            "By 2024, over 50% of code pushed to GitHub was estimated to be AI-assisted, primarily through tools like GitHub Copilot, revolutionizing software development productivity.",
            "Over 85% of financial institutions use AI-powered fraud detection systems. These systems can identify suspicious patterns in milliseconds, preventing billions of dollars in fraudulent transactions annually.",
            "Supervised learning uses labeled data like past expiration dates to make predictions.",
            "Clustering finds patterns or groups in data without labels, like user viewing habits.",
            "Online learning updates the model incrementally as it receives new data.",
            "All listed features can provide useful patterns to improve handwriting recognition.",
            "Generative AI models are designed to create new, original content based on patterns.",
            "The Transformer architecture, introduced in the 'Attention is All You Need' paper (2017), became the foundation for modern language models.",
            "The stride defines how many pixels the filter shifts when sliding over the input image. A larger stride reduces the spatial dimensions of the output feature map.",
            "Deep learning models automatically discover relevant features, removing the need for manual design, which makes them powerful for complex data like images, speech, or text.",
            "CNNs scan small local regions to detect patterns like edges, textures, and shapes, then combine these to understand the whole image.",
            "Activation functions add non-linearity, enabling networks to model complex relationships. Without them, multiple layers collapse into a single linear transformation.",
            "Without activation functions, neural networks would behave like linear models and couldn't learn complex patterns.",
            "Deep networks with many layers can learn complex game strategies, as seen in AI systems like AlphaZero.",
            "In many real-world applications, interpretability is critical — especially in healthcare, finance, and law. Stakeholders need to understand how a model makes decisions.",
            "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation, allowing AI to access external knowledge bases for more accurate responses."
        ]
    }
    
    # Verify all arrays have the same length
    expected_length = 65
    for key, value in data.items():
        if len(value) != expected_length:
            print(f"Warning: {key} has {len(value)} elements, expected {expected_length}")
    
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
    st.session_state.show_explanation = False
    st.session_state.explanation_start_time = None
    st.session_state.explanation_data = None

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
    
    # Add two single logo quiz questions (7th and 8th questions)
    selected_questions.append(get_single_logo_question())  # 7th question
    selected_questions.append(get_single_logo_question())  # 8th question
    
    return selected_questions

def get_time_remaining():
    """Calculate remaining time for current question"""
    if st.session_state.start_time:
        elapsed = time.time() - st.session_state.start_time
        # Questions 7 and 8 (index 6 and 7) get 25 seconds, others get 20
        if st.session_state.current_question in [6, 7]:  # 7th and 8th questions
            remaining = max(0, 25 - int(elapsed))
        else:
            remaining = max(0, 20 - int(elapsed))
        return remaining
    return 20

def get_explanation_time_remaining():
    """Calculate remaining time for explanation display"""
    if st.session_state.explanation_start_time:
        elapsed = time.time() - st.session_state.explanation_start_time
        remaining = max(0, 7 - int(elapsed))
        return remaining
    return 7

# Main app
st.markdown("<h1>Insight Engine Booth</h1>", unsafe_allow_html=True)

# Email input screen
if not st.session_state.quiz_started and not st.session_state.quiz_completed:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="question-card">
                <h2 style="text-align: center; color: #667eea;">Welcome to the AI Quiz! 🧠</h2>
                <p style="text-align: center; font-size: 1.1rem; color: #4a5568; margin: 1rem 0;">
                    You'll have 20 to 25 seconds to answer each question.<br>
                    <strong style="color: #e53e3e;">🎁 Exciting goodies will be provided to Winners!</strong><br>
                    <strong style="color: #d69e2e;">⚠️ Rewards will be provided only to valid users, carefully fill your email.</strong><br>
                    <strong style="color: #d69e2e;">⚠️ </strong> Remember to click the <strong>Submit Answer</strong> button for each question!<br><br>
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
        
        # Show explanation screen if active
        if st.session_state.show_explanation and st.session_state.explanation_data:
            explanation_time_remaining = get_explanation_time_remaining()
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                
                # Show selected answer as disabled radio
                selected = st.session_state.explanation_data['selected']
                
                # Show answer feedback
                is_correct = st.session_state.explanation_data['is_correct']
                status_color = "#51cf66" if is_correct else "#ff6b6b"
                status_text = "✅ Correct!" if is_correct else "❌ Incorrect"
                
                # Check if it's a logo quiz or regular question
                current_question = st.session_state.questions[st.session_state.current_question]
                if current_question.get('type') == 'single_logo_quiz':
                    # For single logo quiz, show different message
                    st.markdown(f"""
                        <div style="background: {'#f0fff4' if is_correct else '#fff5f5'}; 
                                    border: 2px solid {'#9ae6b4' if is_correct else '#fed7d7'}; 
                                    border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                            <div style="color: {status_color}; font-size: 1.3rem; font-weight: bold; text-align: center;">
                                {status_text}
                            </div>
                            <div style="color: #2d3748; text-align: center; margin-top: 0.5rem;">
                                {selected}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # For regular questions
                    st.markdown(f"""
                        <div style="background: {'#f0fff4' if is_correct else '#fff5f5'}; 
                                    border: 2px solid {'#9ae6b4' if is_correct else '#fed7d7'}; 
                                    border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                            <div style="color: {status_color}; font-size: 1.3rem; font-weight: bold; text-align: center;">
                                {status_text}
                            </div>
                            <div style="color: #2d3748; text-align: center; margin-top: 0.5rem;">
                                <strong>Your answer:</strong> {selected} &nbsp; | &nbsp;
                                <strong>Correct answer:</strong> {current_question['Correct_Answer']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Show explanation
                st.markdown(f"""
                    <div class="explanation-card">
                        <div class="explanation-title">💡 Explanation</div>
                        <div class="explanation-text">
                            {st.session_state.explanation_data['explanation']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Auto-proceed after 7 seconds (no timer shown on UI)
                if explanation_time_remaining > 0:
                    time.sleep(1)
                    st.rerun()
                else:
                    # Explanation time finished, move to next question
                    st.session_state.show_explanation = False
                    st.session_state.explanation_data = None
                    st.session_state.current_question += 1
                    
                    if st.session_state.current_question < len(st.session_state.questions):
                        st.session_state.start_time = time.time()
                    else:
                        st.session_state.quiz_completed = True
                    st.rerun()
        else:
            # Normal question screen with timer
            # Timer
            time_remaining = get_time_remaining()
            timer_class = "timer warning" if time_remaining <= 5 else "timer"
            timer_placeholder = st.empty()
            timer_placeholder.markdown(f'<div class="{timer_class}">⏱️ {time_remaining}s</div>', unsafe_allow_html=True)
            
            # Auto-submit if time runs out
            if time_remaining == 0:
                if question_data.get('type') == 'single_logo_quiz':
                    # For single logo quiz, store empty answer
                    st.session_state.answers.append({
                        'question': question_data['question'],
                        'type': 'single_logo_quiz',
                        'logo': question_data['logo'],
                        'user_answer': '',
                        'correct_answer': question_data['correct_answer'],
                        'is_correct': False,
                        'timed_out': True
                    })
                else:
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
                if question_data.get('type') == 'single_logo_quiz':
                    # Single logo quiz question (questions 7 and 8)
                    st.markdown(f"""
                        <div class="question-card">
                            <h2 style="color: #000000;">{question_data['question']}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display the single logo in the center
                    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
                    with col_img2:
                        st.image(question_data['logo']['image'], width=200)
                        user_input = st.text_input("Enter the logo name:", key=f"logo_{current_q_idx}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Submit button for single logo quiz
                    if st.button("✅ Submit Answer", key=f"submit_{current_q_idx}"):
                        # Check the answer
                        is_correct = check_logo_answer(user_input, question_data['correct_answer'], question_data['logo'].get('alt', []))
                        
                        if is_correct:
                            st.session_state.score += 1
                        
                        st.session_state.answers.append({
                            'question': question_data['question'],
                            'type': 'single_logo_quiz',
                            'logo': question_data['logo'],
                            'user_answer': user_input,
                            'correct_answer': question_data['correct_answer'],
                            'is_correct': is_correct,
                            'timed_out': False
                        })
                        
                        # Show explanation for 7 seconds
                        st.session_state.show_explanation = True
                        st.session_state.explanation_start_time = time.time()
                        st.session_state.explanation_data = {
                            'explanation': f"Correct answer: {question_data['correct_answer']}",
                            'is_correct': is_correct,
                            'selected': f"Your answer: {user_input if user_input else 'No answer provided'}"
                        }
                        st.rerun()

                else:
                    # Regular multiple choice question
                    st.markdown(f"""
                        <div class="question-card">
                            <h2 style="color: #000000;">{question_data['Question']}</h2>
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
                    
                    # Submit button
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
                        
                        # Show explanation for 7 seconds
                        st.session_state.show_explanation = True
                        st.session_state.explanation_start_time = time.time()
                        st.session_state.explanation_data = {
                            'explanation': question_data['Explanation'],
                            'is_correct': is_correct,
                            'selected': selected
                        }
                        st.rerun()
            
            # Auto-refresh for timer
            time.sleep(1)
            st.rerun()
    
# Results screen
elif st.session_state.quiz_completed:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Calculate score based on attempted questions only
        attempted_questions = [ans for ans in st.session_state.answers if ans.get('selected') is not None or ans.get('user_answer') or ans.get('timed_out')]
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
            elif answer.get('selected') is None and not answer.get('user_answer'):
                status = "⏭️ Not Attempted"
                color = "#a0aec0"
            else:
                status = "❌ Incorrect"
                color = "#ff6b6b"
            
            if answer.get('type') == 'single_logo_quiz':
                # Single logo quiz result
                st.markdown(f"""
                    <div class="question-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="color: #000000; margin: 0;">Question {idx} (Logo Identification)</h3>
                            <span style="color: {color}; font-weight: bold; font-size: 1.2rem;">{status}</span>
                        </div>
                        <p style="font-size: 1.1rem; margin: 1rem 0; color: #000000;"><strong>{answer['question']}</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display the single logo
                col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
                with col_img2:
                    st.image(answer['logo']['image'], width=150)
                    is_logo_correct = check_logo_answer(answer['user_answer'], answer['correct_answer'], answer['logo'].get('alt', []))
                    answer_color = "#51cf66" if is_logo_correct else "#ff6b6b"
                    st.markdown(f"""
                        <p style="color: {answer_color}; font-weight: bold; text-align: center;">
                            Your answer: {answer['user_answer'] if answer['user_answer'] else 'No answer'}<br>
                            Correct: {answer['correct_answer']}
                        </p>
                    """, unsafe_allow_html=True)
            else:
                # Regular question result
                st.markdown(f"""
                    <div class="question-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="color: #000000; margin: 0;">Question {idx}</h3>
                            <span style="color: {color}; font-weight: bold; font-size: 1.2rem;">{status}</span>
                        </div>
                        <p style="font-size: 1.1rem; margin: 1rem 0; color: #000000;"><strong>{answer['question']}</strong></p>
                        <p style="color: #000000;">
                            <strong>Your answer:</strong> {answer.get('selected', 'No answer (timeout)')}<br>
                            <strong>Correct answer:</strong> {answer['correct']}
                        </p>
                        {f'<p style="color: #000000; margin-top: 1rem;"><em>{answer.get("explanation", "")}</em></p>' if answer.get('selected') is not None and not answer.get('timed_out') else ''}
                    </div>
                """, unsafe_allow_html=True)