# SMU NLP-Research: LLM Sentiment Analysis on Statistics Exam Questions  

**Repository:** [https://github.com/cabrerajulian401/NLP-Research](https://github.com/cabrerajulian401/NLP-Research)  

## 📌 Overview  
This research project is part of the SMU FYRE Program with Dr. Monnie McGee as Julian's Research Mentor. Their research seeks to understand the differences of output reponses in different LLMs. This NPL File evaluates the **sentiment and response characteristics** of four leading Large Language Models (LLMs)—**Grok (xAI), Claude (Anthropic), Gemini (Google), and OpenAI (GPT-4)**—when prompted with **statistics exam questions** from four different sources:  

1. **ACTM Exam** (American College Testing Mathematics)  
2. **Caos Exam** (Comprehensive Assessment of Outcomes in Statistics)  
3. **SMU PhD Statistics Exam** (Southern Methodist University)  
4. **AP Statistics Exam** (Advanced Placement)  

The goal is to:  
✔ Measure **sentiment polarity** (positive, neutral, negative) in LLM responses  
✔ Analyze **response length, complexity, and confidence**  
✔ Compare **performance across different exam difficulty levels**  
✔ Generate **summary statistics & visualizations**  

## 📂 Repository Structure  
<img width="481" alt="Screenshot 2025-04-29 at 4 27 51 PM" src="https://github.com/user-attachments/assets/f03a2ff4-7612-4c10-94fd-27aae9c1066f" />



## 🛠️ Installation & Setup  

### Prerequisites  
- Python 3.10+  
- API keys for:  
  - OpenAI (`OPENAI_API_KEY`)  
  - Anthropic (`ANTHROPIC_API_KEY`)  
  - Google Gemini (`GOOGLE_API_KEY`)  
  - Grok (via X/Twitter API if applicable)  

### Installation Steps  
1. Clone the repo:  
   ```bash  
   git clone https://github.com/cabrerajulian401/NLP-Research.git  
   cd NLP-Research
2. Instal Dependencies python -m venv venv  
source venv/bin/activate  # Linux/Mac | Windows: `venv\Scripts\activate`  
pip install -r requirements.txt

3. Add API KEys to a .env file

### 🚀 Usage Steps
   
##### 1. Query LLMs with Exam Questions
   
Run the following automated prompts across all LLM models and save responses in Word Documents:

python scripts/query_llms.py \  
    --exam_actm data/ACTM_questions.txt \  
    --exam_caos data/Caos_questions.txt \  
    --exam_smu data/SMU_PhD_questions.txt \  
    --exam_ap data/AP_Stats_questions.txt \  
    --output_dir results/raw_responses  
    
#### 2. Analyze Sentiment & Metrics
Extract polarity, subjectivity, and linguistic features:

python scripts/analyze_sentiment.py \  
    --input_dir results/raw_responses \  
    --output_dir results/sentiment_scores  
    
#### 3. Generate Comparative Statistics
Compute aggregate metrics (mean sentiment, response length, etc.):

python scripts/generate_metrics.py \  
    --sentiment_dir results/sentiment_scores \  
    --output_csv results/summary_stats.csv  
    
#### 4. Visualize Results
Explore Jupyter notebooks for interactive analysis:

jupyter notebook notebooks/Sentiment_Analysis.ipynb  

### 📊 Key Metrics Analyzed:

-Sentiment Polarity	[-1, 1] scale (negative → positive)	VADER

-TextBlob Subjectivity	[0, 1] scale (objective → subjective)

-TextBlob Response Length	

-Word/token count - NLTK

-Lexical Diversity	Unique word ratio	len(set(words)) / len(words)

#### 📝 Simple Example Output:
Summary Table of Avg. Sentiment Score [-1,1] and Avg. Word Count (Hypothetical Results)

GPT-4	0.21	142 words	

Claude	0.15	178 words	

Gemini	0.12	115 words	

Grok	-0.03	94 words	

#### 🤝 Contributing
Report issues or suggest enhancements via GitHub Issues.

Fork & submit PRs for:

Additional LLMs (e.g., Mistral, LLaMA)

New exam datasets

Enhanced visualization tools

📜 License
MIT License. See LICENSE.

#### 📬 Contact
Julian Cabrera

Email: cabrerajulian401@gmail.com

GitHub: @cabrerajulian401
