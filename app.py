import streamlit as st
import nltk
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Resume Screening App",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto"
)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models
@st.cache_resource
def load_models():
    try:
        clf = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        return clf, tfidf
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

clf, tfidf = load_models()

# Category mapping
CATEGORY_MAPPING = {
    0: 'Advocate',
    1: 'Arts',
    2: 'Automation Testing',
    3: 'Blockchain',
    4: 'Business Analyst',
    5: 'Civil Engineer',
    6: 'Data Science',
    7: 'Database',
    8: 'DevOps Engineer',
    9: 'DotNet Developer',
    10: 'ETL Developer',
    11: 'Electrical Engineering',
    12: 'HR',
    13: 'Hadoop',
    14: 'Health and fitness',
    15: 'Java Developer',
    16: 'Mechanical Engineer',
    17: 'Network Security Engineer',
    18: 'Operations Manager',
    19: 'PMO',
    20: 'Python Developer',
    21: 'SAP Developer',
    22: 'Sales',
    23: 'Testing',
    24: 'Web Designing'
}

def clean_resume_text(txt):
    """Clean and preprocess resume text."""
    cleantxt = re.sub('http\S+', ' ', txt)  
    cleantxt = re.sub('RT|cc', ' ', cleantxt)
    cleantxt = re.sub('@\S+', ' ', cleantxt)
    cleantxt = re.sub('#\S+', ' ', cleantxt)
    cleantxt = re.sub('[%s]' % re.escape('''!"#$%^&*()*+,-./:;<=>?@[\]^_`{|}~'''), ' ', cleantxt)
    cleantxt = re.sub(r'[^\x00-\x7f]', ' ', cleantxt)
    cleantxt = re.sub('\s+', ' ', cleantxt)  
    return cleantxt.strip()

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file."""
    try:
        resume_bytes = uploaded_file.read()
        try:
            return resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return resume_bytes.decode('latin-1')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def main():
    # App header
    st.title("📄 Resume Screening App")
    st.markdown("""
    Upload a resume (PDF or TXT) to predict the job category it belongs to.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This app uses machine learning to classify resumes into different job categories.
        - Upload a resume file
        - Get predicted job category
        """)
        
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a resume file", 
        type=['txt', 'pdf'],
        accept_multiple_files=False,
        help="Upload a PDF or TXT file containing the resume text"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing your resume..."):
            # Extract and clean text
            resume_text = extract_text_from_file(uploaded_file)
            
            if resume_text:
                cleaned_resume = clean_resume_text(resume_text)
                
                # Transform and predict
                try:
                    input_features = tfidf.transform([cleaned_resume])
                    prediction_id = clf.predict(input_features)[0]
                    category_name = CATEGORY_MAPPING.get(prediction_id, "Unknown")
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Category ID", prediction_id)
                    with col2:
                        st.metric("Predicted Category", category_name)
                        
                    # Show some cleaned text
                    with st.expander("View processed text"):
                        st.text(cleaned_resume[:500] + "...")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()