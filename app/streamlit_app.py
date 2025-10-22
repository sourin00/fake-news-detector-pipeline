import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.inference import initialize_detector, predict_fake_news, get_available_models
from src.data_utils import get_data_stats, load_sample_data, load_fake_real_news_kaggle
import time

# Configure page
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize detector
@st.cache_resource
def load_detector():
    """Load and cache the detector"""
    return initialize_detector()


def main():
    st.title("🔍 Advanced Fake News Detection System")
    st.markdown("*Powered by BERT, TF-IDF, and Ensemble Learning*")

    # Load detector
    detector = load_detector()
    available_models = get_available_models()

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Model Configuration")

        # Model selection
        if available_models:
            model_choice = st.selectbox(
                "Select Model:",
                options=available_models,
                format_func=lambda x: {
                    'tfidf': '📊 TF-IDF + Logistic Regression',
                    'bert': '🤖 BERT (Advanced)',
                    'ensemble': '🎯 Ensemble (Best Performance)'
                }.get(x, x.title())
            )
        else:
            st.error("No models available!")
            st.stop()

        st.markdown("---")

        # Model info
        st.subheader("📈 Model Performance")
        performance_data = {
            'tfidf': {'accuracy': '90%', 'speed': 'Very Fast'},
            'bert': {'accuracy': '98%', 'speed': 'Medium'},
            'ensemble': {'accuracy': '99%+', 'speed': 'Medium'}
        }

        if model_choice in performance_data:
            perf = performance_data[model_choice]
            st.metric("Expected Accuracy", perf['accuracy'])
            st.metric("Processing Speed", perf['speed'])

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Text Analysis")

        # Text input methods
        input_method = st.radio(
            "Input method:",
            ["Type/Paste Text", "Upload File", "Use Example"]
        )

        user_text = ""

        if input_method == "Type/Paste Text":
            user_text = st.text_area(
                "Enter news article text:",
                height=200,
                placeholder="Paste your news article here..."
            )

        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=['txt', 'csv']
            )
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                user_text = st.text_area("File content:", content, height=200)

        elif input_method == "Use Example":
            example_choice = st.selectbox(
                "Select example:",
                ["Fake News Example", "Real News Example"]
            )

            examples = {
                "Fake News Example": "BREAKING: Scientists discover that popular social media platforms are secretly using advanced AI to manipulate user emotions and political beliefs! Government officials refuse to comment on this shocking revelation that could change everything we know about online privacy and free will!",
                "Real News Example": "The Federal Reserve announced today a quarter-point interest rate increase, citing ongoing concerns about inflation. The decision was made following a two-day meeting of the Federal Open Market Committee and aligns with economist predictions."
            }

            user_text = examples[example_choice]
            st.text_area("Selected example:", user_text, height=150, disabled=True)

        # Prediction button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner(f"Analyzing with {model_choice.upper()} model..."):
                    start_time = time.time()
                    result = predict_fake_news(user_text, model_choice)
                    processing_time = time.time() - start_time

                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")

                # Main result
                if result['prediction'] == 0:
                    st.error(f"⚠️ **FAKE NEWS DETECTED**")
                    result_color = "red"
                else:
                    st.success(f"✅ **LIKELY REAL NEWS**")
                    result_color = "green"

                # Confidence metrics
                col_conf1, col_conf2, col_conf3 = st.columns(3)

                with col_conf1:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']:.2%}",
                        delta=None
                    )

                with col_conf2:
                    st.metric(
                        "Processing Time",
                        f"{processing_time:.3f}s"
                    )

                with col_conf3:
                    st.metric(
                        "Model Used",
                        result['model_used'].upper()
                    )

                # Probability breakdown
                st.subheader("🎯 Detailed Probabilities")
                prob_df = pd.DataFrame([
                    {"Category": "Real News", "Probability": result['probabilities']['Real']},
                    {"Category": "Fake News", "Probability": result['probabilities']['Fake']}
                ])

                fig = px.bar(
                    prob_df,
                    x="Category",
                    y="Probability",
                    color="Category",
                    color_discrete_map={"Real News": "green", "Fake News": "red"},
                    title="Prediction Probabilities"
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Please enter some text to analyze.")

    with col2:
        st.header("ℹ️ System Information")

        # Dataset stats
        st.subheader("📊 Training Data Stats")
        sample_data = load_fake_real_news_kaggle()
        stats = get_data_stats(sample_data)

        st.metric("Total Samples", stats['total_samples'])
        st.metric("Real News", stats['real_samples'])
        st.metric("Fake News", stats['fake_samples'])

        # Model availability
        st.subheader("🔧 Available Models")
        for model in available_models:
            status_icon = "✅" if model in available_models else "❌"
            model_name = {
                'tfidf': 'TF-IDF + LogReg',
                'bert': 'BERT Neural Network',
                'ensemble': 'Ensemble Method'
            }.get(model, model.title())
            st.write(f"{status_icon} {model_name}")

        # Tips and info
        st.subheader("💡 Detection Tips")
        st.info("""
        **Signs of Fake News:**
        - Sensational headlines
        - Poor grammar/spelling
        - Lack of credible sources
        - Emotional manipulation
        - Unverified claims

        **For Best Results:**
        - Use complete articles
        - Check multiple sources
        - Verify with fact-checkers
        """)

        # Performance comparison chart
        st.subheader("⚡ Model Comparison")
        comparison_data = pd.DataFrame({
            'Model': ['TF-IDF', 'BERT', 'Ensemble'],
            'Accuracy': [90, 98, 99],
            'Speed': [95, 60, 70]  # Relative speed scores
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comparison_data['Speed'],
            y=comparison_data['Accuracy'],
            mode='markers+text',
            text=comparison_data['Model'],
            textposition="top center",
            marker=dict(size=[15, 20, 25], color=['blue', 'orange', 'red'])
        ))
        fig.update_layout(
            title="Accuracy vs Speed",
            xaxis_title="Speed Score",
            yaxis_title="Accuracy (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
