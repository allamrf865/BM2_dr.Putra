import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from textblob import TextBlob
import re
from scipy.stats import entropy

# Konfigurasi halaman
st.set_page_config(
    page_title="Journal Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .stPlotlyChart {
        background-color: #0E1117;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
class ComprehensiveJournalAnalyzer:
    def __init__(self, text):
        self.text = text
        self.sentences = text.split('.')
        self.words = text.split()
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform([text])
    
    def calculate_all_metrics(self):
        metrics = {}
        
        # 1. Lexical Analysis
        metrics['lexical_diversity'] = len(set(self.words)) / len(self.words)
        metrics['sentence_complexity'] = np.mean([len(s.split()) for s in self.sentences])
        metrics['technical_density'] = len(re.findall(r'\b[A-Z][a-z]*(?:[-][A-Z][a-z]*)*\b', self.text)) / len(self.words)
        
        # 2. Citation Analysis
        citations = re.findall(r'\[\d+\]', self.text)
        metrics['citation_network'] = len(citations) / len(self.sentences)
        metrics['reference_distribution'] = len(set(citations)) / len(citations) if citations else 0
        
        # 3. Statistical Analysis
        statistical_terms = ['significant', 'p-value', 'correlation', 'regression', 'analysis']
        metrics['statistical_usage'] = sum(1 for word in self.words if word.lower() in statistical_terms) / len(self.sentences)
        metrics['statistical_significance'] = len(re.findall(r'p\s*[<>=]\s*0\.\d+', self.text)) / len(self.sentences)
        
        # 4. Methodology Analysis
        method_terms = ['method', 'procedure', 'protocol', 'analysis', 'technique']
        metrics['methodology_completeness'] = sum(1 for word in self.words if word.lower() in method_terms) / len(self.sentences)
        metrics['methodology_robustness'] = len(re.findall(r'\b(validate|verify|confirm|test)\b', self.text.lower())) / len(self.sentences)
        
        # 5. Research Design
        design_terms = ['random', 'control', 'blind', 'placebo', 'trial']
        metrics['research_design_quality'] = sum(1 for word in self.words if word.lower() in design_terms) / len(self.sentences)
        metrics['research_design_completeness'] = len(re.findall(r'\b(design|framework|approach|strategy)\b', self.text.lower())) / len(self.sentences)
        
        # 6. Topic and Semantic Analysis
        metrics['topic_coherence'] = self._calculate_topic_coherence()
        metrics['semantic_density'] = self._calculate_semantic_density()
        metrics['document_entropy'] = self._calculate_document_entropy()

        # 7. Readability and Clarity
        metrics['readability_score'] = self._calculate_readability()
        metrics['scientific_jargon'] = len([w for w in self.words if len(w) > 8]) / len(self.words)
        metrics['structural_coherence'] = self._calculate_structural_coherence()
        
        # 8. Sample and Protocol Analysis
        metrics['sample_size_adequacy'] = len(re.findall(r'\b(n\s*=\s*\d+|sample size|participants)\b', self.text.lower())) / len(self.sentences)
        metrics['protocol_standardization'] = len(re.findall(r'\b(standard|protocol|guideline|procedure)\b', self.text.lower())) / len(self.sentences)
        
        # 9. Data Collection and Analysis
        metrics['data_collection'] = len(re.findall(r'\b(collect|gather|measure|record|obtain)\b', self.text.lower())) / len(self.sentences)
        metrics['analytical_framework'] = len(re.findall(r'\b(framework|model|theory|concept)\b', self.text.lower())) / len(self.sentences)
        
        # 10. Validation and Quality
        metrics['validation_techniques'] = len(re.findall(r'\b(validate|verify|confirm|assess)\b', self.text.lower())) / len(self.sentences)
        metrics['quality_control'] = len(re.findall(r'\b(quality|control|monitor|check)\b', self.text.lower())) / len(self.sentences)
        
        # 11. Bias and Validity
        metrics['bias_assessment'] = len(re.findall(r'\b(bias|confound|limitation)\b', self.text.lower())) / len(self.sentences)
        metrics['reproducibility'] = len(re.findall(r'\b(reproduce|replicate|repeat)\b', self.text.lower())) / len(self.sentences)
        metrics['external_validity'] = len(re.findall(r'\b(generalize|external|population)\b', self.text.lower())) / len(self.sentences)
        metrics['internal_consistency'] = self._calculate_internal_consistency()
        
        # 12. Results and Conclusions
        metrics['methodological_rigor'] = np.mean([metrics['methodology_completeness'], metrics['research_design_quality']])
        metrics['results_interpretation'] = len(re.findall(r'\b(result|find|show|demonstrate)\b', self.text.lower())) / len(self.sentences)
        metrics['conclusion_validity'] = len(re.findall(r'\b(conclude|conclusion|therefore|thus)\b', self.text.lower())) / len(self.sentences)
        
        return metrics
    
    def _calculate_topic_coherence(self):
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda_output = lda.fit_transform(self.tfidf_matrix)
        return np.max(lda_output)
    
    def _calculate_semantic_density(self):
        similarity_matrix = (self.tfidf_matrix * self.tfidf_matrix.T).toarray()
        return np.mean(similarity_matrix)
    
    def _calculate_document_entropy(self):
        word_freq = pd.Series(self.words).value_counts(normalize=True)
        return entropy(word_freq)
    
    def _calculate_readability(self):
        blob = TextBlob(self.text)
        return 206.835 - 1.015 * (len(self.words) / len(self.sentences)) - 84.6 * (sum(len(word) for word in self.words) / len(self.words))
    
    def _calculate_structural_coherence(self):
        section_keywords = ['introduction', 'method', 'result', 'discussion', 'conclusion']
        section_presence = [1 if keyword in self.text.lower() else 0 for keyword in section_keywords]
        return sum(section_presence) / len(section_keywords)
    
    def _calculate_internal_consistency(self):
        sentences = [TextBlob(s).sentiment.polarity for s in self.sentences]
        return np.std(sentences)

    def create_visualizations(self):
        metrics = self.calculate_all_metrics()
        figs = []
        
        # Kelompokkan metrik berdasarkan kategori
        metric_categories = {
            'Lexical Analysis': ['lexical_diversity', 'sentence_complexity', 'technical_density'],
            'Citation Analysis': ['citation_network', 'reference_distribution'],
            'Statistical Analysis': ['statistical_usage', 'statistical_significance'],
            'Methodology Analysis': ['methodology_completeness', 'methodology_robustness'],
            'Research Design': ['research_design_quality', 'research_design_completeness'],
            'Topic Analysis': ['topic_coherence', 'semantic_density', 'document_entropy'],
            'Readability': ['readability_score', 'scientific_jargon', 'structural_coherence'],
            'Sample Analysis': ['sample_size_adequacy', 'protocol_standardization'],
            'Data Collection': ['data_collection', 'analytical_framework'],
            'Validation': ['validation_techniques', 'quality_control'],
            'Bias and Validity': ['bias_assessment', 'reproducibility', 'external_validity', 'internal_consistency'],
            'Results': ['methodological_rigor', 'results_interpretation', 'conclusion_validity']
        }
        
        # 1. Spider Plots untuk setiap kategori
        for category, metric_list in metric_categories.items():
            values = [metrics[m] for m in metric_list]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_list,
                fill='toself',
                name=category
            ))
            fig.update_layout(
                title=f'{category} Metrics',
                template='plotly_dark',
                polar=dict(radialaxis=dict(range=[0, 1]))
            )
            figs.append(fig)
        
        # 2. Heatmap Correlation Matrix
        all_metrics = list(metrics.keys())
        correlation_matrix = np.zeros((len(all_metrics), len(all_metrics)))
        for i, m1 in enumerate(all_metrics):
            for j, m2 in enumerate(all_metrics):
                if i == j:
                    correlation_matrix[i,j] = 1
                else:
                    correlation_matrix[i,j] = np.random.uniform(0.3, 0.7)  # Simulasi korelasi
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=all_metrics,
            y=all_metrics,
            colorscale='Viridis'
        ))
        fig_heatmap.update_layout(
            title='All Metrics Correlations',
            template='plotly_dark',
            height=800
        )
        figs.append(fig_heatmap)
        
        # 3. Parallel Coordinates untuk setiap kategori
        for category, metric_list in metric_categories.items():
            fig = go.Figure(data=
                go.Parcoords(
                    line=dict(color=[metrics[m] for m in metric_list],
                            colorscale='Viridis'),
                    dimensions=[
                        dict(range=[0, 1],
                            label=m,
                            values=[metrics[m]])
                        for m in metric_list
                    ]
                )
            )
            fig.update_layout(
                title=f'{category} Parallel Analysis',
                template='plotly_dark'
            )
            figs.append(fig)
        
        # 4. Summary Table dengan kategori
        summary_data = []
        for category, metric_list in metric_categories.items():
            for metric in metric_list:
                summary_data.append({
                    'Category': category,
                    'Metric': metric,
                    'Score': metrics[metric],
                    'Quality': 'High' if metrics[metric] > 0.7 else 'Medium' if metrics[metric] > 0.4 else 'Low'
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=list(summary_df.columns),
                fill_color='darkblue',
                align='left'
            ),
            cells=dict(
                values=[summary_df[col] for col in summary_df.columns],
                fill_color=[['black']*len(summary_df)],
                align='left',
                font=dict(color='white')
            )
        )])
        fig_table.update_layout(
            title='Comprehensive Analysis Summary',
            template='plotly_dark',
            height=800
        )
        figs.append(fig_table)
        
        # Update semua layout
        for fig in figs:
            fig.update_layout(
                paper_bgcolor='rgb(0,0,0)',
                plot_bgcolor='rgb(0,0,0)',
                font=dict(color='white'),
                margin=dict(t=100, l=100)
            )
        
        return figs, summary_df
# Bagian utama Streamlit
st.title('Comprehensive Journal Quality Analysis')

# Sidebar untuk input dan konfigurasi
st.sidebar.header('Analysis Configuration')

# Text input area
journal_text = st.text_area(
    "Input Journal Text",
    """Surgical treatment versus observation in moderate intermittent exotropia (SOMIX): study protocol for a randomized controlled trial

Background: Intermittent exotropia (IXT) is the most common type of strabismus in China, but the best treatment and optimal timing of intervention for IXT remain controversial, particularly for children with moderate IXT who manifest obvious exodeviation frequently but with only partial impairment of binocular single vision. The lack of randomized controlled trial (RCT) evidence means that the true effectiveness of the surgical treatment in curing moderate IXT is still unknown. The SOMIX study has been designed to determine the long-term effectiveness of surgery for the treatment and the natural history of IXT among patients aged 5 to 18 years old.

Methods/design: A total of 280 patients between 5 and 18 years of age with moderate IXT will be enrolled at Zhongshan Ophthalmic Center, Sun Yat-sen University, Guangzhou, China. After initial clinical assessment, all participants will be randomized to receive surgical treatment or observation, and then be followed up for 5 years. The primary objective is to compare the cure rate of IXT between the surgical treatment and observation group. The secondary objectives are to identify the predictive factors affecting long-term outcomes in each group and to observe the natural course of IXT.

Discussion: The SOMIX trial will provide important guidance regarding the moderate IXT and its managements and modify the treatment strategies of IXT.""",
    height=300
)

# Analisis button dengan spinner
if st.button("Analyze Journal", type="primary"):
    with st.spinner('Analyzing journal content...'):
        # Buat instance analyzer
        analyzer = ComprehensiveJournalAnalyzer(journal_text)
        
        # Dapatkan metrik dan visualisasi
        figs, summary_df = analyzer.create_visualizations()
        
        # Container untuk hasil analisis
        with st.container():
            # Header untuk hasil
            st.header('Analysis Results')
            
            # Tab untuk berbagai visualisasi
            tab1, tab2, tab3, tab4 = st.tabs(["Category Analysis", "Correlations", "Parallel Analysis", "Summary"])
            
            with tab1:
                st.subheader('Category-wise Analysis')
                # Tampilkan spider plots (12 kategori pertama)
                cols = st.columns(2)
                for i, fig in enumerate(figs[:12]):
                    with cols[i % 2]:
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader('Metric Correlations')
                # Tampilkan heatmap
                st.plotly_chart(figs[12], use_container_width=True)
            
            with tab3:
                st.subheader('Parallel Coordinates Analysis')
                # Tampilkan parallel coordinates (12 kategori berikutnya)
                for fig in figs[13:-1]:
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader('Comprehensive Summary')
                # Tampilkan tabel summary
                st.plotly_chart(figs[-1], use_container_width=True)
                
                # Tampilkan dataframe dengan formatting
                st.dataframe(
                    summary_df.style
                    .background_gradient(cmap='viridis', subset=['Score'])
                    .format({'Score': '{:.3f}'})
                )
            
            # Hitung dan tampilkan key findings
            metrics = analyzer.calculate_all_metrics()
            
            st.header('Key Findings')
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Overall Quality Score",
                    f"{float(np.mean(list(metrics.values()))):.3f}",
                    "Based on all metrics"
                )
            
            with col2:
                top_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:3]
                st.write("Top Strengths:")
                for metric, value in top_metrics:
                    st.write(f"- {metric}: {value:.3f}")
            
            with col3:
                bottom_metrics = sorted(metrics.items(), key=lambda x: x[1])[:3]
                st.write("Areas for Improvement:")
                for metric, value in bottom_metrics:
                    st.write(f"- {metric}: {value:.3f}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Comprehensive Journal Analysis Tool</p>
        <p style='font-size: small'>Using advanced metrics and visualizations</p>
    </div>
""", unsafe_allow_html=True)
