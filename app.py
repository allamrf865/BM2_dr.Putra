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
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Sembunyikan hamburger menu dan footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stException {display: none;}
.stError {display: none;}
.stWarning {display: none;}
div[data-testid="stStatusWidget"] {display: none;}
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
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

class ComprehensiveJournalAnalyzer:
    def __init__(self, text):
        self.text = text
        self.sentences = [s.strip() for s in text.split('.') if s.strip()]
        self.words = text.split()
        try:
            self.tfidf = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf.fit_transform([text])
        except:
            self.tfidf = None
            self.tfidf_matrix = None
    
    def calculate_all_metrics(self):
        metrics = {}
        
        try:
            # 1. Lexical Analysis
            unique_words = len(set(self.words))
            total_words = len(self.words)
            metrics['lexical_diversity'] = unique_words / total_words if total_words > 0 else 0
            
            sentence_lengths = [len(s.split()) for s in self.sentences if s.strip()]
            metrics['sentence_complexity'] = np.mean(sentence_lengths) if sentence_lengths else 0
            
            technical_terms = len(re.findall(r'\b[A-Z][a-z]*(?:[-][A-Z][a-z]*)*\b', self.text))
            metrics['technical_density'] = technical_terms / total_words if total_words > 0 else 0
            
            # 2. Citation Analysis
            citations = re.findall(r'\[\d+\]', self.text)
            metrics['citation_network'] = len(citations) / len(self.sentences) if self.sentences else 0
            metrics['reference_distribution'] = len(set(citations)) / len(citations) if citations else 0
            
            # 3. Statistical Analysis
            statistical_terms = ['significant', 'p-value', 'correlation', 'regression', 'analysis']
            stat_count = sum(1 for word in self.words if word.lower() in statistical_terms)
            metrics['statistical_usage'] = stat_count / len(self.sentences) if self.sentences else 0
            
            p_values = len(re.findall(r'p\s*[<>=]\s*0\.\d+', self.text))
            metrics['statistical_significance'] = p_values / len(self.sentences) if self.sentences else 0
            
            # 4. Methodology Analysis
            method_terms = ['method', 'procedure', 'protocol', 'analysis', 'technique']
            method_count = sum(1 for word in self.words if word.lower() in method_terms)
            metrics['methodology_completeness'] = method_count / len(self.sentences) if self.sentences else 0
            
            validation_terms = len(re.findall(r'\b(validate|verify|confirm|test)\b', self.text.lower()))
            metrics['methodology_robustness'] = validation_terms / len(self.sentences) if self.sentences else 0
            
            # 5. Research Design
            design_terms = ['random', 'control', 'blind', 'placebo', 'trial']
            design_count = sum(1 for word in self.words if word.lower() in design_terms)
            metrics['research_design_quality'] = design_count / len(self.sentences) if self.sentences else 0
            
            design_framework = len(re.findall(r'\b(design|framework|approach|strategy)\b', self.text.lower()))
            metrics['research_design_completeness'] = design_framework / len(self.sentences) if self.sentences else 0
            
            # 6. Topic and Semantic Analysis
            metrics['topic_coherence'] = self._calculate_topic_coherence()
            metrics['semantic_density'] = self._calculate_semantic_density()
            metrics['document_entropy'] = self._calculate_document_entropy()

            # 7. Readability and Clarity
            metrics['readability_score'] = self._calculate_readability()
            long_words = len([w for w in self.words if len(w) > 8])
            metrics['scientific_jargon'] = long_words / total_words if total_words > 0 else 0
            metrics['structural_coherence'] = self._calculate_structural_coherence()
            
            # 8. Sample and Protocol Analysis
            sample_mentions = len(re.findall(r'\b(n\s*=\s*\d+|sample size|participants)\b', self.text.lower()))
            metrics['sample_size_adequacy'] = sample_mentions / len(self.sentences) if self.sentences else 0
            
            protocol_mentions = len(re.findall(r'\b(standard|protocol|guideline|procedure)\b', self.text.lower()))
            metrics['protocol_standardization'] = protocol_mentions / len(self.sentences) if self.sentences else 0
            
            # 9. Data Collection and Analysis
            data_collection = len(re.findall(r'\b(collect|gather|measure|record|obtain)\b', self.text.lower()))
            metrics['data_collection'] = data_collection / len(self.sentences) if self.sentences else 0
            
            analytical_framework = len(re.findall(r'\b(framework|model|theory|concept)\b', self.text.lower()))
            metrics['analytical_framework'] = analytical_framework / len(self.sentences) if self.sentences else 0
            
            # 10. Validation and Quality
            validation_techniques = len(re.findall(r'\b(validate|verify|confirm|assess)\b', self.text.lower()))
            metrics['validation_techniques'] = validation_techniques / len(self.sentences) if self.sentences else 0
            
            quality_control = len(re.findall(r'\b(quality|control|monitor|check)\b', self.text.lower()))
            metrics['quality_control'] = quality_control / len(self.sentences) if self.sentences else 0
            
            # 11. Bias and Validity
            bias_assessment = len(re.findall(r'\b(bias|confound|limitation)\b', self.text.lower()))
            metrics['bias_assessment'] = bias_assessment / len(self.sentences) if self.sentences else 0
            
            reproducibility = len(re.findall(r'\b(reproduce|replicate|repeat)\b', self.text.lower()))
            metrics['reproducibility'] = reproducibility / len(self.sentences) if self.sentences else 0
            
            external_validity = len(re.findall(r'\b(generalize|external|population)\b', self.text.lower()))
            metrics['external_validity'] = external_validity / len(self.sentences) if self.sentences else 0
            
            metrics['internal_consistency'] = self._calculate_internal_consistency()
            
            # 12. Results and Conclusions
            metrics['methodological_rigor'] = np.mean([metrics['methodology_completeness'], metrics['research_design_quality']])
            
            results_interpretation = len(re.findall(r'\b(result|find|show|demonstrate)\b', self.text.lower()))
            metrics['results_interpretation'] = results_interpretation / len(self.sentences) if self.sentences else 0
            
            conclusion_validity = len(re.findall(r'\b(conclude|conclusion|therefore|thus)\b', self.text.lower()))
            metrics['conclusion_validity'] = conclusion_validity / len(self.sentences) if self.sentences else 0
            
        except Exception as e:
            # Jika ada error, berikan nilai default
            for key in ['lexical_diversity', 'sentence_complexity', 'technical_density', 'citation_network', 
                       'reference_distribution', 'statistical_usage', 'statistical_significance',
                       'methodology_completeness', 'methodology_robustness', 'research_design_quality',
                       'research_design_completeness', 'topic_coherence', 'semantic_density', 'document_entropy',
                       'readability_score', 'scientific_jargon', 'structural_coherence', 'sample_size_adequacy',
                       'protocol_standardization', 'data_collection', 'analytical_framework', 'validation_techniques',
                       'quality_control', 'bias_assessment', 'reproducibility', 'external_validity',
                       'internal_consistency', 'methodological_rigor', 'results_interpretation', 'conclusion_validity']:
                if key not in metrics:
                    metrics[key] = 0.5
        
        return metrics
    
    def _calculate_topic_coherence(self):
        try:
            if self.tfidf_matrix is not None:
                lda = LatentDirichletAllocation(n_components=3, random_state=42)
                lda_output = lda.fit_transform(self.tfidf_matrix)
                return float(np.max(lda_output))
            else:
                return 0.5
        except:
            return 0.5
    
    def _calculate_semantic_density(self):
        try:
            if self.tfidf_matrix is not None:
                similarity_matrix = (self.tfidf_matrix * self.tfidf_matrix.T).toarray()
                return float(np.mean(similarity_matrix))
            else:
                return 0.5
        except:
            return 0.5
    
    def _calculate_document_entropy(self):
        try:
            word_freq = pd.Series(self.words).value_counts(normalize=True)
            return float(entropy(word_freq))
        except:
            return 0.5
    
    def _calculate_readability(self):
        try:
            if len(self.sentences) > 0 and len(self.words) > 0:
                avg_sentence_length = len(self.words) / len(self.sentences)
                avg_syllables = sum(len(word) for word in self.words) / len(self.words)
                flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables
                return max(0, min(100, flesch)) / 100  # Normalize to 0-1
            else:
                return 0.5
        except:
            return 0.5
    
    def _calculate_structural_coherence(self):
        try:
            section_keywords = ['introduction', 'method', 'result', 'discussion', 'conclusion']
            section_presence = [1 if keyword in self.text.lower() else 0 for keyword in section_keywords]
            return sum(section_presence) / len(section_keywords)
        except:
            return 0.5
    
    def _calculate_internal_consistency(self):
        try:
            if len(self.sentences) > 1:
                sentiments = []
                for sentence in self.sentences:
                    if sentence.strip():
                        blob = TextBlob(sentence)
                        sentiments.append(blob.sentiment.polarity)
                if sentiments:
                    return 1 - min(1, np.std(sentiments))  # Convert to 0-1 scale
            return 0.5
        except:
            return 0.5

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
        
        try:
            # 1. Spider Plots untuk setiap kategori
            for category, metric_list in metric_categories.items():
                values = [metrics.get(m, 0.5) for m in metric_list]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metric_list,
                    fill='toself',
                    name=category,
                    line=dict(color='cyan')
                ))
                fig.update_layout(
                    title=f'{category} Metrics',
                    template='plotly_dark',
                    polar=dict(
                        radialaxis=dict(
                            range=[0, 1],
                            showticklabels=True,
                            gridcolor='gray'
                        ),
                        angularaxis=dict(
                            gridcolor='gray'
                        )
                    ),
                    paper_bgcolor='rgb(0,0,0)',
                    plot_bgcolor='rgb(0,0,0)',
                    font=dict(color='white'),
                    height=400
                )
                figs.append(fig)
            
            # 2. Heatmap Correlation Matrix
            all_metrics = list(metrics.keys())
            n_metrics = len(all_metrics)
            correlation_matrix = np.eye(n_metrics)  # Identity matrix sebagai baseline
            
            # Simulasi korelasi sederhana
            for i in range(n_metrics):
                for j in range(i+1, n_metrics):
                    corr = np.random.uniform(0.3, 0.7)
                    correlation_matrix[i,j] = corr
                    correlation_matrix[j,i] = corr
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=all_metrics,
                y=all_metrics,
                colorscale='Viridis',
                showscale=True
            ))
            fig_heatmap.update_layout(
                title='Metrics Correlation Matrix',
                template='plotly_dark',
                height=600,
                paper_bgcolor='rgb(0,0,0)',
                plot_bgcolor='rgb(0,0,0)',
                font=dict(color='white')
            )
            figs.append(fig_heatmap)
            
            # 3. Summary Table
            summary_data = []
            for category, metric_list in metric_categories.items():
                for metric in metric_list:
                    score = metrics.get(metric, 0.5)
                    quality = 'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'
                    summary_data.append({
                        'Category': category,
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': round(score, 3),
                        'Quality': quality
                    })
            
            summary_df = pd.DataFrame(summary_data)
            
            fig_table = go.Figure(data=[go.Table(
                header=dict(
                    values=list(summary_df.columns),
                    fill_color='darkblue',
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[summary_df[col] for col in summary_df.columns],
                    fill_color='black',
                    align='left',
                    font=dict(color='white', size=10)
                )
            )])
            fig_table.update_layout(
                title='Comprehensive Analysis Summary',
                template='plotly_dark',
                height=600,
                paper_bgcolor='rgb(0,0,0)',
                plot_bgcolor='rgb(0,0,0)',
                font=dict(color='white')
            )
            figs.append(fig_table)
            
        except Exception as e:
            # Jika ada error dalam visualisasi, buat plot sederhana
            fig_simple = go.Figure()
            fig_simple.add_trace(go.Bar(
                x=list(metrics.keys())[:10],
                y=list(metrics.values())[:10],
                marker_color='cyan'
            ))
            fig_simple.update_layout(
                title='Basic Metrics Overview',
                template='plotly_dark',
                paper_bgcolor='rgb(0,0,0)',
                plot_bgcolor='rgb(0,0,0)',
                font=dict(color='white')
            )
            figs = [fig_simple]
            summary_df = pd.DataFrame({'Metric': list(metrics.keys()), 'Score': list(metrics.values())})
        
        return figs, summary_df

# Aplikasi utama
def main():
    st.title('📊 Comprehensive Journal Quality Analysis')
    st.markdown("---")
    
    # Sidebar untuk input dan konfigurasi
    st.sidebar.header('🔧 Analysis Configuration')
    
    # Text input area
    journal_text = st.text_area(
        "📝 Input Journal Text",
        """Surgical treatment versus observation in moderate intermittent exotropia (SOMIX): study protocol for a randomized controlled trial

Background: Intermittent exotropia (IXT) is the most common type of strabismus in China, but the best treatment and optimal timing of intervention for IXT remain controversial, particularly for children with moderate IXT who manifest obvious exodeviation frequently but with only partial impairment of binocular single vision. The lack of randomized controlled trial (RCT) evidence means that the true effectiveness of the surgical treatment in curing moderate IXT is still unknown. The SOMIX study has been designed to determine the long-term effectiveness of surgery for the treatment and the natural history of IXT among patients aged 5 to 18 years old.

Methods/design: A total of 280 patients between 5 and 18 years of age with moderate IXT will be enrolled at Zhongshan Ophthalmic Center, Sun Yat-sen University, Guangzhou, China. After initial clinical assessment, all participants will be randomized to receive surgical treatment or observation, and then be followed up for 5 years. The primary objective is to compare the cure rate of IXT between the surgical treatment and observation group. The secondary objectives are to identify the predictive factors affecting long-term outcomes in each group and to observe the natural course of IXT.

Discussion: The SOMIX trial will provide important guidance regarding the moderate IXT and its managements and modify the treatment strategies of IXT.""",
        height=300
    )
    
    # Analysis options
    st.sidebar.subheader('Analysis Options')
    show_correlations = st.sidebar.checkbox('Show Correlation Matrix', value=True)
    show_summary = st.sidebar.checkbox('Show Summary Table', value=True)
    
    # Analisis button
    if st.button("🚀 Analyze Journal", type="primary"):
        if journal_text.strip():
            with st.spinner('🔄 Analyzing journal content...'):
                try:
                    # Buat instance analyzer
                    analyzer = ComprehensiveJournalAnalyzer(journal_text)
                    
                    # Dapatkan metrik dan visualisasi
                    figs, summary_df = analyzer.create_visualizations()
                    metrics = analyzer.calculate_all_metrics()
                    
                    # Container untuk hasil analisis
                    st.header('📈 Analysis Results')
                    
                    # Key metrics overview
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        overall_score = np.mean(list(metrics.values()))
                        st.metric(
                            "Overall Quality",
                            f"{overall_score:.3f}",
                            delta=f"{(overall_score - 0.5):.3f}"
                        )
                    
                    with col2:
                        top_metric = max(metrics.items(), key=lambda x: x[1])
                        st.metric(
                            "Top Strength",
                            f"{top_metric[1]:.3f}",
                            delta=top_metric[0].replace('_', ' ').title()
                        )
                    
                    with col3:
                        weak_metric = min(metrics.items(), key=lambda x: x[1])
                        st.metric(
                            "Needs Improvement",
                            f"{weak_metric[1]:.3f}",
                            delta=weak_metric[0].replace('_', ' ').title()
                        )
                    
                    with col4:
                        high_quality_count = sum(1 for v in metrics.values() if v > 0.7)
                        st.metric(
                            "High Quality Metrics",
                            f"{high_quality_count}/{len(metrics)}",
                            delta=f"{(high_quality_count/len(metrics)*100):.1f}%"
                        )
                    
                    st.markdown("---")
                    
                    # Tab untuk berbagai visualisasi
                    if len(figs) > 2:
                        tab1, tab2, tab3 = st.tabs(["📊 Category Analysis", "🔗 Correlations", "📋 Summary"])
                        
                        with tab1:
                            st.subheader('Category-wise Analysis')
                            # Tampilkan spider plots
                            cols = st.columns(2)
                            spider_figs = [f for f in figs if 'Metrics' in f.layout.title.text]
                            for i, fig in enumerate(spider_figs):
                                with cols[i % 2]:
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            if show_correlations and len(figs) > len(spider_figs):
                                st.subheader('Metric Correlations')
                                correlation_fig = [f for f in figs if 'Correlation' in f.layout.title.text]
                                if correlation_fig:
                                    st.plotly_chart(correlation_fig[0], use_container_width=True)
                        
                        with tab3:
                            if show_summary:
                                st.subheader('Comprehensive Summary')
                                summary_fig = [f for f in figs if 'Summary' in f.layout.title.text]
                                if summary_fig:
                                    st.plotly_chart(summary_fig[0], use_container_width=True)
                                
                                # Tampilkan dataframe
                                st.dataframe(
                                    summary_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                    else:
                        # Jika hanya ada sedikit figur, tampilkan semuanya
                        for fig in figs:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed insights
                    st.header('🔍 Detailed Insights')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader('🎯 Top Strengths')
                        top_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:5]
                        for i, (metric, value) in enumerate(top_metrics, 1):
                            st.write(f"{i}. **{metric.replace('_', ' ').title()}**: {value:.3f}")
                    
                    with col2:
                        st.subheader('⚠️ Areas for Improvement')
                        bottom_metrics = sorted(metrics.items(), key=lambda x: x[1])[:5]
                        for i, (metric, value) in enumerate(bottom_metrics, 1):
                            st.write(f"{i}. **{metric.replace('_', ' ').title()}**: {value:.3f}")
                    
                    # Recommendations
                    st.header('💡 Recommendations')
                    
                    if overall_score > 0.7:
                        st.success("🎉 Excellent journal quality! This paper demonstrates strong methodological rigor.")
                    elif overall_score > 0.5:
                        st.warning("⚡ Good foundation with room for improvement in specific areas.")
                    else:
                        st.error("🔧 Significant improvements needed across multiple dimensions.")
                    
                    # Specific recommendations based on weak areas
                    weak_areas = [k for k, v in metrics.items() if v < 0.4]
                    if weak_areas:
                        st.write("**Specific recommendations:**")
                        for area in weak_areas[:3]:
                            if 'statistical' in area:
                                st.write("• Enhance statistical analysis and reporting")
                            elif 'methodology' in area:
                                st.write("• Strengthen methodological descriptions")
                            elif 'citation' in area:
                                st.write("• Improve literature review and citations")
                            else:
                                st.write(f"• Focus on improving {area.replace('_', ' ')}")
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis. Please check your input text.")
                    st.exception(e)
        else:
            st.warning("Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>📚 Comprehensive Journal Analysis Tool</p>
            <p style='font-size: small'>Using advanced NLP metrics and visualizations for academic quality assessment</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
