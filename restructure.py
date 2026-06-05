import re

with open('draft.tex', 'r') as f:
    text = f.read()

# 5.1 - 5.5
text = text.replace(r'\subsection{Overall Classification Performance}', r'\subsection{Classification Performance and Feature Analysis}' + '\n' + r'\subsubsection{Overall Classification Performance}')
text = text.replace(r'\subsection{Training Dynamics}', r'\subsubsection{Training Dynamics}')
text = text.replace(r'\subsection{Confusion Matrix Analysis}', r'\subsubsection{Confusion Matrix Analysis}')
text = text.replace(r'\subsection{Per-Class Performance}', r'\subsubsection{Per-Class Performance}')
text = text.replace(r'\subsection{Feature Space Visualization}', r'\subsubsection{Feature Space Visualization}')

# 5.6 - 5.7
text = text.replace(r'\subsection{Spectral Attribution and Molecular Target Localization}', r'\subsection{Attribution and Modality Weighting}' + '\n' + r'\subsubsection{Spectral Attribution and Molecular Target Localization}')
text = text.replace(r'\subsection{Cross-Attention and Adaptive Modality Weighting}', r'\subsubsection{Cross-Attention and Adaptive Modality Weighting}')

# 5.8 - 5.10
text = text.replace(r'\subsection{Robustness to Noise and Baseline Drift}', r'\subsection{Robustness Analysis}' + '\n' + r'\subsubsection{Robustness to Noise and Baseline Drift}')
text = text.replace(r'\subsection{Band Occlusion Experiments}', r'\subsubsection{Band Occlusion Experiments}')
text = text.replace(r'\subsection{Data Scarcity and Class Imbalance}', r'\subsubsection{Data Scarcity and Class Imbalance}')

# 5.13 - 5.16
text = text.replace(r'\subsection{Comprehensive Architectural Ablations (30-Class Taxonomy)}', r'\subsection{Extended Scalability Study}' + '\n' + r'\subsubsection{Comprehensive Architectural Ablations (30-Class Taxonomy)}')
text = text.replace(r'\subsection{Visualization of Learned Branch Representations}', r'\subsubsection{Visualization of Learned Branch Representations}')
text = text.replace(r'\subsection{Cross-Dataset Generalization}', r'\subsubsection{Cross-Dataset Generalization}')
text = text.replace(r'\subsection{Open-World Scalability: 30-Species Taxonomy}', r'\subsubsection{Open-World Scalability: 30-Species Taxonomy}')

with open('draft.tex', 'w') as f:
    f.write(text)

print('Restructured subsections')
