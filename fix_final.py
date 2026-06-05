with open('draft.tex', 'r') as f:
    text = f.read()

# 1. Delete speculative section in conclusion
text = text.replace(
    r"The SeparationCross paradigm provides a blueprint for feature-level" + "\n" +
    r"fusion of any 1D biomedical signal with its 2D scale-domain" + "\n" +
    r"representation, with natural extensions to ECG--spectrogram and" + "\n" +
    r"EEG--scalogram fusion. ",
    ""
)

# 2. Update RamanNet baseline in Table (tab:overall) and Baselines
text = text.replace(
    r"\item \textbf{Raman-only}: 1D CNN on preprocessed spectra only.",
    r"\item \textbf{Raman-only}: 1D CNN (following the RamanNet architecture~\cite{zhou2022ramannet}) on preprocessed spectra only."
)
text = text.replace(
    r"Raman-only (1D)",
    r"Raman-only (RamanNet~\cite{zhou2022ramannet})"
)

# 3. British/Typo spellings
text = text.replace("Denoizing.", "Denoising.")
text = text.replace("comprizing", "comprising")
text = text.replace("programmes", "programs")

with open('draft.tex', 'w') as f:
    f.write(text)

print('Fixed final issues.')
