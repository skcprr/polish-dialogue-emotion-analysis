# polish-dialogue-emotion-analysis

## 1. Project Description & Scope
This project delivers an automated solution for analyzing Polish customer support interactions. It acts as a dual-engine analytical pipeline that ingests raw customer text and simultaneously performs two distinct Natural Language Processing (NLP) tasks: identifying the overall emotional state of the user and extracting individual business aspects along with their specific sentiment polarity.

The scope of the project encompasses:
* **End-to-end data preprocessing:** Parsing complex nested structures and aligning linguistic tokens with sequence labels.
* **Model Fine-Tuning:** Training a transformer-based token classification network on domain-specific Polish corpora.
* **Production-ready Deployment:** Creating a stateless, high-performance REST API capable of serving parallel neural network inferences in real-time.

## 2. Project Objective
The main objective was to design and implement a multi-class and aspect-based sentiment analysis system optimized for Polish consumer dialogues. 

The system implements the following core functionalities:
* **Global Emotion Detection:** Classifies the overall text into one of six primary categories: *anger (złość), fear (strach), disgust (wstręt), sadness (smutek), joy (radość), or neutral (neutralny)*.
* **Aspect Extraction:** Automatically isolates specific nouns, noun phrases, or corporate entities that represent the target of the customer's opinion.
* **Aspect-Based Sentiment Analysis (ABSA):** Assigns an independent emotional polarity (*positive, negative, or neutral*) to each extracted aspect, resolving mixed-sentiment sentences.
* **Confidence Scoring:** Computes percentage-based model certainty metrics for every prediction, allowing downstream filtering of low-confidence results.
* **Real-time REST API:** Exposes a high-throughput endpoint built with FastAPI, returning structured, predictable JSON payloads.

## 3. Glossary of Terms
1. **Aspect:** A key object, feature, service, or business element that acts as the target of a customer's evaluation, praise, or complaint within the text.
2. **Aspect-Based Sentiment Analysis (ABSA):** An advanced NLP task that moves beyond global document-level sentiment to identify individual targets within an utterance and map distinct sentiment polarities to each.
3. **BIO Format (Beginning, Inside, Outside):** A standardized sequence tagging token schema where:
   * `B-` denotes the initial token of an aspect phrase.
   * `I-` denotes subsequent tokens inside the same aspect phrase.
   * `O` indicates tokens outside any defined aspect.
4. **HerBERT:** A multi-task, state-of-the-art contextual language model for Polish based on the BERT architecture, developed and open-sourced by Allegro [1].
5. **CLARIN-PL ASPECTEMO Dataset:** A multi-domain consumer review and dialogue corpus specifically annotated for ABSA tasks, provided by the Polish CLARIN scientific consortium [2].
6. **Emotion_RoBERTa_polish6 (Visegradmedia):** A compact, deeply trained sequence classification model based on the RoBERTa architecture, optimized to categorize Polish text into 6 core emotional classes [3].

## 4. Technologies Used
* **Python 3.12:** The primary execution environment and development language.
* **Google Colab:** Cloud computing platform equipped with GPU accelerators used for model training and fine-tuning.
* **FastAPI:** A modern, fast web framework for building APIs with Python.
* **Hugging Face Transformers:** The core deep learning library used to load, train, serialize, and run transformer architectures.
* **Uvicorn:** A lightning-fast ASGI server implementation used to run the FastAPI application.
* **Postman:** API client used for functional, structural, and integration testing of JSON outputs.

### Embedded Language Models:
* `allegro/herbert-base-cased` (Fine-tuned on CLARIN-PL ASPECTEMO)
* `visegradmedia-emotion/Emotion_RoBERTa_polish6`

## 5. System Implementation

### Data Preparation & Tokenization
The input processing pipeline ingests raw JSON data from the CLARIN-PL ASPECTEMO corpus and converts it into explicit training vectors compatible with HerBERT's token classification head. The system defines a precise numerical mapping for aspect tags:
* `O` – Outside (No aspect present)
* `a_minus_m`, `a_minus_s` – Aspects carrying strong or weak negative sentiment.
* `a_zero` – Neutral aspect.
* `a_plus_s`, `a_plus_m` – Aspects carrying weak or strong positive sentiment.
* `a_amb` – Ambivalent aspect.


### Model Training
The optimization of HerBERT's weights is driven by a standalone `train.py` script using the Hugging Face `Trainer` API.

#### Hyperparameters & Strategy:
* **Epochs (`num_train_epochs`):** 5 epochs, allowing full weight convergence without overfitting.
* **Learning Rate (`learning_rate`):** Optimized for gentle, controlled fine-tuning of pre-trained transformer layers.
* **Batch Size (`batch_size`):** 8 samples per device.
* **Evaluation Metric:** Optimized against **F1-Macro**. This ensures that rare, critical emotional and aspect tags are treated with equal weight compared to dominant neutral background tokens (`O`), preventing class imbalance issues.
* **Checkpoint Selection:** Enforced via `load_best_model_at_end=True` paired with `metric_for_best_model="f1_macro"`. At the end of training, the system discards the final step weights and rolls back to the exact checkpoint that scored the highest F1-Macro on the validation split.

### API Application
The runtime server is defined in `api.py`. Upon application startup, the system instantiates both models and pre-caches weights on the target compute device. When the POST endpoint receives a text payload, it distributes the input to both sub-engines simultaneously. The global emotion labels are localized into clear Polish text strings, and their raw output tensors are normalized into human-readable confidence percentages.


## 6. References

* Mroczkowski, R., Rybak, P., Wróblewska, A., Gawlik, I. (2021). HerBERT: Efficiently Pretrained Transformer-based Language Model for Polish. Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing, Association for Computational Linguistics, pp. 1–10. Available at: https://www.aclweb.org/anthology/2021.bsnlp-1.1

* Kocoń, J., Radom, J., Kaczmarz-Wawryk, E., Wabnic, K., Zajączkowska, A., Zaśko-Zielińska, M. (2021). AspectEmo 1.0: Multi-Domain Corpus of Consumer Reviews for Aspect-Based Sentiment Analysis. CLARIN-PL digital repository. Available at: http://hdl.handle.net/11321/849

* Üveges, I. & Ring, O. (2025). Evaluating the Impact of Synthetic Data on Emotion Classification: A Linguistic and Structural Analysis. Information, 16(4), 330. DOI: 10.3390/info16040330




