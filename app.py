import json
from pathlib import Path

import requests
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "llama3.2:3b"
VALID_LABELS = {"PERSON", "ORG", "LOC", "DATE", "MISC"}
EXAMPLES_PATH = Path(__file__).parent / "data" / "examples.json"

st.set_page_config(
    page_title="Retrieval-Augmented NER",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def load_retrieval_index():
    """Load labeled examples and build a TF-IDF search index."""
    with EXAMPLES_PATH.open("r", encoding="utf-8") as file:
        examples = json.load(file)

    if not examples:
        raise ValueError("data/examples.json contains no examples.")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
    )
    example_vectors = vectorizer.fit_transform(
        [example["text"] for example in examples]
    )
    return examples, vectorizer, example_vectors


@st.cache_resource
def load_spacy_model():
    """Load spaCy's small English NER model for the fallback mode."""
    return spacy.load("en_core_web_sm")


def retrieve_examples(text, examples, vectorizer, example_vectors, top_k):
    """Return the labeled examples with the highest TF-IDF similarity."""
    text_vector = vectorizer.transform([text])
    scores = cosine_similarity(text_vector, example_vectors)[0]
    best_indices = scores.argsort()[::-1][:top_k]

    return [
        {
            **examples[index],
            "similarity": float(scores[index]),
        }
        for index in best_indices
    ]


def extract_entities_with_ollama(text, retrieved_examples):
    """Run retrieval-augmented NER using the local Ollama model."""
    example_context = json.dumps(
        [
            {
                "text": example["text"],
                "entities": example["entities"],
            }
            for example in retrieved_examples
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
Extract named entities from the input text.

Use the retrieved examples as guidance, but only return entities that
appear in the input text. Copy each entity's exact wording from the input.
Use only these labels: PERSON, ORG, LOC, DATE, MISC.

Retrieved labeled examples:
{example_context}

Input text:
{text}

Return a JSON object in this format:
{{"entities": [{{"text": "entity phrase", "label": "ORG"}}]}}
If there are no entities, return {{"entities": []}}.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "system": (
                "You perform named entity recognition. "
                "Return only the requested JSON object."
            ),
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    response.raise_for_status()

    model_output = response.json()["response"]
    parsed_output = json.loads(model_output)
    raw_entities = parsed_output.get("entities", [])

    if not isinstance(raw_entities, list):
        return []

    # Keep only valid labels and exact entity spans found in the input.
    valid_entities = []
    seen = set()

    for entity in raw_entities:
        if not isinstance(entity, dict):
            continue

        phrase = str(entity.get("text", "")).strip()
        label = str(entity.get("label", "")).upper()

        if (
            phrase
            and phrase in text
            and label in VALID_LABELS
            and (phrase, label) not in seen
        ):
            valid_entities.append({"text": phrase, "label": label})
            seen.add((phrase, label))

    return valid_entities


def extract_entities_with_spacy(text):
    """Run spaCy NER when the local Ollama service is unavailable."""
    nlp = load_spacy_model()
    doc = nlp(text)

    label_map = {
        "PERSON": "PERSON",
        "ORG": "ORG",
        "GPE": "LOC",
        "LOC": "LOC",
        "FAC": "LOC",
        "DATE": "DATE",
        "TIME": "DATE",
        "NORP": "MISC",
        "PRODUCT": "MISC",
        "EVENT": "MISC",
        "WORK_OF_ART": "MISC",
        "LAW": "MISC",
        "LANGUAGE": "MISC",
    }

    return [
        {"text": entity.text, "label": label_map[entity.label_]}
        for entity in doc.ents
        if entity.label_ in label_map
    ]


st.title("Retrieval-Augmented Named Entity Recognition")
st.write(
    "The app retrieves similar labeled examples and provides them to a "
    "locally running language model. If Ollama is unavailable, the app "
    "uses a spaCy NER baseline instead."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Similar examples to retrieve",
        min_value=1,
        max_value=5,
        value=3,
    )
    st.markdown(
        "**Entity labels**  \n"
        "PERSON · ORG · LOC · DATE · MISC"
    )
    st.caption(
        "Local mode uses Ollama on this computer. "
        "The fallback uses spaCy and does not call a hosted LLM API."
    )

sample_text = (
    "Microsoft CEO Satya Nadella announced a research center "
    "in London in 2025."
)

input_text = st.text_area(
    "Enter text to analyze",
    value=sample_text,
    height=150,
    max_chars=4000,
)

if st.button("Extract entities", type="primary"):
    text = input_text.strip()

    if not text:
        st.warning("Enter some text first.")
    else:
        try:
            examples, vectorizer, example_vectors = load_retrieval_index()

            retrieved = retrieve_examples(
                text,
                examples,
                vectorizer,
                example_vectors,
                top_k,
            )

            try:
                with st.spinner("Analyzing with the local Llama model..."):
                    entities = extract_entities_with_ollama(text, retrieved)
                extraction_mode = "Local Llama 3.2 3B with retrieved examples"
            except requests.exceptions.RequestException:
                with st.spinner("Ollama is unavailable; using spaCy NER..."):
                    entities = extract_entities_with_spacy(text)
                extraction_mode = "spaCy NER baseline fallback"

            st.caption(f"Extraction mode: {extraction_mode}")
            st.subheader("Detected entities")

            if entities:
                st.dataframe(entities, hide_index=True)
            else:
                st.info("No entities were returned for this text.")

            with st.expander("Retrieved examples", expanded=True):
                if retrieved and retrieved[0]["similarity"] == 0:
                    st.warning(
                        "No lexical overlap was found. The examples below "
                        "may not be useful for this input."
                    )

                for number, example in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**{number}. Similarity: "
                        f"{example['similarity']:.2f}**"
                    )
                    st.write(example["text"])
                    st.caption(
                        ", ".join(
                            f"{entity['text']} ({entity['label']})"
                            for entity in example["entities"]
                        )
                    )

        except Exception as error:
            st.error(f"Something went wrong: {error}")

st.divider()
st.caption(
    "TF-IDF example retrieval · Few-shot prompting · Local LLM inference "
    "with a spaCy fallback · Streamlit"
)