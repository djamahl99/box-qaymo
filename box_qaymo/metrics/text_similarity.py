from typing import Dict, Any, List, Type, Optional, Union
import numpy as np
from collections import Counter
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from sentence_transformers import SentenceTransformer
import torch

from .base import BaseMetric
from box_qaymo.answers.raw_text import RawTextAnswer


class TextSimilarityMetric(BaseMetric[RawTextAnswer]):
    """
    Evaluates text responses using various similarity metrics.
    Supports multiple similarity methods including:
    - Exact match
    - Token overlap (F1)
    - TF-IDF cosine similarity
    - Embedding similarity using Sentence Transformers
    """

    def __init__(
        self,
        methods: List[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        case_sensitive: bool = False,
        use_stemming: bool = False,
        use_lemmatization: bool = True,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
    ):
        """
        Initialize the text similarity metric.

        Args:
            methods: List of similarity methods to use, options include:
                    ["exact_match", "token_overlap", "tfidf", "embedding"]
            embedding_model: Name of the SentenceTransformer model to use
            case_sensitive: Whether comparisons should be case-sensitive
            use_stemming: Whether to apply stemming to tokens
            use_lemmatization: Whether to apply lemmatization to tokens
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
        """
        super().__init__(RawTextAnswer)

        if methods is None:
            self.methods = ["token_overlap", "embedding"]
        else:
            self.methods = methods

        self.case_sensitive = case_sensitive
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation

        # Initialize NLP tools
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None

        # Load spaCy model for more advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback if model not available
            self.nlp = None
            print("Warning: spaCy model not available. Some features may be limited.")

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer() if "tfidf" in self.methods else None

        # Initialize embedding model if needed
        self.embedding_model = None
        # if "embedding" in self.methods:
        #     try:
        #         self.embedding_model = SentenceTransformer(embedding_model)
        #     except:
        #         print(f"Warning: Could not load embedding model {embedding_model}. Embedding similarity will be disabled.")
        #         if "embedding" in self.methods:
        #             self.methods.remove("embedding")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by applying case normalization, removing stopwords,
        punctuation, and applying stemming or lemmatization as configured.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not self.case_sensitive:
            text = text.lower()

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords if configured
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Apply stemming if configured
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]

        # Apply lemmatization if configured
        if self.use_lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)

    def get_tokens(self, text: str) -> List[str]:
        """
        Get tokens from text after preprocessing.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        preprocessed = self.preprocess_text(text)
        return preprocessed.split()

    def exact_match(self, pred_text: str, gt_text: str) -> float:
        """
        Check if texts match exactly after preprocessing.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        pred_processed = self.preprocess_text(pred_text)
        gt_processed = self.preprocess_text(gt_text)
        return float(pred_processed == gt_processed)

    def token_overlap_f1(self, pred_text: str, gt_text: str) -> Dict[str, float]:
        """
        Calculate token overlap metrics (precision, recall, F1) between texts.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        pred_tokens = self.get_tokens(pred_text)
        gt_tokens = self.get_tokens(gt_text)

        # Convert to Counter for easy intersection calculation
        pred_counter = Counter(pred_tokens)
        gt_counter = Counter(gt_tokens)

        # Find common tokens (intersection)
        common = sum((pred_counter & gt_counter).values())

        # Calculate precision, recall, F1
        precision = common / max(sum(pred_counter.values()), 1)
        recall = common / max(sum(gt_counter.values()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        return {"precision": precision, "recall": recall, "f1": f1}

    def tfidf_similarity(self, pred_text: str, gt_text: str) -> float:
        """
        Calculate TF-IDF cosine similarity between texts.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text

        Returns:
            Cosine similarity score
        """
        if self.tfidf_vectorizer is None:
            return 0.0

        # Preprocess texts
        pred_processed = self.preprocess_text(pred_text)
        gt_processed = self.preprocess_text(gt_text)

        # Fit and transform on the corpus
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            [pred_processed, gt_processed]
        )

        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)

    def embedding_similarity(self, pred_text: str, gt_text: str) -> float:
        """
        Calculate embedding-based cosine similarity between texts.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text

        Returns:
            Cosine similarity score based on text embeddings
        """
        if self.embedding_model is None:
            return 0.0

        # Generate embeddings
        pred_embedding = self.embedding_model.encode(pred_text, convert_to_tensor=True)
        gt_embedding = self.embedding_model.encode(gt_text, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(pred_embedding, gt_embedding).item()
        return float(similarity)

    def evaluate(
        self, prediction: RawTextAnswer, ground_truth: RawTextAnswer
    ) -> Dict[str, Any]:
        """
        Evaluate a text prediction against ground truth using configured methods.

        Args:
            prediction: Predicted text answer
            ground_truth: Ground truth text answer

        Returns:
            Dictionary with similarity scores for each method
        """
        # Validate types
        prediction, ground_truth = self.validate_types(prediction, ground_truth)

        pred_text = prediction.text
        gt_text = ground_truth.text

        # Calculate metrics based on configured methods
        results = {}

        if "exact_match" in self.methods:
            results["exact_match"] = self.exact_match(pred_text, gt_text)

        if "token_overlap" in self.methods:
            token_metrics = self.token_overlap_f1(pred_text, gt_text)
            results.update(
                {
                    "token_precision": token_metrics["precision"],
                    "token_recall": token_metrics["recall"],
                    "token_f1": token_metrics["f1"],
                }
            )

        if "tfidf" in self.methods and self.tfidf_vectorizer is not None:
            results["tfidf_similarity"] = self.tfidf_similarity(pred_text, gt_text)

        if "embedding" in self.methods and self.embedding_model is not None:
            results["embedding_similarity"] = self.embedding_similarity(
                pred_text, gt_text
            )

        return results

    def summarise(self, metric_results: List[Dict]) -> Dict[str, Any]:
        """
        Summarize evaluation results across multiple predictions.

        Args:
            metric_results: List of dictionaries containing evaluation results
                           for each prediction-ground truth pair.

        Returns:
            A dictionary with average scores for each metric.
        """
        if not metric_results:
            return {}

        # Initialize summary with keys from first result
        summary = {k: [] for k in metric_results[0].keys()}

        # Collect all values
        for result in metric_results:
            for k, v in result.items():
                if k in summary:
                    summary[k].append(v)

        # Calculate averages
        for k in summary:
            values = summary[k]
            if values:
                summary[k] = sum(values) / len(values)
            else:
                summary[k] = 0.0

        # Add count
        summary["count"] = len(metric_results)

        return summary

    def validate_types(self, prediction, ground_truth):
        """
        Validate that inputs are RawTextAnswer instances.
        Convert strings to RawTextAnswer if needed.
        """
        if isinstance(prediction, str):
            prediction = RawTextAnswer(prediction)
        if isinstance(ground_truth, str):
            ground_truth = RawTextAnswer(ground_truth)

        return super().validate_types(prediction, ground_truth)


class RougeMetric(BaseMetric[RawTextAnswer]):
    """
    Evaluates text responses using ROUGE metrics.
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is commonly
    used for evaluating summarization and translation tasks.
    """

    def __init__(self, rouge_types: List[str] = None):
        """
        Initialize the ROUGE metric.

        Args:
            rouge_types: List of ROUGE metrics to calculate, options include:
                        ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        """
        super().__init__(RawTextAnswer)
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]

        try:
            from rouge_score import rouge_scorer

            self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
        except ImportError:
            print(
                "Warning: rouge_score package not available. Please install with: pip install rouge-score"
            )
            self.scorer = None

    def evaluate(
        self, prediction: RawTextAnswer, ground_truth: RawTextAnswer
    ) -> Dict[str, Any]:
        """
        Evaluate a text prediction against ground truth using ROUGE metrics.

        Args:
            prediction: Predicted text answer
            ground_truth: Ground truth text answer

        Returns:
            Dictionary with ROUGE scores
        """
        if self.scorer is None:
            return {}

        # Validate types
        prediction, ground_truth = self.validate_types(prediction, ground_truth)

        pred_text = prediction.text
        gt_text = ground_truth.text

        # Calculate ROUGE scores
        scores = self.scorer.score(gt_text, pred_text)

        # Extract precision, recall, F1 for each ROUGE type
        results = {}
        for rouge_type, score in scores.items():
            results[f"{rouge_type}_precision"] = score.precision
            results[f"{rouge_type}_recall"] = score.recall
            results[f"{rouge_type}_f1"] = score.fmeasure

        return results

    def summarise(self, metric_results: List[Dict]) -> Dict[str, Any]:
        """
        Summarize ROUGE evaluation results across multiple predictions.

        Args:
            metric_results: List of dictionaries containing evaluation results
                           for each prediction-ground truth pair.

        Returns:
            A dictionary with average ROUGE scores.
        """
        if not metric_results:
            return {}

        # Initialize summary with keys from first result
        summary = {k: [] for k in metric_results[0].keys()}

        # Collect all values
        for result in metric_results:
            for k, v in result.items():
                if k in summary:
                    summary[k].append(v)

        # Calculate averages
        for k in summary:
            values = summary[k]
            if values:
                summary[k] = sum(values) / len(values)
            else:
                summary[k] = 0.0

        # Add count
        summary["count"] = len(metric_results)

        return summary


class BERTScoreMetric(BaseMetric[RawTextAnswer]):
    """
    Evaluates text responses using BERTScore.
    BERTScore leverages pre-trained contextual embeddings from BERT and matches
    words in candidate and reference sentences by cosine similarity.
    """

    def __init__(self, model_type: str = "roberta-large", lang: str = "en"):
        """
        Initialize the BERTScore metric.

        Args:
            model_type: Pre-trained model type to use for BERTScore
            lang: Language code for the text
        """
        super().__init__(RawTextAnswer)
        self.model_type = model_type
        self.lang = lang

        try:
            import bert_score

            self.bert_score = bert_score
        except ImportError:
            print(
                "Warning: bert_score package not available. Please install with: pip install bert-score"
            )
            self.bert_score = None

    def evaluate(
        self, prediction: RawTextAnswer, ground_truth: RawTextAnswer
    ) -> Dict[str, Any]:
        """
        Evaluate a text prediction against ground truth using BERTScore.

        Args:
            prediction: Predicted text answer
            ground_truth: Ground truth text answer

        Returns:
            Dictionary with BERTScore metrics (precision, recall, F1)
        """
        if self.bert_score is None:
            return {}

        # Validate types
        prediction, ground_truth = self.validate_types(prediction, ground_truth)

        pred_text = prediction.text
        gt_text = ground_truth.text

        # Calculate BERTScore
        P, R, F1 = self.bert_score.score(
            [pred_text], [gt_text], lang=self.lang, model_type=self.model_type
        )

        return {
            "bertscore_precision": P.item(),
            "bertscore_recall": R.item(),
            "bertscore_f1": F1.item(),
        }

    def summarise(self, metric_results: List[Dict]) -> Dict[str, Any]:
        """
        Summarize BERTScore evaluation results across multiple predictions.

        Args:
            metric_results: List of dictionaries containing evaluation results
                           for each prediction-ground truth pair.

        Returns:
            A dictionary with average BERTScore metrics.
        """
        if not metric_results:
            return {}

        # Initialize summary with keys from first result
        summary = {k: [] for k in metric_results[0].keys()}

        # Collect all values
        for result in metric_results:
            for k, v in result.items():
                if k in summary:
                    summary[k].append(v)

        # Calculate averages
        for k in summary:
            values = summary[k]
            if values:
                summary[k] = sum(values) / len(values)
            else:
                summary[k] = 0.0

        # Add count
        summary["count"] = len(metric_results)

        return summary


# Simpler implementation that doesn't require external dependencies
class SimpleTextSimilarityMetric(BaseMetric[RawTextAnswer]):
    """
    A simple text similarity metric that doesn't require external dependencies.
    Implements exact match, token overlap, and character-level n-gram similarity.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        remove_punctuation: bool = True,
        ngram_sizes: List[int] = None,
    ):
        """
        Initialize the simple text similarity metric.

        Args:
            case_sensitive: Whether comparisons should be case-sensitive
            remove_punctuation: Whether to remove punctuation
            ngram_sizes: List of n-gram sizes to use for character-level similarity
        """
        super().__init__(RawTextAnswer)
        self.case_sensitive = case_sensitive
        self.remove_punctuation = remove_punctuation
        self.ngram_sizes = ngram_sizes or [2, 3]

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by applying case normalization and removing punctuation.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not self.case_sensitive:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        return text

    def get_tokens(self, text: str) -> List[str]:
        """
        Get tokens from text after preprocessing.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        preprocessed = self.preprocess_text(text)
        return preprocessed.split()

    def get_character_ngrams(self, text: str, n: int) -> List[str]:
        """
        Get character-level n-grams from text.

        Args:
            text: Input text
            n: Size of n-grams

        Returns:
            List of character n-grams
        """
        text = self.preprocess_text(text)
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def exact_match(self, pred_text: str, gt_text: str) -> float:
        """
        Check if texts match exactly after preprocessing.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        pred_processed = self.preprocess_text(pred_text)
        gt_processed = self.preprocess_text(gt_text)
        return float(pred_processed == gt_processed)

    def token_overlap_f1(self, pred_text: str, gt_text: str) -> Dict[str, float]:
        """
        Calculate token overlap metrics (precision, recall, F1) between texts.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        pred_tokens = self.get_tokens(pred_text)
        gt_tokens = self.get_tokens(gt_text)

        # Convert to sets for intersection calculation
        pred_set = set(pred_tokens)
        gt_set = set(gt_tokens)

        # Find common tokens (intersection)
        common = len(pred_set.intersection(gt_set))

        # Calculate precision, recall, F1
        precision = common / max(len(pred_set), 1)
        recall = common / max(len(gt_set), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        return {"precision": precision, "recall": recall, "f1": f1}

    def character_ngram_similarity(self, pred_text: str, gt_text: str, n: int) -> float:
        """
        Calculate character-level n-gram similarity between texts.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text
            n: Size of n-grams

        Returns:
            Similarity score based on n-gram overlap
        """
        pred_ngrams = set(self.get_character_ngrams(pred_text, n))
        gt_ngrams = set(self.get_character_ngrams(gt_text, n))

        if not pred_ngrams or not gt_ngrams:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(pred_ngrams.intersection(gt_ngrams))
        union = len(pred_ngrams.union(gt_ngrams))

        return intersection / union

    def evaluate(
        self, prediction: RawTextAnswer, ground_truth: RawTextAnswer
    ) -> Dict[str, Any]:
        """
        Evaluate a text prediction against ground truth.

        Args:
            prediction: Predicted text answer
            ground_truth: Ground truth text answer

        Returns:
            Dictionary with similarity scores
        """
        # Validate types
        prediction, ground_truth = self.validate_types(prediction, ground_truth)

        pred_text = prediction.text
        gt_text = ground_truth.text

        # Calculate metrics
        results = {"exact_match": self.exact_match(pred_text, gt_text)}

        # Token overlap metrics
        token_metrics = self.token_overlap_f1(pred_text, gt_text)
        results.update(
            {
                "token_precision": token_metrics["precision"],
                "token_recall": token_metrics["recall"],
                "token_f1": token_metrics["f1"],
            }
        )

        # Character n-gram similarity
        for n in self.ngram_sizes:
            similarity = self.character_ngram_similarity(pred_text, gt_text, n)
            results[f"char_{n}gram_similarity"] = similarity

        return results

    def summarise(self, metric_results: List[Dict]) -> Dict[str, Any]:
        """
        Summarize evaluation results across multiple predictions.

        Args:
            metric_results: List of dictionaries containing evaluation results
                           for each prediction-ground truth pair.

        Returns:
            A dictionary with average scores for each metric.
        """
        if not metric_results:
            return {}

        # Initialize summary with keys from first result
        summary = {k: [] for k in metric_results[0].keys()}

        # Collect all values
        for result in metric_results:
            for k, v in result.items():
                if k in summary:
                    summary[k].append(v)

        # Calculate averages
        for k in summary:
            values = summary[k]
            if values:
                summary[k] = sum(values) / len(values)
            else:
                summary[k] = 0.0

        # Add count
        summary["count"] = len(metric_results)

        return summary


# Example usage
def example_usage():
    # Create sample text answers
    pred = RawTextAnswer("The quick brown fox jumps over the lazy dog.")
    gt = RawTextAnswer("A quick brown fox jumped over a lazy dog.")

    # Use the simple metric (no dependencies)
    simple_metric = SimpleTextSimilarityMetric()
    simple_result = simple_metric.evaluate(pred, gt)
    print("Simple metric result:", simple_result)

    # Use the full-featured metric (with dependencies)
    try:
        text_metric = TextSimilarityMetric(
            methods=["exact_match", "token_overlap", "tfidf"]
        )
        text_result = text_metric.evaluate(pred, gt)
        print("Text similarity result:", text_result)
    except Exception as e:
        print("Full metric unavailable:", e)

    # Use ROUGE metric if available
    try:
        rouge_metric = RougeMetric()
        rouge_result = rouge_metric.evaluate(pred, gt)
        print("ROUGE result:", rouge_result)
    except Exception as e:
        print("ROUGE metric unavailable:", e)


if __name__ == "__main__":
    example_usage()
