import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score
from typing import List, Dict, Callable, Optional
import re
import string

class ResponseEvaluator:
    def __init__(self):
        """Initialize the response evaluator with different comparison strategies"""
        pass
    
    def exact_match(self, response: str, ground_truth: str) -> bool:
        """Exact string match (case-insensitive)"""
        return str(response).strip().lower() == str(ground_truth).strip().lower()
    
    def normalized_match(self, response: str, ground_truth: str) -> bool:
        """Match after removing punctuation and extra whitespace"""
        def normalize_text(text):
            # Remove punctuation and extra whitespace
            text = str(text).lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = ' '.join(text.split())
            return text
        
        return normalize_text(response) == normalize_text(ground_truth)
    
    def contains_match(self, response: str, ground_truth: str) -> bool:
        """Check if response contains the ground truth (or vice versa)"""
        response_clean = str(response).strip().lower()
        gt_clean = str(ground_truth).strip().lower()
        
        return gt_clean in response_clean or response_clean in gt_clean
    
    def keyword_match(self, response: str, ground_truth: str, min_keywords: int = 1) -> bool:
        """Match based on shared keywords"""
        def extract_keywords(text):
            # Simple keyword extraction (words longer than 2 chars, no punctuation)
            text = str(text).lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            words = [word for word in text.split() if len(word) > 2]
            return set(words)
        
        response_keywords = extract_keywords(response)
        gt_keywords = extract_keywords(ground_truth)
        
        shared_keywords = response_keywords.intersection(gt_keywords)
        return len(shared_keywords) >= min_keywords
    
    def numeric_match(self, response: str, ground_truth: str, tolerance: float = 0.01) -> bool:
        """Match numeric values within tolerance"""
        def extract_numbers(text):
            numbers = re.findall(r'-?\d+\.?\d*', str(text))
            return [float(n) for n in numbers if n]
        
        response_nums = extract_numbers(response)
        gt_nums = extract_numbers(ground_truth)
        
        if not response_nums or not gt_nums:
            return False
        
        # Check if any number in response matches any number in ground truth
        for r_num in response_nums:
            for gt_num in gt_nums:
                if abs(r_num - gt_num) <= tolerance:
                    return True
        return False
    
    def semantic_similarity_match(self, response: str, ground_truth: str, threshold: float = 0.8) -> bool:
        """
        Semantic similarity using sentence transformers (requires additional dependency)
        This is a placeholder - you'd need to install sentence-transformers
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embeddings = model.encode([str(response), str(ground_truth)])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return similarity >= threshold
        except ImportError:
            print("sentence-transformers not installed. Falling back to contains_match.")
            return self.contains_match(response, ground_truth)
    
    def evaluate_responses(self, 
                          df: pd.DataFrame, 
                          response_col: str = 'response', 
                          ground_truth_col: str = 'ground_truth',
                          method: str = 'normalized',
                          **kwargs) -> pd.DataFrame:
        """
        Evaluate responses against ground truth using specified method
        
        Args:
            df: DataFrame with response and ground truth columns
            response_col: Name of response column
            ground_truth_col: Name of ground truth column  
            method: Evaluation method ('exact', 'normalized', 'contains', 'keyword', 'numeric', 'semantic')
            **kwargs: Additional parameters for specific methods
            
        Returns:
            DataFrame with added 'is_correct' column
        """
        
        # Select comparison method
        method_map = {
            'exact': self.exact_match,
            'normalized': self.normalized_match,
            'contains': self.contains_match,
            'keyword': lambda r, gt: self.keyword_match(r, gt, kwargs.get('min_keywords', 1)),
            'numeric': lambda r, gt: self.numeric_match(r, gt, kwargs.get('tolerance', 0.01)),
            'semantic': lambda r, gt: self.semantic_similarity_match(r, gt, kwargs.get('threshold', 0.8))
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")
        
        comparison_func = method_map[method]
        
        # Apply comparison
        df = df.copy()
        df['is_correct'] = df.apply(
            lambda row: comparison_func(row[response_col], row[ground_truth_col]), 
            axis=1
        )
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame, correct_col: str = 'is_correct') -> Dict:
        """Calculate classification metrics"""
        
        if correct_col not in df.columns:
            raise ValueError(f"Column '{correct_col}' not found in DataFrame")
        
        # Convert to binary labels (assuming True/False or 1/0)
        y_true = np.ones(len(df))  # Assume all should be correct
        y_pred = df[correct_col].astype(bool).astype(int)
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate additional metrics
        total_samples = len(df)
        correct_predictions = y_pred.sum()
        
        metrics = {
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'error_rate': 1 - accuracy
        }
        
        return metrics
    
    def detailed_analysis(self, 
                         df: pd.DataFrame, 
                         response_col: str = 'response',
                         ground_truth_col: str = 'ground_truth',
                         correct_col: str = 'is_correct') -> None:
        """Print detailed analysis of the evaluation results"""
        
        metrics = self.calculate_metrics(df, correct_col)
        
        print("=== Response Evaluation Results ===")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Correct predictions: {metrics['correct_predictions']}")
        print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"Error rate: {metrics['error_rate']:.3f} ({metrics['error_rate']:.1%})")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"                  Predicted")
        print(f"                Wrong  Correct")
        print(f"Expected Wrong    {cm[1,0]:5d}     {cm[1,1]:4d}")
        print(f"        Correct   {cm[0,0]:5d}     {cm[0,1]:4d}")
        
        # Show examples of incorrect predictions
        incorrect_samples = df[~df[correct_col]]
        if len(incorrect_samples) > 0:
            print(f"\n=== Sample Incorrect Predictions (showing up to 5) ===")
            for idx, row in incorrect_samples.head(5).iterrows():
                print(f"\nSample {idx}:")
                print(f"Response: {str(row[response_col])[:100]}...")
                print(f"Ground Truth: {str(row[ground_truth_col])[:100]}...")
        
        # Show examples of correct predictions  
        correct_samples = df[df[correct_col]]
        if len(correct_samples) > 0:
            print(f"\n=== Sample Correct Predictions (showing up to 3) ===")
            for idx, row in correct_samples.head(3).iterrows():
                print(f"\nSample {idx}:")
                print(f"Response: {str(row[response_col])[:100]}...")
                print(f"Ground Truth: {str(row[ground_truth_col])[:100]}...")


def compare_evaluation_methods(df: pd.DataFrame, 
                              response_col: str = 'response',
                              ground_truth_col: str = 'ground_truth') -> pd.DataFrame:
    """Compare different evaluation methods on the same dataset"""
    
    evaluator = ResponseEvaluator()
    methods = ['exact', 'normalized', 'contains', 'keyword', 'numeric']
    
    results = []
    
    for method in methods:
        print(f"\n=== Evaluating with {method.upper()} method ===")
        try:
            df_eval = evaluator.evaluate_responses(df, response_col, ground_truth_col, method)
            metrics = evaluator.calculate_metrics(df_eval)
            
            result = {
                'method': method,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'correct_predictions': metrics['correct_predictions']
            }
            results.append(result)
            
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"F1-Score: {metrics['f1_score']:.3f}")
            
        except Exception as e:
            print(f"Error with {method} method: {e}")
    
    comparison_df = pd.DataFrame(results)
    print(f"\n=== Method Comparison Summary ===")
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    return comparison_df


# Usage examples
if __name__ == "__main__":
    # Example usage
    
    # 1. Load your CSV file
    # df = pd.read_csv('your_responses.csv')
    
    # 2. Create sample data for demonstration
    sample_data = {
        'response': [
            "The capital of France is Paris",
            "Paris is the capital",  
            "The answer is 42",
            "Forty-two",
            "I don't know",
            "The result is 3.14159"
        ],
        'ground_truth': [
            "Paris",
            "Paris", 
            "42",
            "42",
            "Unknown",
            "3.14"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # 3. Initialize evaluator
    evaluator = ResponseEvaluator()
    
    # 4. Evaluate using different methods
    print("=== EXACT MATCH ===")
    df_exact = evaluator.evaluate_responses(df, method='exact')
    evaluator.detailed_analysis(df_exact)
    
    print("\n" + "="*50)
    print("=== NORMALIZED MATCH ===")  
    df_normalized = evaluator.evaluate_responses(df, method='normalized')
    evaluator.detailed_analysis(df_normalized)
    
    print("\n" + "="*50)
    print("=== CONTAINS MATCH ===")
    df_contains = evaluator.evaluate_responses(df, method='contains')
    evaluator.detailed_analysis(df_contains)
    
    # 5. Compare all methods
    print("\n" + "="*50)
    comparison = compare_evaluation_methods(df)
    
    # 6. For your actual use case:
    """
    # Load your CSV
    your_df = pd.read_csv('your_file.csv')
    
    # Evaluate with the method that makes most sense for your data
    evaluated_df = evaluator.evaluate_responses(
        your_df, 
        response_col='actual_response',  # your column name
        ground_truth_col='ground_truth', # your column name  
        method='normalized'  # or whichever method works best
    )
    
    # Get detailed metrics
    evaluator.detailed_analysis(evaluated_df)
    
    # Save results
    evaluated_df.to_csv('evaluated_results.csv', index=False)
    """
