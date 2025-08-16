import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    roc_auc_score
)
from typing import Dict, List

class BooleanResponseEvaluator:
    def __init__(self):
        """Evaluator for True/False responses"""
        pass
    
    def normalize_boolean(self, value) -> bool:
        """
        Convert various boolean representations to True/False
        
        Handles: true/false, True/False, 1/0, yes/no, etc.
        """
        if pd.isna(value):
            return False
        
        # Convert to string and normalize
        str_val = str(value).strip().lower()
        
        # True values
        true_values = {
            'true', '1', 'yes', 'y', 't', 
            'correct', 'right', 'positive'
        }
        
        # False values  
        false_values = {
            'false', '0', 'no', 'n', 'f',
            'incorrect', 'wrong', 'negative'
        }
        
        if str_val in true_values:
            return True
        elif str_val in false_values:
            return False
        else:
            # Try to handle as boolean directly
            try:
                return bool(value)
            except:
                print(f"Warning: Could not parse '{value}' as boolean, treating as False")
                return False
    
    def prepare_data(self, df: pd.DataFrame, 
                    response_col: str = 'response',
                    ground_truth_col: str = 'ground_truth') -> pd.DataFrame:
        """
        Prepare data by converting both columns to boolean values
        
        Returns DataFrame with normalized boolean columns
        """
        df = df.copy()
        
        # Normalize both columns to boolean
        df['response_bool'] = df[response_col].apply(self.normalize_boolean)
        df['ground_truth_bool'] = df[ground_truth_col].apply(self.normalize_boolean)
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate all classification metrics for boolean responses
        """
        y_true = df['ground_truth_bool']
        y_pred = df['response_bool']
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Positive/Negative predictive values
        ppv = precision  # Same as precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics = {
            'total_samples': len(df),
            'true_positives': int(tp),
            'true_negatives': int(tn), 
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def detailed_report(self, df: pd.DataFrame, 
                       response_col: str = 'response',
                       ground_truth_col: str = 'ground_truth') -> Dict:
        """
        Generate a comprehensive evaluation report
        """
        # Prepare data
        df_prepared = self.prepare_data(df, response_col, ground_truth_col)
        
        # Calculate metrics
        metrics = self.calculate_metrics(df_prepared)
        
        # Print detailed report
        print("=== Boolean Response Evaluation Report ===")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"")
        
        print("=== Confusion Matrix ===")
        print(f"                    Predicted")
        print(f"                False    True")
        print(f"Actual False    {metrics['true_negatives']:5d}   {metrics['false_positives']:5d}")
        print(f"       True     {metrics['false_negatives']:5d}   {metrics['true_positives']:5d}")
        print(f"")
        
        print("=== Classification Metrics ===")
        print(f"Accuracy:           {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
        print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.3f} ({metrics['balanced_accuracy']:.1%})")
        print(f"Precision:          {metrics['precision']:.3f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.3f}")
        print(f"Specificity:        {metrics['specificity']:.3f}")
        print(f"F1-Score:           {metrics['f1_score']:.3f}")
        print(f"")
        
        print("=== Predictive Values ===")
        print(f"Positive Predictive Value: {metrics['positive_predictive_value']:.3f}")
        print(f"Negative Predictive Value: {metrics['negative_predictive_value']:.3f}")
        print(f"")
        
        # Distribution analysis
        true_count = df_prepared['ground_truth_bool'].sum()
        false_count = len(df_prepared) - true_count
        pred_true_count = df_prepared['response_bool'].sum()
        pred_false_count = len(df_prepared) - pred_true_count
        
        print("=== Data Distribution ===")
        print(f"Ground Truth - True:  {true_count:3d} ({true_count/len(df_prepared):.1%})")
        print(f"Ground Truth - False: {false_count:3d} ({false_count/len(df_prepared):.1%})")
        print(f"Predicted - True:     {pred_true_count:3d} ({pred_true_count/len(df_prepared):.1%})")
        print(f"Predicted - False:    {pred_false_count:3d} ({pred_false_count/len(df_prepared):.1%})")
        print(f"")
        
        # Error analysis
        errors = df_prepared[df_prepared['response_bool'] != df_prepared['ground_truth_bool']]
        if len(errors) > 0:
            print("=== Error Analysis ===")
            false_positives = errors[errors['response_bool'] == True]
            false_negatives = errors[errors['response_bool'] == False]
            
            print(f"False Positives: {len(false_positives)} (predicted True, actually False)")
            print(f"False Negatives: {len(false_negatives)} (predicted False, actually True)")
            
            # Show some examples
            if len(false_positives) > 0:
                print(f"\nSample False Positives:")
                for idx, row in false_positives.head(3).iterrows():
                    print(f"  Row {idx}: Response='{row[response_col]}', Truth='{row[ground_truth_col]}'")
            
            if len(false_negatives) > 0:
                print(f"\nSample False Negatives:")
                for idx, row in false_negatives.head(3).iterrows():
                    print(f"  Row {idx}: Response='{row[response_col]}', Truth='{row[ground_truth_col]}'")
        
        return metrics, df_prepared
    
    def quick_evaluation(self, df: pd.DataFrame,
                        response_col: str = 'response', 
                        ground_truth_col: str = 'ground_truth') -> None:
        """Quick evaluation with just the key metrics"""
        
        df_prepared = self.prepare_data(df, response_col, ground_truth_col)
        metrics = self.calculate_metrics(df_prepared)
        
        print("=== Quick Evaluation ===")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1-Score:  {metrics['f1_score']:.3f}")
        print(f"Samples:   {metrics['total_samples']}")


def evaluate_boolean_csv(csv_file: str, 
                        response_col: str = 'response',
                        ground_truth_col: str = 'ground_truth',
                        output_file: str = None) -> pd.DataFrame:
    """
    Convenience function to evaluate a CSV file with boolean responses
    
    Args:
        csv_file: Path to CSV file
        response_col: Name of response column
        ground_truth_col: Name of ground truth column  
        output_file: Optional path to save results
        
    Returns:
        DataFrame with evaluation results
    """
    
    # Load data
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} samples from {csv_file}")
    print(f"Response column: '{response_col}'")
    print(f"Ground truth column: '{ground_truth_col}'")
    print()
    
    # Initialize evaluator
    evaluator = BooleanResponseEvaluator()
    
    # Run evaluation
    metrics, df_results = evaluator.detailed_report(df, response_col, ground_truth_col)
    
    # Save results if requested
    if output_file:
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    return df_results


# Usage example
if __name__ == "__main__":
    
    # Create sample data for demonstration
    sample_data = {
        'response': ['true', 'false', 'True', 'False', '1', '0', 'yes', 'no'],
        'ground_truth': ['true', 'true', 'true', 'false', 'true', 'false', 'true', 'false']
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample data:")
    print(df)
    print("\n" + "="*50)
    
    # Quick evaluation
    evaluator = BooleanResponseEvaluator()
    evaluator.quick_evaluation(df)
    
    print("\n" + "="*50)
    
    # Detailed evaluation  
    metrics, df_results = evaluator.detailed_report(df)
    
    print("\n" + "="*50)
    print("Results with normalized boolean columns:")
    print(df_results[['response', 'ground_truth', 'response_bool', 'ground_truth_bool']])
    
    # For your actual usage:
    """
    # Evaluate your CSV file
    results = evaluate_boolean_csv(
        'your_file.csv',
        response_col='your_response_column',
        ground_truth_col='your_ground_truth_column', 
        output_file='evaluation_results.csv'
    )
    """
