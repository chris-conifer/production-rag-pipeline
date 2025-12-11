"""
Golden Dataset Manager
Creates and manages golden Q&A datasets with difficulty stratification
SHARED across all 4 projects
"""

from typing import List, Dict, Any, Optional
import json
import random
from pathlib import Path
from collections import Counter


class GoldenDatasetManager:
    """
    Manages golden QA datasets for evaluation
    
    Features:
    - Create stratified samples (by difficulty, length, type)
    - Save/load golden datasets
    - Validate dataset quality
    - Support for SQuAD and custom formats
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize Golden Dataset Manager
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def create_golden_dataset(
        self,
        dataset,
        n_samples: int = 100,
        stratify_by: str = 'difficulty',
        difficulty_bins: List[str] = ['easy', 'medium', 'hard']
    ) -> List[Dict[str, Any]]:
        """
        Create golden dataset from SQuAD-like dataset
        
        Args:
            dataset: HuggingFace dataset or list of examples
            n_samples: Number of samples to create
            stratify_by: Stratification strategy ('difficulty', 'length', 'type', 'random')
            difficulty_bins: Difficulty levels for stratification
            
        Returns:
            List of golden QA examples
        """
        print(f"\n📊 Creating Golden Dataset ({n_samples} samples)")
        print(f"   Stratification: {stratify_by}")
        
        # Convert to list if needed
        if hasattr(dataset, '__iter__') and not isinstance(dataset, list):
            dataset_list = list(dataset)
        else:
            dataset_list = dataset
        
        # Filter valid examples (has answers)
        valid_examples = [
            ex for ex in dataset_list
            if ex.get('answers') and len(ex['answers']['text']) > 0
        ]
        
        print(f"   Valid examples: {len(valid_examples)}")
        
        if stratify_by == 'difficulty':
            golden_examples = self._stratify_by_difficulty(
                valid_examples, n_samples, difficulty_bins
            )
        elif stratify_by == 'length':
            golden_examples = self._stratify_by_length(valid_examples, n_samples)
        elif stratify_by == 'type':
            golden_examples = self._stratify_by_question_type(valid_examples, n_samples)
        else:  # random
            golden_examples = random.sample(valid_examples, min(n_samples, len(valid_examples)))
        
        # Format golden dataset
        formatted_examples = []
        for i, ex in enumerate(golden_examples):
            formatted_ex = {
                'id': ex.get('id', f'golden_{i}'),
                'question': ex['question'],
                'context': ex['context'],
                'answers': ex['answers']['text'][0] if isinstance(ex['answers'], dict) else ex['answers'][0],
                'all_answers': ex['answers']['text'] if isinstance(ex['answers'], dict) else ex['answers'],
                'difficulty': self._estimate_difficulty(ex),
                'question_type': self._classify_question_type(ex['question']),
                'context_length': len(ex['context']),
                'answer_length': len(ex['answers']['text'][0] if isinstance(ex['answers'], dict) else ex['answers'][0])
            }
            formatted_examples.append(formatted_ex)
        
        print(f"✓ Created golden dataset with {len(formatted_examples)} examples")
        self._print_distribution(formatted_examples)
        
        return formatted_examples
    
    def _stratify_by_difficulty(
        self,
        examples: List[Dict],
        n_samples: int,
        difficulty_bins: List[str]
    ) -> List[Dict]:
        """Stratify by estimated difficulty"""
        # Estimate difficulty for all examples
        examples_with_difficulty = []
        for ex in examples:
            difficulty = self._estimate_difficulty(ex)
            examples_with_difficulty.append((ex, difficulty))
        
        # Group by difficulty
        difficulty_groups = {level: [] for level in difficulty_bins}
        for ex, diff in examples_with_difficulty:
            if diff in difficulty_groups:
                difficulty_groups[diff].append(ex)
        
        # Sample evenly from each difficulty
        samples_per_bin = n_samples // len(difficulty_bins)
        remainder = n_samples % len(difficulty_bins)
        
        selected = []
        for i, level in enumerate(difficulty_bins):
            group = difficulty_groups[level]
            n_to_sample = samples_per_bin + (1 if i < remainder else 0)
            n_to_sample = min(n_to_sample, len(group))
            selected.extend(random.sample(group, n_to_sample))
        
        return selected
    
    def _stratify_by_length(self, examples: List[Dict], n_samples: int) -> List[Dict]:
        """Stratify by answer length"""
        # Sort by answer length
        examples_with_length = [
            (ex, len(ex['answers']['text'][0] if isinstance(ex['answers'], dict) else ex['answers'][0]))
            for ex in examples
        ]
        examples_with_length.sort(key=lambda x: x[1])
        
        # Divide into thirds (short, medium, long)
        n = len(examples_with_length)
        thirds = [
            examples_with_length[:n//3],
            examples_with_length[n//3:2*n//3],
            examples_with_length[2*n//3:]
        ]
        
        # Sample evenly
        samples_per_third = n_samples // 3
        selected = []
        for third in thirds:
            n_to_sample = min(samples_per_third, len(third))
            sampled = random.sample(third, n_to_sample)
            selected.extend([ex for ex, _ in sampled])
        
        return selected
    
    def _stratify_by_question_type(self, examples: List[Dict], n_samples: int) -> List[Dict]:
        """Stratify by question type (what, when, who, where, why, how)"""
        # Classify questions
        type_groups = {
            'what': [], 'when': [], 'who': [], 'where': [], 'why': [], 'how': [], 'other': []
        }
        
        for ex in examples:
            q_type = self._classify_question_type(ex['question'])
            if q_type in type_groups:
                type_groups[q_type].append(ex)
            else:
                type_groups['other'].append(ex)
        
        # Sample proportionally
        total = sum(len(g) for g in type_groups.values())
        selected = []
        
        for q_type, group in type_groups.items():
            if len(group) == 0:
                continue
            proportion = len(group) / total
            n_to_sample = max(1, int(n_samples * proportion))
            n_to_sample = min(n_to_sample, len(group))
            selected.extend(random.sample(group, n_to_sample))
        
        # Ensure we have exactly n_samples
        if len(selected) < n_samples:
            remaining = [ex for ex in examples if ex not in selected]
            selected.extend(random.sample(remaining, n_samples - len(selected)))
        elif len(selected) > n_samples:
            selected = random.sample(selected, n_samples)
        
        return selected
    
    def _estimate_difficulty(self, example: Dict) -> str:
        """Estimate question difficulty based on heuristics"""
        question = example['question'].lower()
        context = example['context']
        answer = example['answers']['text'][0] if isinstance(example['answers'], dict) else example['answers'][0]
        
        # Factors that increase difficulty
        score = 0
        
        # Long context
        if len(context) > 500:
            score += 1
        
        # Long answer
        if len(answer.split()) > 5:
            score += 1
        
        # Complex question words
        complex_words = ['why', 'how', 'explain', 'describe', 'compare']
        if any(word in question for word in complex_words):
            score += 1
        
        # Answer not directly in context
        if answer.lower() not in context.lower():
            score += 1
        
        # Classify
        if score <= 1:
            return 'easy'
        elif score == 2:
            return 'medium'
        else:
            return 'hard'
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question by type"""
        question_lower = question.lower().strip()
        
        if question_lower.startswith('what'):
            return 'what'
        elif question_lower.startswith('when'):
            return 'when'
        elif question_lower.startswith('who'):
            return 'who'
        elif question_lower.startswith('where'):
            return 'where'
        elif question_lower.startswith('why'):
            return 'why'
        elif question_lower.startswith('how'):
            return 'how'
        else:
            return 'other'
    
    def _print_distribution(self, examples: List[Dict]):
        """Print distribution statistics"""
        difficulties = Counter(ex['difficulty'] for ex in examples)
        question_types = Counter(ex['question_type'] for ex in examples)
        
        print(f"\n   Difficulty Distribution:")
        for level, count in difficulties.items():
            print(f"     - {level}: {count} ({count/len(examples)*100:.1f}%)")
        
        print(f"\n   Question Type Distribution:")
        for q_type, count in question_types.most_common():
            print(f"     - {q_type}: {count} ({count/len(examples)*100:.1f}%)")
    
    def save_golden_dataset(
        self,
        golden_dataset: List[Dict],
        path: str,
        format: str = 'json'
    ):
        """
        Save golden dataset to file
        
        Args:
            golden_dataset: List of golden examples
            path: Output file path
            format: 'json' or 'jsonl'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(golden_dataset, f, indent=2, ensure_ascii=False)
        elif format == 'jsonl':
            with open(path, 'w', encoding='utf-8') as f:
                for ex in golden_dataset:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved golden dataset to: {path}")
    
    def load_golden_dataset(self, path: str) -> List[Dict]:
        """
        Load golden dataset from file
        
        Args:
            path: Input file path
            
        Returns:
            List of golden examples
        """
        path = Path(path)
        
        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def validate_dataset(self, golden_dataset: List[Dict]) -> Dict[str, Any]:
        """
        Validate golden dataset quality
        
        Args:
            golden_dataset: List of golden examples
            
        Returns:
            Validation report
        """
        report = {
            'total_examples': len(golden_dataset),
            'valid_examples': 0,
            'issues': []
        }
        
        for i, ex in enumerate(golden_dataset):
            valid = True
            
            if not ex.get('question'):
                report['issues'].append(f"Example {i}: Missing question")
                valid = False
            
            if not ex.get('context'):
                report['issues'].append(f"Example {i}: Missing context")
                valid = False
            
            if not ex.get('answers'):
                report['issues'].append(f"Example {i}: Missing answers")
                valid = False
            
            if valid:
                report['valid_examples'] += 1
        
        report['valid_percentage'] = (report['valid_examples'] / report['total_examples']) * 100
        
        return report



