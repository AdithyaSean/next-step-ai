"""Test data preprocessing pipeline."""

import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.data.generators import StudentDataGenerator

class TestPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor."""
    
    def setUp(self):
        """Set up test cases."""
        self.preprocessor = DataPreprocessor()
        self.generator = StudentDataGenerator(seed=42)
        
        # Generate sample data
        self.sample_data = self.generator.generate_dataset(100)
    
    def test_single_profile_processing(self):
        """Test processing of a single profile."""
        profile = self.sample_data[0]
        processed = self.preprocessor.preprocess_single(profile)
        
        # Check basic structure
        self.assertIsInstance(processed, dict)
        self.assertTrue(any(k.startswith('ol_') for k in processed.keys()))
        
        # Check numeric conversion
        numeric_fields = [v for v in processed.values() 
                        if isinstance(v, (int, float))]
        self.assertTrue(len(numeric_fields) > 0)
    
    def test_fit_transform(self):
        """Test fitting and transforming data."""
        transformed = self.preprocessor.fit_transform(self.sample_data)
        
        # Check output type
        self.assertIsInstance(transformed, pd.DataFrame)
        
        # Check scaling
        grade_cols = [col for col in transformed.columns 
                     if col.startswith(('ol_', 'al_'))
                     and col not in ['ol_total_passed', 'al_stream']]
        if grade_cols:
            grades = transformed[grade_cols].values
            self.assertTrue(np.abs(grades.mean()) < 1e-10)
            self.assertTrue(np.abs(grades.std() - 1) < 1e-10)
    
    def test_categorical_encoding(self):
        """Test encoding of categorical variables."""
        transformed = self.preprocessor.fit_transform(self.sample_data)
        
        # Check stream encoding
        if 'al_stream' in transformed:
            self.assertTrue(transformed['al_stream'].dtype in [np.int32, np.int64])
        
        # Check field encoding
        if 'degree_field' in transformed:
            self.assertTrue(transformed['degree_field'].dtype in [np.int32, np.int64])
    
    def test_save_load(self):
        """Test saving and loading preprocessor."""
        # Fit preprocessor
        self.preprocessor.fit(self.sample_data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save preprocessor
            self.preprocessor.save(tmpdir)
            
            # Load preprocessor
            loaded = DataPreprocessor.load(tmpdir)
            
            # Compare transformations
            original_transform = self.preprocessor.transform(self.sample_data)
            loaded_transform = loaded.transform(self.sample_data)
            
            pd.testing.assert_frame_equal(original_transform, loaded_transform)
    
    def test_invalid_operations(self):
        """Test handling of invalid operations."""
        # Test transform before fit
        with self.assertRaises(ValueError):
            self.preprocessor.transform(self.sample_data)
        
        # Test save before fit
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tmpdir:
                self.preprocessor.save(tmpdir)
        
        # Test loading from non-existent path
        with self.assertRaises(ValueError):
            DataPreprocessor.load("/non/existent/path")

if __name__ == '__main__':
    unittest.main()
