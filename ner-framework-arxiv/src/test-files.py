"""
Unit tests for text preprocessing modules
"""

import unittest
from src.data_processing import text_preprocessing


class TestTextPreprocessing(unittest.TestCase):
    
    def test_clean_text(self):
        """Test text cleaning function"""
        # Test LaTeX removal
        text_with_latex = "This is text $$ \\alpha + \\beta $$ more text"
        cleaned = text_preprocessing.clean_text(text_with_latex)
        self.assertNotIn("$$", cleaned)
        self.assertNotIn("\\alpha", cleaned)
        
        # Test HTML tag removal
        text_with_html = "Text <sup>1</sup> and <table>data</table> here"
        cleaned = text_preprocessing.clean_text(text_with_html)
        self.assertNotIn("<sup>", cleaned)
        self.assertNotIn("<table>", cleaned)
    
    def test_sentence_split(self):
        """Test sentence splitting function"""
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = text_preprocessing.fun_sentece_split(text)
        self.assertEqual(len(sentences), 3)
        
        # Test long sentence splitting
        long_text = "A" * 350  # Sentence longer than 300 chars
        sentences = text_preprocessing.fun_sentece_split(long_text)
        self.assertTrue(all(len(s) <= 350 for s in sentences))
    
    def test_split_by_abstract(self):
        """Test abstract splitting"""
        text = "Title and authors here\nAbstract\nThis is the abstract. Introduction follows."
        before, after = text_preprocessing.split_text_by_abstract(text)
        self.assertIn("Title and authors", before)
        self.assertIn("Abstract", after)
        self.assertIn("Introduction", after)
    
    def test_normalize_text(self):
        """Test text normalization"""
        text = "Hello WORLD! 123 #special @chars"
        normalized = text_preprocessing.normalize_text(text)
        self.assertEqual(normalized, "hello world 123 special chars")
        
        # Test accent removal
        text_with_accents = "café naïve"
        normalized = text_preprocessing.normalize_text(text_with_accents)
        self.assertEqual(normalized, "cafe naive")
    
    def test_fuzzy_match(self):
        """Test fuzzy string matching"""
        # Exact match
        self.assertTrue(text_preprocessing.fuzzy_match("hello", "hello", 0.8))
        
        # Close match
        self.assertTrue(text_preprocessing.fuzzy_match("hello world", "helo world", 0.8))
        
        # No match
        self.assertFalse(text_preprocessing.fuzzy_match("hello", "goodbye", 0.8))


if __name__ == '__main__':
    unittest.main()