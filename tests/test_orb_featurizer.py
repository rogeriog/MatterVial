#!/usr/bin/env python3
"""
Comprehensive unit tests for ORB featurizer integration.
"""

import unittest
import pandas as pd
import numpy as np
from pymatgen.core import Structure, Lattice
import warnings
import os
import sys

# Add the parent directory to the path to import mattervial
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestORBFeaturizer(unittest.TestCase):
    """Test suite for ORB featurizer functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create test structures
        cls.structures = cls._create_test_structures()
        cls.structure_strings = [s.to_json() for s in cls.structures]
        cls.df = pd.DataFrame({'structure': cls.structure_strings})
        
        # Try to import mattervial
        try:
            import mattervial.featurizers
            cls.mattervial_available = True
            cls.orb_featurizer = getattr(mattervial.featurizers, 'orb_v3', None)
        except ImportError:
            cls.mattervial_available = False
            cls.orb_featurizer = None
    
    @staticmethod
    def _create_test_structures():
        """Create test structures for testing."""
        structures = []
        
        # Simple cubic structure (Fe)
        lattice1 = Lattice.cubic(2.87)
        structure1 = Structure(lattice1, ["Fe"], [[0, 0, 0]])
        structures.append(structure1)
        
        # Binary compound (NaCl-like)
        lattice2 = Lattice.cubic(5.64)
        structure2 = Structure(lattice2, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        structures.append(structure2)
        
        # Ternary compound (perovskite-like)
        lattice3 = Lattice.cubic(4.0)
        structure3 = Structure(lattice3, ["Ca", "Ti", "O", "O", "O"], 
                              [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        structures.append(structure3)
        
        return structures
    
    def test_import_availability(self):
        """Test that mattervial can be imported."""
        self.assertTrue(self.mattervial_available, "mattervial.featurizers should be importable")
    
    def test_orb_featurizer_instance(self):
        """Test that ORB featurizer instance exists."""
        if not self.mattervial_available:
            self.skipTest("mattervial not available")
        
        import mattervial.featurizers
        # The instance might be None if dependencies are missing, but it should exist
        self.assertTrue(hasattr(mattervial.featurizers, 'orb_v3'), "orb_v3 should be available")
    
    def test_orb_featurizer_interface(self):
        """Test that ORB featurizer has the correct interface."""
        if not self.mattervial_available or self.orb_featurizer is None:
            self.skipTest("ORB featurizer not available")
        
        # Test that it has get_features method
        self.assertTrue(hasattr(self.orb_featurizer, 'get_features'), 
                       "ORB featurizer should have get_features method")
        self.assertTrue(callable(self.orb_featurizer.get_features), 
                       "get_features should be callable")
    
    def test_orb_feature_extraction(self):
        """Test ORB feature extraction with real structures."""
        if not self.mattervial_available or self.orb_featurizer is None:
            self.skipTest("ORB featurizer not available")
        
        try:
            # Test feature extraction
            features = self.orb_featurizer.get_features(self.df)
            
            # Verify output format
            self.assertIsInstance(features, pd.DataFrame, "Features should be a DataFrame")
            self.assertEqual(features.shape[0], len(self.structures), 
                           f"Should have {len(self.structures)} rows")
            self.assertGreater(features.shape[1], 0, "Should have at least one feature column")
            
            # Verify feature names follow ORB pattern
            feature_names = features.columns.tolist()
            orb_features = [name for name in feature_names if name.startswith('ORB_v3_')]
            self.assertGreater(len(orb_features), 0, "Should have ORB_v3_ prefixed features")
            
            # Verify no NaN values for valid structures
            self.assertFalse(features.isnull().all().any(), 
                           "Should not have columns with all NaN values for valid structures")
            
        except ImportError as e:
            self.skipTest(f"ORB dependencies not available: {e}")
    
    def test_input_format_compatibility(self):
        """Test different input formats."""
        if not self.mattervial_available or self.orb_featurizer is None:
            self.skipTest("ORB featurizer not available")
        
        try:
            # Test with DataFrame
            df_features = self.orb_featurizer.get_features(self.df)
            self.assertIsInstance(df_features, pd.DataFrame)
            
            # Test with Series
            series = pd.Series(self.structure_strings)
            series_features = self.orb_featurizer.get_features(series)
            self.assertIsInstance(series_features, pd.DataFrame)
            
            # Test with list of structures
            list_features = self.orb_featurizer.get_features(self.structures)
            self.assertIsInstance(list_features, pd.DataFrame)
            
            # All should produce the same results
            pd.testing.assert_frame_equal(df_features, series_features)
            pd.testing.assert_frame_equal(df_features, list_features)
            
        except ImportError as e:
            self.skipTest(f"ORB dependencies not available: {e}")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        if not self.mattervial_available or self.orb_featurizer is None:
            self.skipTest("ORB featurizer not available")
        
        try:
            # Test with invalid structure format
            invalid_df = pd.DataFrame({'structure': ['invalid_structure', 'another_invalid']})
            
            # This should either handle gracefully or raise a clear error
            try:
                features = self.orb_featurizer.get_features(invalid_df)
                # If it succeeds, features should contain NaN values
                self.assertTrue(features.isnull().any().any(), 
                              "Invalid structures should produce NaN features")
            except Exception as e:
                # If it fails, the error should be informative
                self.assertIsInstance(e, (ValueError, TypeError, ImportError))
                
        except ImportError as e:
            self.skipTest(f"ORB dependencies not available: {e}")
    
    def test_feature_consistency(self):
        """Test that feature extraction is consistent across runs."""
        if not self.mattervial_available or self.orb_featurizer is None:
            self.skipTest("ORB featurizer not available")
        
        try:
            # Extract features twice
            features1 = self.orb_featurizer.get_features(self.df)
            features2 = self.orb_featurizer.get_features(self.df)
            
            # Results should be identical (or very close due to floating point)
            pd.testing.assert_frame_equal(features1, features2, check_exact=False, rtol=1e-10)
            
        except ImportError as e:
            self.skipTest(f"ORB dependencies not available: {e}")
    
    def test_lazy_loading_behavior(self):
        """Test that lazy loading works correctly."""
        if not self.mattervial_available:
            self.skipTest("mattervial not available")
        
        import mattervial.featurizers
        
        # Test that we can get featurizer status
        if hasattr(mattervial.featurizers, 'get_available_featurizers'):
            status = mattervial.featurizers.get_available_featurizers()
            self.assertIsInstance(status, dict, "Status should be a dictionary")
            self.assertIn('orb_v3', status, "orb_v3 should be in status")
        
        if hasattr(mattervial.featurizers, 'get_featurizer_errors'):
            errors = mattervial.featurizers.get_featurizer_errors()
            self.assertIsInstance(errors, dict, "Errors should be a dictionary")


class TestFeaturizerIntegration(unittest.TestCase):
    """Test integration with other featurizers."""

    def test_selective_availability(self):
        """Test that featurizers are selectively available based on dependencies."""
        try:
            import mattervial.featurizers

            # Check that the module imports successfully
            self.assertTrue(hasattr(mattervial.featurizers, '__all__'))

            # Check that utility functions are available
            if hasattr(mattervial.featurizers, 'get_available_featurizers'):
                status = mattervial.featurizers.get_available_featurizers()
                self.assertIsInstance(status, dict)

                # At least some featurizers should be listed
                self.assertGreater(len(status), 0, "Should have at least one featurizer listed")

                # Check that new multi-layer instances are included
                expected_new = ['mvl_all', 'adj_megnet_all']
                for name in expected_new:
                    self.assertIn(name, status, f"New instance '{name}' should be in available featurizers")

                # Verify that multi-layer instances are configured correctly
                if hasattr(mattervial.featurizers, 'mvl_all') and mattervial.featurizers.mvl_all is not None:
                    self.assertEqual(mattervial.featurizers.mvl_all.layers, ['layer32', 'layer16'])
                if hasattr(mattervial.featurizers, 'adj_megnet_all') and mattervial.featurizers.adj_megnet_all is not None:
                    self.assertEqual(mattervial.featurizers.adj_megnet_all.layers, ['layer32', 'layer16'])

        except ImportError as e:
            self.fail(f"mattervial.featurizers should be importable: {e}")

    def test_multi_layer_featurizer_interfaces(self):
        """Test that multi-layer featurizers have correct interfaces."""
        try:
            import mattervial.featurizers

            # Test MVLFeaturizer class
            mvl_class = mattervial.featurizers.MVLFeaturizer

            # Test default initialization (should use both layers)
            mvl_default = mvl_class()
            self.assertEqual(mvl_default.layers, ['layer32', 'layer16'],
                           "Default MVLFeaturizer should use both layers")

            # Test single layer initialization
            mvl_single = mvl_class(layers='layer32')
            self.assertEqual(mvl_single.layers, ['layer32'],
                           "Single layer MVLFeaturizer should work")

            # Test list initialization
            mvl_list = mvl_class(layers=['layer16'])
            self.assertEqual(mvl_list.layers, ['layer16'],
                           "List initialization should work")

            # Test AdjacentMEGNetFeaturizer class
            adj_class = mattervial.featurizers.AdjacentMEGNetFeaturizer

            # Test default initialization
            adj_default = adj_class()
            self.assertEqual(adj_default.layers, ['layer32', 'layer16'],
                           "Default AdjacentMEGNetFeaturizer should use both layers")

            # Test single layer initialization
            adj_single = adj_class(layers='layer16')
            self.assertEqual(adj_single.layers, ['layer16'],
                           "Single layer AdjacentMEGNetFeaturizer should work")

        except ImportError as e:
            self.skipTest(f"mattervial not available: {e}")

    def test_backward_compatibility(self):
        """Test that existing single-layer instances still work."""
        try:
            import mattervial.featurizers

            # Test that old instances exist and have correct configuration
            if hasattr(mattervial.featurizers, 'mvl32') and mattervial.featurizers.mvl32 is not None:
                mvl32 = mattervial.featurizers.mvl32
                self.assertEqual(mvl32.layers, ['layer32'],
                               "mvl32 should be configured for layer32 only")

            if hasattr(mattervial.featurizers, 'mvl16') and mattervial.featurizers.mvl16 is not None:
                mvl16 = mattervial.featurizers.mvl16
                self.assertEqual(mvl16.layers, ['layer16'],
                               "mvl16 should be configured for layer16 only")

            if hasattr(mattervial.featurizers, 'adj_megnet') and mattervial.featurizers.adj_megnet is not None:
                adj_megnet = mattervial.featurizers.adj_megnet
                self.assertEqual(adj_megnet.layers, ['layer32'],
                               "adj_megnet should be configured for layer32 only")

            if hasattr(mattervial.featurizers, 'adj_megnet_layer16') and mattervial.featurizers.adj_megnet_layer16 is not None:
                adj_megnet16 = mattervial.featurizers.adj_megnet_layer16
                self.assertEqual(adj_megnet16.layers, ['layer16'],
                               "adj_megnet_layer16 should be configured for layer16 only")

        except ImportError as e:
            self.skipTest(f"mattervial not available: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
