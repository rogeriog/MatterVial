#!/usr/bin/env python3
"""
Tests for partial-occupancy (disordered) ORB featurization.

The handle_disorder flag feeds occupancy-weighted soft one-hot atom
embeddings (h_i = sum_Z occ_Z * W[Z]) through the ORB atom-embedding
matrix. For ordered structures the soft one-hot is an exact one-hot row,
so results must reproduce stock ORB bit-for-bit.
"""

import unittest
import numpy as np
from pymatgen.core import Structure, Lattice
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import orb_models  # noqa: F401
    from mattervial.featurizers.mlip_models import ORBModelHandler, _soft_onehot
    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False


def _nacl():
    return Structure(Lattice.cubic(5.64), ["Na", "Cl"],
                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def _kcl():
    return Structure(Lattice.cubic(5.64), ["K", "Cl"],
                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def _nakcl(x):
    """Disordered Na(1-x)K(x)Cl at alchemical composition x."""
    return Structure(Lattice.cubic(5.64),
                     [{"Na": 1.0 - x, "K": x}, "Cl"],
                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def _feats_to_vec(features_dict):
    return np.array([features_dict[k] for k in sorted(features_dict)])


@unittest.skipUnless(ORB_AVAILABLE, "orb_models/mattervial not available")
class TestPartialOccupancyORB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.handler_stock = ORBModelHandler(
            device="cpu", handle_disorder=False)
        cls.handler_dis = ORBModelHandler(
            device="cpu", handle_disorder=True)

    # ------------------------------------------------------------------
    def test_soft_onehot_ordered(self):
        """Soft one-hot of an ordered site is an exact one-hot row."""
        S = _soft_onehot(_nacl())
        self.assertEqual(S.shape, (2, 118))
        self.assertEqual(S[0].sum(), 1.0)
        self.assertEqual(S[0, 11], 1.0)  # Na
        self.assertEqual(S[1, 17], 1.0)  # Cl

    def test_soft_onehot_disordered(self):
        """Occupancy fractions are preserved per element."""
        S = _soft_onehot(_nakcl(0.3))
        np.testing.assert_allclose(S[0, 11], 0.7)  # Na
        np.testing.assert_allclose(S[0, 19], 0.3)  # K
        self.assertEqual(S[1, 17], 1.0)

    # ------------------------------------------------------------------
    def test_ordered_endpoint_regression(self):
        """handle_disorder=True on an ordered structure == stock ORB."""
        f_stock = _feats_to_vec(
            self.handler_stock.extract_features_single(_nacl()))
        f_dis = _feats_to_vec(
            self.handler_dis.extract_features_single(_nacl()))
        self.assertTrue(np.all(np.isfinite(f_dis)))
        np.testing.assert_allclose(f_dis, f_stock, rtol=0, atol=1e-5)

    def test_disordered_initial_embedding_linear(self):
        """Pooled initial_atom_embedding interpolates exactly with occupancy."""
        def init_block(handler, s):
            f = handler.extract_features_single(s)
            return np.array([v for k, v in f.items()
                             if "initial_atom_embedding" in k])

        i0 = init_block(self.handler_dis, _nacl())
        i1 = init_block(self.handler_dis, _kcl())
        im = init_block(self.handler_dis, _nakcl(0.5))
        np.testing.assert_allclose(im, 0.5 * (i0 + i1), rtol=0, atol=1e-5)

    def test_disordered_features_finite_and_distinct(self):
        """Midpoint features are finite and differ from both endpoints."""
        f0 = _feats_to_vec(
            self.handler_dis.extract_features_single(_nacl()))
        f1 = _feats_to_vec(
            self.handler_dis.extract_features_single(_kcl()))
        fm = _feats_to_vec(
            self.handler_dis.extract_features_single(_nakcl(0.5)))
        self.assertTrue(np.all(np.isfinite(fm)))
        self.assertFalse(np.allclose(fm, f0, atol=1e-3))
        self.assertFalse(np.allclose(fm, f1, atol=1e-3))
        # Midpoint should lie inside the endpoint distance
        d01 = np.linalg.norm(f1 - f0)
        self.assertLess(np.linalg.norm(fm - f0), d01)
        self.assertLess(np.linalg.norm(fm - f1), d01)

    def test_atomic_batch_disordered(self):
        """Batched per-atom path returns finite (n_atoms, 256) arrays."""
        out = self.handler_dis.extract_atomic_features_batch(
            [_nakcl(0.5), _nacl()], batch_size=8)
        self.assertEqual(len(out), 2)
        for feats, n in zip(out, (2, 2)):
            self.assertEqual(feats.shape, (n, 256))
            self.assertTrue(np.all(np.isfinite(feats)))

    def test_atomic_single_disordered(self):
        """Single per-atom path on a disordered structure."""
        feats = self.handler_dis.extract_atomic_features_single(_nakcl(0.5))
        self.assertEqual(feats.shape, (2, 256))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_disorder_off_preserves_stock_error(self):
        """handle_disorder=False keeps stock behaviour: the ASE adaptor
        rejects disordered structures, so features come back NaN."""
        f_stock = _feats_to_vec(
            self.handler_stock.extract_features_single(_nakcl(0.3)))
        self.assertTrue(np.all(np.isnan(f_stock)))
        # ... while the flag-enabled handler featurizes the same structure fine
        f_dis = _feats_to_vec(
            self.handler_dis.extract_features_single(_nakcl(0.3)))
        self.assertTrue(np.all(np.isfinite(f_dis)))

    def test_vacancy(self):
        """A vacancy (occupancy < 1) produces finite features."""
        svac = Structure(Lattice.cubic(5.64),
                         [{"Na": 0.9}, "Cl"],
                         [[0, 0, 0], [0.5, 0.5, 0.5]])
        f = _feats_to_vec(
            self.handler_dis.extract_features_single(svac))
        self.assertTrue(np.all(np.isfinite(f)))


@unittest.skipUnless(ORB_AVAILABLE, "orb_models/mattervial not available")
class TestPublicAPIs(unittest.TestCase):

    def test_orb_featurizer_flag(self):
        from mattervial.featurizers.structure import ORBFeaturizer
        feats = ORBFeaturizer(device="cpu", handle_disorder=True) \
            .get_features([_nakcl(0.5)])
        self.assertEqual(feats.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(feats.values)))

    def test_atomic_featurizer_flag(self):
        from mattervial.featurizers.atomic import AtomicFeaturizer
        out = AtomicFeaturizer(models=["ORB_v3"], device="cpu",
                               verbose=False,
                               orb_handle_disorder=True) \
            .get_features(_nakcl(0.5))
        feats = out["ORB_v3"]
        self.assertEqual(feats.shape, (2, 256))
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_atomic_featurizer_default_off(self):
        """Default (flag off): disordered structures fail in batch and
        return an empty (0, 256) array, matching stock behaviour."""
        from mattervial.featurizers.atomic import AtomicFeaturizer
        out = AtomicFeaturizer(models=["ORB_v3"], device="cpu",
                               verbose=False).get_features(_nakcl(0.3))
        feats = out["ORB_v3"]
        self.assertEqual(feats.shape, (0, 256))


if __name__ == "__main__":
    unittest.main()
