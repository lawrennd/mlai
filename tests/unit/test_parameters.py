"""Unit tests for Paramz-lite parameters (CIP-000F)."""

import unittest

import numpy as np

import mlai
from mlai.parameters import POSITIVE, Param, ParamGroup, format_param_table


class TestParam(unittest.TestCase):
    def test_positive_pack_unpack_roundtrip(self):
        p = Param('lengthscale', 2.5, transform=POSITIVE, path='gp.rbf.lengthscale')
        u = p.pack()
        np.testing.assert_allclose(u, [np.log(2.5)])
        p.unpack([np.log(3.0)])
        self.assertAlmostEqual(p.value, 3.0)

    def test_positive_rejects_nonpositive(self):
        p = Param('sigma2', 1.0, transform=POSITIVE)
        with self.assertRaises(ValueError):
            p.value = -1.0
        with self.assertRaises(ValueError):
            Param('sigma2', -0.1, transform=POSITIVE).pack()

    def test_unconstrained_array_pack(self):
        Z = np.array([[0.0], [1.0], [2.0]])
        p = Param('Z', Z, transform=None, path='sparse_gp.Z')
        np.testing.assert_array_equal(p.pack(), Z.ravel())
        p.unpack(np.array([0.5, 1.5, 2.5]))
        np.testing.assert_allclose(p.value, np.array([[0.5], [1.5], [2.5]]))

    def test_scale_natural_gradient(self):
        p = Param('variance', 4.0, transform=POSITIVE)
        g_nat = np.array([0.5])
        g_unc = p.scale_natural_gradient(g_nat)
        np.testing.assert_allclose(g_unc, [2.0])

    def test_value_shape_mismatch_and_scalar_reshape(self):
        p = Param('Z', np.ones((2, 1)), transform=None)
        with self.assertRaises(ValueError):
            p.value = np.ones(3)
        # size-1 reshape path: (1,) storage accepts a scalar
        q = Param('b', np.array([2.0]), transform=None)
        q.value = 3.0
        np.testing.assert_allclose(q.value, [3.0])
        # 0-d natural assignment via setter (arr.ndim == 0 branch)
        r = Param('s', 1.5, transform=None)
        r.value = np.float64(2.5)
        self.assertAlmostEqual(r.value, 2.5)

    def test_unpack_and_grad_size_errors(self):
        p = Param('Z', np.ones(2), transform=None)
        with self.assertRaises(ValueError):
            p.unpack([1.0])
        with self.assertRaises(ValueError):
            p.scale_natural_gradient([1.0])

    def test_display_value_large_array(self):
        p = Param('Z', np.arange(6.0).reshape(3, 2), transform=None)
        self.assertEqual(p.display_value(), '(3, 2)')


class TestParamGroup(unittest.TestCase):
    def test_pack_unpack_and_display(self):
        group = ParamGroup(name='gp')
        group.add(Param('sigma2', 0.1, transform=POSITIVE, path='gp.sigma2'))
        group.add(
            Param('lengthscale', 1.0, transform=POSITIVE, path='gp.rbf.lengthscale')
        )
        packed = group.pack()
        np.testing.assert_allclose(packed, [np.log(0.1), np.log(1.0)])
        group.unpack([np.log(0.2), np.log(2.0)])
        self.assertAlmostEqual(group._params[0].value, 0.2)
        self.assertAlmostEqual(group._params[1].value, 2.0)

        text = repr(group)
        self.assertIn('gp.sigma2', text)
        self.assertIn('gp.rbf.lengthscale', text)
        self.assertIn('+ve', text)
        self.assertIn('Constraint', text)

        html = group._repr_html_()
        self.assertIn('<table>', html)
        self.assertIn('gp.sigma2', html)
        self.assertIn('+ve', html)

    def test_scale_natural_gradients_mixed(self):
        group = ParamGroup()
        group.add(Param('sigma2', 2.0, transform=POSITIVE, path='s'))
        group.add(Param('Z', np.ones(2), transform=None, path='Z'))
        natural = np.array([1.0, 0.3, 0.4])
        unc = group.scale_natural_gradients(natural)
        np.testing.assert_allclose(unc, [2.0, 0.3, 0.4])

    def test_format_empty(self):
        text = format_param_table([])
        self.assertIn('Name', text)

    def test_iter_len_empty_pack_and_errors(self):
        empty = ParamGroup()
        self.assertEqual(len(empty), 0)
        self.assertEqual(list(empty), [])
        np.testing.assert_array_equal(empty.pack(), np.zeros(0))

        group = ParamGroup()
        group.add(Param('a', 1.0, transform=POSITIVE))
        self.assertEqual(len(group), 1)
        self.assertEqual([p.name for p in group], ['a'])
        with self.assertRaises(TypeError):
            group.add('not-a-param')
        with self.assertRaises(ValueError):
            group.scale_natural_gradients(np.ones(2))
        with self.assertRaises(ValueError):
            group.unpack(np.ones(2))


class TestGPParamzLite(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.randn(8, 1)
        self.y = np.sin(self.X[:, 0]) + 0.05 * rng.randn(8)
        self.kern = mlai.Kernel(mlai.eq_cov, lengthscale=1.2, variance=0.8)
        self.gp = mlai.GP(self.X, self.y, sigma2=0.05, kernel=self.kern)

    def test_roundtrip_and_positivity(self):
        p0 = self.gp.parameters.copy()
        self.gp.parameters = p0 + 0.1
        self.assertGreater(self.gp.sigma2, 0.0)
        for name in self.gp._scalar_kernel_param_names():
            self.assertGreater(float(self.gp.kernel.parameters[name]), 0.0)
        self.gp.parameters = p0
        np.testing.assert_allclose(self.gp.parameters, p0)

    def test_repr_contains_names(self):
        text = repr(self.gp)
        self.assertIn('gp.sigma2', text)
        self.assertIn('+ve', text)
        self.assertIn('lengthscale', text)
        html = self.gp._repr_html_()
        self.assertIn('gp.sigma2', html)

    def test_natural_attributes_still_work(self):
        self.gp.sigma2 = 0.2
        self.gp.kernel.parameters['lengthscale'] = 3.0
        self.gp.update_kernel_matrix()
        self.assertAlmostEqual(self.gp.sigma2, 0.2)
        # packed log space reflects natural values
        np.testing.assert_allclose(self.gp.parameters[0], np.log(0.2))

    def test_kernel_display_name_and_shortname(self):
        named = mlai.Kernel(mlai.eq_cov, name='rbf', lengthscale=1.0, variance=1.0)
        gp_named = mlai.GP(self.X, self.y, sigma2=0.1, kernel=named)
        self.assertEqual(gp_named._kernel_display_name(), 'rbf')
        self.assertIn('gp.rbf.', repr(gp_named))

        short = mlai.Kernel(
            mlai.eq_cov, shortname='eq', lengthscale=1.0, variance=1.0
        )
        gp_short = mlai.GP(self.X, self.y, sigma2=0.1, kernel=short)
        self.assertEqual(gp_short._kernel_display_name(), 'eq')
        self.assertIn('gp.eq.', repr(gp_short))


class TestSparseGPParamzLite(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.X = np.linspace(-2, 2, 12)[:, None]
        self.y = np.sin(self.X[:, 0]) + 0.05 * rng.randn(12)
        self.Z = np.linspace(-2, 2, 4)[:, None]
        self.kern = mlai.Kernel(mlai.eq_cov, lengthscale=1.0, variance=1.0)
        self.model = mlai.SparseGP(
            self.X, self.y, self.Z, sigma2=0.1, kernel=self.kern, optimize_Z=True
        )

    def test_roundtrip_layout(self):
        p0 = self.model.parameters.copy()
        self.assertEqual(p0.size, 1 + 2 + self.Z.size)
        self.model.parameters = p0
        np.testing.assert_allclose(self.model.parameters, p0)
        self.assertGreater(self.model.sigma2, 0.0)

    def test_repr_includes_Z(self):
        text = repr(self.model)
        self.assertIn('sparse_gp.sigma2', text)
        self.assertIn('sparse_gp.Z', text)
        self.assertIn('+ve', text)

    def test_expected_size_html_and_kernel_names(self):
        self.assertEqual(
            self.model._expected_parameter_size(), self.model.param_group().size
        )
        html = self.model._repr_html_()
        self.assertIn('<table>', html)
        self.assertIn('sparse_gp.sigma2', html)

        named = mlai.Kernel(mlai.eq_cov, name='rbf', lengthscale=1.0, variance=1.0)
        m_named = mlai.SparseGP(
            self.X, self.y, self.Z, sigma2=0.1, kernel=named, optimize_Z=False
        )
        self.assertEqual(m_named._kernel_display_name(), 'rbf')
        self.assertIn('sparse_gp.rbf.', repr(m_named))

        short = mlai.Kernel(
            mlai.eq_cov, shortname='eq', lengthscale=1.0, variance=1.0
        )
        m_short = mlai.SparseGP(
            self.X, self.y, self.Z, sigma2=0.1, kernel=short, optimize_Z=False
        )
        self.assertEqual(m_short._kernel_display_name(), 'eq')
        self.assertIn('sparse_gp.eq.', repr(m_short))


if __name__ == '__main__':
    unittest.main()
