from numpy.lib.scimath import sqrt
from qlearnkit.encodings import AmplitudeEncoding
import pytest
import numpy as np


@pytest.fixture
def classical_data():
    return [
        [1, 1, 1, 1],
        [-1, -1, -1, -1],
        [1 / 2, -1 / 2, 0, 0]
    ]


@pytest.fixture
def expected_normalised_data():
    return [
        [1 / 2, 1 / 2, 1 / 2, 1 / 2],
        [-1 / 2, -1 / 2, -1 / 2, -1 / 2],
        [1 / sqrt(2), -1 / sqrt(2), 0, 0]
    ]


def test_fidelity(classical_data, tolerance=1.0e-5):
    amp_enc = AmplitudeEncoding(n_features=4)
    normalised_data = np.array([amp_enc.state_vector(data)
                                for data in classical_data
                                ])

    assert (normalised_data ** 2).sum(axis=1) == pytest.approx(1.0, tolerance)


def test_amplitude_encoding(classical_data, expected_normalised_data):
    amp_enc = AmplitudeEncoding(n_features=4)
    normalised_data = np.array([amp_enc.state_vector(data)
                                for data in classical_data
                                ])
    assert (normalised_data == expected_normalised_data).all()
