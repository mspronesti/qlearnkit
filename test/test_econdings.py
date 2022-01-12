from numpy.lib.scimath import sqrt
from qlearnkit.encodings import AmplitudeEncoding
import pytest


@pytest.fixture
def classical_data():
    return [
            [1, 1, 1, 1],
            [-1, -1, -1, -1],
            [1/2, -1/2, 0, 0]
           ]


@pytest.fixture
def expected_normalised_data():
    return [
            [1 / 2, 1 / 2, 1 / 2, 1 / 2],
            [-1 / 2, -1 / 2, -1 / 2, -1 / 2],
            [1/sqrt(2), -1/sqrt(2), 0, 0]
           ]


def assert_fidelity(classical_data, tolerance=1.0e-5):
    amp_enc = AmplitudeEncoding()
    normalised_data = amp_enc.encode_dataset(classical_data)
    assert (normalised_data ** 2).sum() == pytest.approx(1.0, tolerance)


def test_amplitude_encoding(classical_data, expected_normalised_data):
    amp_enc = AmplitudeEncoding()
    normalised_data = amp_enc.encode_dataset(classical_data)
    assert (normalised_data == expected_normalised_data).all()
