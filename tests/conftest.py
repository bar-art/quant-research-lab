import pytest

from quant_research_lab.data import synthetic_dataset
from quant_research_lab.models import ResearchConfig
from quant_research_lab.research import run_research


@pytest.fixture(scope="session")
def dataset():
    return synthetic_dataset(13, 100)


@pytest.fixture(scope="session")
def config():
    return ResearchConfig(
        training_sessions=40, test_sessions=20, windows=(3, 5, 8), scenario_paths=30
    )


@pytest.fixture(scope="session")
def result(dataset, config):
    return run_research(dataset, config)
