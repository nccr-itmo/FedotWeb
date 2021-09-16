def test_quality_endpoint(client):
    case_id = 'scoring'
    quality_plot = client.get(f'api/analytics/quality/{case_id}').json
    assert 'series' in quality_plot
    assert 'options' in quality_plot


def test_results_endpoint(client):
    pipeline_id = 'best_scoring_pipeline'
    case_id = 'scoring'
    results_plot = client.get(f'api/analytics/results/{case_id}/{pipeline_id}').json

    assert 'series' in results_plot
    assert 1 <= len(results_plot['series']) <= 3
    assert 'options' in results_plot


def test_population_endpoint(client):
    case_id = 'scoring'
    results_plot = client.get(f'api/analytics/generations/{case_id}/geno').json

    assert 'series' in results_plot

    results_plot = client.get(f'api/analytics/generations/{case_id}/pheno').json

    assert 'series' in results_plot
