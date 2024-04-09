from .aggregator import SumAggregator, ConcatAggregator, NeighAggregator, SingleAggregator

Aggregator = {
    'sum': SumAggregator,
    'concat': ConcatAggregator,
    'neigh': NeighAggregator,
    'single': SingleAggregator,
}
