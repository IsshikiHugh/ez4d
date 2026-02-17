import torch
from collections import defaultdict


class EvaluatorBase():
    ''' To use this class, you should inherit it and implement the `eval` method. '''
    def __init__(self):
        self.aggregator = defaultdict(list)

    def eval(self, **kwargs):
        ''' Evaluate the metrics on the data. '''
        raise NotImplementedError

    def get_results(self, chosen_metric=None):
        ''' Get the current mean results. '''
        # Only chosen metrics will be compacted and returned.
        compacted = self._compact_aggregator(chosen_metric)
        ret = {}
        for k, v in compacted.items():
            ret[k] = v.mean(dim=0).item()
        return ret

    def _compact_aggregator(self, chosen_metric=None):
        ''' Compact the aggregator list and return the compacted results. '''
        ret = {}
        for k, v in self.aggregator.items():
            # Only chosen metrics will be compacted.
            if chosen_metric is None or k in chosen_metric:
                ret[k] = torch.cat(v, dim=0)
                self.aggregator[k] = [ret[k]]
        return ret
