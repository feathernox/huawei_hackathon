import torch
from torch.nn import functional as F
from poutyne.framework.callbacks import Callback


class LosswiseSessionHandler:
    def __init__(self, api_key, tag='', max_iter=None, params=None):
        import losswise
        losswise.set_api_key(api_key)
        self._session = losswise.Session(tag=tag, max_iter=max_iter,
                                         params=params if params is not None else {}, track_git=False)
        self._graphs = dict()

    def create_graph(self, graph_name, xlabel='', ylabel='', kind=None, display_interval=1):
        assert isinstance(graph_name, str)
        if graph_name not in self._graphs:
            self._graphs[graph_name] = self._session.graph(title=graph_name, xlabel=xlabel, ylabel=ylabel, kind=kind,
                                                           display_interval=display_interval)
        return self._graphs[graph_name]

    def __getitem__(self, graph_name):
        if graph_name not in self._graphs:
            self.create_graph(graph_name)
        return self._graphs[graph_name]

    def done(self):
        self._session.done()


class LosswiseCallback(Callback):

    def __init__(self,
                 api_key: str = None,
                 losswise_session: LosswiseSessionHandler = None,
                 prefix='',
                 tag='my awesome DL',
                 keep_open=False,
                 training_params=None,
                 tracked_params=[]):
        super().__init__()
        assert (api_key is None) ^ (losswise_session is None)

        if losswise_session is None:
            self._session = LosswiseSessionHandler(api_key, tag=tag, params=training_params)
        else:
            self._session = losswise_session
        self._keep_open = keep_open
        self.prefix = prefix
        self.tracked_params = tracked_params
        self.steps_elapsed = 0

    def on_train_begin(self, logs):
        self.metrics = ['loss'] + self.model.metrics_names + self.tracked_params
        self._session.create_graph('loss', xlabel='epoch', kind='min')
        self._session.create_graph('learning rate', xlabel='batch')
        for name in self.model.metrics_names:
            self._session.create_graph(name, xlabel='epoch')
            self._session.create_graph(name + '_iter', xlabel='batch',
                                       display_interval=self.params['steps'])

    def on_train_end(self, logs):
        if not self._keep_open:
            self._session.done()

    def on_epoch_end(self, epoch, logs):
        for name in self.metrics:
            graph_args = dict()
            if name in logs:
                graph_args[self.prefix + name] = logs[name]
            if 'val_' + name in logs:
                graph_args[self.prefix + 'val_' + name] = logs['val_' + name]
            self._session[name].append(epoch, graph_args)

    def on_batch_end(self, batch, logs):
        self.steps_elapsed += 1
        for name in self.metrics:
            if name in logs:
                self._session[name + '_iter'].append(self.steps_elapsed, {self.prefix + name: logs[name]})

        if hasattr(self.model.optimizer, 'get_lr'):
            learning_rates = [self.model.optimizer.get_lr()[0]]
        else:
            learning_rates = (param_group['lr'] for param_group in self.model.optimizer.param_groups)
        for group_idx, lr in enumerate(learning_rates):
            self._session['learning rate'].append(self.steps_elapsed, {'lr_param_group_' + str(group_idx): lr})
