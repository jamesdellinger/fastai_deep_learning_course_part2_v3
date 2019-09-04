
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/06_cuda_cnn_hooks_init_my_reimplementation.ipynb

from exports.nb_05b import *
torch.set_num_threads(2)

def normalize_to(train, valid):
    m, s = train.mean(), train.std()
    return normalize(train, m, s), normalize(valid, m, s)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def flatten(x): return x.view(x.shape[0], -1)


# Less flexible but convenient
class CudaCallback(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb, self.run.yb = self.xb.cuda(), self.yb.cuda()

class BatchTransformXCallback(Callback):
    _order=2
    def __init__(self, tfm): self.tfm = tfm
    def begin_batch(self): self.run.xb = self.tfm(self.xb)

# Helper function to reshape a batch of input images so that
# they're organized as PyTorch expects: (batch size, channels, height, width)
def view_tfm(*size):
    def _inner(x): return x.view(*((-1,) + size))
    return _inner

def get_runner(model, data, lr=0.6, cbs=None, opt_func=None, loss_func = F.cross_entropy):
    if opt_func is None: opt_func = optim.SGD
    opt = opt_func(model.parameters(), lr=lr)
    learn = Learner(model, opt, loss_func, data)
    return learn, Runner(cb_funcs=listify(cbs))

def children(l): return list(l.children())

class ForwardHook():
    def __init__(self, l, f): self.hook = l.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, 'stats'): hook.stats = ([], [])
    means, stds = hook.stats
    if mod.training:
        means.append(outp.data.mean())
        stds .append(outp.data.std())

class ListContainer():
    def __init__(self, items): self.items = listify(items)
    def __getitem__(self, idx):
        # 1., 2. Indexing via a single index or a slice
        try: return self.items[idx]
        except TypeError:
            # 4. If indexing via a mask
            if isinstance(idx[0], bool):
                assert len(idx)==len(self)
                return [o for m,o in zip(idx,self.items) if m]
            # 3. If indexing via a list of indices
            return [self.items[i] for i in idx]

    # Do other useful list operations.
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items) # to do things like 'for x in ...'
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self) > 10: res = res[:-1] + '...]'
        return res

from torch.nn import init

class ForwardHooks(ListContainer):
    def __init__(self, ms, f): super().__init__([ForwardHook(m, f) for m in ms])
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()
    def __del__(self): self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self: h.remove()

# Convenience function for labelling our plots
def get_labels(ax, metric, title):
    return ax.set_xlabel('Iterations'), ax.set_ylabel(f'Activation Output {metric}'), ax.set_title(title)

class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, max_val=None):
        super().__init__()
        self.leak, self.sub, self.max_val = leak, sub, max_val

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x.sub_(self.sub)
        if self.max_val is not None: x.clamp_max_(self.max_val)
        return x

def conv_layer(n_in, n_out, ks=3, stride=2, **kwargs):
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, ks, padding=ks//2, stride=stride),
        GeneralRelu(**kwargs)
    )

def get_cnn_layers(data, n_outs, layer, **kwargs):
    n_outs = [1] + n_outs
    return [layer(n_outs[i], n_outs[i+1], 5 if i==0 else 3, **kwargs)
            for i in range(len(n_outs) - 1)] + [
        nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(n_outs[-1], data.c)]

def get_cnn_model(data, n_outs, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(data, n_outs, layer, **kwargs))

def init_cnn(m, uniform=False):
    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
    for l in m:
        if isinstance(l, nn.Sequential):
            f(l[0].weight, a = 0.1)
            l[0].bias.data.zero_()

def get_learn_run(n_outs, data, lr, layer, cbs=None, opt_func=None, uniform=False, **kwargs):
    model = get_cnn_model(data, n_outs, layer, **kwargs)
    init_cnn(model, uniform=uniform)
    return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)

from IPython.display import display, Javascript
def nb_auto_export():
    display(Javascript("""{
const ip = IPython.notebook
if (ip) {
    ip.save_notebook()
    console.log('a')
    const s = `!python notebook2script_my_reimplementation.py ${ip.notebook_name}`
    if (ip.kernel) { ip.kernel.execute(s) }
}
}"""))