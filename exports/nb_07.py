
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/07_batchnorm_my_reimplementation.ipynb

from exports.nb_06 import *

def init_cnn_(m, f): # m stands for module, f represents the initialization kind
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, 'bias', None) is not None: m.bias.data.zero_()
    for l in m.children(): init_cnn_(l, f)

def init_cnn(m, uniform=False):
    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
    init_cnn_(m, f)

def get_learn_run(n_outs, data, lr, layer, cbs=None, opt_func=None, uniform=False, **kwargs):
    model = get_cnn_model(data, n_outs, layer, **kwargs)
    init_cnn(model, uniform=uniform)
    return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)

def conv_layer(n_in, n_out, ks=3, stride=2, bn=True, **kwargs):
    layers = [nn.Conv2d(n_in, n_out, ks, padding=ks//2, stride=stride, bias=not bn),
              GeneralRelu(**kwargs)]
    if bn: layers.append(nn.BatchNorm2d(n_out, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)

class RunningBatchNorm(nn.Module):
    def __init__(self, n_out, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones (n_out,1,1))
        self.adds  = nn.Parameter(torch.zeros(n_out,1,1))
        self.register_buffer('sums', torch.zeros(1,n_out,1,1))
        self.register_buffer('sqrs', torch.zeros(1,n_out,1,1))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('factor',tensor(0.))
        self.register_buffer('offset',tensor(0.))
        self.batch = 0

    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        s    = x    .sum(dims, keepdim=True)
        ss   = (x*x).sum(dims, keepdim=True)
        c    = s.new_tensor(x.numel()/nc)
        mom1 = s.new_tensor(1 - (1 - self.mom) / math.sqrt(bs - 1))
        self.sums .lerp_(s , mom1)
        self.sqrs .lerp_(ss, mom1)
        self.count.lerp_(c , mom1)
        self.batch += bs
        means = self.sums / self.count
        variances = (self.sqrs / self.count).sub_(means * means)
        if bool(self.batch < 20): variances.clamp_min_(0.01)
        self.factor = self.mults / (variances + self.eps).sqrt()
        self.offset = self.adds - means * self.factor

    def forward(self, x):
        if self.training: self.update_stats(x)
        return x * self.factor + self.offset